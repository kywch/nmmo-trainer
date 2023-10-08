# pylint: disable=bad-builtin, no-member, protected-access
import argparse
import logging
import os
import time
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from itertools import cycle

import numpy as np
import torch
import pandas as pd

from nmmo.render.replay_helper import FileReplayHelper

import pufferlib
from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import pufferlib.policy_ranker
import pufferlib.utils

import environment

from reinforcement_learning import config, clean_pufferl

def setup_policy_store(policy_store_dir):
    # CHECK ME: can be custom models with different architectures loaded here?
    if not os.path.exists(policy_store_dir):
        raise ValueError("Policy store directory does not exist")
    if os.path.exists(os.path.join(policy_store_dir, "trainer.pt")):
        raise ValueError("Policy store directory should not contain trainer.pt")
    logging.info("Using policy store from %s", policy_store_dir)
    policy_store = DirectoryPolicyStore(policy_store_dir)
    return policy_store

def save_replays(policy_store_dir, save_dir):
    # load the checkpoints into the policy store
    policy_store = setup_policy_store(policy_store_dir)
    num_policies = len(policy_store._all_policies())

    # setup the replay path
    save_dir = os.path.join(save_dir, policy_store_dir)
    os.makedirs(save_dir, exist_ok=True)
    logging.info("Replays will be saved to %s", save_dir)

    # Use 1 env and 1 buffer for replay generation
    # TODO: task-condition agents when generating replays
    args = SimpleNamespace(**config.Config.asdict())
    args.num_envs = 1
    args.num_buffers = 1
    args.use_serial_vecenv = True
    args.learner_weight = 0  # evaluate mode
    args.selfplay_num_policies = num_policies + 1
    args.early_stop_agent_num = 0  # run the full episode
    args.resilient_population = 0  # no resilient agents

    # NOTE: This creates a dummy learner agent. Is it necessary?
    from reinforcement_learning import policy  # import your policy
    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
        )
        return cleanrl.Policy(learner_policy)

    # Setup the evaluator. No training during evaluation
    evaluator = clean_pufferl.CleanPuffeRL(
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        vectorization=Serial,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.selfplay_num_policies,
        policy_store=policy_store,
        data_dir=save_dir,
    )

    # Load the policies into the policy pool
    evaluator.policy_pool.update_policies({
        p.name: p.policy(
            policy_args=[evaluator.buffers[0]], 
            device=evaluator.device
        ) for p in list(policy_store._all_policies().values())
    })

    # Set up the replay helper
    o, r, d, i = evaluator.buffers[0].recv()  # reset the env
    replay_helper = FileReplayHelper()
    nmmo_env = evaluator.buffers[0].envs[0].envs[0].env
    nmmo_env.realm.record_replay(replay_helper)
    # Replace the names of the agents with the policy names
    # TODO: Get the id-to-name mapping from the evaluator or policy pool
    for samp, (policy_name, _) in zip(evaluator.policy_pool._sample_idxs, evaluator.policy_pool._policies.items()):
        for idx in samp:
            agent_id = nmmo_env.possible_agents[idx]
            nmmo_env.realm.players[agent_id].name = policy_name + '_' + str(agent_id)

    # Run an episode to generate the replay
    replay_helper.reset()
    while True:
        with torch.no_grad():
            actions, logprob, value, _ = evaluator.policy_pool.forwards(
                torch.Tensor(o).to(evaluator.device),
                None,  # dummy lstm state
                torch.Tensor(d).to(evaluator.device),
            )
            value = value.flatten()
        evaluator.buffers[0].send(actions.cpu().numpy(), None)
        o, r, d, i = evaluator.buffers[0].recv()

        num_alive = len(nmmo_env.realm.players)
        print('Tick:', nmmo_env.realm.tick, ", alive agents:", num_alive)
        if num_alive == 0 or nmmo_env.realm.tick == args.max_episode_length:
            break

    episode_stats = nmmo_env.get_episode_stats()
    num_agents = len(nmmo_env.possible_agents)
    print(f'Score: {float(episode_stats["total_agent_steps"])/(num_agents*args.max_episode_length):.3f},',
          f'norm_progress_to_center {episode_stats["norm_progress_to_center"]:.3f}')

    # Save the replay file
    replay_file = os.path.join(save_dir, f"replay_{time.strftime('%Y%m%d_%H%M%S')}")
    logging.info("Saving replay to %s", replay_file)
    replay_helper.save(replay_file, compress=False)
    evaluator.close()

def create_policy_ranker(policy_store_dir, ranker_file="openskill.pickle"):
    file = os.path.join(policy_store_dir, ranker_file)
    if os.path.exists(file):
        if os.path.exists(file + ".lock"):
            raise ValueError("Policy ranker file is locked. Delete the lock file.")
        logging.info("Using policy ranker from %s", file)
        policy_ranker = pufferlib.utils.PersistentObject(
            file,
            pufferlib.policy_ranker.OpenSkillRanker,
        )
    else:
        policy_ranker = pufferlib.utils.PersistentObject(
            file,
            pufferlib.policy_ranker.OpenSkillRanker,
            "anchor",
        )
    return policy_ranker

class AllPolicySelector(pufferlib.policy_ranker.PolicySelector):
    def select_policies(self, policies):
        # Return all policy names in the alpahebetical order
        # Loops circularly if more policies are needed than available
        loop = cycle([
            policies[name] for name in sorted(policies.keys()
        )])
        return [next(loop) for _ in range(self._num)]

def rank_policies(policy_store_dir, eval_curriculum_file, device):
    # CHECK ME: can be custom models with different architectures loaded here?
    policy_store = setup_policy_store(policy_store_dir)
    policy_ranker = create_policy_ranker(policy_store_dir)
    num_policies = len(policy_store._all_policies())
    policy_selector = AllPolicySelector(num_policies)

    args = SimpleNamespace(**config.Config.asdict())
    args.data_dir = policy_store_dir
    args.eval_mode = True
    args.num_envs = 5  # sample a bit longer in each env
    args.num_buffers = 1
    args.learner_weight = 0  # evaluate mode
    args.selfplay_num_policies = num_policies + 1
    args.early_stop_agent_num = 0  # run the full episode
    args.resilient_population = 0  # no resilient agents
    args.tasks_path = eval_curriculum_file  # task-conditioning

    # NOTE: This creates a dummy learner agent. Is it necessary?
    from reinforcement_learning import policy  # import your policy
    def make_policy(envs):
        learner_policy = policy.Baseline(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
        )
        return cleanrl.Policy(learner_policy)

    # Setup the evaluator. No training during evaluation
    evaluator = clean_pufferl.CleanPuffeRL(
        device=torch.device(device),
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        data_dir=policy_store_dir,
        vectorization=Multiprocessing,
        num_envs=args.num_envs,
        num_cores=args.num_envs,
        num_buffers=args.num_buffers,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.selfplay_num_policies,
        batch_size=args.eval_batch_size,
        policy_store=policy_store,
        policy_ranker=policy_ranker, # so that a new ranker is created
        policy_selector=policy_selector,
    )

    rank_file = os.path.join(policy_store_dir, "ranking.txt")
    with open(rank_file, "w") as f:
        pass

    results = defaultdict(list)
    while evaluator.global_step < args.eval_num_steps:
        _, stats, infos = evaluator.evaluate()

        for pol, vals in infos.items():
            results[pol].extend([
                e[1] for e in infos[pol]['team_results']
            ])

        ratings = evaluator.policy_ranker.ratings()
        dataframe = pd.DataFrame(
            {
                ("Rating"): [ratings.get(n).mu for n in ratings],
                ("Policy"): ratings.keys(),
            }
        )

        with open(rank_file, "a") as f:
            f.write(
                "\n\n"
                + dataframe.round(2)
                .sort_values(by=["Rating"], ascending=False)
                .to_string(index=False)
                + "\n\n"
            )

        # Reset the envs and start the new episodes
        # NOTE: The below line will probably end the episode in the middle, 
        #   so we won't be able to sample scores from the successful agents.
        #   Thus, the scores will be biased towards the agents that die early.
        #   Still, the numbers we get this way is better than frequently
        #   updating the scores because the openskill ranking only takes the mean.
        #evaluator.buffers[0]._async_reset()

        # CHECK ME: delete the policy_ranker lock file
        Path(evaluator.policy_ranker.lock.lock_file).unlink(missing_ok=True)

    evaluator.close()
    for pol, res in results.items():
        aggregated = {}
        keys = asdict(res[0]).keys()
        for k in keys:
            if k == 'policy_id':
                continue
            aggregated[k] = np.mean([asdict(e)[k] for e in res])
        results[pol] = aggregated
    print('Evaluation complete. Average stats:\n', results)


if __name__ == "__main__":
    """Usage: python evaluate.py -p <policy_store_dir> -s <replay_save_dir>

    -p, --policy-store-dir: Directory to load policy checkpoints from for evaluation/ranking
    -s, --replay-save-dir: Directory to save replays (Default: replays/)
    -r, --replay-mode: Replay save mode (Default: False)
    -d, --device: Device to use for evaluation/ranking (Default: cuda if available, otherwise cpu)

    To generate replay from your checkpoints, put them together in policy_store_dir, run the following command, 
    and replays will be saved under the replays/. The script will only use 1 environment.
    $ python evaluate.py -p <policy_store_dir>

    To rank your checkpoints, set the eval-mode to true, and the rankings will be printed out.
    The replay files will NOT be generated in the eval mode.:
    $ python evaluate.py -p <policy_store_dir> -e true

    TODO: Pass in the task embedding?
    """
    logging.basicConfig(level=logging.INFO)

    # Process the evaluate.py command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--policy-store-dir",
        dest="policy_store_dir",
        type=str,
        default=None,
        help="Directory to load policy checkpoints from",
    )
    parser.add_argument(
        "-s",
        "--replay-save-dir",
        dest="replay_save_dir",
        type=str,
        default="replays",
        help="Directory to save replays (Default: replays/)",
    )
    parser.add_argument(
        "-r",
        "--replay-mode",
        dest="replay_mode",
        action="store_true",
        help="Replay mode (Default: False)",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation/ranking (Default: cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "-t",
        "--task-file",
        dest="task_file",
        type=str,
        default="reinforcement_learning/eval_task_with_embedding.pkl",
        help="Task file to use for evaluation",
    )

    # Parse and check the arguments
    eval_args = parser.parse_args()
    assert eval_args.policy_store_dir is not None, "Policy store directory must be specified"

    if getattr(eval_args, "replay_mode", False):
        logging.info("Generating replays from the checkpoints in %s", eval_args.policy_store_dir)
        save_replays(eval_args.policy_store_dir, eval_args.replay_save_dir)
    else:
        logging.info("Ranking checkpoints from %s", eval_args.policy_store_dir)
        logging.info("Replays will NOT be generated")
        rank_policies(eval_args.policy_store_dir, eval_args.task_file, eval_args.device)
