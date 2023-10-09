from argparse import Namespace
from collections import Counter
from collections import defaultdict
import math

import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo

class Config(nmmo.config.MiniGame):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.PROVIDE_DEATH_FOG_OBS = True
        self.MAP_FORCE_GENERATION = False
        #self.HORIZON = args.max_episode_length
        self.PLAYER_N = args.num_agents
        self.TEAMS = {i: [i*args.num_agents_per_team+j+1 for j in range(args.num_agents_per_team)]
                          for i in range(args.num_agents // args.num_agents_per_team)}

        self.PATH_MAPS = f"{args.maps_path}/"
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size

        self.COMMUNICATION_SYSTEM_ENABLED = False

        # Currently testing
        self.TEAM_TASK_EPISODE_PROB = args.team_mode_prob
        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity

def make_env_creator(args: Namespace):
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
            },
        )
        return env
    return env_creator

class Postprocessor(pufferlib.emulation.Postprocessor):
    def __init__(
            self, env, is_multiagent, agent_id,
            eval_mode=False):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self.eval_mode = eval_mode
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self.epoch_return = 0
        self.epoch_length = 0

        self._task_completed = 0
        self._max_task_progress = 0
        self._task_with_2_reward_signal = 0
        self._task_with_0p2_max_progress = 0
        self._curriculum = defaultdict(list)

    def _update_stats(self, agent):
        task = self.env.agent_task_map[agent.ent_id][0]
        # For each task spec, record whether its max progress and reward count
        self._curriculum[task.spec_name].append((task._max_progress, task.reward_signal_count))
        self._max_task_progress = task._max_progress
        if task.reward_signal_count >= 2:
            self._task_with_2_reward_signal = 1.0
        if task._max_progress >= 0.2:
            self._task_with_0p2_max_progress = 1.0
        if task.completed:
            self._task_completed = 1.0

    # @property
    # def observation_space(self):
    #     """If you modify the shape of features, you need to specify the new obs space"""
    #     obs_space = super().observation_space
    #     return obs_space
    
    # def observation(self, obs):
    #     """Called before observations are returned from the environment

    #     Use this to define custom featurizers. Changing the space itself requires you to
    #     define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
    #     """
    #     return obs

    # def action(self, action):
    #     """Called before actions are passed from the model to the environment"""
    #     return action

    def reward_done_info(self, reward, done, info):
        """Update stats + info and save replays."""
        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop('task', None)

        # Stop early when there is a winner (i.e. only one team left)
        # They should get rewarded too
        # if len(self.env.agents) <= self.early_stop_agent_num:
        #     done = True

        if not done:
            self.epoch_length += 1
            self.epoch_return += reward
            return reward, done, info

        if 'stats' not in info:
            info['stats'] = {}

        agent = self.env.realm.players.dead_this_tick.get(
            self.agent_id, self.env.realm.players.get(self.agent_id)
        )
        assert agent is not None
        self._update_stats(agent)
        #log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        #attr_to_col = self.env.realm.event_log.attr_to_col

        info["return"] = self.epoch_return
        info["length"] = self.epoch_length
        info["inflicted_damage"] = agent.history.damage_inflicted

        if self.eval_mode:
            # "return" is used for ranking in the eval mode, so put the task progress here
            info["return"] = self._max_task_progress  # this is 1 if done

        # if self.detailed_stat and self.is_env_done():
        #     info["stats"].update(get_market_stat(self.env.realm))
        #     info["stats"].update(get_supply_stat(self.env.realm))
        #     for key, val in self.env.get_episode_stats().items():
        #         info["stats"]["supply/"+key] = val  # supply is a placeholder

        return reward, done, info

    def is_env_done(self):
        # Trigger only when the episode is done, and has the lowest agent id in agents
        if self.agent_id > min(self.env.agents):
            return False

        # TODO: done when one team "wins"

        if self.env.realm.tick >= self.env.config.HORIZON:  # reached the end
            return True
        for player_id in self.env.agents:  # any alive agents?
            if player_id in self.env.realm.players:
                return False
        return True
