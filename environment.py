from argparse import Namespace

import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo
from nmmo.lib import material
from nmmo.lib.log import EventCode
from nmmo.entity.entity import EntityState

from minigame_postproc import MiniGamePostprocessor

EntityAttr = EntityState.State.attr_name_to_col
IMPASSIBLE = list(material.Impassible.indices)

PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
ENEMY_REPR = 4
DESTROY_TARGET_REPR = 5
TEAMMATE_REPR = 6
PROTECT_TARGET_REPR = 7


class Config(nmmo.config.MiniGame):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.PROVIDE_DEATH_FOG_OBS = True
        self.MAP_FORCE_GENERATION = False
        self.HORIZON = args.max_episode_length
        self.PLAYER_N = args.num_agents
        self.TEAMS = {i: [i*args.num_agents_per_team+j+1 for j in range(args.num_agents_per_team)]
                          for i in range(args.num_agents // args.num_agents_per_team)}

        self.MAP_N = args.num_maps
        self.PATH_MAPS = f"{args.maps_path}/"
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size

        self.COMMUNICATION_SYSTEM_ENABLED = False

        # Currently testing
        self.TEAM_TASK_EPISODE_PROB = args.team_mode_prob
        self.TEAM_BATTLE_EPISODE_PROB = args.team_battle_prob
        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity

def make_env_creator(args: Namespace):
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "runaway_fog_weight": args.runaway_fog_weight,
                "local_superiority_weight": args.local_superiority_weight,
                "local_area_dist": args.local_area_dist,
                "concentrate_fire_weight": args.concentrate_fire_weight,
            },
        )
        return env
    return env_creator

class Postprocessor(MiniGamePostprocessor):
    def __init__(
            self, env, is_multiagent, agent_id,
            eval_mode=False,
            runaway_fog_weight=0,
            local_superiority_weight=0,
            local_area_dist=0,
            concentrate_fire_weight=0,
            ):
        super().__init__(env, agent_id, eval_mode)
        self.config = env.config
        self.runaway_fog_weight = runaway_fog_weight
        self.local_superiority_weight = local_superiority_weight
        self.local_area_dist = local_area_dist
        self.concentrate_fire_weight = concentrate_fire_weight
        self._reset_reward_vars()

        # placeholder for the entity maps
        self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)

        # dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            l, r = i, self.config.MAP_SIZE - i
            self._dist_map[l:r, l:r] = center - i - 1

    def reset(self, observation):
        self._reset_reward_vars()
        # get target_protect, target_destroy from the task, for ProtectAgent and HeadHunting
        if self._my_task is not None:
            if "target_protect" in self._my_task.kwargs:
                target = self._my_task.kwargs["target_protect"]
                self._target_protect = [target] if isinstance(target, int) else target
            for key in ["target", "target_destroy"]:
                if key in self._my_task.kwargs:
                    target = self._my_task.kwargs[key]
                    self.target_destroy = [target] if isinstance(target, int) else target

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add informative tile maps: dist, obstacle, entity
        add_dim = 3
        tile_dim = obs_space["Tile"].shape[1] + add_dim
        obs_space["Tile"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                           shape=(self.config.MAP_N_OBS, tile_dim))
        return obs_space

    def observation(self, obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        # Parse and augment tile obs
        obs["Tile"] = self._augment_tile_obs(obs)

        # Do NOT attack teammates
        obs["ActionTargets"]["Attack"]["Target"] = self._process_attack_mask(obs)
        return obs

    def _augment_tile_obs(self, obs):
        # Process entity obs
        self._entity_map[:] = 0
        entity_idx = obs["Entity"][:, EntityAttr["id"]] != 0
        for entity in obs["Entity"][entity_idx]:
            ent_pos = (entity[EntityAttr["row"]], entity[EntityAttr["col"]])
            if entity[EntityAttr["id"]] > 0:
                self._entity_map[ent_pos] = max(ENEMY_REPR, self._entity_map[ent_pos])
                if entity[EntityAttr["id"]] in self._target_destroy:
                    self._entity_map[ent_pos] = max(DESTROY_TARGET_REPR, self._entity_map[ent_pos])
                if entity[EntityAttr["id"]] in self._my_task.assignee:
                    self._entity_map[ent_pos] = max(TEAMMATE_REPR, self._entity_map[ent_pos])
                if entity[EntityAttr["id"]] in self._target_protect:
                    self._entity_map[ent_pos] = max(PROTECT_TARGET_REPR, self._entity_map[ent_pos])
        entity = self._entity_map[obs["Tile"][:,0], obs["Tile"][:,1]]

        dist = self._dist_map[obs["Tile"][:,0], obs["Tile"][:,1]]
        obstacle = np.isin(obs["Tile"][:,2], IMPASSIBLE)
        maps = [obs["Tile"], dist[:,None], obstacle[:,None], entity[:,None]]
        return np.concatenate(maps, axis=1).astype(np.int16)

    def _process_attack_mask(self, obs):
        mask = obs["ActionTargets"]["Attack"]["Target"]
        if sum(mask) == 1 and mask[-1] == 1:  # no valid target
            return mask
        target_idx = np.where(mask[:-1] == 1)[0]
        teammate = np.in1d(obs["Entity"][target_idx,EntityAttr["id"]], self._my_task.assignee)
        # Do NOT attack teammates
        mask[target_idx[teammate]] = 0
        if sum(mask) == 0:
            mask[-1] = 1  # if no valid target, make sure to turn on no-op
        return mask

    # def action(self, action):
    #     """Called before actions are passed from the model to the environment"""
    #     return action

    def reward_done_info(self, reward, done, info):
        """Called on reward, done, and info before they are returned from the environment"""
        reward, done, info = super().reward_done_info(reward, done, info)  # DO NOT REMOVE

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.
        if not done:
            # Update the reward vars that are used to calculate the below bonuses
            agent = self.env.realm.players[self.agent_id]
            self._update_reward_vars(agent)

            # Run away from death fog
            reward += self.runaway_fog_weight if 1 < self._curr_death_fog < self._prev_death_fog else 0

            # Local superiority bonus
            reward += self.local_superiority_weight * self._local_superiority

            # Concentrate fire bonus
            reward += self.concentrate_fire_weight * self._concentrate_fire

        return reward, done, info

    def _reset_reward_vars(self):
        self._prev_death_fog = 0
        self._curr_death_fog = 0
        self._local_superiority = 0
        self._concentrate_fire = 0
        self._target_protect = []
        self._target_destroy = []

    def _update_reward_vars(self, agent):
        # Death fog
        self._prev_death_fog = self._curr_death_fog
        self._curr_death_fog = self.env.realm.fog_map[agent.pos]

        # Local superiority, get from the agent's entity map
        local_map = self._entity_map[agent.pos[0]-self.local_area_dist:agent.pos[0]+self.local_area_dist+1,
                                      agent.pos[1]-self.local_area_dist:agent.pos[1]+self.local_area_dist+1]
        num_enemy = np.sum(local_map == ENEMY_REPR)
        # TODO: add the distance-based bonus?
        self._local_superiority = np.sum(local_map == TEAMMATE_REPR) - num_enemy if num_enemy > 0 else 0

        # Concentrate fire, get from the agent's log
        self._concentrate_fire = 0
        log = self.env.realm.event_log.get_data(agents=self._my_task.assignee,  # get team log
                                                event_code=EventCode.SCORE_HIT, tick=-1)
        attr_to_col = self.env.realm.event_log.attr_to_col
        my_hit = log[:,attr_to_col["ent_id"]] == self.agent_id
        if sum(my_hit) > 0:
            my_target = log[my_hit,attr_to_col["target_ent"]]
            target_hits = log[:,attr_to_col["target_ent"]] == my_target[0]
            # reward the single hit as well
            self._concentrate_fire = sum(target_hits)
