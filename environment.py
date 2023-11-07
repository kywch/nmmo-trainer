from argparse import Namespace
from collections import Counter

import math
import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo
import nmmo.core.config as cfg
from nmmo.lib import material, utils
from nmmo.lib.event_log import EventCode
from nmmo.entity.entity import EntityState

from minigame_postproc import MiniGamePostprocessor
import team_games as tg

EntityAttr = EntityState.State.attr_name_to_col
IMPASSIBLE = list(material.Impassible.indices)

RESOURCE_EVENTS = [EventCode.EAT_FOOD, EventCode.DRINK_WATER]

PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
ENEMY_REPR = 4
DESTROY_TARGET_REPR = 5
TEAMMATE_REPR = 6
PROTECT_TARGET_REPR = 7


class Config(cfg.Medium, cfg.Terrain, cfg.Resource, cfg.Combat):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("PROVIDE_DEATH_FOG_OBS", True)
        self.set("MAP_FORCE_GENERATION", False)
        self.set("COMMUNICATION_SYSTEM_ENABLED", False)
        self.set("HORIZON", args.max_episode_length)
        self.set("PLAYER_N", args.num_agents)
        self.set("TEAMS", {i: [i*args.num_agents_per_team+j+1 for j in range(args.num_agents_per_team)]
                           for i in range(args.num_agents // args.num_agents_per_team)})

        self.set("MAP_N", args.num_maps)
        self.set("PATH_MAPS", f"{args.maps_path}/")
        self.set("CURRICULUM_FILE_PATH", args.tasks_path)
        self.set("TASK_EMBED_DIM", args.task_size)

        self.set("GAME_PACKS", [(tg.MiniAgentTraining, 1), (tg.MiniTeamTraining, 1), (tg.MiniTeamBattle, 1),
                                (tg.RacetoCenter, 1), (tg.KingoftheHill, 1), (tg.EasyKingoftheHill, 1),
                                (tg.EasyKingoftheQuad, 1),])

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
                "superior_fire_weight": args.superior_fire_weight,
                "key_achievement_weight": args.key_achievement_weight,
                "survival_mode_criteria": args.survival_mode_criteria,
                "get_resource_criteria": args.get_resource_criteria,
                "get_resource_weight": args.get_resource_weight,
                "heal_bonus_weight": args.heal_bonus_weight,
                "meander_bonus_weight": args.meander_bonus_weight,
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
            superior_fire_weight=0,
            key_achievement_weight=0,
            survival_mode_criteria=35,
            get_resource_criteria=75,
            get_resource_weight=0,
            heal_bonus_weight=0,
            meander_bonus_weight=0,
        ):
        super().__init__(env, agent_id, eval_mode)
        self.config = env.config

        self.runaway_fog_weight = runaway_fog_weight

        self.local_superiority_weight = local_superiority_weight
        self.local_area_dist = local_area_dist
        self.concentrate_fire_weight = concentrate_fire_weight
        self.superior_fire_weight = superior_fire_weight
        self.key_achievement_weight = key_achievement_weight

        self.survival_mode_criteria = survival_mode_criteria
        self.get_resource_criteria = get_resource_criteria
        self.get_resource_weight = get_resource_weight
        self.heal_bonus_weight = heal_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight

        self._reset_reward_vars()

        # placeholder for the maps
        self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        self._rally_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        self._rally_target = None

        # dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            l, r = i, self.config.MAP_SIZE - i
            self._dist_map[l:r, l:r] = center - i - 1

    def reset(self, observation):
        super().reset(observation)
        self._reset_reward_vars()

        # Set the task-related vars
        self._rally_target = None
        self._rally_map[:] = 0
        if self._my_task is not None:
            # get target_protect, target_destroy from the task, for ProtectAgent and HeadHunting
            if "target_protect" in self._my_task.kwargs:
                target = self._my_task.kwargs["target_protect"]
                self._target_protect = [target] if isinstance(target, int) else target
            for key in ["target", "target_destroy"]:
                if key in self._my_task.kwargs:
                    target = self._my_task.kwargs[key]
                    self.target_destroy = [target] if isinstance(target, int) else target
            if "SeizeCenter" in self._my_task.name or "ProgressTowardCenter" in self._my_task.name:
                self._rally_target = self.env.realm.map.center_coord
                self._rally_map = np.copy(self._dist_map)
            if "SeizeQuadCenter" in self._my_task.name:
                target = self._my_task.kwargs["quadrant"]
                self._rally_target = self.env.realm.map.quad_centers[target]
                for r in range(self.config.MAP_SIZE):
                    for c in range(self.config.MAP_SIZE):
                        self._rally_map[r,c] = utils.linf_single((r,c), self._rally_target)

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add informative tile maps: dist, obstacle, food, water, entity, rally dist & point
        add_dim = 7
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
        obstacle = np.isin(obs["Tile"][:,2], [material.Stone.index, material.Void.index])
        food = obs["Tile"][:,2] == material.Foilage.index
        water = obs["Tile"][:,2] == material.Water.index

        # Rally point-related obs
        rally_dist = self._rally_map[obs["Tile"][:,0], obs["Tile"][:,1]]  # all zero if no rally point
        if self._rally_target:
            rally_point = self._rally_map[obs["Tile"][:,0], obs["Tile"][:,1]] == 0
        else:
            rally_point = np.zeros_like(rally_dist)

        maps = [obs["Tile"], dist[:,None], obstacle[:,None], food[:,None], water[:,None],
                entity[:,None], rally_dist[:,None], rally_point[:,None]]
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

    def action(self, action):
        """Called before actions are passed from the model to the environment"""
        self._prev_moves.append(action[2])  # 2 is the index for move direction
        return action

    def reward_done_info(self, reward, done, info):
        """Called on reward, done, and info before they are returned from the environment"""
        reward, done, info = super().reward_done_info(reward, done, info)  # DO NOT REMOVE

        # Default reward shaper sums team rewards from the task system.
        # Add custom reward shaping here.
        # NOTE: The case (done and reward > 0)=True comes from team games to NOT penalize sacrifice for team
        if not done or (done and reward > 0):
            # Update the reward vars that are used to calculate the below bonuses
            if self.agent_id in self.env.realm.players:
                agent = self.env.realm.players[self.agent_id]
            elif self.agent_id in self.env.dead_this_tick:
                agent = self.env.dead_this_tick[self.agent_id]
            else: # This should not happen
                raise ValueError(f"Agent {self.agent_id} not found in the realm")
            self._update_reward_vars(agent)

            # Run away from death fog
            reward += self.runaway_fog_weight if 1 < self._curr_death_fog < self._prev_death_fog else 0

            if self.env.config.COMBAT_SYSTEM_ENABLED:
                # Local superiority bonus
                reward += self.local_superiority_weight * self._local_superiority
                # Concentrate fire bonus
                reward += self.concentrate_fire_weight * self._concentrate_fire
                # Fire during superiority -- try to make agents aggressive when having number advantage
                if (self._local_superiority > 0 or self._vof_superiority > 0) and self._concentrate_fire > 0:
                    reward += self.superior_fire_weight
                # Team bonus for higher fire utilization, when superior
                if self._vof_superiority > 0 and self._team_fire_utilization > 0:
                    reward += self.superior_fire_weight*self._team_fire_utilization
                # Score kill
                reward += self.key_achievement_weight * self._player_kill
                # Penalize dying futilely
                if done and (self._local_superiority < 0 or self._vof_superiority < 0):
                    reward = -1

            if self.env.config.RESOURCE_SYSTEM_ENABLED and self.get_resource_weight:
                reward += self._eat_progress_bonus()

                if agent.resources.health_restore > 5:  # health restored when water, food >= 50
                    reward += self.heal_bonus_weight

            if self.env.realm.map.seize_targets:
                for tile, hist in self.env.realm.map.seize_status.items():
                    # check if the agent have just seized the target
                    if hist[0] == self.agent_id and hist[1] == self.env.realm.tick:
                        reward += self.key_achievement_weight if not done else 0

            if self._my_task.reward_to == "agent":
                if len(self._prev_moves) > 5:
                  move_entropy = calculate_entropy(self._prev_moves[-8:])  # of last 8 moves
                  reward += self.meander_bonus_weight * (move_entropy - 1)

        return reward, done, info

    def _eat_progress_bonus(self):
        eat_progress_bonus = 0
        for idx, event_code in enumerate(RESOURCE_EVENTS):
            if self._prev_basic_events[idx] > 0:
                if event_code == EventCode.EAT_FOOD:
                    # bonus for eating
                    eat_progress_bonus += self.get_resource_weight
                    # extra bonus for eating when hungry
                    if self._prev_food_level <= self.survival_mode_criteria:
                        eat_progress_bonus += self.get_resource_weight
                    # extra bonus for eat and progress
                    if self._curr_dist < self._prev_eat_dist:
                        eat_progress_bonus += 2*self.get_resource_weight
                        self._prev_eat_dist = self._curr_dist

                if event_code == EventCode.DRINK_WATER:
                    # bonus for drinking
                    eat_progress_bonus += self.get_resource_weight
                    # extra bonus for eating when hungry
                    if self._prev_water_level <= self.survival_mode_criteria:
                        eat_progress_bonus += self.get_resource_weight
                    # extra bonus for eat and progress
                    if self._curr_dist < self._prev_drink_dist:
                        eat_progress_bonus += 2*self.get_resource_weight
                        self._prev_drink_dist = self._curr_dist

        return eat_progress_bonus

    def _reset_reward_vars(self):
        self._prev_death_fog = 0
        self._curr_death_fog = 0
        self._prev_moves = []

        self._local_superiority = 0
        self._vof_superiority = 0
        self._concentrate_fire = 0
        self._team_fire_utilization = 0
        self._player_kill = 0
        self._target_protect = []
        self._target_destroy = []

        # Eat & progress bonuses: eat & progress, drink & progress
        # (reward when agents eat/drink the farthest so far)
        self._prev_basic_events = np.zeros(2, dtype=np.int16)  # EAT_FOOD, DRINK_WATER
        self._prev_food_level = self._curr_food_level = 100
        self._prev_water_level = self._curr_water_level = 100
        self._prev_health_level = self._curr_health_level = 100
        self._prev_eat_dist = np.inf
        self._prev_drink_dist = np.inf
        self._curr_dist = np.inf

    def _update_reward_vars(self, agent):
        tick_log = self.env.realm.event_log.get_data(agents=self._my_task.assignee, tick=-1)
        attr_to_col = self.env.realm.event_log.attr_to_col

        # Death fog
        self._prev_death_fog = self._curr_death_fog
        self._curr_death_fog = self.env.realm.fog_map[agent.pos]
        self._curr_dist = self._dist_map[agent.pos]

        # System-dependent reward vars
        self._update_combat_reward_vars(agent, tick_log, attr_to_col)
        self._update_resource_reward_vars(agent, tick_log, attr_to_col)

    def _update_combat_reward_vars(self, agent, tick_log, attr_to_col):
        if not self.env.config.COMBAT_SYSTEM_ENABLED:
            return

        # Local superiority, get from the agent's entity map
        local_map = self._entity_map[agent.pos[0]-self.local_area_dist:agent.pos[0]+self.local_area_dist+1,
                                     agent.pos[1]-self.local_area_dist:agent.pos[1]+self.local_area_dist+1]
        # TODO: include all enemies and allies
        # how about their health too?
        num_enemy = np.sum(local_map == ENEMY_REPR)
        # TODO: add the distance-based bonus?
        self._local_superiority = np.sum(local_map == TEAMMATE_REPR) - num_enemy if num_enemy > 0 else 0

        # Visual field superioirty, but count only when enemies are nearby
        self._vof_superiority = np.sum(self._entity_map == TEAMMATE_REPR) - np.sum(self._entity_map == ENEMY_REPR)\
                                if num_enemy > 0 else 0

        # Concentrate fire, get from the agent's log
        self._concentrate_fire = 0
        my_hit = (tick_log[:,attr_to_col["event"]] == EventCode.SCORE_HIT) & \
                 (tick_log[:,attr_to_col["ent_id"]] == self.agent_id)
        if sum(my_hit) > 0:
            my_target = tick_log[my_hit,attr_to_col["target_ent"]]
            target_hits = tick_log[:,attr_to_col["target_ent"]] == my_target[0]
            # reward the single hit as well
            self._concentrate_fire = sum(target_hits)

        # Team fire utilization
        my_team = self._my_task.assignee
        self._team_fire_utilization = 0
        team_fire = (tick_log[:,attr_to_col["event"]] == EventCode.SCORE_HIT) & \
                    np.in1d(tick_log[:,attr_to_col["ent_id"]], my_team)
        if len(my_team) > 1 and sum(team_fire) >= max(2,int(len(my_team)**.5)):
            self._team_fire_utilization = float(sum(team_fire)) / len(my_team)

        # Being hit
        got_hit = (tick_log[:,attr_to_col["event"]] == EventCode.SCORE_HIT) & \
                  (tick_log[:,attr_to_col["target_ent"]] == self.agent_id)
        self._got_hit = sum(got_hit)

        # Player kill, from the agent's log
        my_kill = (tick_log[:,attr_to_col["event"]] == EventCode.PLAYER_KILL) & \
                  (tick_log[:,attr_to_col["ent_id"]] == self.agent_id) & \
                  ~np.in1d(tick_log[:,attr_to_col["target_ent"]], my_team)
        self._player_kill = float(sum(my_kill) > 0)

    def _update_resource_reward_vars(self, agent, tick_log, attr_to_col):
        if not self.env.config.RESOURCE_SYSTEM_ENABLED:
            return

        for idx, event_code in enumerate(RESOURCE_EVENTS):
            event_idx = (tick_log[:,attr_to_col["event"]] == event_code) & \
                        (tick_log[:,attr_to_col["ent_id"]] == self.agent_id)
            self._prev_basic_events[idx] = int(sum(event_idx) > 0)

        # agent-based vars
        self._prev_food_level = self._curr_food_level
        self._curr_food_level = agent.resources.food.val
        self._prev_water_level = self._curr_water_level
        self._curr_water_level = agent.resources.water.val
        self._prev_health_level = self._curr_health_level
        self._curr_health_level = agent.resources.health.val

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy
