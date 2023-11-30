from argparse import Namespace
from collections import Counter
from copy import copy

import math
import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo
import nmmo.core.config as cfg
from nmmo.core import game_api
from nmmo.lib import material, utils
from nmmo.lib.event_log import EventCode
from nmmo.entity.entity import EntityState, CommAttr

from minigame_postproc import MiniGamePostprocessor
import team_games as tg
import postproc_helper as pph

EntityAttr = EntityState.State.attr_name_to_col
CommAttr = {"id": 0, "row": 1, "col": 2, "message": 3}
IMPASSIBLE = list(material.Impassible.indices)

RESOURCE_EVENTS = [EventCode.EAT_FOOD, EventCode.DRINK_WATER]

PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
ENEMY_REPR = 4
DESTROY_TARGET_REPR = 5
TEAMMATE_REPR = 6
PROTECT_TARGET_REPR = 7

class Config(cfg.Medium, cfg.Terrain, cfg.Resource, cfg.Combat, cfg.NPC, cfg.Communication):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("PROVIDE_DEATH_FOG_OBS", True)
        self.set("MAP_FORCE_GENERATION", False)
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
                                (tg.EasyKingoftheQuad, 1), (tg.Sandwich, 1), (tg.CommTogether, 1),
                                (tg.RadioRaid, 1)])

def make_env_creator(args: Namespace, game_cls: game_api.Game=None):
    def env_creator():
        """Create an environment."""
        config = Config(args)
        if game_cls and isinstance(game_cls, game_api.Game):
            config.set("GAME_PACKS", [(game_cls, 1)])
        env = nmmo.Env(config)
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "runaway_fog_weight": args.runaway_fog_weight,
                "local_superiority_weight": args.local_superiority_weight,
                "local_area_dist": args.local_area_dist,
                "superior_fire_weight": args.superior_fire_weight,
                "kill_bonus_weight": args.kill_bonus_weight,
                "comm_grouping_weight": args.comm_grouping_weight,
                "key_achievement_weight": args.key_achievement_weight,
                "task_progress_weight": args.task_progress_weight,
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
            superior_fire_weight=0,
            kill_bonus_weight=0,
            comm_grouping_weight=0,
            key_achievement_weight=0,
            task_progress_weight=0,
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
        self.superior_fire_weight = superior_fire_weight
        self.kill_bonus_weight = kill_bonus_weight
        self.key_achievement_weight = key_achievement_weight
        self.task_progress_weight = task_progress_weight
        self.comm_grouping_weight = comm_grouping_weight

        self.survival_mode_criteria = survival_mode_criteria
        self.get_resource_criteria = get_resource_criteria
        self.get_resource_weight = get_resource_weight
        self.heal_bonus_weight = heal_bonus_weight
        self.meander_bonus_weight = meander_bonus_weight

        self._reset_reward_vars()

        # team/agent, system states, task embedding
        self._task_obs = np.zeros(1+len(self.config.system_states)+self.config.TASK_EMBED_DIM,
                                  dtype=np.float16)

        # placeholders
        self._entity_obs = None  # placeholder
        self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        self._rally_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        self._rally_target = None
        self._can_see_target = False
        self._vof_num_team = None
        self._vof_num_enemy = None

        # dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            l, r = i, self.config.MAP_SIZE - i
            self._dist_map[l:r, l:r] = center - i - 1

    def reset(self, observation):
        super().reset(observation)
        self._reset_reward_vars()
        self._task_obs[0] = float(self._my_task.reward_to == "team")
        self._task_obs[1:1+len(self.config.system_states)] = self.config.system_states
        self._task_obs[1+len(self.config.system_states):] = observation["Task"]

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
                    self._target_destroy = [target] if isinstance(target, int) else target
            if "SeizeCenter" in self._my_task.name or "ProgressTowardCenter" in self._my_task.name:
                self._rally_target = self.env.realm.map.center_coord
                self._rally_map[self._rally_target] = 1
            if "SeizeQuadCenter" in self._my_task.name:
                target = self._my_task.kwargs["quadrant"]
                self._rally_target = self.env.realm.map.quad_centers[target]
                self._rally_map[self._rally_target] = 1

        self.const_dict = {
            "my_team": set(self._my_task.assignee),
            "target_destroy": set(self._target_destroy),
            "target_protect": set(self._target_protect),
            "ENEMY_REPR": ENEMY_REPR,
            "DESTROY_TARGET_REPR": TEAMMATE_REPR,
            "TEAMMATE_REPR": TEAMMATE_REPR,
            "PROTECT_TARGET_REPR": PROTECT_TARGET_REPR,
        }

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add system states to the task obs
        obs_space["Task"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.float16,
                                           shape=self._task_obs.shape)
        # Add informative tile maps: dist, obstacle, food, water, entity, rally point
        add_dim = 6
        tile_dim = obs_space["Tile"].shape[1] + add_dim
        obs_space["Tile"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                           shape=(self.config.MAP_N_OBS, tile_dim))
        return obs_space

    def observation(self, obs):
        """Called before observations are returned from the environment

        Use this to define custom featurizers. Changing the space itself requires you to
        define the observation space again (i.e. Gym.spaces.Dict(gym.spaces....))
        """
        self._entity_obs = obs["Entity"]  # save for later use
        # create a shallow copy, since the original obs is immutable
        mod_obs = {k: v for k, v in obs.items()}
        mod_obs["Task"] = self._task_obs  # system states added to task embedding

        # Parse and augment tile obs
        # see tests/test_update_entity_map.py for the reference python implementation
        self._vof_num_team, self._vof_num_enemy = \
          pph.update_entity_map(self._entity_map, obs["Entity"], EntityAttr, self.const_dict)
        mod_obs["Tile"] = self._augment_tile_obs(obs)

        # Do NOT attack teammates
        mod_obs["ActionTargets"]["Attack"]["Target"] = self._process_attack_mask(obs)
        return mod_obs

    def _augment_tile_obs(self, obs):
        # assume updated entity map
        entity = self._entity_map[obs["Tile"][:,0], obs["Tile"][:,1]]
        dist = self._dist_map[obs["Tile"][:,0], obs["Tile"][:,1]]
        obstacle = (obs["Tile"][:,2] == material.Stone.index) | \
                   (obs["Tile"][:,2] == material.Void.index)
        food = obs["Tile"][:,2] == material.Foilage.index
        water = obs["Tile"][:,2] == material.Water.index

        # Rally point-related obs
        rally_point = self._rally_map[obs["Tile"][:,0], obs["Tile"][:,1]]  # all zero if no rally point

        # To communicate if the agent can see the target
        self._can_see_target = (entity == DESTROY_TARGET_REPR).sum() > 0 or \
                               (self._rally_target is not None and rally_point.sum() > 0)

        maps = [obs["Tile"], dist[:,None], obstacle[:,None], food[:,None], water[:,None],
                entity[:,None], rally_point[:,None]]
        return np.concatenate(maps, axis=1).astype(np.int16)

    def _process_attack_mask(self, obs):
        whole_mask = obs["ActionTargets"]["Attack"]["Target"]
        entity_mask = whole_mask[:-1]
        if entity_mask.sum() == 0 and whole_mask[-1] == 1:  # no valid target
            return whole_mask
        if len(self._my_task.assignee) == 1:  # no team
            return whole_mask
        # the order of entities in obs["Entity"] is the same as in the mask
        teammate = np.in1d(obs["Entity"][:, EntityAttr["id"]], self._my_task.assignee)
        # Do NOT attack teammates
        entity_mask[teammate] = 0
        if entity_mask.sum() == 0:
            whole_mask[-1] = 1  # if no valid target, make sure to turn on no-op
        return whole_mask

    def action(self, action):
        """Called before actions are passed from the model to the environment"""
        self._prev_moves.append(action[3])  # 3 is the index for move direction

        # Override communication with manually computed one
        # NOTE: Can this be learned from scratch?
        action[2] = self._compute_comm_action()
        return action

    def _compute_comm_action(self):
        # comm action values range from 0 - 127, 0: dummy obs
        if self.agent_id not in self.env.realm.players:
            return 0
        agent = self.env.realm.players[self.agent_id]
        return pph.compute_comm_action(self._can_see_target, agent.resources.health.val,
                                       self._entity_obs, EntityAttr, self.const_dict)

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

            # Reward task progress, i.e., setting the new max progress
            if self._new_max_progress:
                # Reward from this task seems too small to encourage learning
                # Provide extra reward when the agents beat the prev max progress
                reward += self.task_progress_weight

            # Extra reward for eating and drinking for survival
            if self.env.config.RESOURCE_SYSTEM_ENABLED and self.get_resource_weight:
                reward += self._eat_progress_bonus()

                if agent.resources.health_restore > 5:  # health restored when water, food >= 50
                    reward += self.heal_bonus_weight

            # Reward key achievements toward team winning
            if self.env.realm.map.seize_targets and self._seize_tile > 0:
                # _seize_tile > 0 if the agent have just seized the target
                reward += self.key_achievement_weight

            # Careful with the combat bonus, by trying not to give too much
            if self.env.config.COMBAT_SYSTEM_ENABLED and \
               not isinstance(self.env.game, tg.CommTogether):  # Use different reward scheme for CommTogether
                # Local superiority bonus -- NOTE: make it small so that agents sit tight together and do nothing
                if self._local_superiority > 0:
                    reward += self.local_superiority_weight * min(self._local_superiority, 3)

                # Get reward for any attack, and bonus for concentrated fire
                reward += self.superior_fire_weight * self._concentrate_fire
                # Fire during superiority -- try to make agents aggressive when having number advantage
                if (self._local_superiority > 0 or self._vof_superiority > 0) and self._concentrate_fire > 0:
                    reward += self.superior_fire_weight * self._concentrate_fire

                # Score kill
                reward += self.kill_bonus_weight * self._player_kill

                # Penalize dying futilely
                if done and (self._local_superiority < 0 or self._vof_superiority < 0):
                    reward = -1

            if isinstance(self.env.game, tg.CommTogether):
                # Get reward for hanging around with teammates
                if self._vof_grouping > 0:
                    reward += self.comm_grouping_weight * min(self._local_grouping, 3)

            # NOTE: this may be why agents are not going straint to the goal in the center race?
            # if self._my_task.reward_to == "agent":
            #     if len(self._prev_moves) > 5:
            #       move_entropy = calculate_entropy(self._prev_moves[-8:])  # of last 8 moves
            #       reward += self.meander_bonus_weight * (move_entropy - 1)

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
        self._seize_tile = 0

        self._local_superiority = 0
        self._local_grouping = 0
        self._vof_num_team = None
        self._vof_num_enemy = None
        self._vof_superiority = 0
        self._vof_grouping = 0
        self._concentrate_fire = 0
        self._player_kill = 0
        self._target_protect = []
        self._target_destroy = []

        # Eat & progress bonuses: eat & progress, drink & progress
        # (reward when agents eat/drink the farthest so far)
        self._prev_basic_events = [0, 0]  # EAT_FOOD, DRINK_WATER
        self._prev_food_level = self._curr_food_level = 100
        self._prev_water_level = self._curr_water_level = 100
        self._prev_health_level = self._curr_health_level = 100
        self._prev_eat_dist = np.inf
        self._prev_drink_dist = np.inf
        self._curr_dist = np.inf

        # task progress
        self._max_task_progress = 0
        self._new_max_progress = False

    def _update_reward_vars(self, agent):
        tick_log = self.env.realm.event_log.get_data(agents=self._my_task.assignee, tick=-1)
        attr_to_col = self.env.realm.event_log.attr_to_col

        # Death fog
        self._prev_death_fog = self._curr_death_fog
        self._curr_death_fog = self.env.realm.fog_map[agent.pos]
        self._curr_dist = self._dist_map[agent.pos]

        # Seize tile
        if self.env.realm.map.seize_targets:
            my_sieze = (tick_log[:,attr_to_col["ent_id"]] == self.agent_id) & \
                       (tick_log[:,attr_to_col["event"]] == EventCode.SEIZE_TILE)
            self._seize_tile = my_sieze.sum() > 0

        # Task progress
        if self._my_task.progress > self._max_task_progress:
          self._new_max_progress = True if self.env.realm.tick > 1 else False
          self._max_task_progress = self._my_task.progress

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
        local_enemy = (local_map == ENEMY_REPR).sum()
        # TODO: add the distance-based bonus?
        self._local_grouping = (local_map == TEAMMATE_REPR).sum()
        self._local_superiority = self._local_grouping - local_enemy if local_enemy > 0 else 0

        # Visual field superioirty, but count only when enemies are nearby
        self._vof_grouping = self._vof_num_team
        self._vof_superiority = self._vof_num_team - self._vof_num_enemy if local_enemy > 0 else 0

        # Concentrate fire, get from the agent's log
        self._concentrate_fire = 0
        my_hit = (tick_log[:, attr_to_col["event"]] == EventCode.SCORE_HIT) & \
                 (tick_log[:, attr_to_col["ent_id"]] == self.agent_id)
        if my_hit.sum() > 0:
            my_target = tick_log[my_hit, attr_to_col["target_ent"]]
            target_hits = tick_log[:, attr_to_col["target_ent"]] == my_target[0]
            # reward the single hit as well
            self._concentrate_fire = target_hits.sum()

        # Team fire utilization
        # my_team = self._my_task.assignee
        # self._team_fire_utilization = 0
        # team_fire = (tick_log[:,attr_to_col["event"]] == EventCode.SCORE_HIT) & \
        #             np.in1d(tick_log[:,attr_to_col["ent_id"]], my_team)
        # if len(my_team) > 1 and team_fire.sum()) >= max(2,int(len(my_team)**.5)):
        #     self._team_fire_utilization = float(team_fire.sum()) / len(my_team)

        # Being hit
        # got_hit = (tick_log[:,attr_to_col["event"]] == EventCode.SCORE_HIT) & \
        #           (tick_log[:,attr_to_col["target_ent"]] == self.agent_id)
        # self._got_hit = got_hit.sum()

        # Player kill, from the agent's log -- ONLY consider players, not npcs
        my_kill = (tick_log[:,attr_to_col["event"]] == EventCode.PLAYER_KILL) & \
                  (tick_log[:,attr_to_col["ent_id"]] == self.agent_id) & \
                  (tick_log[:,attr_to_col["target_ent"]] > 0)
        self._player_kill = my_kill.sum() > 0

    def _update_resource_reward_vars(self, agent, tick_log, attr_to_col):
        if not self.env.config.RESOURCE_SYSTEM_ENABLED:
            return

        for idx, event_code in enumerate(RESOURCE_EVENTS):
            event_idx = (tick_log[:,attr_to_col["event"]] == event_code) & \
                        (tick_log[:,attr_to_col["ent_id"]] == self.agent_id)
            self._prev_basic_events[idx] = event_idx.sum() > 0

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
