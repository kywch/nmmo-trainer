from argparse import Namespace
from collections import Counter
import math

import gym.spaces
import numpy as np

import pufferlib
import pufferlib.emulation

import nmmo
from nmmo.lib import material
from nmmo.lib.log import EventCode
import nmmo.systems.item as Item
from nmmo.entity.entity import EntityState
from nmmo.systems.item import ItemState

from leader_board import StatPostprocessor, extract_unique_event

EntityAttr = EntityState.State.attr_name_to_col
ItemAttr = ItemState.State.attr_name_to_col
IMPASSIBLE = list(material.Impassible.indices)
ARMOR_LIST = [Item.Hat.ITEM_TYPE_ID, Item.Top.ITEM_TYPE_ID, Item.Bottom.ITEM_TYPE_ID]

PASSIVE_REPR = 1  # matched to npc_type
NEUTRAL_REPR = 2
HOSTILE_REPR = 3
ENEMY_REPR = 4
TEAMMATE_REPR = 5

# We can use the following mapping from task name (skill/item name as arg) to profession
TASK_TO_SKILL_MAP = {
    ":melee_": "melee",  # skils
    ":range_": "range",
    ":mage_": "mage",
    ":spear_": "melee",  # weapons
    ":bow_": "range",
    ":wand_": "mage",
    ":pickaxe_": "melee",  # tools
    ":axe_": "range",
    ":chisel_": "mage",
    ":whetstone": "melee",  # ammo
    ":arrow_": "range",
    ":runes_": "mage",
}
SKILL_LIST = sorted(list(set(skill for skill in TASK_TO_SKILL_MAP.values())))
SKILL_TO_AMMO_MAP = {
    "melee": Item.Whetstone.ITEM_TYPE_ID,
    "range": Item.Arrow.ITEM_TYPE_ID,
    "mage": Item.Runes.ITEM_TYPE_ID,
}
SKILL_TO_TOOL_MAP = {
    "melee": Item.Pickaxe.ITEM_TYPE_ID,
    "range": Item.Axe.ITEM_TYPE_ID,
    "mage": Item.Chisel.ITEM_TYPE_ID,
}
SKILL_TO_WEAPON_MAP = {
    "melee": Item.Spear.ITEM_TYPE_ID,
    "range": Item.Bow.ITEM_TYPE_ID,
    "mage": Item.Wand.ITEM_TYPE_ID,
}
SKILL_TO_TILE_MAP = {
    "melee": material.Ore.index,
    "range": material.Tree.index,
    "mage": material.Crystal.index,
}
SKILL_TO_MASK = {
    "melee": np.array([1, 0, 0], dtype=np.int8),
    "range": np.array([0, 1, 0], dtype=np.int8),
    "mage": np.array([0, 0, 1], dtype=np.int8),
}
BASIC_BONUS_EVENTS = [EventCode.EAT_FOOD, EventCode.DRINK_WATER, EventCode.GO_FARTHEST]


#class Config(nmmo.config.Default):
class Config(nmmo.config.Tutorial):
    """Configuration for Neural MMO."""

    def __init__(self, args: Namespace):
        super().__init__()

        self.PROVIDE_ACTION_TARGETS = True
        self.PROVIDE_NOOP_ACTION_TARGET = True
        self.PROVIDE_DEATH_FOG_OBS = True
        self.MAP_FORCE_GENERATION = False
        self.PLAYER_N = args.num_agents
        self.TEAMS = {i: [i*8+j+1 for j in range(8)]  # 8 players per team
                      for i in range(args.num_agents//8)}
        self.HORIZON = args.max_episode_length
        self.MAP_N = args.num_maps
        self.PATH_MAPS = f"{args.maps_path}/{args.map_size}/"
        self.MAP_CENTER = args.map_size
        self.NPC_N = args.num_npcs
        self.CURRICULUM_FILE_PATH = args.tasks_path
        self.TASK_EMBED_DIM = args.task_size
        self.RESOURCE_RESILIENT_POPULATION = args.resilient_population

        self.COMMUNICATION_SYSTEM_ENABLED = False

        # Currently testing
        self.TEAM_TASK_EPISODE_PROB = args.team_mode_prob
        self.COMBAT_SPAWN_IMMUNITY = args.spawn_immunity
        self.PROGRESSION_EXP_THRESHOLD = nmmo.config.default_exp_threshold(
            base_exp = args.base_exp, max_level = 10
        )

        self.NPC_POWER_MULTIPLIER = args.npc_power
        self.NPC_ARMOR_DROP_PROB = args.armor_drop
        # self.EXCHANGE_ACTION_TARGET_DISABLE_LISTING
        self.EQUIPMENT_ARMOR_EXPERIMENTAL = True if args.experimental_armor else False


def make_env_creator(args: Namespace):
    # TODO: Max episode length
    def env_creator():
        """Create an environment."""
        env = nmmo.Env(Config(args))
        env = pufferlib.emulation.PettingZooPufferEnv(env,
            postprocessor_cls=Postprocessor,
            postprocessor_kwargs={
                "eval_mode": args.eval_mode,
                "detailed_stat": args.detailed_stat,
                "early_stop_agent_num": args.early_stop_agent_num,
                "only_use_main_skill": args.only_use_main_skill,
                "survival_mode_criteria": args.survival_mode_criteria,
                "death_fog_criteria": args.death_fog_criteria,
                "survival_heal_weight": args.survival_heal_weight,
                "survival_resource_weight": args.survival_resource_weight,
                "get_resource_weight": args.get_resource_weight,
                "progress_bonus_weight": args.progress_bonus_weight,
                "runaway_bonus_weight": args.runaway_bonus_weight,
                "meander_bonus_weight": args.meander_bonus_weight,
                "combat_bonus_weight": args.combat_bonus_weight,
                "upgrade_bonus_weight": args.upgrade_bonus_weight,
                "unique_event_bonus_weight": args.unique_event_bonus_weight,
                #"underdog_bonus_weight": args.underdog_bonus_weight,
            },
        )
        return env
    return env_creator

class Postprocessor(StatPostprocessor):
    def __init__(self, env, is_multiagent, agent_id,
        eval_mode=False,
        detailed_stat=False,
        early_stop_agent_num=0,
        only_use_main_skill=False,
        survival_mode_criteria=35,
        get_resource_criteria=70,
        death_fog_criteria=1,
        survival_heal_weight=0,
        survival_resource_weight=0,
        get_resource_weight=0,
        progress_bonus_weight=0,
        runaway_bonus_weight=0,
        progress_refractory_period=5,
        meander_bonus_weight=0,
        combat_bonus_weight=0,
        upgrade_bonus_weight=0,
        unique_event_bonus_weight=0,
        clip_unique_event=3,
        underdog_bonus_weight = 0,
    ):
        super().__init__(env, agent_id, eval_mode, detailed_stat, early_stop_agent_num)
        self.config = env.config
        self.survival_mode_criteria = survival_mode_criteria  # for health, food, water
        self.get_resource_criteria = get_resource_criteria
        self.death_fog_criteria = death_fog_criteria
        self.only_use_main_skill = only_use_main_skill
        self.survival_heal_weight = survival_heal_weight
        self.survival_resource_weight = survival_resource_weight
        self.get_resource_weight = get_resource_weight
        self.progress_bonus_weight = progress_bonus_weight
        self.runaway_bonus_weight = runaway_bonus_weight
        self.progress_refractory_period = progress_refractory_period
        self.meander_bonus_weight = meander_bonus_weight
        self.combat_bonus_weight = combat_bonus_weight
        self.upgrade_bonus_weight = upgrade_bonus_weight
        self.unique_event_bonus_weight = unique_event_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.underdog_bonus_weight = underdog_bonus_weight

        self._main_combat_skill = None
        self._skill_task_embedding = None
        self._noop_inventry_item = np.zeros(self.config.ITEM_INVENTORY_CAPACITY + 1, dtype=np.int8)
        self._noop_inventry_item[-1] = 1
        self._ignore_items = None
        self._main_skill_items = None

        # dist map should not change from episode to episode
        self._dist_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        center = self.config.MAP_SIZE // 2
        for i in range(center):
            l, r = i, self.config.MAP_SIZE - i
            self._dist_map[l:r, l:r] = center - i - 1

        # placeholder for the entity maps
        self._entity_map = np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE), dtype=np.int16)
        self._my_team = None

    def reset(self, obs):
        """Called at the start of each episode"""
        super().reset(obs)
        self._reset_reward_vars()
        task_name = self.env.agent_task_map[self.agent_id][0].name
        self._main_combat_skill = self._choose_combat_skill(task_name)
        self._combat_embedding = np.zeros(9, dtype=np.int16)  # copy CombatAttr to [3:]
        self._combat_embedding[SKILL_LIST.index(self._main_combat_skill)] = 1

        # NOTE: The items that are not used by the main combat skill are ignored
        # TODO: Revisit this
        not_my_ammo = [ammo for skill, ammo in SKILL_TO_AMMO_MAP.items() if skill != self._main_combat_skill]
        not_my_tool = [tool for skill, tool in SKILL_TO_TOOL_MAP.items() if skill != self._main_combat_skill]
        not_my_weapon = [weapon for skill, weapon in SKILL_TO_WEAPON_MAP.items() if skill != self._main_combat_skill]
        self._ignore_items = not_my_ammo + not_my_tool + not_my_weapon + [Item.Rod, Item.Gloves]
        self._main_skill_items = [SKILL_TO_AMMO_MAP[self._main_combat_skill], SKILL_TO_TOOL_MAP[self._main_combat_skill],
                                  SKILL_TO_WEAPON_MAP[self._main_combat_skill]]

        # NOTE: this is hacky but works for both agent and team tasks
        self._my_team = self.env.agent_task_map[self.agent_id][0]._assignee

    @staticmethod
    def _choose_combat_skill(task_name):
        task_name = task_name.lower()
        # if task_name contains specific skill or item, choose the corresponding skill
        for hint, skill in TASK_TO_SKILL_MAP.items():
            if hint in task_name:
                return skill
        # otherwise, chooose randomly
        return np.random.choice(SKILL_LIST)

    @property
    def observation_space(self):
        """If you modify the shape of features, you need to specify the new obs space"""
        obs_space = super().observation_space
        # Add main combat skill (3) to the combat attr
        combat_dim = 3 + obs_space["CombatAttr"].shape[0]
        obs_space["CombatAttr"] = gym.spaces.Box(low=-2**15, high=2**15-1, dtype=np.int16,
                                           shape=(combat_dim,))
        # Add informative tile maps: dist, obstacle, food, water, ammo, entity
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
        # Add main combat skill to the combat embedding
        self._combat_embedding[3:] = obs["CombatAttr"]
        obs["CombatAttr"] = self._combat_embedding

        # Parse and augment tile obs
        obs["Tile"] = self._augment_tile_obs(obs)

        # Mask out Give, Destroy, Sell when there are less than 7 items
        # NOTE: Can this be learned from scratch?
        # Without this, agents get rid of items and cannot learn to use and benefit from them
        num_item = sum(obs["Inventory"][:, ItemAttr["id"]] != 0)
        if num_item <= 7:
            obs["ActionTargets"]["Sell"]["InventoryItem"] = self._noop_inventry_item
            obs["ActionTargets"]["Give"]["InventoryItem"] = self._noop_inventry_item
            obs["ActionTargets"]["Destroy"]["InventoryItem"] = self._noop_inventry_item

        # Use the heuristic mask for "Use" action
        obs["ActionTargets"]["Use"]["InventoryItem"] = self._heuristic_use_mask(obs)

        # Mask out the last selected price
        obs["ActionTargets"]["Sell"]["Price"][self._last_price] = 0

        # NOTE: Can this be learned from scratch?
        # Without this, agents use all skills and don't specialize & level up
        if self.only_use_main_skill:
            obs["ActionTargets"]["Attack"]["Style"] = SKILL_TO_MASK[self._main_combat_skill]
        # Do NOT attack teammates
        obs["ActionTargets"]["Attack"]["Target"] = self._process_attack_mask(obs)

        return obs

    def _augment_tile_obs(self, obs):
        dist = self._dist_map[obs["Tile"][:,0], obs["Tile"][:,1]]
        obstacle = np.isin(obs["Tile"][:,2], IMPASSIBLE)
        food = obs["Tile"][:,2] == material.Foilage.index
        water = obs["Tile"][:,2] == material.Water.index
        ammo = obs["Tile"][:,2] == SKILL_TO_TILE_MAP[self._main_combat_skill]

        # Process entity obs
        self._entity_map[:] = 0
        entity_idx = obs["Entity"][:, EntityAttr["id"]] != 0
        for entity in obs["Entity"][entity_idx]:
            if entity[EntityAttr["id"]] == self.agent_id:
                continue
            # Without this, adding the ally map hampered the agent training
            ent_pos = (entity[EntityAttr["row"]], entity[EntityAttr["col"]])
            if entity[EntityAttr["id"]] > 0:
                if entity[EntityAttr["id"]] in self._my_team:
                    self._entity_map[ent_pos] = max(TEAMMATE_REPR, self._entity_map[ent_pos])
                else:
                    self._entity_map[ent_pos] = max(ENEMY_REPR, self._entity_map[ent_pos])
                # If a player is on the resource tile, assume the resource is harvested
                ent_idx = np.where((obs["Tile"][:,0] == entity[EntityAttr["row"]]) &
                                   (obs["Tile"][:,1] == entity[EntityAttr["col"]]))[0]
                food[ent_idx] = False
                ammo[ent_idx] = False
            else:
                npc_type = entity[EntityAttr["npc_type"]]
                self._entity_map[ent_pos] = max(npc_type, self._entity_map[ent_pos])
        entity = self._entity_map[obs["Tile"][:,0], obs["Tile"][:,1]]

        maps = [obs["Tile"], dist[:,None], obstacle[:,None], food[:,None], water[:,None], ammo[:,None], entity[:,None]]
        return np.concatenate(maps, axis=1).astype(np.int16)

    def _process_attack_mask(self, obs):
        mask = obs["ActionTargets"]["Attack"]["Target"]
        if sum(mask) == 1 and mask[-1] == 1:  # no valid target
            return mask
        target_idx = np.where(mask[:-1] == 1)[0]
        teammate = np.in1d(obs["Entity"][target_idx,EntityAttr["id"]], self._my_team)
        # Do NOT attack teammates
        mask[target_idx[teammate]] = 0
        if sum(mask) == 0:
            mask[-1] = 1  # if no valid target, make sure to turn on no-op
        return mask

    # NOTE: Can this be learned from scratch?
    def _heuristic_use_mask(self, obs):
        mask = obs["ActionTargets"]["Use"]["InventoryItem"]
        if sum(obs["Inventory"][:,ItemAttr["id"]] != 0) == 0:
            return mask
        # The mask returns 1 for all the "usable" items
        # Strategy: Assuming the auto-equip is on, equip an item and let it automatically level up

        # Do NOT issue "Use" on the equipped items
        equipped = np.where(obs["Inventory"][:,ItemAttr["equipped"]] == 1)
        mask[equipped] = 0

        # If any of these are equipped, do NOT issue "Use" on the same type of item unless it has higher level
        for type_id in ARMOR_LIST + self._main_skill_items:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == type_id)
            # if there is an item of the same type that is not equipped
            if np.sum(obs["Inventory"][type_idx,ItemAttr["equipped"]]) > 0:
                mask[type_idx] = 0
                if len(type_idx[0]) > 1: # if there are more than 1 items of the same type
                    type_equipped = np.intersect1d(type_idx, equipped)
                    level_equipped = np.max(obs["Inventory"][type_equipped,ItemAttr["level"]])
                    max_level = np.max(obs["Inventory"][type_idx,ItemAttr["level"]])
                    if max_level > level_equipped:
                        # NOTE: This actions is ignored when the agent cannot equip the item due to lower level
                        use_this = np.argmax(obs["Inventory"][type_idx,ItemAttr["level"]])
                        mask[type_idx[0][use_this]] = 1

        # Ignore the items that are not used by the main combat skill
        # NOTE: Revisit the items later, especially Rod and Gloves
        type_idx = np.where(np.in1d(obs["Inventory"][:,ItemAttr["type_id"]], self._ignore_items) == True)
        mask[type_idx] = 0

        # Use ration or potion ONLY when necessary
        starve_or_hydrate = min(self._curr_food_level, self._curr_water_level) == 0 and \
                            max(self._curr_food_level, self._curr_water_level) <= self.survival_mode_criteria
        if not starve_or_hydrate:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == Item.Ration.ITEM_TYPE_ID)
            mask[type_idx] = 0
        if not starve_or_hydrate and self._curr_health_level > self.survival_mode_criteria:
            type_idx = np.where(obs["Inventory"][:,ItemAttr["type_id"]] == Item.Potion.ITEM_TYPE_ID)
            mask[type_idx] = 0

        # Remove no-op when there is something to use
        if np.sum(mask) > 1:
            mask[-1] = 0

        return mask

    def action(self, action):
        """Called before actions are passed from the model to the environment"""
        self._last_moves.append(action[8])  # 8 is the index for move direction
        self._last_price = action[10]  # 10 is the index for selling price
        return action

    def reward_done_info(self, reward, done, info):
        """Called on reward, done, and info before they are returned from the environment"""
        reward, done, info = super().reward_done_info(reward, done, info)  # DO NOT REMOVE

        # Default reward shaper sums team rewards.
        # Add custom reward shaping here.
        if not done:
            # Update the reward vars that are used to calculate the below bonuses
            agent = self.env.realm.players[self.agent_id]
            self._update_reward_vars(agent)

            survival_bonus = 0
            # Survival mode: heal bonus
            # NOTE: agents got addicted to this bonus under death fog, so added death fog criteria
            if self._last_health_level <= self.survival_mode_criteria and \
               self._curr_death_fog < self.death_fog_criteria:
                # 10 in case of enough food/water, 50+ for potion
                survival_bonus += self.survival_heal_weight * agent.resources.health_restore

            # Survival & progress bonuses: eat & progress, drink & progress, run away from the death fog
            progress_bonus = 0
            self._progress_refractory_counter -= 1
            for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
                if self._last_basic_events[idx] > 0:
                    if event_code == EventCode.EAT_FOOD:
                        # progress and eat
                        if self._curr_dist < self._last_eat_dist:
                            progress_bonus += self.progress_bonus_weight
                            self._last_eat_dist = self._curr_dist
                        # eat when starve
                        if self._last_food_level <= self.survival_mode_criteria:
                            survival_bonus += self.survival_resource_weight
                        elif self._last_food_level <= self.get_resource_criteria:
                            # under death fog, better move to the center
                            if self._curr_death_fog < self.death_fog_criteria:
                                survival_bonus += self.get_resource_weight

                    if event_code == EventCode.DRINK_WATER:
                        # progress and drink
                        if self._curr_dist < self._last_drink_dist:
                            progress_bonus += self.progress_bonus_weight
                            self._last_drink_dist = self._curr_dist
                        # drink when dehydrated
                        if self._last_water_level <= self.survival_mode_criteria:
                            survival_bonus += self.survival_resource_weight
                        elif self._last_water_level <= self.get_resource_criteria:
                            # under death fog, better move to the center
                            if self._curr_death_fog < self.death_fog_criteria:
                                survival_bonus += self.get_resource_weight

                    # run away from death fog
                    if event_code == EventCode.GO_FARTHEST:
                        if self._curr_death_fog > 0 or self._progress_refractory_counter <= 0:
                            progress_bonus += self.runaway_bonus_weight
                            self._progress_refractory_counter = self.progress_refractory_period
            # run away from death fog (can get duplicate bonus, but worth rewarding)
            if self._curr_dist < min(self._last_dist[-8:]):
                if self._curr_death_fog > 0 or self._progress_refractory_counter <= 0:
                    progress_bonus += self.runaway_bonus_weight
                    self._progress_refractory_counter = self.progress_refractory_period

            # Add meandering bonus to encourage meandering (to prevent entropy collapse)
            meander_bonus = 0
            if len(self._last_moves) > 5:
                move_entropy = calculate_entropy(self._last_moves[-8:])  # of last 8 moves
                meander_bonus += self.meander_bonus_weight * (move_entropy - 1)

            # Add combat bonus to encourage combat activities that increase exp
            combat_bonus = self.combat_bonus_weight * (self._curr_combat_exp - self._last_combat_exp)

            # Add upgrade bonus to encourage leveling up offense/defense
            # NOTE: This can be triggered when a higher-level NPC drops an item that gets auto-equipped
            # Thus, it can make agents more aggressive towards npcs & equip more items
            upgrade_bonus = self.upgrade_bonus_weight * (self._new_max_offense + self._new_max_defense)

            # Unique event-based rewards, similar to exploration bonus
            # The number of unique events are available in self._curr_unique_count, self._prev_unique_count
            unique_event_bonus = min(self._curr_unique_count - self._prev_unique_count,
                                     self.clip_unique_event) * self.unique_event_bonus_weight

            # Add "Underdog" bonus to encourage attacking higher level agents
            underdog_bonus = self.underdog_bonus_weight * float(self._last_kill_level > agent.attack_level)

            # Sum up all the bonuses. Under the survival mode, ignore some bonuses
            reward += survival_bonus + progress_bonus + upgrade_bonus
            if not self._survival_mode:
                reward += meander_bonus + combat_bonus + unique_event_bonus + underdog_bonus

        return reward, done, info

    def _reset_reward_vars(self):
        # TODO: check the effectiveness of each bonus
        # highest priority: eat when starve, drink when dehydrate, run away from death fog
        self._last_health_level = 100
        self._curr_health_level = 100
        self._last_food_level = 100
        self._curr_food_level = 100
        self._last_water_level = 100
        self._curr_water_level = 100
        self._curr_death_fog = 0
        self._last_dist = []
        self._curr_dist = np.inf
        self._survival_mode = False

        # progress bonuses: eat & progress, drink & progress, run away from the death fog
        # (reward when agents eat/drink the farthest so far)
        num_basic_events = len(BASIC_BONUS_EVENTS)
        self._last_basic_events = np.zeros(num_basic_events, dtype=np.int16)
        self._last_eat_dist = np.inf
        self._last_drink_dist = np.inf
        self._progress_refractory_counter = 0

        # meander bonus (to prevent entropy collapse)
        self._last_moves = []
        self._last_price = 0  # to encourage changing price

        # main combat exp
        self._last_combat_exp = 0
        self._curr_combat_exp = 0

        # equipment, ammo-fire bonus (to level up offense/defense/ammo of the profession)
        # TODO: reward only the relevant profession
        self._max_offense = 0  # max melee/range/mage equipment offense so far
        self._new_max_offense = 0
        self._max_defense = 0  # max melee/range/mage equipment defense so far
        self._new_max_defense = 0
        self._last_ammo_fire = 0  # if an ammo was used in the last tick
        self._max_item_level = 0

        # unique event bonus (to encourage exploring new actions/items)
        self._prev_unique_count = 0
        self._curr_unique_count = 0

        # underdog bonus (to encourage attacking higher level agents)
        # NOTE: is this good? might be useful in the team setting?
        self._last_kill_level = 0

    def _update_reward_vars(self, agent):
        # From the agent
        self._last_health_level = self._curr_health_level
        self._curr_health_level = agent.resources.health.val
        self._last_food_level = self._curr_food_level
        self._curr_food_level = agent.resources.food.val
        self._last_water_level = self._curr_water_level
        self._curr_water_level = agent.resources.water.val
        self._curr_death_fog = self.env.realm.fog_map[agent.pos]
        self._last_dist.append(self._curr_dist)
        self._curr_dist = self._dist_map[agent.pos]
        self._survival_mode = True if min(self._last_health_level,
                                          self._last_food_level,
                                          self._last_water_level) <= self.survival_mode_criteria or \
                                      self._curr_death_fog >= self.death_fog_criteria \
                                    else False

        self._last_combat_exp = self._curr_combat_exp
        self._curr_combat_exp = getattr(agent.skills, self._main_combat_skill).exp.val
        max_offense = getattr(agent, self._main_combat_skill + "_attack")
        if max_offense > self._max_offense:
            self._new_max_offense = 1.0 if self.env.realm.tick > 1 else 0
            self._max_offense = max_offense
        max_defense = max(agent.melee_defense, agent.range_defense, agent.mage_defense)
        self._new_max_defense = 0
        if max_defense > self._max_defense:
            self._new_max_defense = 1.0 if self.env.realm.tick > 1 else 0
            self._max_defense = max_defense

        # From the event logs
        log = self.env.realm.event_log.get_data(agents=[self.agent_id])
        attr_to_col = self.env.realm.event_log.attr_to_col
        self._prev_unique_count = self._curr_unique_count
        self._curr_unique_count = len(extract_unique_event(log, self.env.realm.event_log.attr_to_col))
        curr_tick = log[:, attr_to_col["tick"]] == self.env.realm.tick
        for idx, event_code in enumerate(BASIC_BONUS_EVENTS):
            event_idx = curr_tick & (log[:, attr_to_col["event"]] == event_code)
            self._last_basic_events[idx] = int(sum(event_idx) > 0)
        last_ammo_idx = curr_tick & (log[:, attr_to_col["event"]] == EventCode.FIRE_AMMO) & \
                        (log[:, attr_to_col["item_type"]] == SKILL_TO_AMMO_MAP[self._main_combat_skill])
        self._last_ammo_fire = int(sum(last_ammo_idx) > 0)
        last_kill_idx = curr_tick & (log[:, attr_to_col["event"]] == EventCode.PLAYER_KILL)
        self._last_kill_level = max(log[last_kill_idx, attr_to_col["level"]]) if sum(last_kill_idx) > 0 else 0

def calculate_entropy(sequence):
    frequencies = Counter(sequence)
    total_elements = len(sequence)
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_elements
        entropy -= probability * math.log2(probability)
    return entropy
