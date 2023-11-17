import dill
from nmmo.task import task_spec
import nmmo.core.game_api as ga
import nmmo.minigames as mg
from nmmo.lib import team_helper


def combat_training_config(config, required_systems = ["TERRAIN", "COMBAT"]):
    config.reset()
    config.toggle_systems(required_systems)
    config.set_for_episode("ALLOW_MOVE_INTO_OCCUPIED_TILE", False)

    # Make the map center. NOTE: MAP_SIZE cannot be changed
    config.set_for_episode("MAP_CENTER", 32)

    # Regenerate the map from fractal to have less obstacles
    config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
    config.set_for_episode("TERRAIN_WATER", 0.05)
    config.set_for_episode("TERRAIN_FOILAGE", 0.95)  # prop of stone tiles: 0.05

    # Activate death fog
    config.set_for_episode("DEATH_FOG_ONSET", 128)
    config.set_for_episode("DEATH_FOG_SPEED", 1/8)
    config.set_for_episode("DEATH_FOG_FINAL_SIZE", 3)

    # Small health regen every tick
    config.set_for_episode("PLAYER_HEALTH_INCREMENT", True)

def check_curriculum_file(config):
    try:
        with open(config.CURRICULUM_FILE_PATH, "rb") as f:
            dill.load(f)  # a list of TaskSpec
        return True
    except:
        return False

class MiniAgentTraining(ga.AgentTraining):
    required_systems = ["TERRAIN", "RESOURCE", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self, np_random):
        self.config.reset()
        self.config.toggle_systems(self.required_systems)

        # The default option does not provide enough resources
        self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)

        # fog setting for the race to center
        self.config.set_for_episode("DEATH_FOG_ONSET", 32)
        self.config.set_for_episode("DEATH_FOG_SPEED", 1/4)
        # Only the center tile is safe
        self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 0)

class MiniTeamTraining(ga.TeamTraining):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self, np_random):
        combat_training_config(self.config)

class MiniTeamBattle(ga.TeamBattle):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self, np_random):
        combat_training_config(self.config)

    def _define_tasks(self, np_random):
        sampled_spec = self._get_cand_team_tasks(np_random, num_tasks=1, tags="team_battle")[0]
        return task_spec.make_task_from_spec(self.config.TEAMS,
                                             [sampled_spec] * len(self.config.TEAMS))

class RacetoCenter(mg.RacetoCenter):
    def __init__(self, env, sampling_weight=None):
        super().__init__(env, sampling_weight)
        self._map_size = 24  # start from a smaller map

    def is_compatible(self):
        return check_curriculum_file(self.config) and super().is_compatible()

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        race_task = [spec for spec in curriculum if "center_race" in spec.tags]
        assert len(race_task) == 1, "There should be only one task with the tag"
        race_task *= self.config.PLAYER_N
        return task_spec.make_task_from_spec(self.config.POSSIBLE_AGENTS, race_task)

class KingoftheHill(mg.KingoftheHill):
    num_game_won = 3  # wins to increase seize duration

    def is_compatible(self):
        return check_curriculum_file(self.config) and super().is_compatible()

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        team_task = [spec for spec in curriculum if "king_hill" in spec.tags and "center" in spec.tags]
        assert len(team_task) == 1, "There should be only one task with the tag"
        team_task[0].eval_fn_kwargs["num_ticks"] = self._seize_duration
        team_task *= len(self.teams)
        return task_spec.make_task_from_spec(self.teams, team_task)

class EasyKingoftheHill(KingoftheHill):
    num_game_won = 1  # wins to increase seize duration

    def _set_config(self, np_random):
        super()._set_config(np_random)
        # make the game easier by decreasing the resource demands/penalty
        self.config.set_for_episode("RESOURCE_DEPLETION_RATE", 2)
        self.config.set_for_episode("RESOURCE_RESILIENT_POPULATION", 1)  # 100%
        self.config.set_for_episode("RESOURCE_DAMAGE_REDUCTION", 0.2)  # reduce to 20%
        self.config.set_for_episode("RESOURCE_HEALTH_RESTORE_FRACTION", .02)

class EasyKingoftheQuad(EasyKingoftheHill):
    _next_seize_target = None
    quadrants = ["first", "second", "third", "fourth"]

    def set_seize_target(self, target):
        assert target in self.quadrants, "Invalid target"
        self._next_seize_target = target

    def _set_realm(self, np_random, map_dict):
        if self._next_seize_target is None:
            self._next_seize_target = np_random.choice(self.quadrants)
        self.realm.reset(np_random, map_dict, custom_spawn=True,
                         seize_targets=[self._next_seize_target])
        # team spawn requires custom spawning
        team_loader = team_helper.TeamLoader(self.config, np_random)
        self.realm.players.spawn(team_loader)

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        team_task = [spec for spec in curriculum
                     if "king_hill" in spec.tags and self._next_seize_target in spec.tags]
        self._next_seize_target = None
        assert len(team_task) == 1, "There should be only one task with the tag"
        team_task[0].eval_fn_kwargs["num_ticks"] = self.seize_duration
        team_task *= len(self.teams)
        return task_spec.make_task_from_spec(self.teams, team_task)

class Sandwich(mg.Sandwich):
    _next_grass_map = None

    def is_compatible(self):
        return check_curriculum_file(self.config) and super().is_compatible()

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def _set_config(self, np_random):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map or np_random.choice([True, False], p=[0.2, 0.8])
        super()._set_config(np_random)

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        team_task = [spec for spec in curriculum if "king_hill" in spec.tags and "center" in spec.tags]
        assert len(team_task) == 1, "There should be only one task with the tag"
        team_task[0].eval_fn_kwargs["num_ticks"] = self.seize_duration
        team_task *= len(self.teams)
        return task_spec.make_task_from_spec(self.teams, team_task)

class CommTogether(mg.CommTogether):
    _next_grass_map = None

    def is_compatible(self):
        return check_curriculum_file(self.config) and super().is_compatible()

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def _set_config(self, np_random):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map or np_random.choice([True, False], p=[0.2, 0.8])
        super()._set_config(np_random)

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        team_task = [spec for spec in curriculum if "sync_seek" in spec.tags]
        assert len(team_task) == 1, "There should be only one task with the tag"
        return task_spec.make_task_from_spec(self.teams, team_task*len(self.teams))
