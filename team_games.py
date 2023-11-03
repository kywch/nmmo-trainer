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
    config.set_for_episode("PLAYER_DEATH_FOG", 128)
    config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/8)
    config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 3)

    # Small health regen every tick
    config.set_for_episode("PLAYER_HEALTH_INCREMENT", True)

class MiniAgentTraining(ga.AgentTraining):
    required_systems = ["TERRAIN", "RESOURCE", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        self.config.reset()
        self.config.toggle_systems(self.required_systems)

        # The default option does not provide enough resources
        self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)

        # fog setting for the race to center
        self.config.set_for_episode("PLAYER_DEATH_FOG", 32)
        self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/4)
        # Only the center tile is safe
        self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 0)

class MiniTeamTraining(ga.TeamTraining):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        combat_training_config(self.config)

class MiniTeamBattle(ga.TeamBattle):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        combat_training_config(self.config)

    def _define_tasks(self, np_random):
        sampled_spec = self._get_cand_team_tasks(np_random, num_tasks=1, tags="team_battle")[0]
        return task_spec.make_task_from_spec(self.config.TEAMS,
                                             [sampled_spec] * len(self.config.TEAMS))

class FourTeamBattle(ga.TeamBattle):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    @staticmethod
    def teams(num_players):
        num_agent = num_players // 4
        return {
            "team1": list(range(1, num_agent+1)),
            "team2": list(range(num_agent+1, 2*num_agent+1)),
            "team3": list(range(2*num_agent+1, 3*num_agent+1)),
            "team4": list(range(3*num_agent+1, num_players+1)),
        }

    def _set_config(self):
        combat_training_config(self.config)
        self.config.set_for_episode("TEAMS", self.teams(self.config.PLAYER_N))

    def _define_tasks(self, np_random):
        sampled_spec = self._get_cand_team_tasks(np_random, num_tasks=1, tags="team_battle")[0]
        return task_spec.make_task_from_spec(self.config.TEAMS,
                                             [sampled_spec] * len(self.config.TEAMS))

    def _set_realm(self, np_random, map_dict):
        self.realm.reset(np_random, map_dict, custom_spawn=True)
        # Custom spawning: candidate_locs should be a list of list of (row, col) tuples
        candidate_locs = [[(70, 70)], [(90, 90)], [(70, 90)], [(90, 70)]]
        # Also, one should make sure these locations are spawnable
        for loc_list in candidate_locs:
            for loc in loc_list:
              self.realm.map.make_spawnable(*loc)
        team_loader = team_helper.TeamLoader(self.config, np_random, candidate_locs)
        self.realm.players.spawn(team_loader)

class RacetoCenter(mg.RacetoCenter):
    def __init__(self, env, sampling_weight=None):
        super().__init__(env, sampling_weight)
        self.map_center = 24  # start from a smaller map

    def is_compatible(self):
        try:
          with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
            dill.load(f)  # a list of TaskSpec
        except:
          return False
        return super().is_compatible()

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        race_task = [spec for spec in curriculum if "center_race" in spec.tags]
        assert len(race_task) == 1, "There should be only one task with the tag"
        race_task *= self.config.PLAYER_N
        return task_spec.make_task_from_spec(self.config.POSSIBLE_AGENTS, race_task)

class UnfairFight(mg.UnfairFight):
    def is_compatible(self):
        try:
          with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
            dill.load(f)  # a list of TaskSpec
        except:
          return False
        return super().is_compatible()

    def _define_tasks(self, np_random):
        # Changed to use the curriculum file
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
          curriculum = dill.load(f) # a list of TaskSpec
        def_task = [spec for spec in curriculum if "unfair_def" in spec.tags]
        off_task = [spec for spec in curriculum if "team_battle" in spec.tags and "all_foes" in spec.tags]
        assert len(def_task) == 1 and len(off_task) == 1, "There should be one and only task with the tags"
        return task_spec.make_task_from_spec(self.teams, def_task + off_task)
