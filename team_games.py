from nmmo.core.game_api import AgentTraining, TeamTraining, TeamBattle
from nmmo.task import task_spec


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
    #config.set_for_episode("TERRAIN_DISABLE_STONE", True)

    # Activate death fog
    config.set_for_episode("PLAYER_DEATH_FOG", 128)
    config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/8)
    config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 3)

    # Small health regen every tick
    config.set_for_episode("PLAYER_HEALTH_INCREMENT", True)

class MiniAgentTraining(AgentTraining):
    required_systems = ["TERRAIN", "RESOURCE", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        self.config.reset()
        self.config.toggle_systems(self.required_systems)

        # fog setting for the race to center
        self.config.set_for_episode("PLAYER_DEATH_FOG", 32)
        self.config.set_for_episode("PLAYER_DEATH_FOG_SPEED", 1/4)
        # Only the center tile is safe
        self.config.set_for_episode("PLAYER_DEATH_FOG_FINAL_SIZE", 0)

class MiniTeamTraining(TeamTraining):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        combat_training_config(self.config)

class MiniTeamBattle(TeamBattle):
    required_systems = ["TERRAIN", "COMBAT"]

    def is_compatible(self):
        return self.config.are_systems_enabled(self.required_systems)

    def _set_config(self):
        combat_training_config(self.config)

    def _define_tasks(self, np_random):
        sampled_spec = self._get_cand_team_tasks(np_random, num_tasks=1, tags="team_battle")[0]
        return task_spec.make_task_from_spec(self.config.TEAMS,
                                             [sampled_spec] * len(self.config.TEAMS))
