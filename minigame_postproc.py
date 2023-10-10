from collections import defaultdict
import pufferlib
import pufferlib.emulation
from nmmo.entity.entity import EntityState

EntityAttr = EntityState.State.attr_name_to_col


class MiniGamePostprocessor(pufferlib.emulation.Postprocessor):
    def __init__(
            self, env, agent_id,
            eval_mode=False):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self.eval_mode = eval_mode
        self._reset_episode_stats()

    def reset(self, observation):
        self._reset_episode_stats()

    @property
    def _my_task(self):
        if self.env.agent_task_map is None:
            return None
        # NOTE: this is hacky but works for both agent and team tasks
        return self.env.agent_task_map[self.agent_id][0]

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

    def _lost_protect_task(self):
        if self.env.realm.tick < self.env.realm.config.COMBAT_SPAWN_IMMUNITY or \
           self._my_task.spec_name is None or \
           "HeadHunting" not in self._my_task.spec_name or \
           "ProtectLeader" not in self._my_task.spec_name:
            return False

        # engaged in the protect tasks
        if self._my_task.progress == 0:
            # lost the leader, lost the game, remove the agent
            agent = self.env.realm.players.get(self.agent_id)
            agent.receive_damage(None, agent.resources.health.val)
            return True

        # if the game is not over by the HORIZON, everyone loses
        if self.env.realm.tick >= self.env.realm.config.HORIZON and not self._my_task.completed:
            return True

        return False

    def reward_done_info(self, reward, done, info):
        """Update stats + info and save replays."""
        # Remove the task from info. Curriculum info is processed in _update_stats()
        info.pop('task', None)

        # This only applies for the head hunting task
        if self._lost_protect_task() is True:
            done = True
            reward = -1.0

        # Competition mode: stop early when there is a winner (i.e. only one team left) -- end the game
        if self.env.team_battle_mode and self.env.battle_winners is not None:
            done = True
            reward = len(self.env.config.TEAMS) if self.agent_id in self.env.battle_winners else -1.0

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

        # When there is a competition winner or the env reached the end
        if self.env.battle_winners is not None or\
           self.env.realm.tick >= self.env.config.HORIZON:
            return True
        for player_id in self.env.agents:  # any alive agents?
            if player_id in self.env.realm.players:
                return False
        return True
