from collections import defaultdict
import pufferlib
import pufferlib.emulation
from nmmo.entity.entity import EntityState
from nmmo.lib.event_log import EventCode
import team_games as tg

EntityAttr = EntityState.State.attr_name_to_col


class MiniGamePostprocessor(pufferlib.emulation.Postprocessor):
    def __init__(
            self, env, agent_id,
            eval_mode=False,
            detailed_stat=True):
        super().__init__(env, is_multiagent=True, agent_id=agent_id)
        self.eval_mode = eval_mode
        self.detailed_stat = detailed_stat
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

        if self.is_env_done():
            done = True
            reward = -1.0
            if self.env.game.winners and self.agent_id in self.env.game.winners:
                reward = self.env.game.winning_score

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

        # For now, we only log EAT_FOOD, DRINK_WATER events
        if self.env.config.RESOURCE_SYSTEM_ENABLED:
            game_name = self.env.game.__class__.__name__
            event_cnt = process_event_log(self.env.realm, self.agent_id)
            for key, val in event_cnt.items():
                info["stats"][game_name+"/"+key] = float(val > 0)  # whether an agent did it

        info["return"] = self.epoch_return
        info["length"] = self.epoch_length
        info["inflicted_damage"] = agent.history.damage_inflicted

        if self.eval_mode:
            # "return" is used for ranking in the eval mode, so put the task progress here
            info["return"] = self._max_task_progress  # this is 1 if done

        if self.detailed_stat and self.is_env_done() and \
           self.agent_id == min(self.env.agents):  # to avoid duplicate stats
            game_name = self.env.game.__class__.__name__
            for key, val in self.env.game.get_episode_stats().items():
                info["stats"][game_name+"/"+key] = val
            info["stats"][game_name+"/finished_tick"] = self.env.realm.tick
            if isinstance(self.env.game, tg.RacetoCenter) or isinstance(self.env.game, tg.KingoftheHill):
                info["stats"][game_name+"/game_won"] = self.env.game.winners is not None
                info["stats"][game_name+"/map_size"] = self.env.game.map_size
                max_progress = [task.progress_info["max_progress"] for task in self.env.game.tasks]
                info["stats"][game_name+"/max_progress"] = max(max_progress)
                info["stats"][game_name+"/avg_max_prog"] = sum(max_progress)/len(max_progress)
                if self.env.game.winners:
                    info["stats"][game_name+"/winning_score"] = self.env.game.winning_score
            if isinstance(self.env.game, tg.UnfairFight):
                num_win_alive = sum(1 for agent_id in self.env.game.winners
                                    if agent_id in self.env.realm.players)
                info["stats"][game_name+"/num_win_alive"] = num_win_alive
                large_won = 1 not in self.env.game.winners
                info["stats"][game_name+"/large_won"] = large_won
                if large_won:
                    info["stats"][game_name+"/large_win_score"] = self.env.game.winning_score
            if isinstance(self.env.game, tg.KingoftheHill):
                  info["stats"][game_name+"/seize_duration"] = self.env.game.seize_duration

        return reward, done, info

    def is_env_done(self):
        # When there are declared winners (i.e. only one team left) or the time is up
        return self.env.game.winners or \
               self.env.realm.tick >= self.env.realm.config.HORIZON or \
               self.env.realm.num_players == 0

# convert the numbers into binary (performed or not) for the key events
KEY_EVENT = [
    "eat_food",
    "drink_water",
    "player_kill",
]

INFO_KEY_TO_EVENT_CODE = {
    evt.lower(): val for evt, val in EventCode.__dict__.items()
    if isinstance(val, int) and evt.lower() in KEY_EVENT
}

def process_event_log(realm, agent_list):
    """Process the event log and extract performed actions and achievements."""
    log = realm.event_log.get_data(agents=agent_list)
    attr_to_col = realm.event_log.attr_to_col
    # count the number of events
    event_cnt = {}
    for key, code in INFO_KEY_TO_EVENT_CODE.items():
        # count the freq of each event
        event_cnt[key] = int(sum(log[:, attr_to_col["event"]] == code))
    return event_cnt
