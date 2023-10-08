from timeit import timeit

import nmmo

from map_helper import MapHelper

config = nmmo.config.Tutorial()
config.PROVIDE_DEATH_FOG_OBS = True
env = nmmo.Env(config)
obs = env.reset()

agent_id = 1

map_helper = MapHelper(config, agent_id)
map_helper.reset()

map_helper.update(obs[agent_id])
img = map_helper.extract_tile_feature()

# @profile
# def test():
#     for _ in range(20000):
#         map_helper.update(obs[agent_id])

# 6.9 sec
print(timeit(lambda: map_helper.update(obs[agent_id]), number=10000))

# 2.3 sec
print(timeit(lambda: map_helper.extract_tile_feature(), number=10000))

#test()

print()