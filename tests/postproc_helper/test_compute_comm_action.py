from timeit import timeit
import numpy as np
import postproc_helper as pph

const_dict = {"ENEMY_REPR": np.array([4, 5]),
              "NPC_REPR": np.array([1, 2, 3]),
              "my_team": tuple([1, 2, 3, 4, 5]),}
can_see_target = True
my_health = 75
entity_map = (np.random.rand(160, 160)*6).astype(np.int16)

entity_attr = {"id": 0, "row": 1, "col": 2, "npc_type": 3}
entity_obs = (np.random.rand(100, 10)*160).astype(np.int16)
entity_obs[:, 0] = np.arange(-20, 80)
# make realistic npc type
entity_obs[entity_obs[:, 0] < 0, 3] = entity_obs[entity_obs[:, 0] < 0, 3]%3 + 1
entity_obs[entity_obs[:, 0] > 0, 3] = 0

# reference python implementation
def compute_comm_action(can_see_target, my_health, entity_map, const_dict):
    # comm action values range from 0 - 127, 0: dummy obs

    # check this outside the function
    # if self.agent_id not in self.env.realm.players:
    #     return 0
    my_health = (my_health // 34) + 1  # 1 - 3
    num_enemy = np.sum(np.isin(entity_map, const_dict["ENEMY_REPR"]))
    peri_enemy = min((num_enemy+3)//4, 3)  # 0: no enemy, 1: 1-4, 2: 5-8, 3: 9+
    num_npc = np.sum(np.isin(entity_map, const_dict["NPC_REPR"]))
    peri_npc = min((num_npc+3)//4, 3)  # 0: no npc, 1: 1-4, 2: 5-8, 3: 9+
    return can_see_target << 5 | int(peri_enemy) << 4 | int(peri_npc) << 2 | int(my_health)

# different python implementation
def alt_comm_action(can_see_target, my_health, entity_obs, entity_attr, const_dict):
    my_health = (my_health // 34) + 1  # 1 - 3
    entity_idx = entity_obs[:,entity_attr["id"]] != 0
    peri_enemy = 0
    peri_npc = 0
    for entity in entity_obs[entity_idx]:
        if entity[entity_attr["id"]] < 0:
            peri_npc += 1
        if entity[entity_attr["id"]] > 0 and \
           entity[entity_attr["id"]] not in const_dict["my_team"]:
            peri_enemy += 1
    peri_enemy = min((peri_enemy+3)//4, 3)  # 0: no enemy, 1: 1-4, 2: 5-8, 3: 9+
    peri_npc = min((peri_npc+3)//4, 3)  # 0: no npc, 1: 1-4, 2: 5-8, 3: 9+
    return can_see_target << 5 | int(peri_enemy) << 4 | int(peri_npc) << 2 | int(my_health)

ref_time = timeit(lambda: compute_comm_action(can_see_target, my_health, entity_map, const_dict),
                  number=10000, globals=globals())
print("Reference:", ref_time)
#ref_map = entity_map.copy()

alt_time = timeit(lambda: alt_comm_action(can_see_target, my_health, entity_obs, entity_attr, const_dict),
                  number=10000, globals=globals())
print("Alt python:", alt_time)
ref_map = entity_map.copy()

imp_time = timeit(lambda: pph.compute_comm_action(can_see_target, my_health, entity_obs, entity_attr, const_dict),
                  number=10000, globals=globals())
print("Cython:", imp_time)

# Reference: 1.0252557860003435
# Alt python: 8.914011212000332
# Cython: 0.1068727919991943

assert np.array_equal(ref_map, entity_map)
