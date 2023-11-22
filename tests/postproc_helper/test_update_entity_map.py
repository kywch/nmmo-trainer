from timeit import timeit
import numpy as np
import postproc_helper as pph

entity_map = np.zeros((160,160), dtype=np.int16)
entity_obs = (np.random.rand(100, 10)*160).astype(np.int16)
entity_obs[:, 0] = np.arange(-20, 80)
# make realistic npc type
entity_obs[entity_obs[:, 0] < 0, 3] = entity_obs[entity_obs[:, 0] < 0, 3]%3 + 1
entity_obs[entity_obs[:, 0] > 0, 3] = 0

entity_attr = {"id": 0, "row": 1, "col": 2, "npc_type": 3}
const_dict = {"ENEMY_REPR": 4,
              "DESTROY_TARGET_REPR": 5,
              "TEAMMATE_REPR": 6,
              "PROTECT_TARGET_REPR": 7,
              "my_team": tuple([1, 2, 3, 4, 5]),
              "target_destroy": tuple([6, 7, 8, 9]),
              "target_protect": tuple([10, 11]),}
my_team = [1, 2, 3, 4, 5]

# reference python implementation
def ref_update_entity_map(entity_map, entity_obs, entity_attr, const_dict):
  entity_map[:] = 0
  entity_idx = entity_obs[:,entity_attr["id"]] != 0
  for entity in entity_obs[entity_idx]:
      ent_pos = (entity[entity_attr["row"]], entity[entity_attr["col"]])
      if entity[entity_attr["id"]] < 0:
          npc_type = entity[entity_attr["npc_type"]]
          entity_map[ent_pos] = max(npc_type, entity_map[ent_pos])
      if entity[entity_attr["id"]] > 0 and entity[entity_attr["npc_type"]] == 0:
          entity_map[ent_pos] = max(const_dict["ENEMY_REPR"], entity_map[ent_pos])
          if entity[entity_attr["id"]] in const_dict["target_destroy"]:
              entity_map[ent_pos] = max(const_dict["DESTROY_TARGET_REPR"], entity_map[ent_pos])
          if entity[entity_attr["id"]] in const_dict["my_team"]:
              entity_map[ent_pos] = max(const_dict["TEAMMATE_REPR"], entity_map[ent_pos])
          if entity[entity_attr["id"]] in const_dict["target_protect"]:
              entity_map[ent_pos] = max(const_dict["PROTECT_TARGET_REPR"], entity_map[ent_pos])

ref_time = timeit(lambda: ref_update_entity_map(entity_map, entity_obs, entity_attr, const_dict),
                  number=10000, globals=globals())
print("Reference:", ref_time)
ref_map = entity_map.copy()

pph.update_entity_map(entity_map, entity_obs, entity_attr, const_dict)
imp_time = timeit(lambda: pph.update_entity_map(entity_map, entity_obs, entity_attr, const_dict),
                  number=10000, globals=globals())
print("Cython:", imp_time)

# Reference: 18.969731437995506
# Cython: 0.4687720790025196

assert np.array_equal(ref_map, entity_map)
