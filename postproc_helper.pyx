#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
import numpy as np

def update_entity_map(short [:, :] entity_map,
                      short [:, ::1] entity_obs,
                      dict entity_attr,
                      dict const_dict):
    cdef short idx, row, col
    entity_map[:] = 0
    for idx in range(entity_obs.shape[0]):
        if entity_obs[idx, entity_attr["id"]] == 0:
            continue
        row, col = entity_obs[idx, entity_attr["row"]], entity_obs[idx, entity_attr["col"]]
        if entity_obs[idx, entity_attr["id"]] < 0:
            entity_map[row, col] = max(entity_obs[idx, entity_attr["npc_type"]], entity_map[row, col])
        if entity_obs[idx, entity_attr["id"]] > 0 and entity_obs[idx, entity_attr["npc_type"]] == 0:
            entity_map[row, col] = max(const_dict["ENEMY_REPR"], entity_map[row, col])
            if entity_obs[idx, entity_attr["id"]] in const_dict["target_destroy"]:
                entity_map[row, col] = max(const_dict["DESTROY_TARGET_REPR"], entity_map[row, col])
            if entity_obs[idx, entity_attr["id"]] in const_dict["my_team"]:
                entity_map[row, col] = max(const_dict["TEAMMATE_REPR"], entity_map[row, col])
            if entity_obs[idx, entity_attr["id"]] in const_dict["target_protect"]:
                entity_map[row, col] = max(const_dict["PROTECT_TARGET_REPR"], entity_map[row, col])

def compute_comm_action(bint can_see_target,
                        short my_health,
                        short [:, ::1] entity_obs,
                        dict entity_attr,
                        dict const_dict):
    cdef short idx, row, col
    cdef short peri_enemy = 0
    cdef short peri_npc = 0

    my_health = (my_health // 34) + 1  # 1 - 3
    for idx in range(entity_obs.shape[0]):
        if entity_obs[idx, entity_attr["id"]] == 0:
            continue
        if entity_obs[idx, entity_attr["id"]] < 0:
            peri_npc += 1
        if entity_obs[idx, entity_attr["id"]] > 0 and \
           entity_obs[idx, entity_attr["id"]] in const_dict["my_team"]:
            peri_enemy += 1
    peri_enemy = min((peri_enemy+3)//4, 3)  # 0: no enemy, 1: 1-4, 2: 5-8, 3: 9+
    peri_npc = min((peri_npc+3)//4, 3)  # 0: no npc, 1: 1-4, 2: 5-8, 3: 9+
    return can_see_target << 5 | peri_enemy << 4 | peri_npc << 2 | my_health
