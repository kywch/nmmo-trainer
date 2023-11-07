"""Manual test for creating learning curriculum manually"""
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin
# pylint: disable=wildcard-import,unused-wildcard-import
from typing import List

from nmmo.task.base_predicates import *
from nmmo.task.task_spec import TaskSpec, check_task_spec
import nmmo.minigames as mg

TICK_GOAL = [30, 50, 70, 100, 150, 200, 256]
ENTITY_GOAL = [1, 2, 3, 5, 7, 10]

curriculum: List[TaskSpec] = []

# Stay alive as long as possible
for ticks in TICK_GOAL:
  curriculum.append(
      TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": ticks})
  )

def ProtectAgent(gs, subject, target_protect, num_tick):  # for num_ticks
    return TickGE(gs, subject, num_tick) *\
           CheckAgentStatus(gs, subject, target_protect, status="alive")

def HeadHunting(gs, subject, target_protect, target_destroy):  # for target
    my_leader = CheckAgentStatus(gs, subject, target_protect, status="alive")
    if my_leader * CheckAgentStatus(gs, subject, target_destroy, status="dead") == 1:
        return 1
    # Give partial reward for protecting my leader
    return TickGE(gs, subject, 2000) * my_leader

for tick in TICK_GOAL:
    curriculum.append(
        TaskSpec(eval_fn=ProtectAgent,
                 eval_fn_kwargs={"target_protect": "my_team_leader",
                                 "num_tick": tick},
                 sampling_weight=10,
                 reward_to="team"))

# want the other team or team leader to die
for target in ["left_team", "left_team_leader", "right_team", "right_team_leader", "all_foes"]:
    curriculum.append(
        TaskSpec(eval_fn=CheckAgentStatus,
                 eval_fn_kwargs={"target": target, "status": "dead"},
                 sampling_weight=10,
                 reward_to="team",
                 tags=["team_battle", target]))

for target in ["left_team_leader", "right_team_leader"]:
    curriculum.append(
        TaskSpec(eval_fn=CanSeeAgent,
                 eval_fn_kwargs={"target": target},
                 reward_to="team"))

    curriculum.append(
        TaskSpec(eval_fn=HeadHunting,
                 eval_fn_kwargs={"target_destroy": target,
                                 "target_protect": "my_team_leader"},
                 sampling_weight=10,
                 reward_to="team",
                 tags=["team_battle", "head_hunt"]))

for target in ["left_team", "right_team"]:
    curriculum.append(
        TaskSpec(eval_fn=CanSeeGroup,
                 eval_fn_kwargs={"target": target},
                 reward_to="team"))

for reward_to in ["agent", "team"]:
    for num_agent in ENTITY_GOAL:
        curriculum.append(
            TaskSpec(eval_fn=DefeatEntity,
                     eval_fn_kwargs={"agent_type": "player", "level": 0, "num_agent": num_agent},
                     sampling_weight=2,
                     reward_to=reward_to))

for event_code in ["EAT_FOOD", "DRINK_WATER"]:
    for num_cnt in range(5, 50, 10):
        curriculum.append(
            TaskSpec(
                eval_fn=CountEvent,
                eval_fn_kwargs={"event": event_code, "N": num_cnt},
                sampling_weight=10,
                reward_to="agent",
            )
        )

curriculum.append(
    TaskSpec(eval_fn=mg.ProgressTowardCenter,
             eval_fn_kwargs={},
             sampling_weight=3,
             reward_to="agent",
             tags=["center_race"]))

for seize_dur in range(10, 91, 10):
    curriculum.append(
        TaskSpec(eval_fn=SeizeCenter,
                 eval_fn_kwargs={"num_ticks": seize_dur},
                 reward_to="team"))

    for quad in ["first", "second", "third", "fourth"]:
        curriculum.append(
            TaskSpec(eval_fn=SeizeQuadCenter,
                     eval_fn_kwargs={"num_ticks": seize_dur, "quadrant": quad},
                     reward_to="team"))

for quad1 in ["first", "second", "third", "fourth"]:
    curriculum.append(
        TaskSpec(eval_fn=SeizeQuadCenter,
                  eval_fn_kwargs={"num_ticks": 100, "quadrant": quad1},
                  reward_to="team",
                  tags=["unfair_fight", quad1]))

    for quad2 in ["first", "second", "third", "fourth"]:
        if quad1 != quad2:
            curriculum.append(
                TaskSpec(eval_fn=mg.SeizeBothQuads,
                         eval_fn_kwargs={"num_ticks": 100, "quadrants": [quad1, quad2]},
                         reward_to="team",
                         tags=["unfair_fight", (quad1, quad2)]))

curriculum.append(
    TaskSpec(eval_fn=SeizeCenter,
             eval_fn_kwargs={"num_ticks": 100},
             reward_to="team",
             tags=["king_hill"]))

if __name__ == "__main__":
    # Import the custom curriculum
    print("------------------------------------------------------------")
    import team_tasks  # which is this file
    CURRICULUM = team_tasks.curriculum
    print("The number of training tasks in the curriculum:", len(CURRICULUM))

    # Check if these task specs are valid in the nmmo environment
    # Invalid tasks will crash your agent training
    print("------------------------------------------------------------")
    print("Checking whether the task specs are valid ...")
    results = check_task_spec(CURRICULUM)
    num_error = 0
    for result in results:
        if result["runnable"] is False:
            print("ERROR: ", result["spec_name"])
            num_error += 1
    assert num_error == 0, "Invalid task specs will crash training. Please fix them."
    print("All training tasks are valid.")

    # The task_spec must be picklable to be used for agent training
    print("------------------------------------------------------------")
    print("Checking if the training tasks are picklable ...")
    CURRICULUM_FILE_PATH = "team_task_with_embedding.pkl"
    with open(CURRICULUM_FILE_PATH, "wb") as f:
        import dill
        dill.dump(CURRICULUM, f)
    print("All training tasks are picklable.")

    # To use the curriculum for agent training, the curriculum, task_spec, should be
    # saved to a file with the embeddings using the task encoder. The task encoder uses
    # a coding LLM to encode the task_spec into a vector.
    print("------------------------------------------------------------")
    print("Generating the task spec with embedding file ...")
    from curriculum_generation.task_encoder import TaskEncoder
    LLM_CHECKPOINT = "Salesforce/codegen-350m-mono"  # "Salesforce/codegen25-7b-instruct"

    # Get the task embeddings for the training tasks and save to file
    # You need to provide the curriculum file as a module to the task encoder
    with TaskEncoder(LLM_CHECKPOINT, team_tasks) as task_encoder:
        assert task_encoder.embed_dim == 1024, "Embed_dim must be 1024 with the codegen-350m model"
        task_encoder.get_task_embedding(CURRICULUM, save_to_file=CURRICULUM_FILE_PATH)
    print("Done.")
