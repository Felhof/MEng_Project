from resources.resource_manager import MultiAgentResourceManager
from resources.test_problems import decomposable_problem


def main(resource_manager, resource_problem_dict, training_steps=50000, steps_per_episode=500, restricted_tasks=None):
    rm = resource_manager(resource_problem_dict, restricted_tasks=restricted_tasks, training_steps=training_steps,
                          steps_per_episode=steps_per_episode)
    #rm = resource_manager(resource_problem_dict, training_steps=training_steps, steps_per_episode=steps_per_episode)
    rm.train_model()
    rm.save_training_results()
    rm.run_model()


restricted_tasks = [3]

main(MultiAgentResourceManager, decomposable_problem, training_steps=20000, steps_per_episode=100,
     restricted_tasks=restricted_tasks)
