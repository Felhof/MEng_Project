import argparse
import csv
import time

from resources.resourcemanager.resource_manager import ResourceManager
from resources.resourcemanager.adp_resource_manager import ADPResourceManager
from resources.resourcemanager.marl_resource_manager import MultiAgentResourceManager
import resources.test_problems
from resources.reward_checkpoints import CheckpointResults

def main(resource_manager, resource_problem_dict, training_config, algorithm="A2C", problem_name="_multi",
         iterations=5):
    model_name = resource_problem_dict["name"] + problem_name + "_" + algorithm
    checkpoint_results = CheckpointResults(model_name=model_name)

    start = time.time()
    for _ in range(iterations):
        rm = resource_manager(resource_problem_dict,
                              training_config=training_config,
                              algorithm=algorithm,
                              checkpoint_results=checkpoint_results)
        rm.train_model()
    end = time.time()
    print("time:", end - start)

    checkpoint_results.save_results()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('problem', type=str,
                        help='The problem to solve')
    parser.add_argument('--hpsearch', action='store_true',
                        help='Train with hyperparameter search')
    parser.add_argument('--baseline', action='store_true',
                        help='calculate baseline result')
    parser.add_argument('--algo', type=str,
                        help="which algorithm to use (A2C or PPO)")
    parser.add_argument('--stage1steps', type=int,
                        help="how many training steps in stage 1")
    parser.add_argument('--stage2steps', type=int,
                        help="how many training steps in stage 2")
    parser.add_argument('--i', type=int,
                        help="how often to run the experiment")
    parser.add_argument('--l', action='store_true',
                        help='load stage 1 models from previous training run')
    parser.add_argument('--ad', action="store_true",
                        help='use Abbad-Daoui decomposition')

    args = parser.parse_args()

    test_problems = {
        "split_on_best_5": resources.test_problems.split_on_best_5,
        "split_on_best_6": resources.test_problems.split_on_best_6,
        "split_on_best_7": resources.test_problems.split_on_best_7,
        "split_on_worst_6": resources.test_problems.split_on_worst_6,
        "split_on_worst_7": resources.test_problems.split_on_worst_7,
        "unequal_rewards_varied_departure_p_5": resources.test_problems.unequal_rewards_varied_departure_p_5,
        "unequal_rewards_varied_departure_p_6": resources.test_problems.unequal_rewards_varied_departure_p_6,
        "unequal_rewards_varied_departure_p_7": resources.test_problems.unequal_rewards_varied_departure_p_7,
        "equal_rewards_varied_departure_p_8": resources.test_problems.equal_rewards_varied_departure_p_8,
        "many_tasks": resources.test_problems.many_tasks,
        "many_tasks_v2": resources.test_problems.many_tasks_v2,
        "many_tasks_small": resources.test_problems.many_tasks_small,
    }

    problem = test_problems[args.problem]

    stage1steps = args.stage1steps if args.stage1steps is not None else 50000
    stage2steps = args.stage2steps if args.stage2steps is not None else 50000

    training_config = {
        "stage1_training_steps": stage1steps,
        "stage2_training_steps": stage2steps,
        "steps_per_episode": 500,
        "training_iterations": 10,
        "search_hyperparameters": args.hpsearch,
        "hpsearch_iterations": 10,
        "show": True,
        "load": args.l
    }

    name = "_dl"
    if args.baseline:
        resource_manager = ResourceManager
        name = "_baseline"
    elif args.ad:
        resource_manager = MultiAgentResourceManager
        name = "_ad"
    else:
        resource_manager = ADPResourceManager

    algo = args.algo if args.algo is not None else "A2C"

    iterations = args.i if args.i is not None else 10

    main(resource_manager, problem, training_config, algorithm=algo, problem_name=name, iterations=iterations)
