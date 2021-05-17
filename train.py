import argparse
import csv

from resources.resourcemanager.resource_manager import ResourceManager
from stable_baselines3 import A2C, PPO, DDPG
from resources.resourcemanager.adp_resource_manager import ADPResourceManager
import resources.test_problems


def main(resource_manager, resource_problem_dict, training_config, algorithm="A2C", problem_name="_multi",
         iterations=10):
    name = resource_problem_dict["name"] + problem_name + "_" + algorithm

    rewards = []
    for _ in range(iterations):
        rm = resource_manager(resource_problem_dict,
                              training_config=training_config,
                              algorithm=algorithm)
        rm.train_model()
        reward = rm.run_model(n_steps=10000, save=True, name=name)
        rewards.append(reward)

    location = 'data/{}_rewards.csv'.format(name)
    with open(location, mode='w') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(rewards)


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

    args = parser.parse_args()

    test_problems = {
        "tricky_problem": resources.test_problems.tricky_problem,
        "deep_decomposable": resources.test_problems.deep_decomposable_problem,
        "deep_decomposable_alt": resources.test_problems.deep_decomposable_problem_alt,
        "wide_decomposable": resources.test_problems.wide_decomposable_problem2,
        "wide_decomposable_alt": resources.test_problems.wide_decomposable_problem2_alt,
        "adp_problem": resources.test_problems.adp_problem,
        "adp_problem2": resources.test_problems.adp_problem2,
        "adp_problem_big": resources.test_problems.adp_problem_big,
        "adp_problem_massive": resources.test_problems.adp_problem_massive,
        "adp_problem_climb": resources.test_problems.adp_problem_climb,
        "adp_problem_massive2": resources.test_problems.adp_problem_massive2,
        "adp_problem_very_big": resources.test_problems.adp_problem_very_big,
        "adp_problem_insane": resources.test_problems.adp_problem_insane,
        "adp_problem_insane2": resources.test_problems.adp_problem_insane2
    }

    problem = test_problems[args.problem]

    training_config = {
        "stage1_training_steps": 50000,
        "stage2_training_steps": 50000,
        "steps_per_episode": 100,
        "training_iterations": 10,
        "search_hyperparameters": args.hpsearch,
        "hpsearch_iterations": 10,
        "show": False
    }

    name = "_multi"
    if args.baseline:
        resource_manager = ResourceManager
        name = "_baseline"
    else:
        #resource_manager = MultiAgentResourceManager
        resource_manager = ADPResourceManager

    if args.algo is None:
        algo = "A2C"
    else:
        algo = args.algo

    main(resource_manager, problem, training_config, algorithm=algo, problem_name=name)
