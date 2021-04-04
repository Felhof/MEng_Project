import argparse

from resources.resource_manager import MultiAgentResourceManager
import resources.test_problems


def main(resource_manager, resource_problem_dict, training_steps=50000, steps_per_episode=500,
         search_hyperparameters=False):
    rm = resource_manager(resource_problem_dict,
                          training_steps=training_steps,
                          steps_per_episode=steps_per_episode,
                          search_hyperparameters=search_hyperparameters)
    rm.train_model()
    rm.plot_training_results()
    rm.run_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('problem', type=str,
                        help='The problem to solve')

    parser.add_argument('--hpsearch', action='store_true',
                        help='Train with hyperparameter search')

    args = parser.parse_args()

    test_problems = {
        "decomposable1": resources.test_problems.deep_decomposable_problem
    }

    problem = test_problems[args.problem]

    search_hyperparameters = args.hpsearch

    main(MultiAgentResourceManager, problem, training_steps=1000, steps_per_episode=100,
         search_hyperparameters=search_hyperparameters)
