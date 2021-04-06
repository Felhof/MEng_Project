import argparse

from resources.resource_manager import MultiAgentResourceManager
import resources.test_problems


def main(resource_manager, resource_problem_dict, training_config):
    rm = resource_manager(resource_problem_dict,
                          training_config=training_config)
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
        "deep_decomposable": resources.test_problems.deep_decomposable_problem,
        "wide_decomposable": resources.test_problems.wide_decomposable_problem2
    }

    problem = test_problems[args.problem]

    training_config = {
        "stage1_training_steps": 200,
        "stage2_training_steps": 500,
        "steps_per_episode": 100,
        "training_iterations": 10,
        "search_hyperparameters": args.hpsearch,
        "hpsearch_iterations": 10
    }

    main(MultiAgentResourceManager, problem, training_config)
