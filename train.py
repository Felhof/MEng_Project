import argparse

from resources.resource_manager import ResourceManager, MultiAgentResourceManager
import resources.test_problems


def main(resource_manager, resource_problem_dict, training_config):
    #rm = resource_manager(resource_problem_dict,
    #                      training_config=training_config)

    rm = ResourceManager(resource_problem_dict)
    rm.train_model()
    rm.run_model()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('problem', type=str,
                        help='The problem to solve')

    parser.add_argument('--hpsearch', action='store_true',
                        help='Train with hyperparameter search')

    args = parser.parse_args()

    test_problems = {
        "tricky_problem": resources.test_problems.tricky_problem,
        "deep_decomposable": resources.test_problems.deep_decomposable_problem,
        "deep_decomposable_alt": resources.test_problems.deep_decomposable_problem_alt,
        "wide_decomposable": resources.test_problems.wide_decomposable_problem2,
        "wide_decomposable_alt": resources.test_problems.wide_decomposable_problem2_alt
    }

    problem = test_problems[args.problem]

    training_config = {
        "stage1_training_steps": 30000,
        "stage2_training_steps": 60000,
        "steps_per_episode": 100,
        "training_iterations": 10,
        "search_hyperparameters": args.hpsearch,
        "hpsearch_iterations": 10
    }

    main(MultiAgentResourceManager, problem, training_config)
