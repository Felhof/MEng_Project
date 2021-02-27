import argparse

from resources.resource_manager import MultiAgentResourceManager
import resources.test_problems


def main(resource_manager, resource_problem_dict, training_steps=50000, steps_per_episode=500):
    rm = resource_manager(resource_problem_dict, training_steps=training_steps,
                          steps_per_episode=steps_per_episode)
    rm.train_model()
    rm.save_training_results()
    rm.run_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('problem', type=str,
                        help='The problem to solve')

    parser.add_argument('--hpsearch', action='store_true',
                        help='Train with hyperparameter search')

    args = parser.parse_args()

    test_problems = {
        "decomposable_problem": resources.test_problems.decomposable_problem
    }

    problem = test_problems[args.problem]

    main(MultiAgentResourceManager, problem, training_steps=20000, steps_per_episode=100)
