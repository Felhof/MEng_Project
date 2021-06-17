from resources.resourcemanager.resourcemanager import ResourceManager
from resources.resourcemanager.adp_resourcemanager import ADPResourceManager
from resources.resourcemanager.marl_resourcemanager import MultiAgentResourceManager
import resources.rap_configurations
from resources.reward_checkpoints import CheckpointResults

import argparse


def main(resource_manager, resource_problem_dict, training_config, algorithm="A2C", problem_name="_multi"):
    model_name = resource_problem_dict["name"] + problem_name + "_" + algorithm
    checkpoint_results = CheckpointResults(checkpoints=resource_problem_dict["Checkpoints"], model_name=model_name)

    rm = resource_manager(resource_problem_dict,
                          training_config=training_config,
                          algorithm=algorithm,
                          checkpoint_results=checkpoint_results)
    rm.train_model()

    checkpoint_results.save_results()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('configuration', type=str,
                        help='The name of the rap configuration to solve. See resources/rap_configurations.py for '
                             'configurations. Example: simple_problem_A')
    parser.add_argument('--hpsearch', action='store_true',
                        help='Train with hyperparameter search')
    parser.add_argument('--baseline', action='store_true',
                        help='Train model without decomposition. '
                             'If neither this flag nor ad is set, DLCA will be used.')
    parser.add_argument('--algo', type=str,
                        help="Which algorithm to use: A2C or PPO. Default: A2C)")
    parser.add_argument('--stage1steps', type=int,
                        help="How many training steps in stage 1. Default: 50000")
    parser.add_argument('--stage2steps', type=int,
                        help="How many training steps in stage 2. Default: 50000")
    parser.add_argument('--l', action='store_true',
                        help='Load stage 1 models from previous training run. Only works for Dean-Lin decomposition.'
                             'Models must be saved in models folder.')
    parser.add_argument('--ad', action="store_true",
                        help='Use A2D2 decomposition. '
                             'If neither this flag nor baseline is set, DLCA will be used.')

    args = parser.parse_args()

    test_problems = {
        "split_on_best_5": resources.rap_configurations.split_on_best_5,
        "split_on_best_6": resources.rap_configurations.split_on_best_6,
        "split_on_best_7": resources.rap_configurations.split_on_best_7,
        "split_on_worst_6": resources.rap_configurations.split_on_worst_6,
        "split_on_worst_7": resources.rap_configurations.split_on_worst_7,
        "unequal_rewards_varied_departure_p_5": resources.rap_configurations.unequal_rewards_varied_departure_p_5,
        "unequal_rewards_varied_departure_p_6": resources.rap_configurations.unequal_rewards_varied_departure_p_6,
        "unequal_rewards_varied_departure_p_7": resources.rap_configurations.unequal_rewards_varied_departure_p_7,
        "unequal_rewards_varied_departure_p_6_A": resources.rap_configurations.unequal_rewards_varied_departure_p_6_A,
        "unequal_rewards_varied_departure_p_7_A": resources.rap_configurations.unequal_rewards_varied_departure_p_7_A,
        "unequal_rewards_varied_departure_p_6_B": resources.rap_configurations.unequal_rewards_varied_departure_p_6_B,
        "unequal_rewards_varied_departure_p_7_B": resources.rap_configurations.unequal_rewards_varied_departure_p_7_B,
        "equal_rewards_varied_departure_p_8": resources.rap_configurations.equal_rewards_varied_departure_p_8,
        "many_tasks": resources.rap_configurations.many_tasks,
        "many_tasks_v2": resources.rap_configurations.many_tasks_v2,
        "many_tasks_small": resources.rap_configurations.many_tasks_small,
        "simple_problem_A": resources.rap_configurations.simple_problem_A,
        "simple_problem_B": resources.rap_configurations.simple_problem_B,
        "simple_problem_C": resources.rap_configurations.simple_problem_C,
        "simple_problem_D": resources.rap_configurations.simple_problem_D,
        "simple_problem_E": resources.rap_configurations.simple_problem_E,
        "varying_departure_p_C": resources.rap_configurations.varying_departure_p_C,
        "varying_departure_p_D": resources.rap_configurations.varying_departure_p_D,
        "varying_departure_p_E": resources.rap_configurations.varying_departure_p_E,
        "varying_cost_C": resources.rap_configurations.varying_cost_C,
        "varying_cost_D": resources.rap_configurations.varying_cost_D,
        "varying_cost_E": resources.rap_configurations.varying_cost_E,
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

    main(resource_manager, problem, training_config, algorithm=algo, problem_name=name)
