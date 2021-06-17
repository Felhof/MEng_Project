import numpy as np

from resources.region import Region, TaskCondition


simple_problem_A = {
    "name": "simple_problem_A",
    "rewards": np.array([2, 3, 4, 5, 1]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

simple_problem_B = {
    "name": "simple_problem_B",
    "rewards": np.array([1, 3, 4, 5, 2]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

simple_problem_C = {
    "name": "simple_problem_C",
    "rewards": np.array([1, 2, 4, 5, 3]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}


simple_problem_D = {
    "name": "simple_problem_D",
    "rewards": np.array([1, 2, 3, 5, 4]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

simple_problem_E = {
    "name": "simple_problem_E",
    "rewards": np.array([1, 2, 3, 4, 5]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

varying_departure_p_E = {
    "name": "varying_departure_p_E",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.1, 0.2]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

varying_departure_p_D = {
    "name": "varying_departure_p_D",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.2, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

varying_departure_p_C = {
    "name": "varying_departure_p_C",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[1], [1], [1], [1], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.2, 0.1, 0.05]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}


varying_cost_C = {
    "name": "varying_cost_C",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[3], [3], [1], [2], [3]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=3, max_value=3)])],
        1: [Region([TaskCondition(task_id=4, min_value=2, max_value=2)])],
        2: [Region([TaskCondition(task_id=4, min_value=1, max_value=1)])],
        3: [Region([TaskCondition(task_id=4, min_value=0, max_value=0)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

varying_cost_D = {
    "name": "varying_cost_D",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[3], [3], [3], [1], [2]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=5, max_value=5)])],
        1: [Region([TaskCondition(task_id=4, min_value=4, max_value=4)])],
        2: [Region([TaskCondition(task_id=4, min_value=3, max_value=3)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=2)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}

varying_cost_E = {
    "name": "varying_cost_E",
    "rewards": np.array([1, 1, 1, 1, 1]),
    "resource_requirements": np.array([[3], [3], [3], [2], [1]]),
    "max_resource_availabilities": np.array([10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    "locked_tasks": [4],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=4, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=4, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=4, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=4, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=4, min_value=0, max_value=1)])]
    },
    "Checkpoints": [2, 4, 6, 8, 10]
}



split_on_best_5 = {
    "name": "split_on_best_5",
    "rewards": np.array([3, 4, 5, 4, 5]),
    "resource_requirements": np.array([[1, 2, 0], [2, 2, 0], [1, 2, 2], [1, 1, 2], [2, 2, 1]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

split_on_best_6 = {
    "name": "split_on_best_6",
    "rewards": np.array([3, 4, 5, 4, 5, 6]),
    "resource_requirements": np.array([[1, 2, 0], [2, 2, 0], [1, 2, 2], [1, 1, 2], [2, 2, 1], [2, 2, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_5 = {
    "name": "unequal_rewards_varied_departure_p_5",
    "rewards": np.array([3, 6, 5, 4, 5]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.05, 0.1, 0.15, 0.1]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_6 = {
    "name": "unequal_rewards_varied_departure_p_6",
    "rewards": np.array([3, 6, 5, 4, 5, 4]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 2, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.05, 0.1, 0.15, 0.1, 0.15]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_6_A = {
    "name": "unequal_rewards_varied_departure_p_6_A",
    "rewards": np.array([6, 5, 4, 5, 4, 3]),
    "resource_requirements": np.array([[2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 2, 2], [2, 2, 0]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.1, 0.15, 0.1, 0.15, 0.2]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_6_B = {
    "name": "unequal_rewards_varied_departure_p_6_B",
    "rewards": np.array([3, 5, 4, 5, 4, 6]),
    "resource_requirements": np.array([[2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 2, 2], [2, 2, 0]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.1, 0.15, 0.1, 0.15, 0.05]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}


split_on_worst_6 = {
    "name": "split_on_worst_6",
    "rewards": np.array([6, 4, 5, 4, 5, 3]),
    "resource_requirements": np.array([[3, 3, 3], [2, 2, 0], [1, 2, 2], [1, 1, 2], [2, 2, 1], [1, 2, 0]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

split_on_best_7 = {
    "name": "split_on_best_7",
    "rewards": np.array([3, 4, 3, 4, 5, 5, 6]),
    "resource_requirements": np.array([[2, 1, 0], [2, 2, 0], [1, 0, 2], [0, 2, 2], [2, 3, 0], [0, 2, 3], [2, 2, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [6],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=6, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=6, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=6, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=6, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=6, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

split_on_worst_7 = {
    "name": "split_on_worst_7",
    "rewards": np.array([4, 4, 5, 6, 5, 3, 3]),
    "resource_requirements": np.array([[2, 2, 0], [0, 2, 2], [2, 3, 0], [2, 2, 2], [0, 2, 3], [2, 1, 0], [1, 0, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [5, 6],
    "n_abstract_actions": 4,
    "n_locked_tasks": 2,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up_1",
        2: "Up_2",
        3: "Up_both",
        4: "Down"
    },
    "direction_to_action": {
        "Up_1": np.array([0, 0, 0, 0, 0, 0, 1]),
        "Up_2": np.array([0, 0, 0, 0, 0, 1, 0]),
        "Up_both": np.array([0, 0, 0, 0, 0, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0])
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_7 = {
    "name": "unequal_rewards_varied_departure_p_7",
    "rewards": np.array([3, 6, 5, 4, 5, 5, 4]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [2, 0, 2],  [0, 2, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.05, 0.1, 0.15, 0.1, 0.1, 0.15]),
    "locked_tasks": [6],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=6, min_value=8, max_value=10),
                    TaskCondition(task_id=5, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=8, max_value=10)]),
            Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        1: [Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=6, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        2: [Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=0, max_value=3)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_7_A = {
    "name": "unequal_rewards_varied_departure_p_7_A",
    "rewards": np.array([6, 5, 4, 5, 5, 4, 3]),
    "resource_requirements": np.array([[2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [2, 0, 2],  [0, 2, 2], [2, 2, 0]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.1, 0.15, 0.1, 0.1, 0.15, 0.2]),
    "locked_tasks": [6],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=6, min_value=8, max_value=10),
                    TaskCondition(task_id=5, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=8, max_value=10)]),
            Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        1: [Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=6, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        2: [Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=0, max_value=3)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

unequal_rewards_varied_departure_p_7_B = {
    "name": "unequal_rewards_varied_departure_p_7_B",
    "rewards": np.array([3, 5, 4, 5, 5, 4, 6]),
    "resource_requirements": np.array([[2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [2, 0, 2],  [0, 2, 2], [2, 2, 0]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.1, 0.15, 0.1, 0.1, 0.15, 0.05]),
    "locked_tasks": [6],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=6, min_value=8, max_value=10),
                    TaskCondition(task_id=5, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=8, max_value=10)]),
            Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        1: [Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=6, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        2: [Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=0, max_value=3)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}



equal_rewards_varied_departure_p_8 = {
    "name": "equal_rewards_varied_departure_p_8",
    "rewards": np.array([3, 6, 5, 4, 5, 5, 7, 6]),
    "resource_requirements": np.array([[2, 1, 0], [3, 3, 0], [1, 2, 2], [1, 1, 2], [2, 2, 1], [3, 2, 0], [5, 0, 2],
                                       [2, 0, 4]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.17, 0.08, 0.11, 0.14, 0.11, 0.11, 0.05, 0.08]),
    "locked_tasks": [6, 7],
    "n_abstract_actions": 5,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up_10",
        2: "Up_01",
        3: "Up_11",
        4: "Down"
    },
    "direction_to_action": {
        "Up_10": np.array([0, 0, 0, 0, 0, 0, 1, 0]),
        "Up_01": np.array([0, 0, 0, 0, 0, 0, 0, 1]),
        "Up_11": np.array([0, 0, 0, 0, 0, 0, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=6, min_value=8, max_value=10),
                    TaskCondition(task_id=5, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=8, max_value=10)]),
            Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        1: [Region([TaskCondition(task_id=6, min_value=4, max_value=7),
                    TaskCondition(task_id=6, min_value=0, max_value=3)]),
            Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=4, max_value=7)])],
        2: [Region([TaskCondition(task_id=6, min_value=0, max_value=3),
                    TaskCondition(task_id=5, min_value=0, max_value=3)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}


many_tasks = {
    "name": "many_tasks",
    "rewards": np.array([3, 6, 5, 4, 5, 3, 4, 5, 6]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 1, 2], [2, 0, 2],
                                       [3, 2, 0], [0, 3, 3]]),
    "max_resource_availabilities": np.array([8, 8, 8]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.3, 0.15, 0.2, 0.25, 0.2, 0.3, 0.25, 0.2, 0.15]),
    "locked_tasks": [7, 8],
    "n_abstract_actions": 5,
    "n_locked_tasks": 2,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up_7",
        2: "Up_8",
        3: "Up_78",
        4: "Down"
    },
    "direction_to_action": {
        "Up_7": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
        "Up_8": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        "Up_78": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                    TaskCondition(task_id=6, min_value=0, max_value=0)]),
            Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                    TaskCondition(task_id=6, min_value=4, max_value=4)])],
        1: [Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                    TaskCondition(task_id=6, min_value=0, max_value=1)]),
            Region([TaskCondition(task_id=5, min_value=0, max_value=1),
                    TaskCondition(task_id=6, min_value=3, max_value=3)])],
        2: [Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                    TaskCondition(task_id=6, min_value=2, max_value=2)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                    TaskCondition(task_id=6, min_value=0, max_value=1)]),
            Region([TaskCondition(task_id=5, min_value=0, max_value=1),
                    TaskCondition(task_id=6, min_value=2, max_value=2)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1),
                    TaskCondition(task_id=6, min_value=0, max_value=1)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}


many_tasks_v2 = {
    "name": "many_tasks_v2",
    "rewards": np.array([1, 3, 2, 2, 2, 1, 2, 5, 6]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 1, 2], [2, 0, 2],
                                       [3, 2, 0], [0, 3, 3]]),
    "max_resource_availabilities": np.array([8, 8, 8]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.3, 0.15, 0.25, 0.25, 0.25, 0.3, 0.25, 0.5, 0.4]),
    "locked_tasks": [7, 8],
    "n_abstract_actions": 5,
    "n_locked_tasks": 2,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up_7",
        2: "Up_8",
        3: "Up_78",
        4: "Down"
    },
    "direction_to_action": {
        "Up_7": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
        "Up_8": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        "Up_78": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=7, min_value=2, max_value=2),
                    TaskCondition(task_id=8, min_value=0, max_value=1)]),
            Region([TaskCondition(task_id=7, min_value=0, max_value=1),
                    TaskCondition(task_id=8, min_value=2, max_value=2)])],
        1: [Region([TaskCondition(task_id=7, min_value=0, max_value=1),
                    TaskCondition(task_id=8, min_value=0, max_value=1)])],
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}


many_tasks_small = {
    "name": "many_tasks_small",
    "rewards": np.array([3, 6, 5, 4, 5, 3, 4]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 1, 2], [2, 0, 2]]),
    "max_resource_availabilities": np.array([8, 8, 8]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.3, 0.15, 0.2, 0.25, 0.2, 0.3, 0.25]),
    "locked_tasks": [7, 8],
    "n_abstract_actions": 5,
    "n_locked_tasks": 2,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up_7",
        2: "Up_8",
        3: "Up_78",
        4: "Down"
    },
    "direction_to_action": {
        "Up_7": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
        "Up_8": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        "Up_78": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}

deep_6 = {
    "name": "split_on_best_6",
    "rewards": np.array([3, 4, 5, 4, 5, 6]),
    "resource_requirements": np.array([[1, 2, 0], [2, 2, 0], [1, 2, 2], [1, 1, 2], [2, 2, 1], [2, 2, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "locked_tasks": [5],
    "n_abstract_actions": 3,
    "n_locked_tasks": 1,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 0, 0, 0, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0])
    },
    "AD_Regions": {
        0: [Region([TaskCondition(task_id=5, min_value=8, max_value=10)])],
        1: [Region([TaskCondition(task_id=5, min_value=6, max_value=7)])],
        2: [Region([TaskCondition(task_id=5, min_value=4, max_value=5)])],
        3: [Region([TaskCondition(task_id=5, min_value=2, max_value=3)])],
        4: [Region([TaskCondition(task_id=5, min_value=0, max_value=1)])]
    },
    "Checkpoints": [4, 8, 12, 16, 20]
}