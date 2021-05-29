import numpy as np

from resources.environments.rap_environment import Region, TaskCondition
from resources.resourcemanager.adp_resource_manager import AbstractActionSpecification

# problem satisfying the uniform resource requirement
urr_problem_dict = {
    "name": None,
    "rewards": np.array([30, 23, 17, 12, 9, 7, 5, 3, 2, 1]),
    "resource_requirements": np.ones((10, 5)),
    "max_resource_availabilities": np.ones(5) * 7,
    "task_arrival_p": np.array([0.6, 0.8, 0.8, 0.7, 0.55, 0.9, 0.9, 0.8, 0.9, 0.9]),
    "task_departure_p": np.array([0.1, 0.2, 0.2, 0.15, 0.15, 0.25, 0.3, 0.3, 0.3, 0.35]),
    "restricted_tasks": None
}

tricky_problem = {
    "name": "tricky",
    "rewards": np.array([10, 1]),
    "resource_requirements": np.ones((2, 2)),
    "max_resource_availabilities": np.ones(2),
    "task_arrival_p": np.array([1, 1]),
    "task_departure_p": np.array([0.01, 0.99]),
    "restricted_tasks": [0],
    "locks": [[1], [0]]
}

test_problem = {
    "name": None,
    "rewards": np.array([3, 2, 1]),
    "resource_requirements": np.ones((3, 1)),
    "max_resource_availabilities": np.ones(1)*2,
    "task_arrival_p": np.array([0.3, 0.4, 0.5]),
    "task_departure_p": np.array([0.6, 0.6, 0.99]),
    "restricted_tasks": None
}

small_problem = {
    "name": None,
    "rewards": np.array([4, 3, 2, 1]),
    "resource_requirements": np.ones((4, 1)),
    "max_resource_availabilities": np.ones(1)*3,
    "task_arrival_p": np.array([0.1, 0.2, 0.3, 0.4]),
    "task_departure_p": np.array([0.6, 0.6, 0.6, 0.6]),
    "restricted_tasks": None
}

deep_decomposable_problem = {
    "name": "deep_decomposable",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [3],
    "locks": [[3], [2], [1], [0]],
}

deep_decomposable_problem2 = {
    "name": "deep_decomposable2",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [2, 3],
    "locks": [[[1], [3]], [[0], [3]], [[2], [2]], [[1], [2]], [[0], [2]], [[3], [1]],  [[2], [1]], [[1], [1]],
              [[0], [1]], [[4], [0]], [[3], [0]], [[2], [0]], [[3], [0]], [[4], [0]]],
}

deep_decomposable_problem_alt = {
    "name": "deep_decomposable_alt",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [0, 2], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.01, 0.5]),
    "restricted_tasks": [2],
    "locks": [[2], [1], [0]],
}

wide_decomposable_problem = {
    "name": "wide_decomposable",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [3],
    "locks": [[3, 2], [1, 0]],
}

wide_decomposable_problem2 = {
    "name": "wide_decomposable2",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [2, 3],
    "locks": [[[1, 2], [3, 2]], [[4, 3, 2], [1, 0]], [[1, 0], [1, 0]]]
}

wide_decomposable_problem2_alt = {
    "name": "wide_decomposable2_alt",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.01, 0.01]),
    "restricted_tasks": [2, 3],
    "locks": [[[1, 2], [3, 2]], [[4, 3, 2], [1, 0]], [[1, 0], [1, 0]]]
}

decomposable_problem2 = {
    "name": "decomposable2",
    "rewards": np.array([1, 1, 1, 1, 10, 10]),
    "resource_requirements": np.array([[0, 1], [1, 0], [0, 1], [1, 0], [2, 0], [0, 2]]),
    "max_resource_availabilities": np.array([3, 3]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.5, 0.6, 0.01, 0.01]),
    "restricted_tasks": None
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
    }
}

unequal_rewards_varied_departure_p_6 = {
    "name": "equal_rewards_varied_departure_p_6",
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
    }
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
    }
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
    }
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
    }
}

unequal_rewards_varied_departure_p_7 = {
    "name": "equal_rewards_varied_departure_p_7",
    "rewards": np.array([3, 6, 5, 4, 5, 4, 5]),
    "resource_requirements": np.array([[2, 2, 0], [2, 2, 0], [1, 2, 1], [1, 1, 2], [2, 1, 1], [0, 2, 2], [2, 0, 2]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.05, 0.1, 0.15, 0.1, 0.15, 0.1]),
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
    }
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
    }
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
    }
}


"""
"abstract_action_to_direction": {
        0: "Stay",
        1: "Up_5",
        2: "Up_6",
        3: "Up_7",
        4: "Up_8",
        5: "Up_56",
        6: "Up_57",
        7: "Up_58",
        8: "Up_67",
        9: "Up_68",
        10: "Up_78",
        11: "Up_567",
        12: "Up_568",
        13: "Up_578",
        14: "Up678",
        15: "Up5678",
        16: "Down"
    },
    "direction_to_action": {
        "Up_5": np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),
        "Up_6": np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),
        "Up_7": np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),
        "Up_8": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
        "Up_56": np.array([0, 0, 0, 0, 0, 1, 1, 0, 0]),
        "Up_57": np.array([0, 0, 0, 0, 0, 1, 0, 1, 0]),
        "Up_58": np.array([0, 0, 0, 0, 0, 1, 0, 0, 1]),
        "Up_67": np.array([0, 0, 0, 0, 0, 0, 1, 1, 0]),
        "Up_68": np.array([0, 0, 0, 0, 0, 0, 1, 0, 1]),
        "Up_78": np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]),
        "Up_567": np.array([0, 0, 0, 0, 0, 1, 1, 1, 0]),
        "Up_568": np.array([0, 0, 0, 0, 0, 1, 1, 0, 1]),
        "Up_578": np.array([0, 0, 0, 0, 0, 1, 0, 1, 1]),
        "Up678": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1]),
        "Up5678": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]),
        "Down": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    }
"""


