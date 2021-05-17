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

adp_problem2 = {
    "name": "adp_problem2",
    "rewards": np.array([1, 4, 4, 8]),
    "resource_requirements": np.array([[0, 0, 1], [2, 0, 1], [1, 2, 0], [1, 2, 2]]),
    "max_resource_availabilities": np.array([5, 10, 10]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.10, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=3, min_value=0, max_value=2)]),
        2: Region([TaskCondition(task_id=3, min_value=3, max_value=5)])
    },
    "region_id_to_abstract_actions": {
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2),
            AbstractActionSpecification(direction="Down", target_region_id=1)],
        1: [AbstractActionSpecification(direction="Up", target_region_id=2),
            AbstractActionSpecification(direction="Stay", target_region_id=1)],
    }
}


adp_problem_big = {
    "name": "adp_problem_big",
    "rewards": np.array([1, 4, 4, 8]),
    "resource_requirements": np.array([[0, 0, 1], [2, 0, 1], [1, 2, 0], [1, 2, 2]]),
    "max_resource_availabilities": np.array([15, 30, 30]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.10, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=3, min_value=0, max_value=3)]),
        2: Region([TaskCondition(task_id=3, min_value=4, max_value=7)]),
        3: Region([TaskCondition(task_id=3, min_value=8, max_value=11)]),
        4: Region([TaskCondition(task_id=3, min_value=12, max_value=15)])
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1),
            AbstractActionSpecification(direction="Up", target_region_id=2)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2),
            AbstractActionSpecification(direction="Up", target_region_id=3),
            AbstractActionSpecification(direction="Down", target_region_id=1)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3),
            AbstractActionSpecification(direction="Up", target_region_id=4),
            AbstractActionSpecification(direction="Down", target_region_id=2)],
        4: [AbstractActionSpecification(direction="Stay", target_region_id=4),
            AbstractActionSpecification(direction="Down", target_region_id=3)]
    }
}

adp_problem_massive = {
    "name": "adp_problem_massive",
    "rewards": np.array([3, 3, 3, 3, 2, 10]),
    "resource_requirements": np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 3, 3]]),
    "max_resource_availabilities": np.array([24, 24, 24]),
    "task_arrival_p": np.array([0.75, 0.75, 0.75, 0.75, 0.75, 1.]),
    "task_departure_p": np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=5, min_value=0, max_value=2)]),
        2: Region([TaskCondition(task_id=5, min_value=3, max_value=5)]),
        3: Region([TaskCondition(task_id=5, min_value=6, max_value=8)]),
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3)]
    },
    "n_abstract_actions": 3,
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


adp_problem_climb = {
    "name": "adp_problem_climb",
    #"rewards": np.array([3, 4, 5, 6, 6, 6, 4, 4]),
    #"resource_requirements": np.array([[3, 0, 0], [1, 0, 3], [3, 3, 0], [3, 1, 3], [1, 1, 6], [3, 1, 6], [1, 2, 3],
    #                                   [1, 1, 2]]),
    "rewards": np.array([3, 3, 3, 3, 2, 10]),
    "resource_requirements": np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 3, 3]]),
    "max_resource_availabilities": np.array([24, 24, 24]),
    "task_arrival_p": np.array([0.66, 0.66, 0.66, 0.66, 0.66, 1.]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=5, min_value=0, max_value=2)]),
        2: Region([TaskCondition(task_id=5, min_value=3, max_value=5)]),
        3: Region([TaskCondition(task_id=5, min_value=6, max_value=8)]),
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3)]
    },
    "n_abstract_actions": 3,
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


adp_problem_massive2 = {
    "name": "adp_problem_massive2",
    "rewards": np.array([3, 4, 5, 6, 4, 4]),
    "resource_requirements": np.array([[3, 0, 0], [1, 0, 3], [3, 2, 0], [2, 3, 1], [1, 3, 0], [1, 0, 3]]),
    "max_resource_availabilities": np.array([30, 30, 30]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=5, min_value=0, max_value=3),
                   TaskCondition(task_id=4, min_value=0, max_value=3)]),
        2: Region([TaskCondition(task_id=5, min_value=0, max_value=3),
                   TaskCondition(task_id=4, min_value=4, max_value=7)]),
        3: Region([TaskCondition(task_id=5, min_value=0, max_value=3),
                   TaskCondition(task_id=4, min_value=8, max_value=10)]),
        4: Region([TaskCondition(task_id=5, min_value=4, max_value=7),
                   TaskCondition(task_id=4, min_value=0, max_value=3)]),
        5: Region([TaskCondition(task_id=5, min_value=4, max_value=7),
                   TaskCondition(task_id=4, min_value=4, max_value=7)]),
        6: Region([TaskCondition(task_id=5, min_value=4, max_value=7),
                   TaskCondition(task_id=4, min_value=8, max_value=10)]),
        7: Region([TaskCondition(task_id=5, min_value=8, max_value=10),
                   TaskCondition(task_id=4, min_value=0, max_value=3)]),
        8: Region([TaskCondition(task_id=5, min_value=8, max_value=10),
                   TaskCondition(task_id=4, min_value=4, max_value=7)]),
        9: Region([TaskCondition(task_id=5, min_value=8, max_value=10),
                   TaskCondition(task_id=4, min_value=8, max_value=10)]),
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3)],
        4: [AbstractActionSpecification(direction="Stay", target_region_id=4)],
        5: [AbstractActionSpecification(direction="Stay", target_region_id=5)],
        6: [AbstractActionSpecification(direction="Stay", target_region_id=6)],
        7: [AbstractActionSpecification(direction="Stay", target_region_id=7)],
        8: [AbstractActionSpecification(direction="Stay", target_region_id=8)],
        9: [AbstractActionSpecification(direction="Stay", target_region_id=9)]
    },
    "n_abstract_actions": 5,
    "abstract_action_to_direction": {
        0: "Stay",
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
        4: (0, -1)
    },
    "direction_to_action": {
        (1, 0): np.array([0, 0, 0, 0, 1, 0]),
        (0, 1): np.array([0, 0, 0, 0, 0, 1]),
        (-1, 0): np.array([0, 0, 0, 0, 0, 0]),
        (0, -1): np.array([0, 0, 0, 0, 0, 0])
    }
}

adp_problem_very_big = {
    "name": "adp_problem_very_big",
    "rewards": np.array([3, 4, 5, 4, 5, 6]),
    "resource_requirements": np.array([[0, 3, 0], [2, 2, 0], [1, 2, 2], [1, 1, 2], [2, 3, 0], [2, 2, 2]]),
    "max_resource_availabilities": np.array([10, 10, 10]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=5, min_value=0, max_value=0)]),
        2: Region([TaskCondition(task_id=5, min_value=1, max_value=1)]),
        3: Region([TaskCondition(task_id=5, min_value=2, max_value=2)]),
        4: Region([TaskCondition(task_id=5, min_value=3, max_value=3)]),
        5: Region([TaskCondition(task_id=5, min_value=4, max_value=4)]),
        6: Region([TaskCondition(task_id=5, min_value=5, max_value=5)])
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3)],
        4: [AbstractActionSpecification(direction="Stay", target_region_id=4)],
        5: [AbstractActionSpecification(direction="Stay", target_region_id=5)],
        6: [AbstractActionSpecification(direction="Stay", target_region_id=6)]
    },
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


adp_problem_insane = {
    "name": "adp_problem_insane",
    # "rewards": np.array([3, 4, 3, 4, 5, 6, 5]),
    # "resource_requirements": np.array([[2, 1, 0], [2, 2, 0], [1, 2, 2], [0, 1, 2], [2, 3, 0], [2, 2, 2], [0, 2, 3]]),
    # "max_resource_availabilities": np.array([10, 10, 10]),
    "rewards": np.array([3, 4, 3, 4, 5, 6, 5]),
    "resource_requirements": np.array([[2, 1, 0], [2, 2, 0], [1, 0, 2], [0, 2, 2], [2, 3, 0], [2, 2, 2], [0, 2, 3]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]),
    "task_departure_p": np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    "regions": {
        1: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        2: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=1, max_value=1)]),
        3: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=2, max_value=2)]),
        4: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=3, max_value=3)]),
        5: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=4, max_value=4)]),
        6: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=5, max_value=5)]),
        7: Region([TaskCondition(task_id=5, min_value=0, max_value=0),
                   TaskCondition(task_id=6, min_value=6, max_value=6)]),
        8: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        9: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                   TaskCondition(task_id=6, min_value=1, max_value=1)]),
        10: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                   TaskCondition(task_id=6, min_value=2, max_value=2)]),
        11: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                    TaskCondition(task_id=6, min_value=3, max_value=3)]),
        12: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                    TaskCondition(task_id=6, min_value=4, max_value=4)]),
        13: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                    TaskCondition(task_id=6, min_value=5, max_value=5)]),
        14: Region([TaskCondition(task_id=5, min_value=1, max_value=1),
                    TaskCondition(task_id=6, min_value=6, max_value=6)]),
        15: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        16: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                   TaskCondition(task_id=6, min_value=1, max_value=1)]),
        17: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                   TaskCondition(task_id=6, min_value=2, max_value=2)]),
        18: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                    TaskCondition(task_id=6, min_value=3, max_value=3)]),
        19: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                    TaskCondition(task_id=6, min_value=4, max_value=4)]),
        20: Region([TaskCondition(task_id=5, min_value=2, max_value=2),
                    TaskCondition(task_id=6, min_value=5, max_value=5)]),
        21: Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        22: Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                   TaskCondition(task_id=6, min_value=1, max_value=1)]),
        23: Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                    TaskCondition(task_id=6, min_value=2, max_value=2)]),
        24: Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                    TaskCondition(task_id=6, min_value=3, max_value=3)]),
        25: Region([TaskCondition(task_id=5, min_value=3, max_value=3),
                    TaskCondition(task_id=6, min_value=4, max_value=4)]),
        26: Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        27: Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                    TaskCondition(task_id=6, min_value=1, max_value=1)]),
        28: Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                    TaskCondition(task_id=6, min_value=2, max_value=2)]),
        29: Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                    TaskCondition(task_id=6, min_value=3, max_value=3)]),
        30: Region([TaskCondition(task_id=5, min_value=4, max_value=4),
                    TaskCondition(task_id=6, min_value=4, max_value=4)]),
        31: Region([TaskCondition(task_id=5, min_value=5, max_value=5),
                   TaskCondition(task_id=6, min_value=0, max_value=0)]),
        32: Region([TaskCondition(task_id=5, min_value=5, max_value=5),
                    TaskCondition(task_id=6, min_value=1, max_value=1)]),
        33: Region([TaskCondition(task_id=5, min_value=5, max_value=5),
                    TaskCondition(task_id=6, min_value=2, max_value=2)]),
        34: Region([TaskCondition(task_id=5, min_value=5, max_value=5),
                    TaskCondition(task_id=6, min_value=3, max_value=3)]),
        35: Region([TaskCondition(task_id=5, min_value=6, max_value=6),
                    TaskCondition(task_id=6, min_value=0, max_value=0)]),
        36: Region([TaskCondition(task_id=5, min_value=6, max_value=6),
                    TaskCondition(task_id=6, min_value=1, max_value=1)]),
        37: Region([TaskCondition(task_id=5, min_value=6, max_value=6),
                    TaskCondition(task_id=6, min_value=2, max_value=2)]),
        38: Region([TaskCondition(task_id=5, min_value=7, max_value=7),
                    TaskCondition(task_id=6, min_value=0, max_value=0)]),
        39: Region([TaskCondition(task_id=5, min_value=7, max_value=7),
                    TaskCondition(task_id=6, min_value=1, max_value=1)]),
        40: Region([TaskCondition(task_id=5, min_value=7, max_value=7),
                    TaskCondition(task_id=6, min_value=2, max_value=2)]),
        41: Region([TaskCondition(task_id=5, min_value=8, max_value=8),
                    TaskCondition(task_id=6, min_value=0, max_value=0)]),
        42: Region([TaskCondition(task_id=5, min_value=8, max_value=8),
                    TaskCondition(task_id=6, min_value=1, max_value=1)]),
        43: Region([TaskCondition(task_id=5, min_value=9, max_value=9),
                    TaskCondition(task_id=6, min_value=0, max_value=0)]),
        44: Region([TaskCondition(task_id=5, min_value=10, max_value=10),
                    TaskCondition(task_id=6, min_value=0, max_value=0)])
    },
    "region_id_to_abstract_actions": {
        1: [AbstractActionSpecification(direction="Stay", target_region_id=1)],
        2: [AbstractActionSpecification(direction="Stay", target_region_id=2)],
        3: [AbstractActionSpecification(direction="Stay", target_region_id=3)],
        4: [AbstractActionSpecification(direction="Stay", target_region_id=4)],
        5: [AbstractActionSpecification(direction="Stay", target_region_id=5)],
        6: [AbstractActionSpecification(direction="Stay", target_region_id=6)],
        7: [AbstractActionSpecification(direction="Stay", target_region_id=7)],
        8: [AbstractActionSpecification(direction="Stay", target_region_id=8)],
        9: [AbstractActionSpecification(direction="Stay", target_region_id=9)],
        10: [AbstractActionSpecification(direction="Stay", target_region_id=10)],
        11: [AbstractActionSpecification(direction="Stay", target_region_id=11)],
        12: [AbstractActionSpecification(direction="Stay", target_region_id=12)],
        13: [AbstractActionSpecification(direction="Stay", target_region_id=13)],
        14: [AbstractActionSpecification(direction="Stay", target_region_id=14)],
        15: [AbstractActionSpecification(direction="Stay", target_region_id=15)],
        16: [AbstractActionSpecification(direction="Stay", target_region_id=16)],
        17: [AbstractActionSpecification(direction="Stay", target_region_id=17)],
        18: [AbstractActionSpecification(direction="Stay", target_region_id=18)],
        19: [AbstractActionSpecification(direction="Stay", target_region_id=19)],
        20: [AbstractActionSpecification(direction="Stay", target_region_id=20)],
        21: [AbstractActionSpecification(direction="Stay", target_region_id=21)],
        22: [AbstractActionSpecification(direction="Stay", target_region_id=22)],
        23: [AbstractActionSpecification(direction="Stay", target_region_id=23)],
        24: [AbstractActionSpecification(direction="Stay", target_region_id=24)],
        25: [AbstractActionSpecification(direction="Stay", target_region_id=25)],
        26: [AbstractActionSpecification(direction="Stay", target_region_id=26)],
        27: [AbstractActionSpecification(direction="Stay", target_region_id=27)],
        28: [AbstractActionSpecification(direction="Stay", target_region_id=28)],
        29: [AbstractActionSpecification(direction="Stay", target_region_id=29)],
        30: [AbstractActionSpecification(direction="Stay", target_region_id=30)],
        31: [AbstractActionSpecification(direction="Stay", target_region_id=31)],
        32: [AbstractActionSpecification(direction="Stay", target_region_id=32)],
        33: [AbstractActionSpecification(direction="Stay", target_region_id=33)],
        34: [AbstractActionSpecification(direction="Stay", target_region_id=34)],
        35: [AbstractActionSpecification(direction="Stay", target_region_id=35)],
        36: [AbstractActionSpecification(direction="Stay", target_region_id=36)],
        37: [AbstractActionSpecification(direction="Stay", target_region_id=37)],
        38: [AbstractActionSpecification(direction="Stay", target_region_id=38)],
        39: [AbstractActionSpecification(direction="Stay", target_region_id=39)],
        40: [AbstractActionSpecification(direction="Stay", target_region_id=40)],
        41: [AbstractActionSpecification(direction="Stay", target_region_id=41)],
        42: [AbstractActionSpecification(direction="Stay", target_region_id=42)],
        43: [AbstractActionSpecification(direction="Stay", target_region_id=43)],
        44: [AbstractActionSpecification(direction="Stay", target_region_id=44)]
    },
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


adp_problem_insane2 = {
    "name": "adp_problem_insane2",
    "rewards": np.array([3, 4, 3, 4, 5, 6, 5]),
    "resource_requirements": np.array([[2, 2, 0], [1, 1, 0], [1, 1, 2], [1, 1, 1], [1, 2, 1], [2, 2, 2], [0, 2, 3]]),
    "max_resource_availabilities": np.array([20, 20, 20]),
    "task_arrival_p": np.array([0.75, 0.75, 0.95, 0.95, 0.7, 0.85, 0.85]),
    "task_departure_p": np.array([0.2, 0.01, 0.1, 0.03, 0.01, 0.05, 0.05]),
    "locked_tasks": [5, 6],
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

adp_problem = {
    "name": "adp_problem",
    "rewards": np.array([1, 1, 1]),
    "resource_requirements": np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
    "max_resource_availabilities": np.array([2, 2, 2]),
    "task_arrival_p": np.array([0.66, 0.66, 0.66]),
    "task_departure_p": np.array([0.05, 0.05, 0.05]),
    "locked_tasks": [2],
    "n_abstract_actions": 3,
    "abstract_action_to_direction": {
        0: "Stay",
        1: "Up",
        2: "Down"
    },
    "direction_to_action": {
        "Up": np.array([0, 0, 1]),
        "Down": np.array([0, 0, 0])
    }
}


