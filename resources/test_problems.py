import numpy as np

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

tricky_problem_dict = {
    "name": None,
    "rewards": np.array([10, 1]),
    "resource_requirements": np.ones((2, 2)),
    "max_resource_availabilities": np.ones(2),
    "task_arrival_p": np.array([1, 1]),
    "task_departure_p": np.array([0.05, 0.99]),
    "restricted_tasks": None
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
    "name": "deep_decomposable",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [2, 3],
    "locks": [[[1], [3]], [[0], [3]], [[2], [2]], [[1], [2]], [[0], [2]], [[3], [1]],  [[2], [1]], [[1], [1]],
              [[0], [1]], [[4], [0]], [[3], [0]], [[2], [0]], [[3], [0]], [[4], [0]]],
}

wide_decomposable_problem = {
    "name": "deep_decomposable",
    "rewards": np.array([2, 1, 2, 10]),
    "resource_requirements": np.array([[0, 2], [1, 0], [1, 1], [2, 1]]),
    "max_resource_availabilities": np.array([7, 4]),
    "task_arrival_p": np.array([0.25, 0.25, 0.25, 0.25]),
    "task_departure_p": np.array([0.6, 0.5, 0.4, 0.01]),
    "restricted_tasks": [3],
    "locks": [[3, 2], [1, 0]],
    "levels": 4
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
