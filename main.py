import os

import matplotlib.pyplot as plt

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resource_manager import ResourceManager
from resource_allocation_problem import ResourceAllocationProblem


def main(resource_manager, resource_problem_dict, training_steps=50000, steps_per_episode=500):
    rm = resource_manager(resource_problem_dict, training_steps=training_steps, steps_per_episode=steps_per_episode)
    rm.train_model()
    rm.plot_training_results()
    rm.test_model()


# problem satisfying the uniform resource requirement
urr_problem_dict = {
    "rewards": np.array([30, 23, 17, 12, 9, 7, 5, 3, 2, 1]),
    "resource_requirements": np.ones((10, 5)),
    "max_resource_availabilities": np.ones(5) * 7,
    "task_arrival_p": np.array([0.6, 0.8, 0.8, 0.7, 0.55, 0.9, 0.9, 0.8, 0.9, 0.9]),
    "task_departure_p": np.array([0.1, 0.2, 0.2, 0.15, 0.15, 0.25, 0.3, 0.3, 0.3, 0.35]),
}

tricky_problem_dict = {
    "rewards": np.array([5, 1]),
    "resource_requirements": np.ones((2, 2)),
    "max_resource_availabilities": np.ones(2),
    "task_arrival_p": np.array([1, 1]),
    "task_departure_p": np.array([0.05, 0.99])
}

main(ResourceManager, tricky_problem_dict, training_steps=50000, steps_per_episode=500)
