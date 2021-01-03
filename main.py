import os

import matplotlib.pyplot as plt

from stable_baselines import A2C
from stable_baselines.common.cmd_util import make_vec_env
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from go_left_env import GoLeftEnv
from gridworld import GridWorld
from resource_manager import ResourceManager


def main(resource_manager, resource_problem_dict, training_steps=50000, steps_per_episode=500):
    rm = resource_manager(resource_problem_dict, training_steps=training_steps, steps_per_episode=steps_per_episode)
    rm.train_model()
    rm.plot_training_results()
    #rm.print_policy()
    #rm.evaluate_model()


def test():
    log_dir = "/tmp/gym"

    # Instantiate the env
    env = GridWorld()
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)

    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # Train the agent
    with ProgressBarManager(5000) as progress_callback:
        model = A2C('MlpPolicy', env, verbose=1).learn(5000, callback=[progress_callback, auto_save_callback])

    # Test the trained agent
    obs = env.reset()
    n_steps = 20
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render(mode='console')
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break


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

test_problem = {
    "rewards": np.array([3, 2, 1]),
    "resource_requirements": np.ones((3, 1)),
    "max_resource_availabilities": np.ones(1)*2,
    "task_arrival_p": np.array([0.3, 0.4, 0.5]),
    "task_departure_p": np.array([0.6, 0.6, 0.99]),
}

small_problem = {
    "rewards": np.array([4, 3, 2, 1]),
    "resource_requirements": np.ones((4, 1)),
    "max_resource_availabilities": np.ones(1)*3,
    "task_arrival_p": np.array([0.1, 0.2, 0.3, 0.4]),
    "task_departure_p": np.array([0.6, 0.6, 0.6, 0.6]),
}

main(ResourceManager, small_problem, training_steps=35000, steps_per_episode=500)
