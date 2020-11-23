import os

import matplotlib.pyplot as plt

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager, PlottingCallback
from resource_manager import ResourceManager
from resource_allocation_problem import ResourceAllocationProblem


def main(training_steps=50000):
    # Create log dir
    log_dir = "/tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)

    ra_problem = ResourceAllocationProblem(rewards, resource_requirements, max_resource_availabilities,
                                           task_arrival_p, task_departure_p)
    env = ResourceManager(ra_problem, max_timesteps)
    # If the environment doesn't follow the interface, an error will be thrown
    check_env(env, warn=True)

    # wrap it
    env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)

    # Create callbacks
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    #plotting_callback = PlottingCallback(log_dir)

    model = ACKTR('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    with ProgressBarManager(training_steps) as progress_callback:
        # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
        model.learn(training_steps, callback=[progress_callback, auto_save_callback])

    x, y = ts2xy(load_results(log_dir), 'timesteps')
    plt.figure(figsize=(20, 10))
    plt.plot(x, y, "b", label="RL Agent")
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("cumulative reward")
    plt.show()
    plt.savefig("reward")

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


rewards = np.array([30, 23, 17, 12, 9, 7, 5, 3, 2, 1])
resource_requirements = np.ones((10, 5))
max_resource_availabilities = np.ones(5) * 7
task_arrival_p = np.array([0.6, 0.8, 0.8, 0.7, 0.55, 0.9, 0.9, 0.8, 0.9, 0.9])
task_departure_p = np.array([0.1, 0.2, 0.2, 0.15, 0.15, 0.25, 0.3, 0.3, 0.3, 0.35])
max_timesteps = 500

#task_count = 2
#rewards = [5, 1]
#resource_limit = 1
#task_arrival_p = [1, 1]
#task_departure_p = [0.05, 0.99]
#max_timesteps = 500


main(training_steps=50000)
