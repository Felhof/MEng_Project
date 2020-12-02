import os
import matplotlib.pyplot as plt

from stable_baselines.common.env_checker import check_env
from stable_baselines import A2C
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resource_allocation_problem import ResourceAllocationProblem
from rap_environment import ResourceAllocationEnvironment


class ResourceManager:

    def __init__(self, rap, training_steps=50000, steps_per_episode=500, log_dir="/tmp/gym"):
        # Create log dir
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        rewards = rap["rewards"]
        resource_requirements = rap["resource_requirements"]
        max_resource_availabilities = rap["max_resource_availabilities"]
        task_arrival_p = rap["task_arrival_p"]
        task_departure_p = rap["task_departure_p"]

        self.ra_problem = ResourceAllocationProblem(rewards, resource_requirements, max_resource_availabilities,
                                                    task_arrival_p, task_departure_p)
        env = ResourceAllocationEnvironment(self.ra_problem, steps_per_episode)
        # If the environment doesn't follow the interface, an error will be thrown
        check_env(env, warn=True)

        # wrap it
        self.environment = make_vec_env(lambda: env, n_envs=1, monitor_dir=self.log_dir)

        self.model = None
        self.training_steps = training_steps

    def train_model(self):
        # Create callbacks
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = A2C('MlpPolicy', self.environment, verbose=1, tensorboard_log=self.log_dir)

        with ProgressBarManager(self.training_steps) as progress_callback:
            # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
            self.model.learn(total_timesteps=self.training_steps, callback=[progress_callback, auto_save_callback])

    def plot_training_results(self, xlabel="episode", ylabel="cumulative reward", filename="reward"):
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        plt.figure(figsize=(20, 10))
        plt.plot(x, y, "b", label="RL Agent")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig(filename)

    def evaluate_model(self, n_episodes=10, episode_length=500, render=False):
        print("Comparing Model to optimal strategy...")
        model_rewards = [
            self.get_model_solution(episode_length=episode_length, render=render) for _ in range(n_episodes)
        ]
        optimal_strategy_rewards = [
            self.calculate_optimal_solution(episode_length=episode_length) for _ in range(n_episodes)
        ]
        print("In {0} episodes the model achieved an average reward of: {1}".format(
            n_episodes,
            np.mean(model_rewards)
        ))
        print("The optimal strategy achieved an average reward of: {1}".format(
            n_episodes,
            np.mean(optimal_strategy_rewards)
        ))

    def get_model_solution(self, episode_length=500, render=False):
        reward = 0
        observation = self.environment.reset()
        for _ in range(episode_length):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, r, _, _ = self.environment.step(action)
            reward += r
            if render:
                self.environment.render(mode='console')

        return reward

    def calculate_optimal_solution(self, episode_length=500):
        self.ra_problem.reset()
        observation = self.environment.reset()
        reward = 0
        for _ in range(episode_length):
            action = self.ra_problem.get_heuristic_solution(observation)
            observation, r, _, _ = self.environment.step(action)
            reward += r

        return reward
