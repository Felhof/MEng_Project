import os
import matplotlib.pyplot as plt

from stable_baselines.common.env_checker import check_env
from stable_baselines import ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resource_allocation_problem import ResourceAllocationProblem
from rap_environment import RAPEnvironment


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
        env = RAPEnvironment(self.ra_problem, steps_per_episode)
        # If the environment doesn't follow the interface, an error will be thrown
        check_env(env, warn=True)

        # wrap it
        self.environment = make_vec_env(lambda: env, n_envs=1, monitor_dir=self.log_dir)

        self.model = None
        self.training_steps = training_steps

    def train_model(self, algorithm=ACKTR):
        # Create callbacks
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = algorithm('MlpPolicy', self.environment, verbose=1, tensorboard_log=self.log_dir)

        with ProgressBarManager(self.training_steps) as progress_callback:
            # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
            self.model.learn(self.training_steps, callback=[progress_callback, auto_save_callback])

    def plot_training_results(self, xlabel="episode", ylabel="cumulative reward", filename="reward"):
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        plt.figure(figsize=(20, 10))
        plt.plot(x, y, "b", label="RL Agent")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
        plt.savefig(filename)

    def test_model(self, n_steps=100, render=False):
        true_rewards = []
        optimal_rewards = []
        obs = self.environment.reset()
        for step in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            optimal_rewards.append(self.ra_problem.get_optimal_reward(obs))
            print("Step {}".format(step + 1))
            print("Action: ", action)
            obs, reward, done, info = self.environment.step(action)
            print('obs=', obs, 'reward=', reward, 'done=', done)
            true_rewards.append(reward)
            if render:
                self.environment.render(mode='console')
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                print("Goal reached!", "reward=", reward)
                break

        plt.figure(figsize=(20, 10))
        plt.plot(range(len(true_rewards)), true_rewards, "b", label="Reward achieved by agent")
        plt.plot(range(len(optimal_rewards)), optimal_rewards, "r", label="Optimum Reward")
        plt.legend()
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.show()
        plt.savefig("test")
