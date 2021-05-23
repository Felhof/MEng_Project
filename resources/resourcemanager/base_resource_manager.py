import os
import matplotlib.pyplot as plt

from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3 import A2C, PPO

from resources.resource_allocation_problem import ResourceAllocationProblem


algorithms = {
    "A2C": A2C,
    "PPO": PPO
}


class BaseResourceManager:

    def __init__(self, rap, log_dir="/tmp/gym", algorithm="A2C", checkpoint_results=None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        rewards = rap["rewards"]
        resource_requirements = rap["resource_requirements"]
        max_resource_availabilities = rap["max_resource_availabilities"]
        task_arrival_p = rap["task_arrival_p"]
        task_departure_p = rap["task_departure_p"]

        self.model_name = ""

        self.model = None
        self.environment = None
        self.algorithm = algorithms.get(algorithm, A2C)

        self.ra_problem = ResourceAllocationProblem(rewards, resource_requirements, max_resource_availabilities,
                                                    task_arrival_p, task_departure_p)

        self.checkpoint_results = checkpoint_results

    def plot_training_results(self, title="Learning Curve", xlabel="episode", ylabel="cumulative reward",
                              filename="reward", log_dir=None, show=False):
        plt.clf()

        if log_dir is None:
            log_dir = self.log_dir
        x, y = ts2xy(load_results(log_dir), 'timesteps')

        plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.plot(x, y, "b", label="Cumulative Reward")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.savefig("img/" + filename)
        if show:
            plt.show()


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

    def save_model(self):
        filename = "models/{}".format(self.model_name)
        path = os.path.abspath(filename)
        self.model.save(path)

    def get_model_solution(self, episode_length=500, render=False):
        reward = 0
        observation = self.vector_environment.reset()
        for _ in range(episode_length):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, r, _, _ = self.vector_environment.step(action)
            reward += r
            if render:
                self.vector_environment.show(mode='console')

        return reward

    def calculate_optimal_solution(self, episode_length=500):
        self.ra_problem.reset()
        observation = self.vector_environment.reset()
        reward = 0
        for _ in range(episode_length):
            action = self.ra_problem.get_heuristic_solution(observation)
            observation, r, _, _ = self.vector_environment.step(action)
            reward += r

        return reward

    def print_policy(self):
        all_observations = self.environment.enumerate_observations()
        policy = {}

        for observation in all_observations:
            action = self.model.predict(observation, deterministic=True)
            policy[tuple(observation)] = action

        for item in policy.items():
            print("{0} : {1}".format(item[0], list(item[1])))

    def run_model(self, n_steps=250, save=False, name="", file_location="data", model_path=None):
        if model_path is not None:
            model = self.algorithm.load(model_path)
        else:
            model = self.model
        log = []
        seeds = list(range(1, n_steps + 1))
        obs = self.environment.reset(deterministic=True, seed=0)
        total_reward = 0
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            log.append("Sate: " + str(obs))
            log.append("Action: " + str(action))
            obs, reward, done, info = self.environment.step(action)
            if "lower level action" in info:
                log.append("lower level action: " + str(info["lower level action"]))
            total_reward += reward
            log.append('reward: ' + str(reward))
            log.append('done: ' + str(done))
            if "lower level state" in info:
                log.append("lower level state:" + str(info["lower level state"]))
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                log.append("Goal reached! Reward= " + str(reward))
                obs = self.environment.reset(deterministic=True, seed=seeds.pop(0))

        log.append("Total reward: " + str(total_reward))

        if not save:
            for line in log:
                print(line)
        else:
            filename = file_location + "/" + name + "_policy.txt"
            text = "\n".join(log)
            with open(filename, 'w') as file:
                file.write(text)

        return total_reward
