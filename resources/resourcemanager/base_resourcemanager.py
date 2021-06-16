import os
import matplotlib.pyplot as plt
import csv

from stable_baselines3.common.results_plotter import load_results, ts2xy
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

    def save_episode_rewards_as_csv(self, data_directory="data/", log_dir=None):

        if log_dir is None:
            log_dir = self.log_dir

        episode, rewards = ts2xy(load_results(log_dir), 'timesteps')

        filename = self.model_name + "_episode_rewards"
        location = data_directory + '{}.csv'.format(filename)

        with open(location, mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(episode)
            results_writer.writerow(rewards)

    def save_model(self):
        filename = "models/{}".format(self.model_name)
        path = os.path.abspath(filename)
        self.model.save(path)

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
