import matplotlib.pyplot as plt

import numpy as np


class LearningCurvePlotter:

    def __init__(self):
        self.stage_results = {}

    def add_results(self, stage, new_results):
        results = self.stage_results.get(stage, [])
        results.append(new_results)
        self.stage_results[stage] = results

    def plot_training_results(self, xlabel="episode", ylabel="cumulative reward", filename="reward"):
        for stage, results in self.stage_results.items():
            rewards_mean = np.array([np.mean(reward) for reward in results])
            rewards_std = np.array([np.std(reward) for reward in results])
            plt.figure(figsize=(20, 10))
            plt.title(stage)
            plt.plot(len(rewards_mean), rewards_mean, "b", label="Average Reward")
            plt.plot(len(rewards_mean), rewards_mean + rewards_std, "o", label="Avg Reward + 1 std")
            plt.plot(len(rewards_mean), rewards_mean - rewards_std, "g", label="Avg Reward - 1 std")
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axhline(y=0, color='r', linestyle='-')
            # plt.yscale("log")
            plt.show()
            plt.savefig(filename)
