import csv
import matplotlib.pyplot as plt
import math


class LearningCurvePlotter:

    def __init__(self, data_dir="data/", img_dir="img/"):
        self.results = []
        self.img_dir = img_dir
        self.data_dir = data_dir

    def add_result(self, result):
        self.results.append(result)

    def save_results(self, filename="results"):
        location = self.data_dir + '{}.csv'.format(filename)
        with open(location, mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for result in self.results:
                episodes, rewards = result
                results_writer.writerow(episodes)
                results_writer.writerow(rewards)

    def plot_average_results(self, title="average reward", filename="average_reward", epoch_length=20000):

        reward_buckets = [[] for _ in range(math.ceil(epoch_length / 100))]

        for result in self.results:
            episodes, rewards = result
            bucket_rewards = []
            bucket = 1

            for idx, episode in enumerate(episodes):
                if episode > bucket * 100:
                    if len(bucket_rewards) >= 1:
                        avg_bucket_reward = sum(bucket_rewards) / len(bucket_rewards)
                    elif bucket > 1:
                        avg_bucket_reward = reward_buckets[bucket - 2][-1]
                    else:
                        avg_bucket_reward = 0
                    reward_buckets[bucket - 1].append(avg_bucket_reward)
                    bucket_rewards = []
                    bucket += 1
                bucket_rewards.append(rewards[idx])

            if len(bucket_rewards) >= 1:
                avg_bucket_reward = sum(bucket_rewards) / len(bucket_rewards)
                reward_buckets[bucket - 1].append(avg_bucket_reward)

        mean_rewards = [sum(bucket_reward) / len(bucket_reward) for bucket_reward in reward_buckets]

        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.plot(range(len(mean_rewards)), mean_rewards, "b", label="Mean Reward")
        plt.legend()
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.savefig(self.img_dir + filename)
