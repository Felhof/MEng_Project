import os
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resource_allocation_problem import ResourceAllocationProblem
from rap_environment import ResourceAllocationEnvironment, MDPResourceAllocationEnvironment, RestrictedMDPResourceAllocationEnvironment
from MDP import MDPBuilder, RestrictedMDP
import MTA
from multistage_model import MultiStageActorCritic


class BaseResourceManager:

    def __init__(self, rap, log_dir="/tmp/gym"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        rewards = rap["rewards"]
        resource_requirements = rap["resource_requirements"]
        max_resource_availabilities = rap["max_resource_availabilities"]
        task_arrival_p = rap["task_arrival_p"]
        task_departure_p = rap["task_departure_p"]

        self.ra_problem = ResourceAllocationProblem(rewards, resource_requirements, max_resource_availabilities,
                                                    task_arrival_p, task_departure_p)

    def plot_training_results(self, xlabel="episode", ylabel="cumulative reward", filename="reward"):
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        plt.figure(figsize=(20, 10))
        plt.plot(x, y, "b", label="RL Agent")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axhline(y=0, color='r', linestyle='-')
        #plt.yscale("log")
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
        observation = self.vector_environment.reset()
        for _ in range(episode_length):
            action, _ = self.model.predict(observation, deterministic=True)
            observation, r, _, _ = self.vector_environment.step(action)
            reward += r
            if render:
                self.vector_environment.render(mode='console')

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



class ResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=50000, steps_per_episode=500, log_dir="/tmp/gym"):
        super(ResourceManager, self).__init__(rap, log_dir=log_dir)

        self.environment = MDPResourceAllocationEnvironment(self.ra_problem, steps_per_episode)
        # If the environment doesn't follow the interface, an error will be thrown
        check_env(self.environment, warn=True)

        # wrap it
        self.vector_environment = make_vec_env(lambda: self.environment, n_envs=1, monitor_dir=self.log_dir)

        self.model = None
        self.training_steps = training_steps

    def train_model(self):
        # Create callbacks
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

        self.model = A2C('MlpPolicy', self.vector_environment, verbose=1, tensorboard_log=self.log_dir)

        with ProgressBarManager(self.training_steps) as progress_callback:
            # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
            self.model.learn(total_timesteps=self.training_steps, callback=[progress_callback, auto_save_callback])


class MultiAgentResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=50000, steps_per_episode=500, log_dir="/tmp/gym"):
        super(MultiAgentResourceManager, self).__init__(rap, log_dir=log_dir)

        self.rap_mdp = MDPBuilder(self.ra_problem).build_mdp()
        state_idx = list(self.rap_mdp.idx_to_state.keys())
        state_dllst = MTA.DLLst(state_idx)

        MTA.mta_for_scc_and_levels(state_dllst, self.rap_mdp)
        scc_lst = MTA.SCC_lst
        scc_lst.sort(key=lambda scc: scc.lvl)
        self.levels = []
        current_level = []
        level = 0
        for scc in scc_lst:
            if scc.lvl != level:
                self.levels.append(current_level)
                level += 1
                current_level = []
            current_level.append(scc)
        self.levels.append(current_level)

        self.model = None
        self.training_steps = training_steps
        self.steps_per_episode = steps_per_episode

    def train_model(self):
        state_idx_to_model = {}
        models = []

        lower_level_values = {}
        for level in self.levels:
            values = {}
            for scc in level:
                state_idxs = scc.get_state_idxs()
                restricted_mdp = RestrictedMDP(self.rap_mdp, state_idxs, lower_level_values)
                environment = RestrictedMDPResourceAllocationEnvironment(self.ra_problem, restricted_mdp)
                vector_environment = make_vec_env(lambda: environment, n_envs=1, monitor_dir=self.log_dir)
                model = A2C('MlpPolicy', vector_environment, verbose=1, tensorboard_log=self.log_dir)

                with ProgressBarManager(self.training_steps) as progress_callback:
                    model.learn(total_timesteps=self.training_steps, callback=progress_callback)

                models.append(model)
                for idx in state_idxs:
                    state_idx_to_model[idx] = model

                values.update({idx: value for idx, value in zip(state_idxs, model.value[state_idxs])})
            lower_level_values = values

        whole_environment = MDPResourceAllocationEnvironment(self.ra_problem, self.steps_per_episode)
        whole_vector_environment = make_vec_env(lambda: whole_environment, n_envs=1, monitor_dir=self.log_dir)
        multistage_model = A2C(MultiStageActorCritic, whole_vector_environment, verbose=1, tensorboard_log=self.log_dir,
                               stage1_models=models)
        with ProgressBarManager(self.training_steps) as progress_callback:
            multistage_model.learn(total_timesteps=self.training_steps, callback=progress_callback)
