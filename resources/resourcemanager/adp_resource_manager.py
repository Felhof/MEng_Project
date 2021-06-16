from resources.resourcemanager.base_resource_manager import BaseResourceManager
import multiprocessing as mp
import os
import time

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EveryNTimesteps
import matplotlib.pyplot as plt
import itertools

from resources.callbacks import ProgressBarManager
from resources.environments.rap.rap_environment import DeanLinRegionalResourceAllocationEnvironment
from resources.environments.rap.adp_environment import ADPResourceAllocationEnvironment
from resources.region import Region
from resources.callbacks import SavePerformanceOnCheckpoints, SaveOnBestTrainingRewardCallback

def binary_sequences(n):
    assert n >= 1
    length_1_sequences = [[0], [1]]
    if n == 1:
        return length_1_sequences
    else:
        lower_length_sequences = binary_sequences(n - 1)
        sequences = [sequence + last_digit
                     for sequence, last_digit
                     in itertools.product(lower_length_sequences, length_1_sequences)]
        return sequences


class ADPResourceManager(BaseResourceManager):

    def __init__(self, rap, log_dir="/tmp/gym", training_config=None, algorithm="A2C", checkpoint_results=None):
        super(ADPResourceManager, self).__init__(rap, log_dir=log_dir, algorithm=algorithm,
                                                 checkpoint_results=checkpoint_results)

        self.save_dir = "/tmp/gym/"

        self.regions = Region.create_regions(rap)
        self.abstract_action_to_direction = rap["abstract_action_to_direction"]
        self.direction_to_action = rap["direction_to_action"]
        self.n_abstract_actions = len(self.abstract_action_to_direction)
        self.n_locked_tasks = len(rap["locked_tasks"])
        self.actions = list(range(self.n_abstract_actions))
        binary_states = [sequence for sequence in binary_sequences(self.n_locked_tasks)]
        region_states = list(range(len(self.regions)))
        self.states = [tuple([region_state] + binary_state)
                       for region_state, binary_state
                       in itertools.product(region_states, binary_states)]
        self.training_config = training_config
        self.model_name = "Dean_Lin_{}".format(rap["name"])
        self.environment = None
        self.policy = None

    def train_model(self):
        setup_start = time.time()
        load = self.training_config["load"]
        regional_policies = {id: {} for id, _ in enumerate(self.regions)}
        if not load:
            num_workers = mp.cpu_count() - 2
            pool = mp.Pool(num_workers)
            abstract_action_results = {}
            for region_id, region in enumerate(self.regions):
                regional_policies[region_id] = {key: value for key, value in self.direction_to_action.items()}
                abstract_action_results[region_id] = pool.apply_async(self.train_abstract_action, (),
                                                                      {"target_region": region, "region_id": region_id})

            pool.close()

            region_ids_and_paths = [(key, aa_result.get()) for key, aa_result in abstract_action_results.items()]
        else:
            region_ids_and_paths = []
            for region_id in regional_policies.keys():
                regional_policies[region_id] = {key: value for key, value in self.direction_to_action.items()}
                region_ids_and_paths.append((region_id, self.get_model_path(region_id)))

        for region_id, aa_path in region_ids_and_paths:
            abstract_action = self.algorithm.load(aa_path)
            self.model = abstract_action
            self.environment = DeanLinRegionalResourceAllocationEnvironment(self.ra_problem, region=self.regions[region_id])
            name = aa_path.split('/')[-1]
            super(ADPResourceManager, self).run_model(save=True, name=name)
            regional_policies[region_id]["Stay"] = abstract_action

        self.train_adp_model_with_deep_learning(regional_policies=regional_policies, setup_start=setup_start)

    def train_abstract_action(self, target_region=None, region_id=0):
        environment = DeanLinRegionalResourceAllocationEnvironment(self.ra_problem, region=target_region)
        abstract_action = self.algorithm('MlpPolicy', environment, verbose=1, tensorboard_log=self.log_dir)

        training_steps = self.training_config["stage1_training_steps"]
        with ProgressBarManager(training_steps) as progress_callback:
            # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
            abstract_action.learn(total_timesteps=training_steps, callback=progress_callback)

        aa_name = "{0}_AA_for_region_{1}".format(self.model_name, region_id)
        aa_path = os.path.abspath(aa_name)
        abstract_action.save(aa_path)

        return aa_path

    def get_model_path(self, region_id):
        aa_name = "{0}_AA_for_region_{1}".format(self.model_name, region_id)
        aa_path = os.path.abspath(aa_name)
        return aa_path

    def train_adp_model_with_deep_learning(self, regional_policies=None, setup_start=0.):
        regions = self.regions
        environment = ADPResourceAllocationEnvironment(self.ra_problem, regions, regional_policies,
                                                       abstract_action_to_direction=self.abstract_action_to_direction,
                                                       n_locked_tasks=self.n_locked_tasks,
                                                       n_abstract_actions=self.n_abstract_actions)
        environment = Monitor(environment, self.log_dir)
        self.environment = environment
        adp_model = self.algorithm('MlpPolicy', environment, verbose=1, tensorboard_log=self.log_dir)
        self.model = adp_model

        auto_save_callback = SaveOnBestTrainingRewardCallback(log_dir=self.log_dir)
        auto_save_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=auto_save_callback)

        name = self.model_name + "_full_model_multi"
        setup_time = time.time() - setup_start
        checkpoint_callback = SavePerformanceOnCheckpoints(stage1_time=setup_time, resource_manager=self, name=name,
                                                           checkpoint_results=self.checkpoint_results)
        checkpoint_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=checkpoint_callback)

        training_steps = self.training_config["stage2_training_steps"]
        with ProgressBarManager(training_steps) as progress_callback:
            adp_model.learn(total_timesteps=training_steps, callback=[progress_callback,
                                                                      auto_save_callback_every_1000_steps,
                                                                      checkpoint_callback_every_1000_steps])

        self.save_episode_rewards_as_csv()
        # self.plot_training_results(filename=name, show=self.training_config.get("show", False))

    def train_adp_model_with_monte_carlo(self, regional_policies=None, show=False):
        self.environment = ADPResourceAllocationEnvironment(self.ra_problem, self.regions, regional_policies,
                                                            abstract_action_to_direction=self.abstract_action_to_direction,
                                                            n_locked_tasks=self.n_locked_tasks,
                                                            n_abstract_actions=self.n_abstract_actions)
        policy, rewards = self.monte_carlo_control(2500)
        self.policy = policy

        if show:
            print(policy)

            title = "Learning Curve"
            xlabel = "episode"
            ylabel = "cumulative reward"
            filename = "{}_mc_learning_curve".format(self.model_name)

            plt.figure(figsize=(20, 10))
            plt.title(title)
            plt.plot(range(len(rewards)), rewards, "b", label="Cumulative Reward")
            plt.legend()
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.savefig("img/" + filename)
            plt.show()

    # Implementation of on-policy e-greedy first-visit MC control
    def monte_carlo_control(self, n, gamma=0.9):
        Q = self.initialize_Q()
        Returns = {}
        policy = {}
        rewards = []

        for s in self.states:
            self.update_epsilon_greedy_policy(policy, s, Q, 1)

        for episode in range(n):
            print("MC Control episode {0}/{1}".format(episode + 1, n), end="\r")
            trace = self.create_trace(policy, self.environment, gamma=gamma)
            rewards.append(sum(r for _, _, r in trace))

            occured_state_action_pairs = []
            for (s, a, r) in trace:
                if (s, a) in occured_state_action_pairs:
                    continue
                occured_state_action_pairs.append((s, a))
                Returns[(s, a)] = Returns.get((s, a), [])
                Returns[(s, a)].append(r)
                Q[(s, a)] = np.mean(Returns[(s, a)])

            e = (n - episode) / (n + 1)
            for s, _, _ in trace:
                self.update_epsilon_greedy_policy(policy, s, Q, e)

        for s in self.states:
            self.update_greedy_policy(policy, s, Q)

        return policy, rewards

    # Given a policy and a discount value gamma create a trace
    # of state, action, reward triples
    def create_trace(self, policy, environment, gamma=0.9):
        trace = []

        state = tuple(environment.reset())

        done = False
        while not done:
            action = self.get_next_action(policy, state)

            next_state, reward, done, info = environment.step(action)
            next_state = tuple(next_state)

            discount = gamma
            for idx, (s, a, r) in enumerate(reversed(trace)):
                trace[-(idx + 1)] = (s, a, r + discount * reward)
                discount *= gamma

            trace.append((state, action, reward))

            state = next_state

        return trace

    # initialize a Q function with Q(s,a) = 0 for all s,a
    def initialize_Q(self):
        Q = {}
        for s in self.states:
            for a in self.actions:
                Q[(s, a)] = 0
        return Q

    # Update policy to be epsilon-greedy for a particular Q function
    def update_epsilon_greedy_policy(self, policy, s, Q, epsilon):
        a_star = np.argmax([Q[(s, a)] for a in self.actions])
        for a in self.actions:
            if a == a_star:
                policy[(s, a)] = 1 - epsilon + epsilon / len(self.actions)
            else:
                policy[(s, a)] = epsilon / len(self.actions)

    # Update policy to be epsilon-greedy for a particular Q function
    def update_greedy_policy(self, policy, s, Q):
        a_star = np.argmax([Q[(s, a)] for a in self.actions])
        for a in self.actions:
            if a == a_star:
                policy[(s, a)] = 1
            else:
                policy[(s, a)] = 0

    # Given a policy and the current state get the next action
    def get_next_action(self, policy, state):
        next_action_probablities = [policy[(state, a)] for a in self.actions]
        return np.random.choice(self.actions, p=next_action_probablities)

"""
    def run_model(self, n_steps=250, save=False, name="", file_location="data"):
        log = []
        state = tuple(self.environment.reset())
        total_reward = 0
        for step in range(n_steps):
            action = self.get_next_action(self.policy, state)
            log.append("Sate: " + str(state))
            log.append("Action: " + str(action))
            state, reward, done, info = self.environment.step(action)
            state = tuple(state)
            log.append("lower level action: " + str(info["lower level action"]))
            total_reward += reward
            log.append('reward: ' + str(reward))
            log.append('done: ' + str(done))
            log.append("lower level state:" + str(info["lower level state"]))
            if done:
                log.append("Goal reached! Reward= " + str(reward))
                state = tuple(self.environment.reset())

        log.append("Total reward: " + str(total_reward))

        filename = "data/{}_mc_policy.txt".format(self.model_name)
        text = "\n".join(log)
        with open(filename, 'w') as file:
            file.write(text)

        return total_reward
"""

class AbstractActionSpecification:

    def __init__(self, direction="Stay", target_region_id=0):
        self.direction = direction
        self.target_region_id = target_region_id
