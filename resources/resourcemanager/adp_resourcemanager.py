from resources.resourcemanager.base_resourcemanager import BaseResourceManager
import multiprocessing as mp
import os
import time

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EveryNTimesteps
import itertools

from resources.callbacks import ProgressBarManager
from resources.environments.rap.rap_restricted import DeanLinRegionalResourceAllocationEnvironment
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

        self.train_adp_model(regional_policies=regional_policies, setup_start=setup_start)

    def train_abstract_action(self, target_region=None, region_id=0):
        environment = DeanLinRegionalResourceAllocationEnvironment(self.ra_problem, region=target_region)
        abstract_action = self.algorithm('MlpPolicy', environment, verbose=1, tensorboard_log=self.log_dir)

        training_steps = self.training_config["stage1_training_steps"]
        with ProgressBarManager(training_steps) as progress_callback:
            abstract_action.learn(total_timesteps=training_steps, callback=progress_callback)

        aa_name = "{0}_AA_for_region_{1}".format(self.model_name, region_id)
        aa_path = os.path.abspath(aa_name)
        abstract_action.save(aa_path)

        return aa_path

    def get_model_path(self, region_id):
        aa_name = "{0}_AA_for_region_{1}".format(self.model_name, region_id)
        aa_path = os.path.abspath(aa_name)
        return aa_path

    def train_adp_model(self, regional_policies=None, setup_start=0.):
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
