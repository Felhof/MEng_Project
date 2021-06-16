import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EveryNTimesteps
import numpy as np
import multiprocessing as mp
import time

from resources.callbacks import ProgressBarManager
from resources.environments.rap.rap_environment import ResourceAllocationEnvironment, \
    AbbadDaouiRegionalResourceAllocationEnvironment
from resources.multistage_model import MultiStageActorCritic
from resources.resourcemanager.base_resource_manager import BaseResourceManager
from resources.callbacks import SavePerformanceOnCheckpoints, SaveOnBestTrainingRewardCallback

import ray
from ray import tune
from ray.tune import CLIReporter


class MultiAgentResourceManager(BaseResourceManager):

    def __init__(self, rap, training_config=None, log_dir="/tmp/gym", algorithm="A2C", checkpoint_results=None):
        super(MultiAgentResourceManager, self).__init__(rap, log_dir=log_dir, algorithm=algorithm,
                                                        checkpoint_results=checkpoint_results)

        self.save_dir = "/tmp/gym/"

        self.regions = rap["AD_Regions"]
        self.training_config = training_config
        self.model_name = "MARL_{}".format(rap["name"])
        if training_config["search_hyperparameters"]:
            self.model_name += "_tuned"

    def train_model(self):
        num_workers = mp.cpu_count() - 2

        stage1_hyperparams = [None] * len(self.regions)
        stage2_hyperparams = None

        stage1_models = []
        lower_lvl_model_paths = {}
        stage1_start_time = time.time()

        lvl = 0
        while lvl in self.regions:
            pool = mp.Pool(num_workers)
            regions = self.regions[lvl]
            regional_model_results = []
            for idx, region in enumerate(regions):
                environment_kwargs = {
                    "region": region,
                    "max_timesteps": self.training_config["steps_per_episode"]
                }

                policy_kwargs = {}
                submodel_name = self.model_name + "_stage1_lvl" + str(idx)
                if self.training_config["search_hyperparameters"] and (stage1_hyperparams[idx] is None):
                    stage1_hyperparams[idx] = self.search_hyperparams(AbbadDaouiRegionalResourceAllocationEnvironment,
                                                                      environment_kwargs,
                                                                      policy_kwargs,
                                                                      training_steps=self.training_config[
                                                                          "stage1_training_steps"])

                regional_model_results.append((region,
                                               pool.apply_async(
                                                   self.train_model_and_save_path,
                                                   (),
                                                   {
                                                    "model_name": submodel_name,
                                                    "training_kwargs": {
                                                        "environment_kwargs": environment_kwargs,
                                                        "policy_kwargs": policy_kwargs,
                                                        "training_steps": self.training_config["stage1_training_steps"],
                                                        "lower_lvl_model_paths": lower_lvl_model_paths
                                                        }
                                                    }
                                               )))

            for region, model_path_result in regional_model_results:
                model_path = model_path_result.get()
                model = PPO.load(model_path)
                stage1_models.append(model)
                task_values_in_region = region.find_task_values_within_region()
                for key in task_values_in_region:
                    lower_lvl_model_paths[key] = model_path

            lvl += 1

        stage1_time_taken = time.time() - stage1_start_time

        environment_kwargs = {
            "max_timesteps": self.training_config["steps_per_episode"]
        }
        policy_kwargs = {"stage1_models": stage1_models}

        if self.training_config["search_hyperparameters"] and (stage2_hyperparams is None):
            stage2_hyperparams = self.search_hyperparams(ResourceAllocationEnvironment,
                                                         environment_kwargs,
                                                         policy_kwargs,
                                                         policy=MultiStageActorCritic,
                                                         training_steps=self.training_config[
                                                             "stage2_training_steps"])

        multistage_model = self.train_stage2_model(environment_kwargs=environment_kwargs,
                                                   policy_kwargs=policy_kwargs,
                                                   training_steps=self.training_config["stage2_training_steps"],
                                                   stage1_time=stage1_time_taken,
                                                   model_name=self.model_name+"_full_model")

        self.model = multistage_model

    def search_hyperparams(self, environment_class, environment_kwargs, policy_kwargs,
                           policy=None, training_steps=200):

        if policy is None:
            policy = "MlpPolicy"

        model_keys = []
        model_paths = []

        if "lower_lvl_models" in environment_kwargs:
            lower_lvl_models = environment_kwargs["lower_lvl_models"]
            if lower_lvl_models:
                for key, model in lower_lvl_models.items():
                    path = os.path.abspath("submodel_" + str(key))
                    model.save(path)
                    model_paths.append(path)
                    model_keys.append(key)
            environment_kwargs["lower_lvl_models"] = {}

        if "stage1_models" in policy_kwargs:
            stage1_models = policy_kwargs["stage1_models"]
            for n, model in enumerate(stage1_models):
                path = os.path.abspath("stage1_model_" + str(n))
                model.save(path)
                model_paths.append(path)
            policy_kwargs["stage1_models"] = []

        ra_problem = self.ra_problem
        log_dir = self.log_dir
        hpsearch_iterations = self.training_config["hpsearch_iterations"]

        def train(config, checkpoint_dir=None):

            if model_keys:
                for key, path in zip(model_keys, model_paths):
                    submodel = PPO.load(path)
                    environment_kwargs["lower_lvl_models"][key] = submodel
            if "stage1_models" in policy_kwargs:
                for path in model_paths:
                    stage1_model = PPO.load(path)
                    policy_kwargs["stage1_models"].append(stage1_model)

            environment = environment_class(ra_problem, **environment_kwargs)
            vector_environment = make_vec_env(lambda: environment, n_envs=1, monitor_dir=log_dir)

            rewards = []

            for n in range(hpsearch_iterations):
                model = PPO(policy,
                            vector_environment,
                            verbose=1,
                            tensorboard_log=log_dir,
                            learning_rate=config["learning_rate"],
                            ent_coef=config["ent_coef"],
                            max_grad_norm=config["max_grad_norm"],
                            policy_kwargs=policy_kwargs)

                model.learn(total_timesteps=training_steps)

                reward = evaluate_policy(model, vector_environment, n_eval_episodes=100)[0]
                rewards.append(reward)

            tune.report(reward=np.mean(rewards), std=np.std(rewards))

        config = {
            "learning_rate": tune.uniform(0.0004, 0.0012),
            "use_entropy_loss": tune.choice([True, False]),
            "ent_coef": tune.sample_from(lambda spec: spec.config.use_entropy_loss * np.random.uniform(0, 0.002)),
            "max_grad_norm": tune.uniform(0.25, 0.75)
        }

        # To print the current trial status
        reporter = CLIReporter(metric_columns=["reward", "std", "evaluation_iteration"])

        ray.init(log_to_driver=False)
        result = tune.run(
            train,
            config=config,
            num_samples=15,
            progress_reporter=reporter
        )

        best_trial_config = result.get_best_trial("reward", mode="max").config

        ray.shutdown()

        if "stage1_models" in policy_kwargs:
            policy_kwargs["stage1_models"] = stage1_models

        del best_trial_config["use_entropy_loss"]

        print("Best config: ", best_trial_config)

        return best_trial_config

    def train_stage1_model(self, environment_kwargs=None, policy_kwargs=None, lower_lvl_model_paths=None,
                           hyperparams=None, training_steps=20000):
        config = {
            "verbose": 1,
            "tensorboard_log": self.log_dir,
            "policy_kwargs": policy_kwargs
        }

        if hyperparams is not None:
            for key in hyperparams.keys():
                config[key] = hyperparams[key]

        lower_lvl_models = {}
        paths_to_models = {}
        for key, path in lower_lvl_model_paths.items():
            model = paths_to_models.get(path, self.algorithm.load(path))
            paths_to_models[path] = model
            lower_lvl_models[key] = model

        environment_kwargs["lower_lvl_models"] = lower_lvl_models

        environment = AbbadDaouiRegionalResourceAllocationEnvironment(self.ra_problem, **environment_kwargs)
        environment = Monitor(environment, self.log_dir)

        model = PPO("MlpPolicy",
                    environment,
                    **config)
        with ProgressBarManager(training_steps) as progress_callback:
            model.learn(total_timesteps=training_steps, callback=progress_callback)

        self.environment = environment

        return model

    def train_stage2_model(self, environment_kwargs=None, policy_kwargs=None, hyperparams=None, training_steps=20000,
                           model_name="", stage1_time=0):

        config = {
            "verbose": 1,
            "tensorboard_log": self.log_dir,
            "policy_kwargs": policy_kwargs
        }

        if hyperparams is not None:
            for key in hyperparams.keys():
                config[key] = hyperparams[key]

        print(environment_kwargs)
        environment = ResourceAllocationEnvironment(self.ra_problem, **environment_kwargs)
        environment = Monitor(environment, self.log_dir)
        self.environment = environment

        model = PPO(MultiStageActorCritic,
                    environment,
                    **config)
        with ProgressBarManager(training_steps) as progress_callback:
            auto_save_callback = SaveOnBestTrainingRewardCallback(log_dir=self.log_dir)
            auto_save_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=auto_save_callback)
            checkpoint_callback = SavePerformanceOnCheckpoints(stage1_time=stage1_time, resource_manager=self,
                                                               name=model_name,
                                                               checkpoint_results=self.checkpoint_results)
            checkpoint_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=checkpoint_callback)
            callbacks = [progress_callback, auto_save_callback_every_1000_steps, checkpoint_callback_every_1000_steps]
            model.learn(total_timesteps=training_steps, callback=callbacks)

        self.environment = environment

        self.save_episode_rewards_as_csv()

        return model

    def train_model_and_save_path(self, model_name="", training_kwargs=None):
        model = self.train_stage1_model(**training_kwargs)
        model_path = os.path.abspath(model_name)
        model.save(model_path)
        return model_path
