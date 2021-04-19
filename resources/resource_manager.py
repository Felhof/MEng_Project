import os
import matplotlib.pyplot as plt
import itertools

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

from resources.callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resources.resource_allocation_problem import ResourceAllocationProblem
from resources.rap_environment import ResourceAllocationEnvironment, MDPResourceAllocationEnvironment, \
    RestrictedMDPResourceAllocationEnvironment, RestrictedResourceAllocationEnvironment
from resources.MDP import MDPBuilder, RestrictedMDP
from resources import MTA
from resources.multistage_model import MultiStageActorCritic
import torch
from resources.plotter import LearningCurvePlotter

import ray
from ray import tune
from ray.tune import CLIReporter


class BaseResourceManager:

    def __init__(self, rap, plotter=None, log_dir="/tmp/gym"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        rewards = rap["rewards"]
        resource_requirements = rap["resource_requirements"]
        max_resource_availabilities = rap["max_resource_availabilities"]
        task_arrival_p = rap["task_arrival_p"]
        task_departure_p = rap["task_departure_p"]

        self.model_name = ""
        self.plotter = plotter

        self.model = None
        self.environment = None

        self.ra_problem = ResourceAllocationProblem(rewards, resource_requirements, max_resource_availabilities,
                                                    task_arrival_p, task_departure_p)

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

    def run_model(self, n_steps=50):
        obs = self.environment.reset()
        total_reward = 0
        for step in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            print("Sate: ", obs)
            print("Action: ", action)
            obs, reward, done, info = self.environment.step(action)
            total_reward += reward
            print('reward: ', reward, 'done: ', done)
            if done:
                # Note that the VecEnv resets automatically
                # when a done signal is encountered
                print("Goal reached!", "reward=", reward)
                obs = self.environment.reset()

        print("Total reward: ", total_reward)


class ResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=60000, steps_per_episode=100, log_dir="/tmp/gym", plotter=None):
        super(ResourceManager, self).__init__(rap, log_dir=log_dir, plotter=plotter)

        self.model_name = rap["name"] + "_baseline"

        self.environment = ResourceAllocationEnvironment(self.ra_problem, steps_per_episode)
        # If the environment doesn't follow the interface, an error will be thrown
        check_env(self.environment, warn=True)

        # wrap it
        self.vector_environment = make_vec_env(lambda: self.environment, n_envs=1, monitor_dir=self.log_dir)

        self.training_steps = training_steps

    def train_model(self):
        plotter = LearningCurvePlotter()

        for _ in range(10):

            # Create callbacks
            auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=self.log_dir)

            vector_environment = make_vec_env(lambda: self.environment, n_envs=1, monitor_dir=self.log_dir)
            self.model = A2C('MlpPolicy', vector_environment, verbose=1, tensorboard_log=self.log_dir)

            with ProgressBarManager(self.training_steps) as progress_callback:
                # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
                self.model.learn(total_timesteps=self.training_steps, callback=[progress_callback, auto_save_callback])

            result = ts2xy(load_results(self.log_dir), 'timesteps')
            plotter.add_result(result)

        csv_name = self.model_name + "_results"
        plot_name = self.model_name + "_average_reward"
        plotter.save_results(csv_name)
        plotter.plot_average_results(filename=plot_name, epoch_length=self.training_steps)


class MultiAgentResourceManager(BaseResourceManager):

    def __init__(self, rap, training_config=None, plotter=None,
                 log_dir="/tmp/gym"):
        super(MultiAgentResourceManager, self).__init__(rap, log_dir=log_dir, plotter=plotter)

        self.save_dir = "/tmp/gym/"

        self.restricted_tasks = rap["restricted_tasks"]
        self.locks = rap["locks"]
        self.training_config = training_config
        self.model_name = "MARL_{}".format(rap["name"])
        if training_config["search_hyperparameters"]:
            self.model_name += "_tuned"
        self.learning_curve_plotter = LearningCurvePlotter()

    def train_model(self, iterations=10):

        stage1_hyperparams = [None] * len(self.locks)
        stage1_plotter = [LearningCurvePlotter() for _ in range(len(self.locks))]
        stage2_hyperparams = None
        stage2_plotter = LearningCurvePlotter()

        for _ in range(self.training_config["training_iterations"]):
            stage1_models = []
            lower_lvl_models = {}

            for idx, task_lock in enumerate(self.locks):
                if len(self.restricted_tasks) == 1:
                    task_locks = {self.restricted_tasks[0]: task_lock}
                else:
                    task_locks = {restricted_task: amount for restricted_task, amount
                                  in zip(self.restricted_tasks, task_lock)}

                environment_kwargs = {
                    "task_locks": task_locks,
                    "lower_lvl_models": lower_lvl_models,
                    "max_timesteps": self.training_config["steps_per_episode"]
                }

                policy_kwargs = {}
                name = self.model_name + "_stage1_lvl" + str(idx)
                if self.training_config["search_hyperparameters"] and (stage1_hyperparams[idx] is None):
                    stage1_hyperparams[idx] = self.search_hyperparams(RestrictedResourceAllocationEnvironment,
                                                                      environment_kwargs,
                                                                      policy_kwargs,
                                                                      training_steps=self.training_config["stage1_training_steps"])

                model = self.train_submodel(RestrictedResourceAllocationEnvironment,
                                            environment_kwargs,
                                            policy_kwargs,
                                            name=name,
                                            hyperparams=stage1_hyperparams[idx],
                                            training_steps=self.training_config["stage1_training_steps"])

                result = ts2xy(load_results(self.log_dir), 'timesteps')
                stage1_plotter[idx].add_result(result)

                stage1_models.append(model)
                if len(self.restricted_tasks) == 1:
                    lower_lvl_models[tuple(task_lock)] = model
                else:
                    for key in itertools.product(*task_lock):
                        lower_lvl_models[key] = model

            environment_kwargs = {
                "max_timesteps": self.training_config["steps_per_episode"]
            }
            policy_kwargs = {"stage1_models": stage1_models}
            name = self.model_name + "_" + "stage2"

            if self.training_config["search_hyperparameters"] and (stage2_hyperparams is None):
                stage2_hyperparams = self.search_hyperparams(ResourceAllocationEnvironment,
                                                             environment_kwargs,
                                                             policy_kwargs,
                                                             policy=MultiStageActorCritic,
                                                             training_steps=self.training_config["stage2_training_steps"])

            multistage_model = self.train_submodel(ResourceAllocationEnvironment,
                                                   environment_kwargs,
                                                   policy_kwargs,
                                                   policy=MultiStageActorCritic,
                                                   name=name,
                                                   hyperparams=stage2_hyperparams,
                                                   training_steps=self.training_config["stage2_training_steps"])

            result = ts2xy(load_results(self.log_dir), 'timesteps')
            stage2_plotter.add_result(result)

        for n in range(len(self.locks)):
            csv_name = self.model_name + "_stage1_lvl{0}_results".format(n)
            plot_name = self.model_name + "_stage1_lvl{0}_average_reward".format(n)
            stage1_plotter[n].save_results(csv_name)
            stage1_plotter[n].plot_average_results(filename=plot_name,
                                                   epoch_length=self.training_config["stage1_training_steps"])

        stage2_plotter.save_results(self.model_name + "_stage2_results")
        stage2_plotter.plot_average_results(filename=self.model_name + "_stage2_average_reward",
                                            epoch_length=self.training_config["stage2_training_steps"])

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
                    submodel = A2C.load(path)
                    environment_kwargs["lower_lvl_models"][key] = submodel
            if "stage1_models" in policy_kwargs:
                for path in model_paths:
                    stage1_model = A2C.load(path)
                    policy_kwargs["stage1_models"].append(stage1_model)
            
            environment = environment_class(ra_problem, **environment_kwargs)
            vector_environment = make_vec_env(lambda: environment, n_envs=1, monitor_dir=log_dir)
            
            rewards = []

            for n in range(hpsearch_iterations):
                model = A2C(policy,
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

    def train_submodel(self, environment_class, environment_kwargs, policy_kwargs,
                       policy=None, name="", hyperparams=None, training_steps=20000):

        if policy is None:
            policy = "MlpPolicy"

        config = {
            "verbose": 1,
            "tensorboard_log": self.log_dir,
            "policy_kwargs": policy_kwargs
        }

        if hyperparams is not None:
            for key in hyperparams.keys():
                config[key] = hyperparams[key]

        env = environment_class(self.ra_problem, **environment_kwargs)
        vector_env = make_vec_env(lambda: env, n_envs=1, monitor_dir=self.log_dir)

        model = A2C(policy,
                    vector_env,
                    **config)
        with ProgressBarManager(training_steps) as progress_callback:
            model.learn(total_timesteps=training_steps, callback=progress_callback)

        #print("Training Reward: ", evaluate_policy(model, vector_env, n_eval_episodes=100)[0])

        self.environment = vector_env

        self.plot_training_results(filename=name)

        return model



class MDPMultiAgentResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=50000, steps_per_episode=500, log_dir="/tmp/gym"):
        super(MDPMultiAgentResourceManager, self).__init__(rap, log_dir=log_dir)

        self.rap_mdp = MDPBuilder(self.ra_problem).build_mdp()
        self.rap_mdp.transform(3)
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

        self.training_steps = training_steps
        self.steps_per_episode = steps_per_episode

    def train_model(self):
        models = []

        lower_level_values = {}
        for lvl_id, level in enumerate(self.levels):
            current_level_values = {}
            for scc_id, scc in enumerate(level):
                state_idxs = scc.get_state_idxs()
                restricted_mdp = RestrictedMDP(self.rap_mdp, state_idxs, lower_level_values)
                environment = RestrictedMDPResourceAllocationEnvironment(self.ra_problem, restricted_mdp)
                vector_environment = make_vec_env(lambda: environment, n_envs=1, monitor_dir=self.log_dir)
                model = A2C('MlpPolicy', vector_environment, verbose=1, tensorboard_log=self.log_dir)

                with ProgressBarManager(self.training_steps) as progress_callback:
                    model.learn(total_timesteps=self.training_steps, callback=progress_callback)

                self.plot_training_results(title="SubAgent {0} {1}".format(lvl_id + 1, scc_id + 1))
                self.environment = environment
                self.model = model
                self.run_model()

                models.append(model)

                states = torch.tensor([restricted_mdp.idx_to_state_list(idx) for idx in state_idxs])
                _, values, _ = model.policy.forward(states)

                current_level_values.update({idx: value.item() for idx, value in zip(state_idxs, values)})
            lower_level_values = current_level_values

        whole_environment = MDPResourceAllocationEnvironment(self.ra_problem, self.steps_per_episode)
        whole_vector_environment = make_vec_env(lambda: whole_environment, n_envs=1, monitor_dir=self.log_dir)
        policy_kwargs = {"stage1_models": models}
        multistage_model = A2C(MultiStageActorCritic, whole_vector_environment, verbose=1, tensorboard_log=self.log_dir,
                               policy_kwargs=policy_kwargs)
        with ProgressBarManager(self.training_steps) as progress_callback:
            multistage_model.learn(total_timesteps=self.training_steps, callback=progress_callback)

        self.environment = whole_vector_environment
        self.model = multistage_model