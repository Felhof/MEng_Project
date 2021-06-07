import time

from resources.resourcemanager.base_resource_manager import BaseResourceManager

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from resources.callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resources.environments.rap_environment import ResourceAllocationEnvironment
from resources.callbacks import SavePerformanceOnCheckpoints
from stable_baselines3.common.callbacks import EveryNTimesteps


class ResourceManager(BaseResourceManager):

    def __init__(self, rap, log_dir="/tmp/gym", training_config=None, algorithm="A2C", checkpoint_results=None):
        super(ResourceManager, self).__init__(rap, log_dir=log_dir, algorithm=algorithm,
                                              checkpoint_results=checkpoint_results)

        self.model_name = rap["name"] + "_baseline"

        self.environment = ResourceAllocationEnvironment(self.ra_problem)
        # If the environment doesn't follow the interface, an error will be thrown
        check_env(self.environment, warn=True)

        # wrap it
        self.vector_environment = make_vec_env(lambda: self.environment, n_envs=1, monitor_dir=self.log_dir)

        self.training_steps = training_config["stage1_training_steps"]

    def train_model(self):

        for _ in range(1):

            auto_save_callback = SaveOnBestTrainingRewardCallback(log_dir=self.log_dir)
            auto_save_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=auto_save_callback)

            self.environment = Monitor(self.environment, self.log_dir)
            self.model = self.algorithm('MlpPolicy', self.environment, verbose=1, tensorboard_log=self.log_dir)

            name = self.model_name + "_full_model"
            checkpoint_callback = SavePerformanceOnCheckpoints(resource_manager=self, name=name,
                                                               checkpoint_results=self.checkpoint_results)
            checkpoint_callback_every_1000_steps = EveryNTimesteps(n_steps=1000, callback=checkpoint_callback)

            with ProgressBarManager(self.training_steps) as progress_callback:
                self.model.learn(total_timesteps=self.training_steps, callback=[progress_callback,
                                                                                auto_save_callback_every_1000_steps,
                                                                                checkpoint_callback_every_1000_steps])

        self.save_episode_rewards_as_csv()
        # self.plot_training_results(filename=self.model_name + "_results", show=True)