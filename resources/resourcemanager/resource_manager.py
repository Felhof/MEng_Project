from resources.resourcemanager.base_resource_manager import BaseResourceManager

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy

from resources.callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from resources.environments.rap_environment import ResourceAllocationEnvironment
from resources.plotter import LearningCurvePlotter


class ResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=60000, steps_per_episode=100, log_dir="/tmp/gym", training_config=None):
        super(ResourceManager, self).__init__(rap, log_dir=log_dir)

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
