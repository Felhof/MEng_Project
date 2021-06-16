import os
import time

import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class SavePerformanceOnCheckpoints(BaseCallback):
    def __init__(self, stage1_time=0, checkpoints=None, resource_manager=None, n_eval_episodes=10000, name="",
                 checkpoint_results=None, log_dir="/tmp/gym"):
        super(SavePerformanceOnCheckpoints, self).__init__()
        self.start_time = 0
        self.checkpoint_scores = []
        self.checkpoint_id = 0
        self.stage1_time = stage1_time
        if checkpoints is None:
            self.checkpoints = [4, 8, 12, 16, 20]
        else:
            self.checkpoints = checkpoints
        self.rm = resource_manager
        self.n_eval_episodes = n_eval_episodes
        self.name = name
        self.checkpoint_results = checkpoint_results

        self.best_model_path = os.path.join(log_dir, 'best_model.zip')

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        keep_training = True
        time_elapsed = time.time() - self.start_time + self.stage1_time
        reward = -1
        while True:
            checkpoint = self.checkpoints[self.checkpoint_id]
            if time_elapsed / 60 > checkpoint:
                if reward == -1:
                    reward = self.rm.run_model(n_steps=self.n_eval_episodes, save=True, name=self.name,
                                               model_path=self.best_model_path)
                self.checkpoint_scores.append(reward)
                if self.checkpoint_id == len(self.checkpoints) - 1:
                    keep_training = False
                    self.checkpoint_results.add_result(self.checkpoint_scores)
                    break
                self.checkpoint_id += 1
            else:
                break
        return keep_training


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    https://stable-baselines.readthedocs.io/en/master/guide/examples.html
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                self.model.save(self.save_path)

        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
