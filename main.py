import os

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

from callbacks import SaveOnBestTrainingRewardCallback, ProgressBarManager
from go_left_env import GoLeftEnv
from gridworld import GridWorld
from resource_manager import ResourceManager

# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

#rewards = {(1, 0): -1, (0,2): -1, (2, 1): -1, (1, 3): -1, (3, 2): -1, (2, 4): -1}

#env = GridWorld(grid_size=5, rewards=rewards, start_state=(0, 0), goal_state=(4, 4))
env = ResourceManager()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)

# wrap it
env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)

# Create callbacks
auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

model = ACKTR('MlpPolicy', env, verbose=1)\

with ProgressBarManager(10000) as progress_callback:
    # This is equivalent to callback=CallbackList([progress_callback, auto_save_callback])
    model.learn(10000, callback=[progress_callback, auto_save_callback])

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
