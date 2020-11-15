import numpy as np
import gym
from gym import spaces


class GridWorld(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, grid_size=5, rewards={}, start_state=(0, 0), goal_state=(4, 4)):
        super(GridWorld, self).__init__()

        # Size of the grid
        self.grid_size = grid_size
        # Initialize the agent in the corner of the grid
        self.agent_pos = start_state
        self.start_state = start_state
        self.goal_state = goal_state

        # Initialize rewards
        rewards[goal_state] = 10
        self.rewards = rewards

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        n_states = grid_size**2
        self.observation_space = spaces.Discrete(n_states)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = self.start_state
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return self.agent_position_to_state()

    def step(self, action):
        x, y = self.agent_pos
        if action == self.UP:
            y = min(self.grid_size - 1, y + 1)
        elif action == self.RIGHT:
            x = min(self.grid_size - 1, x + 1)
        elif action == self.DOWN:
            y = max(0, y - 1)
        elif action == self.LEFT:
            x = max(0, x - 1)
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        self.agent_pos = (x, y)

        observation = self.agent_position_to_state()

        # Are we at the left of the grid?
        done = bool(self.agent_pos == self.goal_state)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = self.rewards.get(self.agent_pos, 0)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return observation, reward, done, info

    def render(self, mode='console'):
        if mode != 'console':
          raise NotImplementedError()

        symbols = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.agent_pos == (x, y):
                    if self.agent_pos == self.goal_state:
                        symbols.append("V")
                    if self.rewards.get(self.agent_pos, 0) >= 0:
                        symbols.append("o")
                    else:
                        symbols.append("x")
                elif self.goal_state == (x, y):
                    symbols.append("G")
                else:
                    if self.rewards.get((x, y), 0) >= 0:
                        symbols.append(".")
                    else:
                        symbols.append("!")
            symbols.append("\n")

        print("".join(symbols))


    def close(self):
        pass

    def agent_position_to_state(self):
        return self.agent_pos[0] + self.grid_size*self.agent_pos[1]