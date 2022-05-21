import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pandas as pd
import gym
import random

from gym import Env, spaces
import time

INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE = 45
INITIAL_PORTFOLIO_BALANCE = 100000
N_DISCRETE_ACTIONS = 101
WINDOW_SIZE = 10


class PortfolioEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, path_to_data_file):
        super(PortfolioEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.allocation = None
        self.balance = None
        self.upro_df = pd.read_csv('./data/UPROSIM_preprocessed.csv')
        self.tmf_df = pd.read_csv('./data/TMFSIM_preprocessed.csv')
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        high = np.zeros(2 + 2 * 101)
        high[1:] = 100
        high[0] = np.inf
        self.observation_space = spaces.Box(low=0, high=high, shape=(2 + 2 * WINDOW_SIZE, ))

    def step(self, action):
        # Execute one time step within the environment
        ...

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_PORTFOLIO_BALANCE
        self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE
        # np.stack([upro_df['Price'].to_numpy()[:WINDOW_SIZE], tmf_df['Price'].to_numpy()[:WINDOW_SIZE]]).T.flatten()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
