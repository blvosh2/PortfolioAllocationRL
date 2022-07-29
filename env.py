import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pandas as pd
import gym
import random

from gym import Env, spaces
import time

INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS = 40.
# USD value
INITIAL_PORTFOLIO_BALANCE = 100000
N_DISCRETE_ACTIONS = 101
# trading days
WINDOW_SIZE = 10
# action frequency in days
STEP_SIZE = 70


class PortfolioEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PortfolioEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.allocation = None
        self.balance = None
        self.done = False
        self.num_stocks_tmf = 0
        self.num_stocks_upro = 0
        self.invested = 0.
        self.current_day = WINDOW_SIZE
        self.upro_df = pd.read_csv('./data/UPROSIM_preprocessed.csv')
        self.tmf_df = pd.read_csv('./data/TMFSIM_preprocessed.csv')
        self.end_day = self.tmf_df['Price'].to_numpy().shape[0]
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.num_tmf_history = []
        self.num_upro_history = []
        # Example for using image as input:
        high = np.zeros(5 + 2 * WINDOW_SIZE)
        high[0:] = np.inf
        high[4] = 100
        self.observation_space = spaces.Box(low=0, high=high, shape=(5 + 2 * WINDOW_SIZE,))

    def step(self, action):
        # Execute one time step within the environment
        self.allocation = action
        self.num_stocks_upro, self.num_stocks_tmf, remainder = self.calculate_num_of_stocks()
        self.num_tmf_history.append(self.num_stocks_tmf)
        self.num_upro_history.append(self.num_stocks_upro)
        self.current_day += STEP_SIZE
        tmf_price = self.tmf_df['Price'].to_numpy()[self.current_day if self.current_day < self.end_day else -1]
        upro_price = self.upro_df['Price'].to_numpy()[self.current_day if self.current_day < self.end_day else -1]
        self.invested = (upro_price * self.num_stocks_upro + tmf_price * self.num_stocks_tmf)
        self.balance = remainder + self.invested
        if self.current_day >= self.end_day:
            self.done = True
            return np.zeros(self.observation_space.shape), self.balance, self.done
        reward = self.balance
        next_state = self.get_state()

        return next_state, reward, self.done

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        self.done = False
        self.current_day = WINDOW_SIZE
        self.balance = INITIAL_PORTFOLIO_BALANCE
        self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS
        self.num_stocks_upro, self.num_stocks_tmf, remainder = self.calculate_num_of_stocks()
        self.invested = self.balance - remainder
        state = self.get_state()

        return state

    def calculate_num_of_stocks(self):
        tmf_price = self.tmf_df['Price'].to_numpy()[self.current_day]
        upro_price = self.upro_df['Price'].to_numpy()[self.current_day]
        num_stocks_tmf = (self.balance * (1. - (self.allocation / 100.))) // tmf_price
        num_stocks_upro = (self.balance * (self.allocation / 100.)) // upro_price
        remainder = self.balance - (upro_price * num_stocks_upro + tmf_price * num_stocks_tmf)
        return num_stocks_upro, num_stocks_tmf, remainder

    def render(self, mode='human', close=False):
        if self.done:
            self.current_day = WINDOW_SIZE
            self.balance = INITIAL_PORTFOLIO_BALANCE
            self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS
            num_stocks_upro, num_stocks_tmf, remainder = self.calculate_num_of_stocks()
            tmf_price = self.tmf_df['Price'].to_numpy()
            upro_price = self.upro_df['Price'].to_numpy()
            default_allocation_balance = num_stocks_tmf * tmf_price + num_stocks_upro * upro_price
            agent_balance = tmf_price[WINDOW_SIZE:] * np.array(self.num_tmf_history) + upro_price[WINDOW_SIZE:] * np.array(self.num_upro_history)
            plt.plot(agent_balance)
            plt.plot(default_allocation_balance)
            plt.show()
            pass
        else:
            print("render can be called only in the end of an episode")

    def get_state(self):
        # get a vector of shape (2 * WINDOWS_SIZE,) of a concatenation of (UTF_price, TMF_price) of each day
        start_pos = self.current_day - WINDOW_SIZE
        end_pos = self.current_day
        history = np.stack([self.upro_df['Price'].to_numpy()[start_pos:end_pos],
                            self.tmf_df['Price'].to_numpy()[start_pos:end_pos]]).T.flatten()

        return np.concatenate([np.array([self.balance, self.invested, self.num_stocks_upro,
                                         self.num_stocks_tmf, self.allocation]), history])


def main(random_action=True):
    stock_env = PortfolioEnv()
    stock_env.reset(seed=0)
    done = False
    rewards = []
    while not done:
        if random_action:
            action = stock_env.action_space.sample()
        else:
            action = 45.
        next_state, reward, done = stock_env.step(action=action)
        rewards.append(reward)
        if done:
            # stock_env.render()
            print("finished")
    return rewards[-1]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rewards = []
    for _ in range(1):
        reward = main(True)
        rewards.append(reward)
    print(np.mean(rewards))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
