import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym

INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS = 40.
# USD value
INITIAL_PORTFOLIO_BALANCE = 100000
N_DISCRETE_ACTIONS = 101
# trading days
WINDOW_SIZE = 10
# action frequency in days
STEP_SIZE = 70
# normalize the data
BALANCE_NORM_FACTOR = 1e8
STOCK_NORM_FACTOR = 2e5
UPRO_MAX = 23.28
TMF_MAX = 19.02





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
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.num_tmf_history = []
        self.num_upro_history = []
        # Example for using image as input:
        high = np.zeros(5 + 2 * WINDOW_SIZE)
        high[0:] = np.inf
        high[4] = 100
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=(5 + 2 * WINDOW_SIZE,))

    def step(self, action):
        # Execute one time step within the environment
        info = {'none': None}
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
            return np.zeros(self.observation_space.shape), self.balance / BALANCE_NORM_FACTOR, self.done, info
        reward = self.balance
        next_state = self.get_state()

        # TODO: add info if needed (e.g. for debugging)
        return next_state, reward / BALANCE_NORM_FACTOR, self.done, info

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
        history = np.stack([self.upro_df['Price'].to_numpy()[start_pos:end_pos] / UPRO_MAX,
                            self.tmf_df['Price'].to_numpy()[start_pos:end_pos]]).T.flatten() / TMF_MAX

        return np.concatenate([np.array([self.balance / BALANCE_NORM_FACTOR, self.invested / BALANCE_NORM_FACTOR, self.num_stocks_upro / STOCK_NORM_FACTOR,
                                         self.num_stocks_tmf / STOCK_NORM_FACTOR, self.allocation / 100.]), history])


def run(random_action=True):
    """Run the environment."""
    stock_env = PortfolioEnv()
    stock_env.reset(seed=0)
    done = False
    rewards = []
    while not done:
        if random_action:
            action = stock_env.action_space.sample()
        else:
            action = 68.
        next_state, reward, done, _ = stock_env.step(action=action)
        rewards.append(reward)
        if done:
            # stock_env.render()
            print("finished")
    return rewards[-1]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rewards = []
    for _ in range(1):
        reward = run(random_action=False)
        rewards.append(reward)
    print(np.mean(rewards))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
