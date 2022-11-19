import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import gym

INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS = 40.
# USD value
# actions : [+5% upro to tmf, -5% upro to tmf, +10% upro to tmf, -10% upro to tmf, do nothing, rebalance]
N_DISCRETE_ACTIONS = 6
# trading days
WINDOW_SIZE = 10
# action frequency in days
STEP_SIZE = 70
# for random environment we define a episode length
EPISODE_LENGTH = 3000


class PortfolioEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, upro_df, tmf_df, step_size=STEP_SIZE, deterministic=False):
        super(PortfolioEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.deterministic = deterministic
        self.current_step = 0
        self.step_size = step_size
        self.allocation = None
        self.done = False
        self.num_stocks_tmf = 0
        self.num_stocks_upro = 0
        self.total_change = 1.
        self.current_day = WINDOW_SIZE
        self.upro_df = upro_df
        self.tmf_df = tmf_df
        self.end_day = self.tmf_df['Change'].to_numpy().shape[0]
        self.action_space = gym.spaces.Discrete(N_DISCRETE_ACTIONS)
        self.allocation_history = []
        # Example for using image as input:
        high = np.zeros(1 + 2 * WINDOW_SIZE)
        high[0:] = np.inf
        self.observation_space = gym.spaces.Box(low=0, high=high, shape=(1 + 2 * WINDOW_SIZE,))

    def step(self, action):
        self.current_step += 1
        # Execute one time step within the environment
        info = {'none': None}
        if action == 0:
            self.allocation += 5.
        elif action == 1:
            self.allocation -= 5.
        elif action == 2:
            self.allocation += 10.
        elif action == 3:
            self.allocation -= 10.
        elif action == 4:
            pass
        elif action == 5:
            self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS

        if self.allocation > 100.:
            self.allocation = 100.
        elif self.allocation < 0.:
            self.allocation = 0.

        self.allocation_history.append(self.allocation)
        self.current_day += self.step_size
        df_index = self.current_day if self.current_day < self.end_day else self.end_day - 1
        tmf_change = np.prod((self.tmf_df['Change'].to_numpy()[df_index - self.step_size:df_index] + 100.) / 100.)
        upro_change = np.prod((self.upro_df['Change'].to_numpy()[df_index - self.step_size:df_index] + 100.) / 100.)
        self.total_change *= ((tmf_change * (100. - self.allocation) + upro_change * self.allocation) / 100.)
        reward = self.total_change / 1000.

        if self.current_step >= EPISODE_LENGTH:
            self.done = True
            return np.zeros(self.observation_space.shape), reward, self.done, info

        if not self.deterministic:
            self.current_day = np.random.randint(WINDOW_SIZE, self.end_day - self.step_size - 1)

        next_state = self.get_state()
        return next_state, reward, self.done, info

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.done = False
        self.current_day = WINDOW_SIZE
        self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS
        self.total_change = 1.
        state = self.get_state()
        return state

    def render(self, mode='human', close=False, compare_to_balance=45., savefig=False, figname=None):
        if self.done:
            self.current_day = WINDOW_SIZE
            self.allocation = INITIAL_PORTFOLIO_ALLOCATION_PERCENTAGE_STOCKS
            tmf_change = (self.tmf_df['Change'].to_numpy() + 100.) / 100.
            upro_change = (self.upro_df['Change'].to_numpy() + 100.) / 100.

            # get balanced history
            truncated_upro_history, truncated_tmf_history = self.get_balanced_history(compare_to_balance)
            default_allocation_balance = ((tmf_change[WINDOW_SIZE:] * truncated_tmf_history + upro_change[
                                                                                              WINDOW_SIZE:] * truncated_upro_history) / 100.).tolist()
            default_allocation_balance = np.cumprod(default_allocation_balance)

            truncated_upro_history = np.repeat(np.array(self.allocation_history), self.step_size)[
                                     :self.end_day - WINDOW_SIZE]
            truncated_tmf_history = np.repeat((100. - np.array(self.allocation_history)), self.step_size)[
                                    :self.end_day - WINDOW_SIZE]
            agent_balance = ((tmf_change[WINDOW_SIZE:] * truncated_tmf_history + upro_change[
                                                                                 WINDOW_SIZE:] * truncated_upro_history) / 100.).tolist()
            agent_balance = np.cumprod(agent_balance)
            plt.plot(agent_balance, label='agent')
            plt.plot(default_allocation_balance, label='balanced')
            plt.legend()
            plt.title(figname if figname else 'agent vs balanced')
            if savefig and figname is not None:
                plt.savefig(figname)
            plt.show()
        else:
            print("render can be called only in the end of an episode")

    def get_state(self):
        # get a vector of shape (2 * WINDOWS_SIZE,) of a concatenation of (UTF_price, TMF_price) of each day
        start_pos = self.current_day - WINDOW_SIZE
        end_pos = self.current_day
        history = np.stack([(self.upro_df['Change'].to_numpy()[start_pos:end_pos] + 100.) / 100.,
                            (self.tmf_df['Change'].to_numpy()[start_pos:end_pos] + 100.) / 100.]).T.flatten()

        return np.concatenate([np.array([self.allocation / 100.]), history])

    def get_balanced_history(self, compare_to_balance=45.):
        truncated_upro_history = None
        truncated_tmf_history = None
        stock_env = PortfolioEnv(upro_df=self.upro_df, tmf_df=self.tmf_df, step_size=self.step_size, deterministic=True)
        stock_env.reset(seed=0)
        done = False
        while not done:
            action = compare_to_balance
            next_state, reward, done, _ = stock_env.step(action=action)
            if done:
                truncated_upro_history = np.repeat(np.array(stock_env.allocation_history), self.step_size)[
                                         :self.end_day - WINDOW_SIZE]
                truncated_tmf_history = np.repeat((100. - np.array(stock_env.allocation_history)), self.step_size)[
                                        :self.end_day - WINDOW_SIZE]

        return truncated_upro_history, truncated_tmf_history


def run(random_action=False):
    """Run the environment."""
    upro_df = pd.read_csv('./data/UPROSIM_preprocessed.csv')
    tmf_df = pd.read_csv('./data/TMFSIM_preprocessed.csv')
    stock_env = PortfolioEnv(upro_df=upro_df, tmf_df=tmf_df)
    stock_env.reset(seed=0)
    done = False
    rewards = []
    while not done:
        if random_action:
            action = stock_env.action_space.sample()
        else:
            action = 45.
        next_state, reward, done, _ = stock_env.step(action=action)
        rewards.append(reward)
        if done:
            # stock_env.render()
            print("finished")
    return rewards


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rewards = []
    for _ in range(1):
        reward = run(random_action=True)
        rewards.append(reward)
    print(np.mean(rewards))
    print(sum(reward))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
