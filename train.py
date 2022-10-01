from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from env import PortfolioEnv
import pandas as pd

upro_df = pd.read_csv('./data/UPROSIM_preprocessed.csv')
tmf_df = pd.read_csv('./data/TMFSIM_preprocessed.csv')


def train(to_index=6714):
    env = make_vec_env(PortfolioEnv, env_kwargs={'upro_df': upro_df[:to_index], 'tmf_df': tmf_df[:to_index]}, n_envs=4,
                       monitor_dir='./logs')
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, batch_size=256)
    model.learn(total_timesteps=1000000.)
    model.save("ppo_portfolio")


def infer(from_index=6714):
    model = PPO.load("ppo_portfolio")
    env = PortfolioEnv(upro_df=upro_df[from_index:], tmf_df=tmf_df[from_index:])
    obs = env.reset()
    rewards = []
    actions = []
    for i in range(100000):
        action, _states = model.predict(obs)
        obs, reward, dones, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        if dones:
            print("done")
            env.render()
            env.close()
            break


if __name__ == '__main__':
    # train()
    infer(from_index=6714)