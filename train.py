import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import PortfolioEnv

if __name__ == '__main__':
    env = PortfolioEnv()
    # env = make_vec_env(PortfolioEnv, n_envs=4)
    # model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save("ppo_portfolio")

    # Enjoy trained agent
    model = PPO.load("ppo_portfolio")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            print("done")
            break
        # env.render()
    pass
