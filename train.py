from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from env import PortfolioEnv
import pandas as pd
import click

upro_df = pd.read_csv('./data/UPROSIM_preprocessed.csv')
tmf_df = pd.read_csv('./data/TMFSIM_preprocessed.csv')


def train(to_index=6714, checkpoint_name='ppo_portfolio', step_size=70):
    env = make_vec_env(PortfolioEnv, env_kwargs={'upro_df': upro_df[:to_index], 'tmf_df': tmf_df[:to_index], 'step_size': step_size}, n_envs=8,
                       monitor_dir='./logs')
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, batch_size=256, policy_kwargs={'net_arch': [dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]})
    model.learn(total_timesteps=1000000. * 5)
    model.save(checkpoint_name)


def infer(from_index=6714, checkpoint_name='ppo_portfolio', step_size=70):
    model = PPO.load(checkpoint_name)
    env = PortfolioEnv(upro_df=upro_df[from_index:], tmf_df=tmf_df[from_index:], step_size=step_size, deterministic=True)
    obs = env.reset()
    rewards = []
    actions = []
    for i in range(100000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        if dones:
            print("done")
            env.render(savefig=True, figname=checkpoint_name)
            env.close()
            break


@click.command()
@click.option('-t', '--istrain', is_flag=True, default=False, help='Train the model')
@click.option('-i', '--isinfer', is_flag=True, default=False, help='Infer')
@click.option('-n', '--checkpoint_name', type=str, default='ppo_portfolio', help='checkpoint name')
@click.option('--split', type=int, default=6714, help='Index to split train and test data')
@click.option('-s', '--step-size', type=int, default=70, help='step size of the environment')
def main(istrain=False, isinfer=False, checkpoint_name='ppo_portfolio', split=6714, step_size=70):
    if istrain:
        train(to_index=split, checkpoint_name=checkpoint_name, step_size=step_size)
    if isinfer:
        infer(from_index=split, checkpoint_name=checkpoint_name, step_size=step_size)


if __name__ == '__main__':
    main()