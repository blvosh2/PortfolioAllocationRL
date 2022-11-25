from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from env import PortfolioEnv
from utils import split_df
import pandas as pd
import click


def train(upro_df, tmf_df, checkpoint_name='ppo_portfolio', step_size=70, drawdown=False):
    env = make_vec_env(PortfolioEnv,
                       env_kwargs={'upro_df': upro_df, 'tmf_df': tmf_df, 'step_size': step_size, 'drawdown': drawdown},
                       n_envs=8,
                       monitor_dir='./logs')
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, batch_size=256,
                policy_kwargs={'net_arch': [dict(pi=[256, 256, 256, 256], vf=[256, 256, 256, 256])]})
    model.learn(total_timesteps=1000000. * 5)
    model.save(checkpoint_name)


def infer(upro_df, tmf_df, checkpoint_name='ppo_portfolio', step_size=70):
    model = PPO.load(checkpoint_name)
    env = PortfolioEnv(upro_df=upro_df, tmf_df=tmf_df, step_size=step_size,
                       deterministic=True)
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
@click.option('--istest', is_flag=True, default=False, help='Test')
@click.option('-d', '--drawdown', is_flag=True, default=False, help='drawdown punishment')
@click.option('-n', '--checkpoint_name', type=str, default='ppo_portfolio', help='checkpoint name')
@click.option('-s', '--step-size', type=int, default=70, help='step size of the environment')
def main(istrain=False, isinfer=False, istest=False, checkpoint_name='ppo_portfolio', step_size=70, drawdown=False):
    upro_df = pd.read_csv('./data/UPRO_final.csv')
    tmf_df = pd.read_csv('./data/TMF_final.csv')
    tmf_train, tmf_val, tmf_test = split_df(tmf_df)
    upro_train, upro_val, upro_test = split_df(upro_df)

    if istrain:
        train(upro_df=upro_train, tmf_df=tmf_train, checkpoint_name=checkpoint_name, step_size=step_size, drawdown=drawdown)
    if isinfer:
        infer(upro_df=upro_val, tmf_df=tmf_val, checkpoint_name=checkpoint_name, step_size=step_size)
    if istest:
        infer(upro_df=upro_test, tmf_df=tmf_test, checkpoint_name=checkpoint_name, step_size=step_size)


if __name__ == '__main__':
    main()
