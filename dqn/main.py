from core.utils import set_global_seed, create_log_files
from dqn import dqn
import argparse
import gym
import envs
import torch


parser = argparse.ArgumentParser('DQN')
parser.add_argument('-e', '--env-name', default='ForagingWorld-v0',
                    help='Name of environment')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--exp-name', required=True,
                    help='Name of experiment')
parser.add_argument('--n-test-runs', default=10, type=int,
                    help='Number of episodes at each test run')
parser.add_argument('--log-dir', default='/data/results',
                    help='Specify path to directory where logs will be'
                    'saved')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='Discount factor')
parser.add_argument('--eps', default=0.1, type=float,
                    help='Exploration parameter')
parser.add_argument('--alpha', default=1e-3, type=float,
                    help='Learning rate for the agent')
parser.add_argument('--n-training-steps', default=1e6, type=int,
                    help='Number of steps for which OK is to be trained')
parser.add_argument('--batch-size', default=10, type=int,
                    help='Batch size for updating Q-values')
parser.add_argument('--pretrained-agent', default='',
                    help='Path to pretrained agent')
parser.add_argument('--test-interval', default=500, type=int,
                    help='Interval at which agent is tested')
parser.add_argument('--scenario', default=1, type=int,
                    help='Scenario (described in the paper)')


def main():
    args = parser.parse_args()

    env = gym.make(args.env_name, scenario=args.scenario)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(args.seed)

    log_dir, log_files = create_log_files(args, 0)

    dqn(env=env,
        eps=args.eps,
        gamma=args.gamma,
        alpha=args.alpha,
        device=device,
        training_steps=args.n_training_steps,
        batch_size=args.batch_size,
        pretrained_agent=args.pretrained_agent,
        test_interval=args.test_interval,
        n_test_runs=args.n_test_runs,
        log_file=log_files['agent'],
        log_dir=log_dir)


if __name__ == '__main__':
    main()
