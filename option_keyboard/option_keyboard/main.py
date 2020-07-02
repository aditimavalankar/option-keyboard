from option_keyboard.option_keyboard.agent import keyboard_player
from option_keyboard.option_keyboard.learn import learn_options
from option_keyboard.core.utils import set_global_seed, create_log_files
import argparse
import gym
from itertools import product
import option_keyboard.envs
import torch
import numpy as np
import os


parser = argparse.ArgumentParser('OK')
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
# Option Keyboard parameters
parser.add_argument('--gamma-ok', default=0.9, type=float,
                    help='Discount factor for the Option Keyboard')
parser.add_argument('--eps1-ok', default=0.2, type=float,
                    help='Probability of changing the cumulant')
parser.add_argument('--eps2-ok', default=0.1, type=float,
                    help='Exploration parameter for the Option Keyboard')
parser.add_argument('--alpha-ok', default=1e-4, type=float,
                    help='Learning rate for the Option Keyboard - '
                    'tried from [1e-1, 1e-2, 1e-3, 1e-4')
parser.add_argument('--max-steps-ok', default=100, type=int,
                    help='Maximum number of steps in an episode when the'
                    'Option Keyboard is being trained')
parser.add_argument('--n-training-steps-ok', default=5e5, type=int,
                    help='Number of steps for which OK is to be trained')
parser.add_argument('--ok-batch-size', default=10, type=int,
                    help='Batch size for updating option Q-values')
parser.add_argument('--pretrained-options', default='',
                    help='Path to pretrained option models (experiment results'
                    'directory)')
parser.add_argument('--test-interval-option', default=1500, type=int,
                    help='Interval at which option is tested')
# Agent parameters
parser.add_argument('-n', '--n-training-steps-agent', default=1e6, type=int,
                    help='Number of training steps for which agent is to be'
                    'trained')
parser.add_argument('--agent-batch-size', default=10, type=int,
                    help='Batch size for updating agent Q-values')
parser.add_argument('--eps-agent', default=0.1, type=float,
                    help='Exploration over weight vector w')
parser.add_argument('--gamma-agent', default=0.99, type=float,
                    help='Discount factor for the agent')
parser.add_argument('--alpha-agent', default=1e-4, type=float,
                    help='Learning rate for the agent - '
                    'tried from [1e-1, 1e-2, 1e-3, 1e-4')
parser.add_argument('--max-steps-agent', default=300, type=int,
                    help='Maximum number of steps in an episode when the'
                    'agent is being trained')
parser.add_argument('--test-interval-agent', default=500, type=int,
                    help='Interval at which agent is tested')
parser.add_argument('--pretrained-agent', default='',
                    help='Path to pretrained agent model (experiment results'
                    'directory)')
parser.add_argument('--scenario', default=1, type=int,
                    help='Scenario (described in the paper)')


def main():
    args = parser.parse_args()

    env = gym.make(args.env_name, scenario=args.scenario)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)
    log_dir, log_files = create_log_files(args, env.num_resources())

    # Learn the Option Keyboard
    env.set_learning_options(np.array([1, 1]), True)
    Q_E = learn_options(env=env,
                        d=env.num_resources(),
                        eps1=args.eps1_ok,
                        eps2=args.eps2_ok,
                        alpha=args.alpha_ok,
                        gamma=args.gamma_ok,
                        max_ep_steps=args.max_steps_ok,
                        device=device,
                        training_steps=args.n_training_steps_ok,
                        batch_size=args.ok_batch_size,
                        pretrained_options=args.pretrained_options,
                        test_interval=args.test_interval_option,
                        n_test_runs=args.n_test_runs,
                        log_files=log_files,
                        log_dir=log_dir)
    env.set_learning_options(np.array([1, 1]), False)

    for i in range(env.num_resources()):
        if args.n_training_steps_ok == 0:
            checkpoint = torch.load(os.path.join(args.pretrained_options,
                                                 'value_fn_%d.pt'
                                                 % (i + 1)))
        else:
            checkpoint = torch.load(os.path.join(log_dir, 'saved_models',
                                                 'best', 'value_fn_%d.pt'
                                                 % (i + 1)))
        Q_E[i].q_net.load_state_dict(checkpoint['Q'])

    W = [x for x in product([-1, 0, 1], repeat=2) if sum(x) >= 0]
    W.remove((0, 0))
    W = np.array(W)

    # Learn the agent
    Q_w = keyboard_player(env=env,
                          W=W,
                          Q=Q_E,
                          alpha=args.alpha_agent,
                          eps=args.eps_agent,
                          gamma=args.gamma_agent,
                          training_steps=args.n_training_steps_agent,
                          batch_size=args.agent_batch_size,
                          pretrained_agent=args.pretrained_agent,
                          max_ep_steps=args.max_steps_agent,
                          device=device,
                          test_interval=args.test_interval_agent,
                          n_test_runs=args.n_test_runs,
                          log_file=log_files['agent'],
                          log_dir=log_dir)


if __name__ == '__main__':
    main()
