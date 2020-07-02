from option_keyboard.core.utils import set_global_seed
from option_keyboard.core.networks import MlpDiscrete
import argparse
import gym
import option_keyboard.envs
import torch
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser('test_ok')
parser.add_argument('-e', '--env-name', default='ForagingWorld-v0',
                    help='Name of environment')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='Random seed')
parser.add_argument('--exp-name', required=True,
                    help='Name of experiment')
parser.add_argument('--n-test-episodes', default=100,
                    help='Number of test episodes')
parser.add_argument('--visualize', action='store_true',
                    help='Flag for visualization')
parser.add_argument('--saved-models', required=True,
                    help='Path to saved models')
parser.add_argument('--save-path', default='',
                    help='Path to file to which results are to be saved')


def main():
    args = parser.parse_args()
    env = gym.make(args.env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    Q = MlpDiscrete(input_dim=env.observation_space.shape[0],
                    output_dim=env.action_space.n,
                    hidden=[64, 128])

    if not torch.cuda.is_available():
        checkpoint = torch.load(os.path.join(args.saved_models, 'agent.pt'),
                                map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(args.saved_models, 'agent.pt'))
    Q.load_state_dict(checkpoint['Q'])
    Q.to(device)

    if args.save_path:
        fp = open(args.save_path, 'wb')

    returns = []
    for _ in range(args.n_test_episodes):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        done = False
        ep_return = 0
        if args.visualize:
            env.render()

        while not done:
            q = Q(s)
            a = torch.argmax(q)
            s, r, done, _ = env.step(a)
            ep_return += r
            s = torch.from_numpy(s).float().to(device)
            if args.visualize:
                env.render()

        print('Episodic return:', ep_return)
        returns.append(ep_return)

    returns = np.array(returns)
    print('Mean: %f, Std. dev: %f' % (returns.mean(), returns.std()))
    if args.save_path:
        pickle.dump({'Seed': args.seed, 'Returns': returns}, fp)


if __name__ == '__main__':
    main()
