import option_keyboard.envs
from option_keyboard.core.utils import set_global_seed
from option_keyboard.option_keyboard.ok import option_keyboard
from option_keyboard.core.value_function import ValueFunction
import argparse
import gym
import torch
import numpy as np
import os
import pickle
from itertools import product
from option_keyboard.core.networks import MlpDiscrete


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

    d = env.num_resources()

    hyperparams_file = open(os.path.join(args.saved_models.split('saved_models')[0],
                            'hyperparams'), 'rb')

    # Loading saved models and constant values
    returns = []
    if args.save_path:
        fp = open(args.save_path, 'a+b')

    W = [x for x in product([-1, 0, 1], repeat=2) if sum(x) >= 0]
    W.remove((0, 0))
    W = np.array(W)

    hyperparams = pickle.load(hyperparams_file)
    gamma = hyperparams.gamma_ok
    max_ep_steps = hyperparams.max_steps_agent

    value_fns = [ValueFunction(input_dim=env.observation_space.shape[0] + d,
                               action_dim=(env.action_space.n + 1),
                               n_options=d,
                               hidden=[64, 128],
                               batch_size=hyperparams.ok_batch_size,
                               gamma=gamma,
                               alpha=hyperparams.alpha_ok)
                 for _ in range(d)]

    Q_w = MlpDiscrete(input_dim=env.observation_space.shape[0],
                      output_dim=W.shape[0],
                      hidden=[64, 128])

    for i in range(env.num_resources()):
        if not torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(args.saved_models,
                                                 'value_fn_%d.pt' %
                                                 (i + 1)),
                                    map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(os.path.join(args.saved_models,
                                                 'value_fn_%d.pt' %
                                                 (i + 1)))

        value_fns[i].q_net.load_state_dict(checkpoint['Q'])
        value_fns[i].q_net.to(device)

    if not torch.cuda.is_available():
        checkpoint = torch.load(os.path.join(args.saved_models, 'agent.pt'),
                                map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(os.path.join(args.saved_models, 'agent.pt'))

    Q_w.load_state_dict(checkpoint['Q'])
    Q_w.to(device)
    # ########

    for _ in range(args.n_test_episodes):
        s = env.reset()
        done = False
        s = torch.from_numpy(s).float().to(device)
        n_steps = 0
        ret = 0

        while not done:
            w = W[torch.argmax(Q_w(s))]
            (s_next, done, _, _, n_steps, info) = option_keyboard(env, s, w,
                                                                  value_fns,
                                                                  gamma,
                                                                  n_steps,
                                                                  max_ep_steps,
                                                                  device,
                                                                  args.visualize)

            ret += sum(info['rewards'])
            s = torch.from_numpy(s_next).float().to(device)

        print('Episode return:', ret)
        returns.append(ret)

    returns = np.array(returns)
    print('Mean: %f, Std. dev: %f' % (returns.mean(), returns.std()))
    pickle.dump({'Seed': args.seed, 'Returns': returns}, fp)
    fp.close()


if __name__ == '__main__':
    main()
