import torch
import numpy as np
from core.value_function import ValueFunction
from core.utils import update, get_cumulant
from test import test_learning_options
from tensorboardX import SummaryWriter
import os


def reset(s, d, device):
    h = torch.from_numpy(np.concatenate((np.zeros(d), s))).float().to(device)
    k = np.random.randint(d)
    return h, k


def learn_options(env, d, eps1, eps2, alpha, gamma, max_ep_steps, device,
                  training_steps, batch_size, pretrained_options,
                  test_interval, n_test_runs, log_files, log_dir):
    """
    Parameters:
        env: Environment
        d : Number of cumulants/resources
        eps1 : Probability of changing cumulant
        eps2 : Exploration parameters
        alpha : Learning rate
        gamma : Discount factor
        max_ep_steps: Maximum steps per episode
        device: cpu or gpu
        training_steps: Total number of steps for which OK is to be trained
        batch size: Batch size for updating the Q-values
        pretrained_options: Path to pretrained option models, if available
        test_interval: Number of steps after which agent is tested
        n_test_runs: Number of episodes to run at test time
        log_files: Files to store episode return logs for each option
        log_dir: Directory where intermediate models are saved
    """

    state_dim = env.observation_space.shape[0]
    if type(env.action_space).__name__ == 'Discrete':
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape

    # One Q-network for each cumulant, where each network gives the Q-values
    # for all actions in a state for each option. So, the output dimension of
    # each Q-network is (|A| + 1) x n_options
    value_fns = [ValueFunction(input_dim=state_dim + d,
                               action_dim=(action_dim + 1),
                               n_options=d,
                               hidden=[64, 128],
                               batch_size=batch_size,
                               gamma=gamma,
                               alpha=alpha)
                 for _ in range(d)]

    for i in range(d):
        value_fns[i].q_net.to(device)

    writer = {}
    writer['writer'] = SummaryWriter(os.path.join(log_dir, 'runs'))

    s = env.reset()
    h, k = reset(s, d, device)
    n_steps = 0
    done = False
    best_avg_return = 0

    # Load pretrained models if they exist
    if pretrained_options:
        for i in range(d):
            checkpoint = torch.load(os.path.join(pretrained_options,
                                                 'value_fn_%d.pt' %
                                                 (i + 1)))
            n_steps = checkpoint['steps']
            value_fns[i].q_net.load_state_dict(checkpoint['Q'])
            value_fns[i].optimizer.load_state_dict(checkpoint['optimizer'])
            best_avg_return = checkpoint['best_avg_return']

    # Start learning
    while n_steps < training_steps:

        if done or n_steps % max_ep_steps == 0:
            s = env.reset()
            h, k = reset(s, d, device)
            done = False

        if np.random.binomial(1, eps1):
            h, k = reset(s, d, device)
            for i in range(d):
                value_fns[i].terminate = True
                loss = value_fns[i].update_batch([], device)
                if loss is not None:
                    writer['writer'].add_scalar('value_fn/Q_%d' % (i + 1),
                                                loss,
                                                n_steps)

        q = value_fns[k].q_net(h)[k * (action_dim + 1):
                                  (k + 1) * (action_dim + 1)]

        # Special case - generate random action under current option if all
        # options have the termination condition as the optimal action, in
        # order to prevent the algorithm from going in loops :)
        all_terminate = True
        for i in range(d):
            q_i = value_fns[i].q_net(h)[i * (action_dim + 1):
                                        (i + 1) * (action_dim + 1)]
            if torch.argmax(q_i) != action_dim:
                all_terminate = False
                break

        if np.random.binomial(1, eps2) or all_terminate:
            a = torch.tensor(np.random.randint(action_dim)).to(device)
        else:
            with torch.no_grad():
                a = torch.argmax(q)

        if a != action_dim:  # a == action_dim => terminal action is chosen.
            s_next, _, done, info = env.step(a)
            n_steps += 1
            food_type = info['food type']
            h_next = update(h, s_next, food_type, device)

            cumulants = torch.tensor([get_cumulant(h, a, action_dim,
                                                   food_type, j)
                                      for j in range(d)]).to(device)

            for i in range(d):
                with torch.no_grad():
                    q_next = value_fns[i].q_net(h_next)
                    q_next = q_next[i * (action_dim + 1):
                                    (i + 1) * (action_dim + 1)]
                with torch.no_grad():
                    a_next = torch.argmax(q_next)
                transition = [h, a, h_next, a_next, cumulants[i]]
                loss = value_fns[i].update_batch(transition, device)
                if loss is not None:
                    writer['writer'].add_scalar('value_fn/Q_%d' % (i + 1),
                                                loss,
                                                n_steps)

            s = s_next
            h = h_next

        else:
            cumulants = torch.tensor([get_cumulant(h, a, action_dim, -1, j)
                                      for j in range(d)]).to(device)
            for i in range(d):
                transition = [h, a, -1, -1, cumulants[i]]
                value_fns[i].terminate = True
                loss = value_fns[i].update_batch(transition, device)
                if loss is not None:
                    writer['writer'].add_scalar('value_fn/Q_%d' % (i + 1),
                                                loss,
                                                n_steps)

            h, k = reset(s, d, device)

        # Save models
        if n_steps % test_interval == 0:

            # Check performance over different configurations of w
            w = np.array([1, 1])
            ep_returns1, _ = test_learning_options(env, value_fns, 0, w, gamma,
                                                   n_steps, max_ep_steps,
                                                   device, n_test_runs,
                                                   log_files['1,1'])
            writer['writer'].add_scalar('returns/w=1,1',
                                        sum(ep_returns1) /
                                        n_test_runs, n_steps)

            w = np.array([1, -1])
            ep_returns2, _ = test_learning_options(env, value_fns, 0, w, gamma,
                                                   n_steps, max_ep_steps,
                                                   device, n_test_runs,
                                                   log_files['1,-1'])
            writer['writer'].add_scalar('returns/w=1,-1',
                                        sum(ep_returns2) /
                                        n_test_runs, n_steps)

            w = np.array([-1, 1])
            ep_returns3, _ = test_learning_options(env, value_fns, 0, w, gamma,
                                                   n_steps, max_ep_steps,
                                                   device, n_test_runs,
                                                   log_files['-1,1'])
            writer['writer'].add_scalar('returns/w=-1,1',
                                        sum(ep_returns3) /
                                        n_test_runs, n_steps)

            # Save the current model
            for i in range(d):
                torch.save({'steps': n_steps,
                            'Q': value_fns[i].q_net.state_dict(),
                            'optimizer': value_fns[i].optimizer.state_dict(),
                            'best_avg_return': best_avg_return
                            },
                           os.path.join(log_dir, 'saved_models',
                                        'value_fn_%d.pt'
                                        % (i + 1)))

            # Save this as the best model if it performs better than
            # previous models
            if sum(ep_returns1) / n_test_runs >= best_avg_return:
                best_avg_return = sum(ep_returns1) / n_test_runs
                print('Changing best avg ret to', best_avg_return)
                for i in range(d):
                    torch.save({'steps': n_steps,
                                'Q': value_fns[i].q_net.state_dict(),
                                'optimizer': value_fns[i].optimizer.state_dict(),
                                'best_avg_return': best_avg_return
                                },
                               os.path.join(log_dir, 'saved_models',
                                            'best', 'value_fn_%d.pt'
                                            % (i + 1)))

    return value_fns
