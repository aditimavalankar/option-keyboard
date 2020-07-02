import torch
import numpy as np
from ok import option_keyboard
from core.networks import MlpDiscrete
from torch.optim import Adam
from tensorboardX import SummaryWriter
from test import test_agent
import os


def keyboard_player(env, W, Q, alpha, eps, gamma, training_steps, batch_size,
                    pretrained_agent, max_ep_steps, device, test_interval,
                    n_test_runs, log_file, log_dir):
    """
        env: Environment
        W: Weight vector to be learnt over cumulants
        Q: Q-functions over all cumulants and options
        alpha: learning rate
        eps: Exploration parameter over weight vector w
        gamma: Discount factor
        training_steps: Number of steps for which agent is to be trained
        batch_size: Batch size for updating Q-values
        pretrained_agent: Path to pretrained agent model
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        test_interval: Number of steps after which agent is tested
        n_test_runs: Number of episodes for which performance is tested
        log_file: File to store episode return logs
        log_dir: Directory where logs and intermediate models are saved
    """

    n = W.shape[0]
    Q_w = MlpDiscrete(input_dim=env.observation_space.shape[0],
                      output_dim=n,
                      hidden=[64, 128])
    Q_w.to(device)
    optimizer = Adam(Q_w.parameters(), lr=alpha)

    s = env.reset()
    s = torch.from_numpy(s).float().to(device)
    n_steps = 0
    best_avg_return = -100  # random low value, needs to be changed!
    q_loss = 0
    n_items_batch = 0
    done = False

    writer = {}
    writer['writer'] = SummaryWriter(os.path.join(log_dir, 'runs'))

    # Load pretrained agent, if available
    if pretrained_agent:
        checkpoint = torch.load(os.path.join(pretrained_agent, 'agent.pt'))
        n_steps = checkpoint['steps']
        Q_w.load_state_dict(checkpoint['Q'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_avg_return = checkpoint['best_avg_return']

    # Start learning
    while n_steps < training_steps:
        if done:
            s = torch.from_numpy(env.reset()).float().to(device)
            done = False

        q = Q_w(s)

        # Epsilon-greedy exploration
        if np.random.binomial(1, eps):
            w_index = torch.tensor(np.random.randint(n)).to(device)
        else:
            w_index = torch.argmax(q)

        w = W[w_index]

        (s_next, done, r_prime,
         gamma_prime, n_steps, _) = option_keyboard(env, s, w, Q, gamma,
                                                    n_steps, max_ep_steps,
                                                    device)

        s_next = torch.from_numpy(s_next).float().to(device)

        q_next = Q_w(s_next).detach()

        td_error = r_prime + gamma_prime * q_next.max() - q[w_index]
        q_loss += 0.5 * (td_error ** 2)
        n_items_batch += 1

        # Update the networks
        if n_items_batch == batch_size:
            optimizer.zero_grad()
            writer['writer'].add_scalar('agent/Q',
                                        q_loss.item(),
                                        n_steps + 1)
            q_loss.backward()
            optimizer.step()
            q_loss = 0
            n_items_batch = 0

        s = s_next

        # Test the agent at intermediate time steps and save current and best
        # models
        if n_steps % test_interval == 0:
            ep_returns = test_agent(env, W, Q_w, Q, gamma, n_steps,
                                    max_ep_steps, device, n_test_runs,
                                    log_file)
            writer['writer'].add_scalar('episode_returns/Agent',
                                        sum(ep_returns) / n_test_runs,
                                        n_steps)

            torch.save({'steps': n_steps,
                        'Q': Q_w.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_avg_return': best_avg_return
                        },
                       os.path.join(log_dir, 'saved_models', 'agent.pt'))

            if sum(ep_returns) / n_test_runs > best_avg_return:
                best_avg_return = sum(ep_returns) / n_test_runs
                torch.save({'steps': n_steps,
                            'Q': Q_w.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_avg_return': best_avg_return
                            },
                           os.path.join(log_dir, 'saved_models', 'best',
                                        'agent.pt'))

    return Q_w
