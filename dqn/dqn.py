import torch
import numpy as np
from core.networks import MlpDiscrete
from torch.optim import Adam
from tensorboardX import SummaryWriter
import os
import pickle


def test_agent(env, Q, device, n_test_runs, test_log_file, n_steps,
               visualize=False):
    """
        env: Environment
        Q: Q-network
        device: cpu or gpu
        n_test_runs: Number of episodes for which performance is tested
        test_log_file: Path to log file for test results
        n_steps: Number of steps for which the DQN has currently been trained
        visualize: Flag for rendering environment
    """
    ep_returns = []
    for _ in range(n_test_runs):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        done = False
        ep_return = 0

        while not done:
            q = Q(s)
            a = torch.argmax(q)
            s, r, done, _ = env.step(a)
            ep_return += r
            s = torch.from_numpy(s).float().to(device)

        ep_returns.append(ep_return)

    logfile = open(test_log_file, 'a+b')
    pickle.dump({'steps': n_steps, 'returns': ep_returns}, logfile)
    logfile.close()

    print('Steps:', n_steps,
          'Avg. return:', sum(ep_returns) / n_test_runs,
          'Episodic return:', ep_returns)

    return ep_returns


def dqn(env, eps, gamma, alpha, device, training_steps, batch_size,
        pretrained_agent, test_interval, n_test_runs, log_file, log_dir,
        visualize=False):
    """
        env: Environment
        eps: Exploration parameter over actions
        gamma: Discount factor
        alpha: Learning rate
        device: cpu or gpu
        training_steps: Number of steps for which agent is to be trained
        batch_size: Batch size for updating Q-values
        pretrained_agent: Path to pretrained agent model
        test_interval: Number of steps after which agent is tested
        n_test_runs: Number of episodes for which performance is tested
        log_file: File to store episode return logs
        log_dir: Directory where logs and intermediate models are saved
        visualize: Flag for rendering environment
    """

    n = env.action_space.n

    Q = MlpDiscrete(input_dim=env.observation_space.shape[0],
                    output_dim=n,
                    hidden=[64, 128])
    Q.to(device)

    optimizer = Adam(Q.parameters(), lr=alpha)

    s = env.reset()
    s = torch.from_numpy(s).float().to(device)
    n_steps = 0
    best_avg_return = -100
    q_loss = 0
    n_items_batch = 0

    writer = {}
    writer['writer'] = SummaryWriter(os.path.join(log_dir, 'runs'))

    # Load pretrained agent, if available
    if pretrained_agent:
        checkpoint = torch.load(os.path.join(pretrained_agent, 'agent.pt'))
        n_steps = checkpoint['steps']
        Q.load_state_dict(checkpoint['Q'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_avg_return = checkpoint['best_avg_return']

    # Start learning
    while n_steps < training_steps:
        q = Q(s)

        # Epsilon-greedy exploration
        if np.random.binomial(1, eps):
            a = torch.tensor(np.random.randint(n)).to(device)
        else:
            a = torch.argmax(q)

        s_next, r, done, _ = env.step(a)
        n_steps += 1

        s_next = torch.from_numpy(s_next).float().to(device)

        with torch.no_grad():
            q_next = Q(s_next)

        td_error = r + gamma * q_next.max() - q[a]
        q_loss += 0.5 * (td_error ** 2)
        n_items_batch += 1

        # Update the network
        if n_items_batch == batch_size:
            optimizer.zero_grad()
            writer['writer'].add_scalar('agent/Q',
                                        q_loss.item(),
                                        n_steps + 1)
            q_loss.backward()
            optimizer.step()
            q_loss = 0
            n_items_batch = 0

        s = (s_next if not done
             else torch.from_numpy(env.reset()).float().to(device))

        # Test the agent at intermediate time steps and save current and best
        # models
        if n_steps % test_interval == 0:
            ep_returns = test_agent(env, Q, device, n_test_runs, log_file,
                                    n_steps)

            writer['writer'].add_scalar('episode_returns/Agent',
                                        sum(ep_returns) / n_test_runs,
                                        n_steps)

            torch.save({'steps': n_steps,
                        'Q': Q.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_avg_return': best_avg_return
                        },
                       os.path.join(log_dir, 'saved_models', 'agent.pt'))

            if sum(ep_returns) / n_test_runs > best_avg_return:
                best_avg_return = sum(ep_returns) / n_test_runs
                torch.save({'steps': n_steps,
                            'Q': Q.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_avg_return': best_avg_return
                            },
                           os.path.join(log_dir, 'saved_models', 'best',
                                        'agent.pt'))
