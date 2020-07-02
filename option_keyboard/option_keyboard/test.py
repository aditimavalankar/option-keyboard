from option_keyboard.option_keyboard.ok import option_keyboard
import torch
import pickle
import numpy as np


def test_agent(env, W, Q_w, Q_E, gamma, training_steps, max_ep_steps, device,
               n_test_runs, test_log_file):

    """
        This function tests the overall performance of the agent, given that
        it has already learnt the keyboard.

        env: Environment
        W: Weight vector to be learnt over cumulants
        Q_w: Q-function over weight vectors
        Q_E: Q-functions over all cumulants and options
        gamma: Discount factor
        training_steps: Number of steps for which agent is to be trained
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        n_test_runs: Number of episodes for which performance is tested
        test_log_file: Path to log file for test results
    """
    ep_returns = []

    for _ in range(n_test_runs):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        n_steps = 0
        ep_return = 0
        done = False

        while n_steps < max_ep_steps and not done:
            with torch.no_grad():
                q = Q_w(s)
            w = W[torch.argmax(q)]
            (s_next, done, r_prime,
             gamma_prime, n_steps, info) = option_keyboard(env, s, w, Q_E,
                                                           gamma, n_steps,
                                                           max_ep_steps,
                                                           device)

            s_next = torch.from_numpy(s_next).float().to(device)
            s = (s_next if not done
                 else torch.from_numpy(env.reset()).float().to(device))

            ep_return += sum(info['rewards'])

        ep_returns.append(ep_return)

    print('Steps:', training_steps,
          'Avg. return:', sum(ep_returns) / n_test_runs,
          'Episodic return:', ep_returns)

    logfile = open(test_log_file, 'a+b')
    pickle.dump({'steps': training_steps, 'returns': ep_returns}, logfile)
    logfile.close()

    return ep_returns


def test_learning_options(env, Q_E, index, w, gamma, training_steps,
                          max_ep_steps, device, n_test_runs,
                          log_file):

    """
        This function tests the performance of the agent for different weight
        vectors w. This is typically used to see how the options being learnt
        perform for w = (1, 1), which would optimize for all types of food
        items. We also record performance for w = (1, -1) and w = (-1, 1) since
        the keyboard learnt should perform reasonably well for all
        configurations of w that we consider.

        env: Environment
        Q_E: Q-functions over all cumulants and options
        index: Index of cumulant for which performance is measured
        w: Weight vector (kept constant to measure keyboard performance)
        gamma: Discount factor
        training_steps: Number of steps for which agent is to be trained
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        n_test_runs: Number of episodes for which performance is tested
        log_file: Path to log file for test results
    """
    ep_returns = []
    cumulant_returns = []

    env.set_learning_options(w, True)

    for _ in range(n_test_runs):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        n_steps = 0
        ep_return = 0
        cumulant_return = 0
        done = False

        while n_steps < max_ep_steps and not done:
            (s_next, done, r_prime,
             gamma_prime, n_steps, info) = option_keyboard(env, s, w, Q_E,
                                                           gamma, n_steps,
                                                           max_ep_steps,
                                                           device)

            s_next = torch.from_numpy(s_next).float().to(device)
            s = (s_next if not done
                 else torch.from_numpy(env.reset()).float().to(device))

            ep_return += sum(info['rewards'])

            cumulant_return += sum([info['env_info'][i]['rewards'][index]
                                    for i in range(len(info['env_info']))])

        ep_returns.append(ep_return)
        cumulant_returns.append(cumulant_return)

    print('w:', w, 'Steps:', training_steps,
          'Avg. return:', sum(ep_returns) / n_test_runs,
          'Episodic return:', ep_returns,
          'Cumulant return:', cumulant_returns)

    logfile = open(log_file, 'a+b')
    pickle.dump({'steps': training_steps, 'returns': ep_returns,
                 'cumulant_returns': cumulant_returns}, logfile)
    logfile.close()

    env.set_learning_options(np.ones(len(w)), True)

    return ep_returns, cumulant_returns
