import torch
from option_keyboard.core.utils import update


def option_keyboard(env, s, w, Q, gamma, n_steps, max_ep_steps, device,
                    visualize=False):
    """
        env: Environment
        s: State provided by the environment
        w: Weight vector provided by the agent
        Q: Q-function over cumulants and options
        gamma: Discount factor
        n_steps: Number of steps currently taken
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        visualize: Flag for rendering environment
    """

    r_prime = 0
    gamma_prime = 1
    d = len(Q)
    h = torch.cat((torch.zeros(d).float().to(device), s))
    log_info = {'rewards': [], 'env_info': [], 'actions': []}
    w = torch.Tensor(w).float().to(device)
    action_dim = env.action_space.n
    s_next = s.to('cpu').numpy()
    done = False

    # Take at least one step in the environment.
    q_values = torch.stack([Q[i].q_net(h) for i in range(d)])
    q_gpe = (q_values.permute(1, 0) * w).permute(1, 0).sum(dim=0)
    indices = [i for i in range(len(q_gpe)) if (i + 1) % (action_dim + 1) != 0]
    indices = torch.tensor(indices).to(device)
    q_gpe = torch.index_select(q_gpe, 0, indices)
    a = torch.argmax(q_gpe) % action_dim

    s_next, r, done, info = env.step(a)
    if visualize:
        env.render()
    n_steps += 1
    food_type = info['food type']
    h_next = update(h, s_next, food_type, device)
    r_prime += gamma_prime * r
    gamma_prime = 0 if done else gamma_prime * gamma
    h = h_next
    log_info['rewards'].append(r)
    log_info['env_info'].append(info)
    log_info['actions'].append(a)

    while not done and n_steps < max_ep_steps:

        q_values = torch.stack([Q[i].q_net(h) for i in range(d)])
        q_gpe = (q_values.permute(1, 0) * w).permute(1, 0).sum(dim=0)
        a = torch.argmax(q_gpe) % (action_dim + 1)

        if a != action_dim:
            s_next, r, done, info = env.step(a)
            if visualize:
                env.render()
            n_steps += 1
            food_type = info['food type']
            h_next = update(h, s_next, food_type, device)
            r_prime += gamma_prime * r
            gamma_prime = 0 if done else gamma_prime * gamma
            h = h_next
            log_info['rewards'].append(r)
            log_info['env_info'].append(info)
            log_info['actions'].append(a)

        if a == action_dim:
            break

        if n_steps == max_ep_steps:
            done = True

    return s_next, done, r_prime, gamma_prime, n_steps, log_info
