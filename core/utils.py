import numpy as np
import torch
import random
import pickle
import os


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def update(h, s_next, food_type, device):
    """
        h: history
        s_next: next state
        food_type: type of food item from {(0, 1), (1, 0), (1, 1)}
        device: cpu or gpu
    """
    m = len(food_type)
    previous_res = h[:m]
    new_res = torch.stack([torch.tensor(food_type[i] * 1.0).to(device)
                           if food_type[i] else previous_res[i]
                           for i in range(m)])
    h_next = torch.cat((new_res, torch.from_numpy(s_next).float().to(device)))
    return h_next


def get_cumulant(h, a, terminal_a, food_type, index):
    """
        h: history
        a: action
        terminal_a: index of the terminal action = |A| + 1
        food_type: type of food item from {(0, 1), (1, 0), (1, 1)}
        index: index of the resource for which the cumulant value is required
    """
    if a == terminal_a:
        return torch.tensor(0.)
    if not h[index] and food_type[index]:
        return torch.tensor(food_type[index] * 1.)
    elif h[index]:
        return torch.tensor(-1.)
    return torch.tensor(0.)


def create_log_files(args, n_cumulants):
    """
        args: all command-line arguments
        n_cumulants: number of cumulants
    """
    # Create base directory to store all experiment results
    try:
        os.mkdir(args.log_dir)
    except FileExistsError:
        pass
    try:
        base_dir = os.path.join(args.log_dir, args.exp_name)
        os.mkdir(base_dir)
    except FileExistsError:
        pass

    # Store hyperparameters
    hyperparam_file = open(os.path.join(base_dir, 'hyperparams'), 'a+b')
    pickle.dump(args, hyperparam_file)
    hyperparam_file.close()

    # Create empty directory for saved models
    try:
        os.mkdir(os.path.join(base_dir, 'saved_models'))
    except FileExistsError:
        pass

    # Create empty directory within saved models for best models
    try:
        os.mkdir(os.path.join(base_dir, 'saved_models', 'best'))
    except FileExistsError:
        pass

    # Create folder to store Tensorboard log
    try:
        os.mkdir(os.path.join(base_dir, 'runs'))
    except FileExistsError:
        pass

    # Create log files for options and agent to store learning curves
    log_files = {'cumulants': []}

    # Agent log file
    log_file = os.path.join(base_dir, 'agent_log_file')
    l_file = open(log_file, 'a+b')
    l_file.close()
    log_files['agent'] = log_file

    # Cumulant log files
    for i in range(n_cumulants):
        log_file = os.path.join(base_dir, 'cumulant_%d_log_file' % (i + 1))
        l_file = open(log_file, 'a+b')
        l_file.close()
        log_files['cumulants'].append(log_file)

    # Combined cumulants log files
    log_file = os.path.join(base_dir, '1,1')
    l_file = open(log_file, 'a+b')
    l_file.close()
    log_files['1,1'] = log_file
    log_file = os.path.join(base_dir, '1,-1')
    l_file = open(log_file, 'a+b')
    l_file.close()
    log_files['1,-1'] = log_file
    log_file = os.path.join(base_dir, '-1,1')
    l_file = open(log_file, 'a+b')
    l_file.close()
    log_files['-1,1'] = log_file

    return base_dir, log_files
