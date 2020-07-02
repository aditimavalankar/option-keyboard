from core.networks import MlpDiscrete
import torch
from torch.optim import Adam


class ValueFunction:
    def __init__(self, input_dim, action_dim, n_options, hidden, batch_size,
                 gamma, alpha):
        """
            input_dim: dimension of input to the Q-network
            action_dim: dimension of the augmented action space = |A| + 1
            n_options: number of options to be learnt
            hidden: dimensions of hidden layers (list)
            batch_size: batch size for updating Q-values
            gamma: discount factor
            alpha: learning rate
        """
        self.q_net = MlpDiscrete(input_dim=input_dim,
                                 output_dim=action_dim * n_options,
                                 hidden=hidden)
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_options = n_options
        self.action_dim = action_dim
        self.terminate = False
        self.batch = {'h': [], 'h_next': [], 'a': [], 'a_next': [], 'c': []}
        self.n_items = 0
        self.optimizer = Adam(self.q_net.parameters(), lr=alpha)

    def update_batch(self, transition, device):
        """
            Updates the batch with the incoming transition, and updates the
            networks if the batch is full.
            transition: one-step transition consisting of history at time t,
                        action at time t, history at time (t + 1), action at
                        time (t + 1), and cumulant
            device: cpu or gpu
        """
        if transition:
            h, a, h_next, a_next, c = transition
            self.batch['h'].append(h)
            self.batch['a'].append(a)
            if not self.terminate:
                self.batch['h_next'].append(h_next)
                self.batch['a_next'].append(a_next)
            self.batch['c'].append(c)
            self.n_items += 1

        if not self.batch['h']:
            return None

        if self.terminate or self.n_items == self.batch_size:
            # terminate update
            self.batch['h'] = torch.stack(self.batch['h'])
            self.batch['a'] = torch.stack(self.batch['a']).unsqueeze(-1)
            try:
                self.batch['h_next'] = torch.stack(self.batch['h_next'])
                self.batch['a_next'] = torch.stack(self.batch['a_next']).unsqueeze(-1)
            except RuntimeError:
                pass
            self.batch['c'] = torch.stack(self.batch['c']).unsqueeze(1)

            indices = [[self.action_dim * j + self.batch['a'][i]
                        for j in range(self.n_options)]
                       for i in range(len(self.batch['a']))]
            indices = torch.tensor(indices).to(device)

            q = torch.gather(self.q_net(self.batch['h']), 1, indices)

            # compute Q-values at next time step for TD update
            with torch.no_grad():
                try:
                    indices = [[self.action_dim * j + self.batch['a_next'][i]
                                for j in range(self.n_options)]
                               for i in range(len(self.batch['a_next']))]
                    indices = torch.tensor(indices).to(device)
                    
                    q_next = torch.gather(self.q_net(self.batch['h_next']), 1,
                                          indices)
                    if (self.terminate and len(self.batch['h']) !=
                            len(self.batch['h_next'])):
                        q_next = torch.cat((q_next,
                                            torch.zeros(1,
                                                        self.n_options).to(device)),
                                           0)

                except AttributeError:
                    q_next = torch.zeros(1, self.n_options).to(device)

            td_error = self.batch['c'] + self.gamma * q_next - q
            loss = (0.5 * (td_error ** 2)).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.n_items = 0
            self.batch = {'h': [], 'h_next': [], 'a': [], 'a_next': [], 'c': []}
            torch.cuda.empty_cache()
            self.terminate = False

            return loss.item()

        else:
            return None
