import torch.nn as nn


class MlpDiscrete(nn.Module):

    def __init__(self, input_dim, output_dim, hidden=[64, 128]):
        super(MlpDiscrete, self).__init__()

        self.hidden1 = nn.Linear(input_dim, hidden[0])
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(hidden[0], hidden[1])
        self.output = nn.Linear(hidden[1], output_dim)

    def forward(self, x):
        self.h1 = self.hidden1(x)
        self.r1 = self.relu(self.h1)
        self.h2 = self.hidden2(self.r1)
        self.r2 = self.relu(self.h2)
        self.o = self.output(self.r2)
        return self.o

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)

    def copy_weights(self, net):
        self.load_state_dict(net.state_dict())

    def soft_update(self, net, tau):
        for old_p, new_p in zip(self.parameters(), net.parameters()):
            old_p.data.copy_(tau * new_p + (1 - tau) * old_p)
