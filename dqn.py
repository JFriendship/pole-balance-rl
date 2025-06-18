import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, x):
        return self.output_layer(self.act1(self.input_layer(x)))