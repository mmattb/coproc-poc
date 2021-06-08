import torch
import torch.nn as nn


class R(nn.Module):
    def __init__(self, num_neurons=4):
        super(R, self).__init__()

        self.h = None

        self.I = nn.Parameter(torch.zeros((1, num_neurons, 1)))
        self.W = nn.Parameter(torch.zeros((1, num_neurons, num_neurons)))
        self.b = nn.Parameter(torch.zeros((1, num_neurons)))
        self.num_neurons = num_neurons
        self.act = nn.Tanh()

        self.fc = nn.Linear(num_neurons, 1)

    def forward(self, batch):
        batch_size = batch.shape[0]
        if self.h is None:
            self.h = torch.zeros((batch_size, self.num_neurons))

        x = self.W @ self.h.reshape(batch_size, self.num_neurons, 1)
        x = x + self.I @ batch.reshape(batch.shape + (1,))
        x = x.squeeze()
        x = x + self.b
        new_h = self.act(x)
        self.h = new_h


        readout = self.fc(self.h.squeeze())
        return readout

    def reset(self):
        self.h = None
