
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import utils


class CPNModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation_func=utils.ReTanh,
            num_neurons=200):
        super(StimModel, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func = activation_func()
        self.num_neurons = num_neurons

        with torch.no_grad():
            # Inter-neuron hidden recurrent weights
            # (num_neurons, num_neurons)
            self.W = nn.Parameter(torch.zeros((num_neurons, num_neurons)))
            nn.init.normal_(self.W[:,:], mean=0.0, std=(1.0 / np.sqrt(num_neurons)))

            # Neuron response to input
            # (num_neurons, in_dim)
            self.I = nn.Parameter(torch.zeros((num_neurons, in_dim)))
            nn.init.normal_(self.I[:,:], mean=0.0, std=(1.0 / np.sqrt(in_dim)))

            # Neuron biases
            self.b = nn.Parameter(torch.zeros((num_neurons,)))
            nn.init.uniform_(self.b, -1.0, 1.0)

            self.fc = nn.Linear(num_neurons, out_dim)


        # (batch, num_neurons)
        self.x = None
        # (batch, num_neurons)
        self.prev_output = None
        self.x0 = None

    def reset(self):
        self.x = None
        self.prev_output = None

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = torch.zeros((batch_size, self.num_neurons))

        x = self.W.reshape((1,) + self.W.shape) @ \
                self.prev_output.reshape(self.prev_output.shape + (1,))
        assert x.shape == (batch_size, self.num_neurons, 1)

        x += self.I.reshape((1,) + self.I.shape) @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        readout = self.fc(rnn_output)

        self.prev_output = rnn_output
        return readout


