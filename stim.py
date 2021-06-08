import copy

import numpy as np
from scipy.stats import norm
import torch

import utils


class Stimulus(object):
    def __init__(self, num_stim_channels, num_neurons, pad_right_neurons=200,
                 batch_size=1):
        self._num_stim_channels = num_stim_channels
        self._num_neurons = num_neurons
        self._pad_right_neurons = pad_right_neurons
        self._batch_size = batch_size

    @property
    def num_stim_channels(self):
        return self._num_stim_channels

    @property
    def batch_size(self):
        return self._batch_size

    def add(self, params):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()

    def reset(self, batch_size=None):
        self._batch_size = batch_size

    def __str__(self):
        raise NotImplementedError()


class Stimulus1to1(Stimulus):
    def __init__(
        self, num_stim_channels, num_neurons, pad_right_neurons=200, kill_thresh=1e-2,
        batch_size=1
    ):
        super(Stimulus1to1, self).__init__(
            num_stim_channels, num_neurons, pad_right_neurons,
            batch_size=batch_size,
        )

        self._buffer = []
        self._buf_offsets = []
        self._kill_thresh = kill_thresh

    def add(self, params):
        # (batch_size, num_neurons, num_neurons)
        W = self.get_neuron_weights()

        if isinstance(params, torch.Tensor):
            P = params.clone().detach()
        else:
            P = torch.tensor(params)

        self._buffer.append((W, P))
        self._buf_offsets.append(0)

    def reset(self, batch_size=None):
        super(Stimulus1to1, self).reset(batch_size=batch_size)
        self._buffer = []
        self._buf_offsets = []

    def get_neuron_weights(self):
        if self._num_stim_channels != self._num_neurons:
            raise ValueError(
                "This Stimulus provides 1 stimulus channel per "
                "neuron; num_stim_channels must equal num_neurons"
            )

        # out: (batch_size, num_neurons, num_neurons)
        return torch.tensor(np.repeat(
            np.identity(self._num_neurons).reshape(
                1, self._num_neurons, self._num_neurons
            ),
            self.batch_size,
            axis=0,
        ))

    def get_next(self):
        stim_out = torch.zeros(
            (self._batch_size, self._num_neurons + self._pad_right_neurons)
        )

        # shortcut
        if not self._buffer:
            return stim_out

        new_buf = []
        new_offsets = []
        for bidx, offset in enumerate(self._buf_offsets):
            W, P = self._buffer[bidx]
            A = utils.alpha(P, offset)
            cur_stim = torch.unsqueeze(A, 2) * W

            if torch.sum(cur_stim > self._kill_thresh) > 0.0 or offset == 0:
                # W: (batch_size, num_stim_channels, num_neurons)
                # A: (batch_size, num_stim_channels)
                cur_stim = torch.unsqueeze(A, 2) * W
                stim_out[:, : self._num_neurons] += torch.sum(cur_stim, axis=1)

                new_buf.append((W, P))
                new_offsets.append(offset + 1)

        self._buffer = new_buf
        self._buf_offsets = new_offsets

        # (batch_size, num_neurons)
        return stim_out

    def __str__(self):
        return "1to1"


class StimulusGaussian(Stimulus1to1):
    def __init__(
        self,
        num_stim_channels,
        num_neurons,
        pad_right_neurons=200,
        sigma=2.5,
        batch_size=1,
    ):
        super(StimulusGaussian, self).__init__(
            num_stim_channels,
            num_neurons,
            pad_right_neurons,
            batch_size=batch_size,
        )

        self._sigma = sigma
        self._norm = norm(0, self._sigma)

    def get_neuron_weights(self):
        win = utils.array_weights(
            self._num_neurons, self._num_stim_channels, distance_func=self._norm.pdf
        )
        wout = np.repeat(
            win.reshape(1, self._num_stim_channels, self._num_neurons),
            self.batch_size,
            axis=0,
        )
        return torch.tensor(wout)

    def __str__(self):
        return f"gaussian{self.num_stim_channels}.{self._sigma}"


class StimulusGaussianExp(Stimulus):
    def __init__(self, num_stim_channels, num_neurons, pad_right_neurons=200,
            sigma=2.5, decay=0.3, batch_size=1):
        super(StimulusGaussianExp, self).__init__(num_stim_channels,
                num_neurons, pad_right_neurons, batch_size=batch_size)

        self._sigma = sigma
        self._vals = torch.zeros((batch_size, num_neurons))
        self._decay = decay
        self._norm = norm(0, self._sigma)

    def get_neuron_weights(self):
        win = utils.array_weights(
            self._num_neurons, self._num_stim_channels, distance_func=self._norm.pdf
        ).T
        wout = np.repeat(
            win.reshape(1, self._num_neurons, self._num_stim_channels),
            self.batch_size,
            axis=0,
        )
        return torch.tensor(wout).float()

    def add(self, params):
        """
        params: (batch_size, num_stim_channels)
        """
        #P = params.clone().detach()
        P = params

        # (batch_size, num_neurons, num_stim_channels)
        W = self.get_neuron_weights()

        # (batch_size, num_neurons)
        new_stim = W @ P.reshape(self.batch_size, self._num_stim_channels, 1)

        self._vals += new_stim[:, :, 0]

    def get_next(self):
        stim_out = torch.zeros((self.batch_size, self._num_neurons + self._pad_right_neurons))
        stim_out[:, :self._num_neurons] = self._vals

        # Update vals according to an exponential decay
        self._vals -= self._vals * self._decay

        return stim_out

    def __str__(self):
        return f"gaussianExp{self.num_stim_channels}.{self._sigma}"
