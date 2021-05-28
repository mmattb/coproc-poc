import numpy as np
from scipy.stats import norm
import torch

import utils


class Stimulus(object):
    def __init__(self, num_stim_channels, num_neurons, pad_right_neurons=200):
        self._num_stim_channels = num_stim_channels
        self._num_neurons = num_neurons
        self._pad_right_neurons = pad_right_neurons

    @property
    def num_stim_channels(self):
        return self._num_stim_channels

    def add(self, params):
        raise NotImplementedError()

    def get_next(self):
        raise NotImplementedError()

    def reset(self):
        pass


class Stimulus1to1(Stimulus):
    def __init__(self, num_stim_channels, num_neurons, pad_right_neurons=200):
        super(Stimulus1to1, self).__init__(
            num_stim_channels, num_neurons, pad_right_neurons
        )

        self._buffer = []
        self._buf_offsets = []

    def add(self, params):
        W = self.get_neuron_weights()
        A = [utils.alpha(p, thresh=1e-1) for p in params]

        self._buffer.append((W, A))
        self._buf_offsets.append(0)

    def reset(self):
        self._buffer = []
        self._buf_offsets = []

    def get_neuron_weights(self):
        if self._num_stim_channels != self._num_neurons:
            raise ValueError(
                "This Stimulus provides 1 stimulus channel per "
                "neuron; num_stim_channels must equal num_neurons"
            )

        return np.identity(self._num_neurons)

    def get_next(self):
        stim_out = torch.zeros((1, self._num_neurons + self._pad_right_neurons))

        new_buf = []
        new_offsets = []
        for bidx, offset in enumerate(self._buf_offsets):
            W, A = self._buffer[bidx]

            As = np.zeros((self._num_stim_channels,))
            used_one = False
            for cidx, a in enumerate(A):
                if offset < len(a):
                    used_one = True
                    As[cidx] = a[offset]

            cur_stim = As @ W
            stim_out[0, : self._num_neurons] += cur_stim

            if used_one:
                new_buf.append((W, A))
                new_offsets.append(offset + 1)

        self._buffer = new_buf
        self._buf_offsets = new_offsets

        # (1, num_neurons)
        return stim_out


class StimulusGaussian(Stimulus1to1):
    def __init__(
        self, num_stim_channels, num_neurons, pad_right_neurons=200, sigma=2.5
    ):
        super(StimulusGaussian, self).__init__(
            num_stim_channels, num_neurons, pad_right_neurons
        )

        self._sigma = sigma
        self._norm = norm(0, self._sigma)

    def get_neuron_weights(self):
        W = utils.array_weights(
            self._num_neurons, self._num_stim_channels, distance_func=self._norm.pdf
        )
        return W
