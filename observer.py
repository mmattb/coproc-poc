import torch

from utils import gaussian_array_weights


class Observer(object):
    def __init__(self):
        self._out_dim = None
        self._in_dim = None

    def reduce(self, x):
        raise NotImplementedError()

    def observe(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().numpy()
        return self.reduce(x)

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def in_dim(self):
        return self._in_dim

    def __call__(self, x):
        return self.observe(x)


class ObserverGaussian1d(Observer):
    def __init__(self, in_dim, out_dim=30, sigma=1):
        super(ObserverGaussian1d, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self.sigma = sigma

        self.norm, self.weights = gaussian_array_weights(in_dim, out_dim, sigma)

    def reduce(self, x):
        # Note: weights may broadcast up if we have batches
        reduced = self.weights.reshape((1,) + self.weights.shape) @ \
                x.reshape(x.shape + (1,))
        return reduced.squeeze(axis=-1)

class ObserverPassthrough(Observer):
    def __init__(self, in_dim):
        self._in_dim = in_dim
        self._out_dim = in_dim

    def reduce(self, x):
        return x
