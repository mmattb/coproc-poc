import enum

import torch

from .utils import gaussian_array_weights


class Observer(object):
    def __init__(self):
        self._out_dim = None
        self._in_dim = None

    def reduce(self, x):
        raise NotImplementedError()

    def observe(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
        return self.reduce(x)

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def in_dim(self):
        return self._in_dim

    def __call__(self, x):
        return self.observe(x)

    def __str__(self):
        raise NotImplementedError()


class ObserverGaussian1d(Observer):
    def __init__(self, in_dim, out_dim=20, sigma=1.75, cuda=None):
        super(ObserverGaussian1d, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._cuda = cuda
        self.sigma = sigma

        self.norm, self.weights = gaussian_array_weights(
            in_dim, out_dim, sigma, normalize=True
        )
        self.weights = torch.tensor(
            self.weights.reshape((1,) + self.weights.shape)
        ).float()

        if cuda is not None:
            self.weights = self.weights.cuda(cuda)

    def reduce(self, x):
        # Note: weights may broadcast up if we have batches
        reduced = self.weights @ x.reshape(x.shape + (1,))
        return reduced.squeeze(axis=-1).detach()

    def __str__(self):
        return f"gaussian{self.out_dim}.{self.sigma}"


class ObserverPassthrough(Observer):
    def __init__(self, in_dim):
        super(ObserverPassthrough, self).__init__()
        self._in_dim = in_dim
        self._out_dim = in_dim

    def reduce(self, x):
        return x

    def __str__(self):
        return "passthrough"

class ObserverType(enum.Enum):
    gaussian = ObserverGaussian1d
    passthrough = ObserverPassthrough

