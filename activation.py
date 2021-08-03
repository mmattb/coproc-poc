import enum

import torch

import utils


class ActivationType(enum.Enum):
    ReLU = torch.nn.ReLU
    ReTanh = utils.ReTanh
    Tanh = torch.nn.Tanh
