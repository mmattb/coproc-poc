import math

import torch
import torch.nn as nn
from scipy.io import loadmat
import scipy.stats
import numpy as np


def dfunc_dsquared(dists, decay=1):
    out = [1 / ((d * decay) ** 2) for d in dists]
    return out


def array_weights(
    in_dim, out_dim, distance_func=dfunc_dsquared, normalize=False, **dfunckwargs
):
    if out_dim >= in_dim:
        raise ValueError(
            "Input dimension (%d) must exceed output dimension "
            "(%d)" % (in_dim, out_dim)
        )

    interval = in_dim / (out_dim + 1)
    centers = np.array([interval * i for i in range(1, out_dim + 1)])

    weights = np.zeros((out_dim, in_dim))
    for c_idx, m in enumerate(centers):
        dists = [i - m for i in range(in_dim)]
        cur_weights = distance_func(dists, **dfunckwargs)
        weights[c_idx, :] = cur_weights

    if normalize:
        # Re-normalize, so each channels's weights sum to 1
        weights = weights / weights.sum(axis=1, keepdims=True)

    return weights


def gaussian_array_weights(in_dim, out_dim, sigma, normalize=False):
    norm = scipy.stats.norm(0, sigma)
    return norm, array_weights(
        in_dim, out_dim, normalize=normalize, distance_func=norm.pdf
    )


def alpha(gmax, tau=5, onset=0, thresh=1e-6, time_scalar=1):
    """
    Inspired by: https://www.sas.upenn.edu/LabManuals/BBB251/NIA/NEUROLAB/HELP/alphasyn.htm
    """

    v_out = []
    i = 0

    v = (
        gmax
        * time_scalar
        * (i - onset)
        / tau
        * math.exp(-1 * time_scalar * (i - onset - tau) / tau)
    )
    v_out.append(v)

    while abs(v) > thresh or i < (onset + tau):
        i += 1
        v = (
            gmax
            * time_scalar
            * (i - onset)
            / tau
            * math.exp(-1 * time_scalar * (i - onset - tau) / tau)
        )
        v_out.append(v)

    return np.array(v_out)


def hash_model_data_name(
    activation, input_name, fr_weight, io_weight, sparsity, repetition
):
    return "-".join(
        [activation, input_name, fr_weight, io_weight, sparsity, repetition]
    )


def fill_jagged_array(x):
    # Pad arrays with NaNs s.t. all arrays in the list are equal length
    feat_dim = x[0].shape[0]
    max_len = max([a.shape[-1] for a in x])
    make_nan_pad = lambda l: np.full((feat_dim, max_len - l), np.nan)
    x = [np.concatenate([a, make_nan_pad(a.shape[-1])], -1) for a in x]
    x = np.stack(x).transpose(0, 2, 1)
    return torch.tensor(x, dtype=torch.float32, requires_grad=False)


def init_from_michaels_model(
    init_data_path, num_input_features, num_neurons_per_module, output_dim
):
    data = loadmat(init_data_path)
    data = {k: v for k, v in data.items() if "__" not in k}

    npm = num_neurons_per_module

    J = torch.tensor(data["J"].copy()).float()

    assert data["B"].shape[1] - 1 == num_input_features
    I, S = np.split(data["B"], [num_input_features], axis=1)

    I = torch.tensor(I.copy()).float()
    S = torch.tensor(S.copy()).float()
    num_neurons = I.shape[0]

    B = data["bx"]
    B = torch.tensor(B.copy()).float()

    x0 = data["x0"]
    x0 = torch.tensor(x0.copy()).float().squeeze()

    fc = nn.Linear(npm, output_dim)
    fc.load_state_dict(
        {
            "weight": torch.tensor(data["W"], dtype=torch.float32)[:, :npm],
            "bias": torch.tensor(data["bz"].squeeze(), dtype=torch.float32),
        }
    )

    return J, I, S, B, fc, x0


class ReTanh(nn.Module):
    """
    ReTanh activation function
    """

    def forward(self, x):
        return torch.tanh(torch.clamp(x, min=0))


class NonAct(nn.Module):
    """
    ReTanh activation function
    """

    def forward(self, x):
        return x
