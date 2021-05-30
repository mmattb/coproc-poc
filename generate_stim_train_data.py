import os
import random
import time

import h5py
import numpy as np
import torch
import scipy.io


import mRNN
import michaels_load
import observer
import stim
import utils


BASE_DIR = os.path.join("michaels", "Networks")
NUM_NEURONS_PER_MODULE = 100
NUM_EXAMPLES = 2000
# Applied only to M1
NUM_STIM_ELECTRODES = 35
# per module
NUM_OBS_ELECTRODES = 35
# Range of stimulus power is (-n, n), where n is:
STIM_MAX_POWER = 7.5


network_path = michaels_load.get_default_path()


def get_in_dim(data_path=network_path):
    network_data = scipy.io.loadmat(data_path, simplify_cells=True)
    dim = network_data["inp"][0].shape[0]
    return dim


def generate(
    out_path,
    mrnn,
    data_path=network_path,
    num_neurons_per_module=NUM_NEURONS_PER_MODULE,
    num_examples=NUM_EXAMPLES,
    stim_max_power=STIM_MAX_POWER,
    observer_instance=None,
):

    dataset = mRNN.MichaelsDataset(data_path)
    out_file = h5py.File(out_path, "w")

    try:
        return _generate(
            out_file,
            mrnn,
            dataset,
            data_path,
            num_neurons_per_module=num_neurons_per_module,
            num_examples=num_examples,
            stim_max_power=stim_max_power,
            observer_instance=observer_instance,
        )
    finally:
        out_file.close()


def _generate(
    out_file,
    mrnn,
    dataset,
    data_path,
    num_neurons_per_module=NUM_NEURONS_PER_MODULE,
    num_examples=NUM_EXAMPLES,
    stim_max_power=STIM_MAX_POWER,
    observer_instance=None,
):

    num_stim_channels = mrnn.stimulus.num_stim_channels

    if observer_instance is None:
        ob = observer.ObserverGaussian1d(
            in_dim=num_neurons_per_module, out_dim=NUM_OBS_ELECTRODES
        )
    else:
        ob = observer_instance

    num_data = len(dataset)
    out_file["params"] = data_path
    input_dim = ob.out_dim * 3 + num_stim_channels
    output_dim = ob.out_dim * 3

    for eidx in range(num_examples):
        if not (eidx % 100):
            print(f"{time.time()} Generating example {eidx}")

        mrnn.reset_hidden()

        # Each example is based on one actual input trace, randomly chosen
        didx = random.randrange(num_data)
        din, dout = dataset[didx]
        steps = din.shape[0]

        # The stimulation model maps observations from all brain regions
        # and the stim params to predicted observations. Remember
        # stim is applied only to M1.
        cur_inps = np.zeros((steps, input_dim))
        cur_obvs = np.zeros((steps, output_dim))
        prev_obvs = []

        for tidx in range(steps):
            # Stimulate sometimes, and not others, so we can observe both conditions.
            if random.random() <= 0.2 and stim_max_power:
                stim = [
                    random.uniform(-1 * stim_max_power, stim_max_power)
                    for _ in range(num_stim_channels)
                ]
                mrnn.stimulate(stim)
            else:
                stim = [0.0] * num_stim_channels

            cur = din[tidx, :].T.squeeze()
            cur = cur.reshape(cur.shape + (1,))
            mrnn(cur)

            for midx, module_obs in enumerate(prev_obvs):
                cur_inps[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs
            cur_inps[tidx, 3 * ob.out_dim :] = stim

            # 3-tuple of (1, ob.out_dim)
            obs = mrnn.observe(ob)
            for midx, module_obs in enumerate(obs):
                cur_obvs[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs

            prev_obvs = obs

        out_file["/%s/obvs" % str(eidx)] = cur_obvs
        out_file["/%s/inputs" % str(eidx)] = cur_inps
        out_file["/%s/class" % str(eidx)] = didx

    return prev_obvs, cur_inps, cur_obvs
