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
NUM_EXAMPLES = 2000


network_path = michaels_load.get_default_path()


def get_in_dim(data_path=network_path):
    network_data = scipy.io.loadmat(data_path, simplify_cells=True)
    dim = network_data["inp"][0].shape[0]
    return dim


def generate(
    out_path,
    mrnn_model_path,
    observer_instance,
    data_path=network_path,
    num_examples=NUM_EXAMPLES,
):

    dataset = mRNN.MichaelsDataset(data_path)
    out_file = h5py.File(out_path, "w")
    mrnn = mRNN.load_from_file(mrnn_model_path, pretrained=True)

    try:
        return _generate(
            out_file,
            mrnn,
            observer_instance,
            dataset,
            data_path,
            num_examples=num_examples,
        )
    finally:
        out_file.close()


def _generate(
    out_file,
    mrnn,
    observer_instance,
    dataset,
    data_path,
    num_examples=NUM_EXAMPLES,
):

    num_data = len(dataset)
    ob = observer_instance

    out_file["params"] = data_path
    input_dim = dataset[0][0].shape[1]
    output_dim = ob.out_dim

    for eidx in range(num_examples):
        if not (eidx % 100):
            print(f"{time.time()} Generating example {eidx}")

        mrnn.reset_hidden()

        # Each example is based on one actual input trace, randomly chosen
        didx = random.randrange(num_data)
        din, dout = dataset[didx]
        steps = din.shape[0]

        cur_inps = np.zeros((steps, input_dim))
        cur_obvs = np.zeros((steps, output_dim))

        for tidx in range(steps):
            cur = din[tidx, :].T.squeeze()
            cur = cur.reshape(cur.shape + (1,))
            mrnn(cur)

            cur_inps[tidx, :] = din[tidx, :]

            # 3-tuple of (1, ob.out_dim)
            obs = mrnn.observe(ob)
            obs_m1 = obs[0]
            cur_obvs[tidx, :] = obs_m1

        out_file["/%s/obvs" % str(eidx)] = cur_obvs
        out_file["/%s/inputs" % str(eidx)] = cur_inps
        out_file["/%s/class" % str(eidx)] = didx

    return cur_inps, cur_obvs
