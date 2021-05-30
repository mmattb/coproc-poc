import os
import random

import h5py
import numpy as np
import torch
import scipy.io

import mRNN
import observer
import utils


BASE_DIR = "_Networks"
NUM_NEURONS_PER_MODULE = 100
NUM_EXAMPLES = 2000
# Applied only to M1
NUM_STIM_ELECTRODES = 35
# per module
NUM_OBS_ELECTRODES = 35
# Range of stimulus power is (-n, n), where n is:
STIM_MAX_POWER = 7.5


# Options are M and Z (These are the two monkeys in
# the experiment and I made separate networks for each of their kinematics)
monkey = "M"

# This variable specifies the non-linearity used in the Hidden layer. Input and Output layers are Linear
activation_type = "recttanh"  # Options are recttanh or rectlinear
# This variable specifies the architecture (Figure 5).
# In order from left to right in Figure 5 we have: CNN, Feedforward, LabeledLine-In, LabeledLine-Out, Ball, Sparse
input_name = (
    "CNN"  # Options are CNN, Feedforward, LabeledLine-In, LabeledLine-Out, Ball, Sparse
)
# This variable specifies the firing rate penalty hyper-parameter (Equation 5)
FR = "1e-3"  # Options are 0, 1e-3, 1e-2, 1e-1
# This variable specifies the input/output weight penalty (Equation 6)
IO = "1e-5"  # Options are 0, 1e-5, 1e-4, 1e-3
# This variable specifies the sparsity of the connections between modules (most results use 1e-1)
sparsity = "1e-1"  # Options are 1e-2, 1e-1, 1e0
# This variable specifies runs with different random initializations of each network
repetition = "1"  # Options are 1, 2, 3, 4
data_folder = os.path.join(BASE_DIR, monkey)
network_dir = "-".join([activation_type, input_name, FR, IO, sparsity, repetition])
network_path = os.path.join(data_folder, os.path.join(network_dir, network_dir))


def generate(
    out_path,
    data_path=network_path,
    num_neurons_per_module=NUM_NEURONS_PER_MODULE,
    num_examples=NUM_EXAMPLES,
    num_obs_electrodes=NUM_OBS_ELECTRODES,
    lesion=True,
):

    network_data = scipy.io.loadmat(data_path, simplify_cells=True)
    out_file = h5py.File(out_path, "w")

    try:
        return _generate(
            out_file,
            network_data,
            data_path=data_path,
            num_neurons_per_module=num_neurons_per_module,
            num_examples=num_examples,
            num_obs_electrodes=num_obs_electrodes,
        )
    finally:
        out_file.close()


def _generate(
    out_file,
    network_data,
    data_path=network_path,
    num_neurons_per_module=NUM_NEURONS_PER_MODULE,
    num_examples=NUM_EXAMPLES,
    num_obs_electrodes=NUM_OBS_ELECTRODES,
):
    rnn = mRNN.MichaelsRNN(init_data_path=data_path)
    ob = observer.ObserverGaussian1d(
        in_dim=num_neurons_per_module, out_dim=num_obs_electrodes
    )

    num_data = network_data["inp"].shape[0]
    out_file["params"] = data_path

    inputs = network_data["inp"]

    # TODO: pick up here. Do we repeat the last sample instead of feeding 0s?

    # We will pad all examples to this length. We feed zeros beyond the example's
    # natural length.
    max_example_len = max([x.shape[1] for x in inputs])

    input_dim = num_obs_electrodes * 3 + num_stim_electrodes
    output_dim = num_obs_electrodes * 3

    for eidx in range(num_examples):
        if not (eidx % 100):
            print(f"Generating example {eidx}")

        rnn.reset_hidden()
        # +1 for the hold signal
        pad = torch.zeros((rnn.num_input_features + 1, 1))

        # Each example is based on one actual input trace, randomly chosen
        didx = random.randrange(num_data)
        example_in = inputs[didx]
        data = utils.fill_jagged_array(
            [
                example_in,
            ]
        )

        steps = data.shape[1]

        # The stimulation model maps observations from all brain regions
        # and the stim params to predicted observations of M1. Remember
        # stim is applied only to M1 as well.
        cur_inps = np.zeros((max_example_len, input_dim))
        cur_obvs = np.zeros((max_example_len, output_dim))
        prev_obvs = []

        for tidx in range(steps):
            cur = data[:, tidx, :]
            rnn(cur.T)

            for midx, module_obs in enumerate(prev_obvs):
                cur_inps[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs
            cur_inps[tidx, 3 * ob.out_dim :] = stim

            # 3-tuple of (1, num_obs_electrodes)
            obs = rnn.observe(ob)
            for midx, module_obs in enumerate(obs):
                cur_obvs[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs

            prev_obvs = obs

        # Let any stimulation play out, and let the network settle to its default
        # (fed with 0s) state. This is better than padding all samples with 0s,
        # since we see the effect of any already established stimulation.
        #
        # Is the unpredictable trial length an unacceptable source of error for us?
        # Should we just patch with 0s (e.g. jagged_array)?
        for tidx in range(steps, max_example_len):
            rnn(pad)
            for midx, module_obs in enumerate(prev_obvs):
                cur_inps[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs

            obs = rnn.observe(ob)
            for midx, module_obs in enumerate(obs):
                cur_obvs[tidx, midx * ob.out_dim : (midx + 1) * ob.out_dim] = module_obs

            prev_obvs = obs

        out_file["/%s/obvs" % str(eidx)] = cur_obvs
        out_file["/%s/inputs" % str(eidx)] = cur_inps
        out_file["/%s/class" % str(eidx)] = didx


class HealthyDataset(Dataset):
    def __init__(self, data_file_path):
        """
        Args:
          * data_file_path - path to an hdf5, formatted by generate_stim_train_data.py
        """
        self.f_in = h5py.File(data_file_path, 'r')
        sample_keys = [k for k in self.f_in.keys() if k != "params"]
        self.sample_count = len(sample_keys)

        # Indexed by idx
        self.data = []
        self._load_data()
        
    def __len__(self):
        return self.sample_count

    def _load_data_single(self, idx):
        sample = self.f_in[str(idx)]

        din = torch.tensor(sample['inputs']).float()
        dout = torch.tensor(sample['obvs']).float()

        return din, dout

    def _load_data(self):
        for i in range(self.sample_count):
            self.data.append(self._load_data_single(i))

    def __getitem__(self, idx):
        return idx, self.data[idx]
