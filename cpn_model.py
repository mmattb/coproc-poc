import sys

import h5py
import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import mRNN
import stim_model
import utils


class CPNModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation_func=utils.ReTanh, num_neurons=None):
        super(CPNModel, self).__init__()

        if num_neurons is None:
            num_neurons = in_dim + 10

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func = activation_func()
        self.num_neurons = num_neurons

        with torch.no_grad():
            # Inter-neuron hidden recurrent weights
            # (num_neurons, num_neurons)
            self.W = nn.Parameter(torch.zeros((num_neurons, num_neurons)))
            nn.init.normal_(self.W[:, :], mean=0.0, std=(1.0 / np.sqrt(num_neurons)))

            # Neuron response to input
            # (num_neurons, in_dim)
            self.I = nn.Parameter(torch.zeros((num_neurons, in_dim)))
            nn.init.normal_(self.I[:, :], mean=0.0, std=(1.0 / np.sqrt(in_dim)))

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

        x = self.W.reshape((1,) + self.W.shape) @ self.prev_output.reshape(
            self.prev_output.shape + (1,)
        )
        assert x.shape == (batch_size, self.num_neurons, 1)

        x += self.I.reshape((1,) + self.I.shape) @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        readout = self.fc(rnn_output)

        self.prev_output = rnn_output
        return readout


class CPNTrainDataset(Dataset):
    def __init__(self, data_file_path):
        f = h5py.File(data_file_path, "r")
        sample_keys = [k for k in f.keys() if k != "params"]
        self.sample_count = len(sample_keys)

        # Indexed by idx
        self.data = []
        self._load_data(f)

    def __len__(self):
        return self.sample_count

    def _load_data_single(self, idx, f):
        sample = f[str(idx)]

        din = torch.tensor(sample["inputs"]).float()
        dout = torch.tensor(sample["obvs"]).float()

        return din, dout

    def _load_data(self, f):
        for i in range(self.sample_count):
            self.data.append(self._load_data_single(i, f))

    def __getitem__(self, idx):
        return self.data[idx]


def train_model(
    dataset,
    model,
    mrnn,
    ben,
    observer_instance,
    optimizer,
    example_len,
    in_dim,
    out_dim,
    train_max_epochs=2000,
    batch_size=64,
    train_pct_stop_thresh=None,
    train_stop_thresh=None,
    model_save_path=None,
):
    obs = observer_instance
    losses = []
    min_loss = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for eidx in range(train_max_epochs):
        print(f"Epoch: {eidx}")
        for i_batch, sampled_batch in enumerate(loader):
            model.reset()
            mrnn.reset()
            ben.reset()
            optimizer.zero_grad()

            # din: input to mRNN, i.e. VGG features + hold signal
            # dout: observation of last module. mRNN was healthy when observed
            #       (batch_size, time, obs feature)
            din, dout = sampled_batch
            batch_size = din.shape[0]

            ben_preds = torch.empty((batch_size, example_len, obs.out_dim))
            for tidx in range(example_len):
                cur_in = din[:, tidx, :]

                # NOTE: if our loss is based on task performance, we can
                # get outputs here.
                mrnn(cur_in)
                observations = mrnn.observe(obs)

                stim_params = model(observations)
                print(cur_in.shape, stim_params.shape)
                ben_in = torch.cat(cur_in, stim_params, axis=1)
                print(ben_in.shape)
                ben_pred = ben(ben_in)

                mrnn.stimulate(stim_params)

                # ben outputs predictions for all modules, but our loss is
                # based only on the last module (smallest indices).
                ben_preds[:, tidx, :] = ben_pred[:, : obs.out_dim]

            # NOTE: here we can have a loss pbased on task performance,
            # or brain state
            loss = torch.nn.MSELoss()(preds, dout)
            losses.append(loss.item())

            if min_loss is None:
                min_loss = loss.item()
            elif min_loss > loss.item():
                print(f"Min loss: {loss}")
                if model_save_path is not None:
                    torch.save(model.state_dict(), model_save_path)
                min_loss = loss.item()

            loss.backward()
            optimizer.step()

        if train_stop_thresh is not None and train_stop_thresh >= min_loss:
            break

    if eidx >= train_max_epochs:
        sys.stderr.write(f"Learning didn't converage after {eidx} epochs\n")


def train_new(
    train_data_path,
    mrnn_model_path,
    ben_model_path,
    observer_instance,
    lesion,
    stimulus,
    activation_func,
    learning_rate=0.008,
    train_max_epochs=2000,
    batch_size=64,
    train_pct_stop_thresh=None,
    train_stop_thresh=0.0001,
    model_save_path=None,
):

    torch.autograd.set_detect_anomaly(True)
    print("Loading dataset; this may take awhile...")
    dataset = CPNTrainDataset(train_data_path)

    example_len = dataset[0][0].shape[0]
    in_dim = dataset[0][0].shape[1]
    out_dim = stimulus.num_stim_channels

    mrnn = mRNN.load_from_file(
        mrnn_model_path, pretrained=True, stimulus=stimulus, lesion=lesion
    )

    # NOTE: if we want to lock ben, do it here with the 'pretrained' kwarg
    ben = stim_model.load_from_file(ben_model_path)
    model = CPNModel(in_dim, out_dim, activation_func=activation_func)

    train_model(
        dataset,
        model,
        mrnn,
        ben,
        observer_instance,
        optimizer,
        example_len,
        in_dim,
        out_dim,
        train_max_epochs=train_max_epochs,
        batch_size=batch_size,
        train_pct_stop_thresh=train_pct_stop_thresh,
        train_stop_thresh=train_stop_thresh,
        model_save_path=model_save_path,
    )
    return model, dataset, optimizer
