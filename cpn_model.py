import sys
import time

import h5py
import numpy as np
import torch
import torch.autograd
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import mRNN
import stim
import stim_model
import utils


DEFAULT_STIM_REG_WEIGHT = 1e-7


class CPNModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation_func=utils.ReTanh, num_neurons=None):
        super(CPNModel, self).__init__()

        if num_neurons is None:
            num_neurons = in_dim + 10

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_func_t = activation_func

        if activation_func is nn.PReLU:
            self.activation_func = activation_func(num_neurons)
        else:
            self.activation_func = activation_func()

        self.num_neurons = num_neurons

        with torch.no_grad():
            # Inter-neuron hidden recurrent weights
            # (num_neurons, num_neurons)
            self.W = nn.Parameter(torch.zeros((1, num_neurons, num_neurons)))
            nn.init.normal_(self.W[:, :, :], mean=0.0, std=(1.0 / np.sqrt(num_neurons)))

            # Neuron response to input
            # (num_neurons, in_dim)
            self.I = nn.Parameter(torch.zeros((1, num_neurons, in_dim)))
            nn.init.normal_(self.I[:, :, :], mean=0.0, std=(1.0 / np.sqrt(in_dim)))

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
        self.x0 = None

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = self.activation_func(
                    torch.zeros((batch_size, self.num_neurons)))

        x = self.W @ self.prev_output.reshape(self.x.shape + (1,))
        assert x.shape == (batch_size, self.num_neurons, 1)

        x = x + self.I @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        readout = self.fc(rnn_output)

        self.prev_output = rnn_output

        return readout

class CPNNoiseyCollection(nn.Module):
    """
    This acts like a CPN, but in-fact varies its parameters for each sample in
    the batch. For N% of the batch it acts exactly like the provided CPN. For
    the rest, it acts the same, but with some amount of mean 0 noise added to
    every param.
    """
    def __init__(self, cpn, noisey_pct=0.90, noise_var=0.3):
        super(CPNNoiseyCollection, self).__init__()

        self.in_dim = cpn.in_dim
        self.out_dim = cpn.out_dim
        self.activation_func_t = cpn.activation_func_t

        if cpn.activation_func_t is nn.PReLU:
            self.activation_func = cpn.activation_func_t(cpn.num_neurons)
        else:
            self.activation_func = cpn.activation_func_t()

        self.num_neurons = cpn.num_neurons
        self.cpn = cpn
        self.noisey_pct = noisey_pct
        self.noise_var = noise_var

        # (batch, num_neurons)
        self.x = None
        # (batch, num_neurons)
        self.prev_output = None
        self.x0 = None

    def reset(self):
        self.x = None
        self.prev_output = None
        self.x0 = None

        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def setup(self, batch_size):
        with torch.no_grad():
            noisey_cnt = int(batch_size * self.noisey_pct)
            cpn = self.cpn

            # Inter-neuron hidden recurrent weights
            # (batch_size, num_neurons, num_neurons)
            self.W = nn.Parameter(torch.empty((batch_size, self.num_neurons,
                    self.num_neurons)))
            self.W[:, :, :] = cpn.W[:, :, :].repeat(batch_size, 1, 1)
            self.W[:noisey_cnt, :, :] += self.noise_var * (
                    torch.rand(noisey_cnt, self.num_neurons,
                    self.num_neurons) - 0.5)

            # Neuron response to input
            # (batch_size, num_neurons, in_dim)
            self.I = nn.Parameter(torch.empty((batch_size, self.num_neurons,
                    self.in_dim)))
            self.I[:, :, :] = cpn.I[:, :, :].repeat(batch_size, 1, 1)
            self.I[:noisey_cnt, :, :] += self.noise_var * (
                    torch.rand(noisey_cnt, self.num_neurons,
                    self.in_dim) - 0.5)

            # Neuron biases
            self.b = nn.Parameter(torch.empty((batch_size, self.num_neurons)))
            self.b[:, :] = cpn.b[:].reshape(1, self.num_neurons).repeat(batch_size, 1)
            self.b[:noisey_cnt, :] += self.noise_var * (
                    torch.rand(noisey_cnt, self.num_neurons) - 0.5)

            # This one is harder... Need to re-implement fc...
            self.fc_w = nn.Parameter(torch.empty(batch_size,
                    self.out_dim, self.num_neurons))
            self.fc_w[:, :, :] = cpn.fc.weight[:, :].reshape(1, self.out_dim,
                    self.num_neurons).repeat(batch_size, 1, 1)
            self.fc_w[:noisey_cnt, :, :] += self.noise_var * (
                    torch.rand(noisey_cnt, self.out_dim, self.num_neurons) - 0.5)

            self.fc_b = nn.Parameter(torch.empty(batch_size, self.out_dim))
            self.fc_b[:, :] = cpn.fc.bias[:].reshape(1, self.out_dim).repeat(
                    batch_size, 1)
            self.fc_b[:noisey_cnt, :] += self.noise_var * (
                    torch.rand(noisey_cnt, self.out_dim) - 0.5)

    def forward(self, din):
        """
        Args:
            din - (batch, in_dim)
        """
        batch_size = din.shape[0]

        if self.x is None:
            self.x0 = torch.zeros((batch_size, self.num_neurons))
            self.x = self.x0
            self.prev_output = self.activation_func(
                    torch.zeros((batch_size, self.num_neurons)))
            self.setup(batch_size)

        x = self.W @ self.prev_output.reshape(self.x.shape + (1,))
        assert x.shape == (batch_size, self.num_neurons, 1)

        x = x + self.I @ din.reshape(din.shape + (1,))
        x = x.squeeze() + self.b
        self.x = x

        rnn_output = self.activation_func(x)
        assert rnn_output.shape == (batch_size, self.num_neurons)

        weighted = self.fc_w @ rnn_output.reshape(rnn_output.shape + (1,))
        readout = weighted.squeeze(dim=2) + self.fc_b

        self.prev_output = rnn_output

        return readout


#class CPNModelLSTM(utils.LSTMModel):
#    pass

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


bi = None
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
    stim_reg_weight=DEFAULT_STIM_REG_WEIGHT,
    batch_size=64,
    train_pct_stop_thresh=None,
    train_stop_thresh=None,
    model_save_path=None,
):
    obs = observer_instance
    min_loss = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for eidx in range(train_max_epochs):
        print(f"Epoch: {eidx}")
        for i_batch, sampled_batch in enumerate(loader):
            model.reset()
            mrnn.reset_hidden()
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
                mrnn(cur_in.T)
                observations = mrnn.observe(obs)

                obs_vecs = [
                    torch.tensor(o, dtype=torch.float, requires_grad=False)
                    for o in observations
                ]
                obs_vec = torch.cat(obs_vecs, axis=1)

                stim_params = model(obs_vec)
                ben_in = torch.cat((obs_vec, stim_params), axis=1)
                ben_in.retain_grad()
                global bi
                bi = ben_in
                ben_pred = ben(ben_in)

                mrnn.stimulate(stim_params)

                # ben outputs predictions for all modules, but our loss is
                # based only on the last module (smallest indices).
                ben_preds[:, tidx, :] = ben_pred[:, : obs.out_dim]

            # NOTE: here we can have a loss based on task performance,
            # or brain state

            loss = torch.nn.MSELoss()(ben_preds, dout)
            # reg_loss = model.stim_regularizer() * stim_reg_weight
            # loss += reg_loss
            print(loss)

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


def prep_new(
    train_data_path,
    mrnn_model_path,
    ben_model_path,
    observer_instance,
    lesion,
    stimulus,
    activation_func,
    learning_rate=0.008,
    train_max_epochs=2000,
    stim_reg_weight=DEFAULT_STIM_REG_WEIGHT,
    batch_size=64,
    train_pct_stop_thresh=None,
    train_stop_thresh=0.0001,
    model_save_path=None,
):

    torch.autograd.set_detect_anomaly(True)
    print("Loading dataset; this may take awhile...")
    dataset = CPNTrainDataset(train_data_path)

    example_len = dataset[0][0].shape[0]
    obs_vec_dim = 3 * observer_instance.out_dim
    in_dim = obs_vec_dim
    out_dim = stimulus.num_stim_channels

    mrnn = mRNN.load_from_file(
        mrnn_model_path, pretrained=True, stimulus=stimulus, lesion=lesion
    )

    # NOTE: if we want to lock ben, do it here with the 'pretrained' kwarg
    ben_in_dim = obs_vec_dim + stimulus.num_stim_channels
    ben_out_dim = obs_vec_dim
    ben = stim_model.load_from_file(
        ben_model_path,
        ben_in_dim,
        ben_out_dim,
    )
    model = CPNModel(in_dim, out_dim, activation_func=activation_func)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    return (
        dataset,
        model,
        mrnn,
        ben,
        observer_instance,
        optimizer,
        example_len,
        in_dim,
        out_dim,
        train_max_epochs,
        stim_reg_weight,
        batch_size,
        train_pct_stop_thresh,
        train_stop_thresh,
        model_save_path,
    )


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
    stim_reg_weight=DEFAULT_STIM_REG_WEIGHT,
    batch_size=64,
    train_pct_stop_thresh=None,
    train_stop_thresh=0.0001,
    model_save_path=None,
):
    train_args = prep_new(
        train_data_path,
        mrnn_model_path,
        ben_model_path,
        observer_instance,
        lesion,
        stimulus,
        activation_func,
        learning_rate=learning_rate,
        train_max_epochs=train_max_epochs,
        batch_size=batch_size,
        train_pct_stop_thresh=train_pct_stop_thresh,
        train_stop_thresh=train_stop_thresh,
        model_save_path=model_save_path,
    )

    train_model(*train_args)
    return train_args[1], train_args[0], train_args[5]
