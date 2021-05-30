import random
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import michaels_load
import stim
import utils


# Length of real-world time for a given time step, in ms
tick = 10
dt = 1
tau = 100 / tick
synaptic_scaling_factor = 1.2
num_neurons_per_module = 100
num_modules = 3
num_input_features = 20
num_output_features = 50
sparsity = 0.1


# Scalars for regularizers
sc_loss_fr = 1e-1
sc_loss_io = 1e-5


class MichaelsRNN(nn.Module):
    def __init__(
        self,
        tau=tau,
        dt=dt,
        num_neurons_per_module=num_neurons_per_module,
        num_modules=num_modules,
        num_input_features=num_input_features,
        sparsity=sparsity,
        synaptic_scaling_factor=synaptic_scaling_factor,
        activation_func=utils.ReTanh,
        sc_loss_fr=sc_loss_fr,
        sc_loss_io=sc_loss_io,
        output_dim=50,
        stimulus=None,
        lesion=None,
        init_data_path=None,
    ):

        super(MichaelsRNN, self).__init__()

        self.tau = tau
        self.dt = dt
        self.sparsity = sparsity
        self.num_neurons_per_module = num_neurons_per_module
        self.num_modules = num_modules
        self.num_input_features = num_input_features
        self.num_neurons = num_neurons_per_module * num_modules
        self.synaptic_scaling_factor = synaptic_scaling_factor
        self.activation_func = activation_func()
        self.output_dim = output_dim

        self.sc_loss_fr = sc_loss_fr
        self.sc_loss_io = sc_loss_io

        npm = self.num_neurons_per_module
        numn = self.num_neurons

        # Used in firing rate regularizer
        self.sum_fr = 0.0
        self.denom_fr = 0.0

        # Recurrent state -----------------
        # Last outputs from the RNN. Setting to 0 for the first time step.
        # Set on first use, since we will keep memory for each trial in the
        # batch, and we don't know the batch size until the first call of
        # forward()
        # Referred to as r_i in the Michaels text.
        # (batch_size, num_neurons) once set
        self.prev_output = None

        # Internal state of the neurons. Referred to as x_i in the text
        self.x = None

        if init_data_path is not None:
            (
                J,
                I,
                S,
                B,
                fc,
                x0,
            ) = utils.init_from_michaels_model(
                init_data_path, num_input_features, num_neurons_per_module, output_dim
            )

            self.J = nn.Parameter(J)
            self.I = nn.Parameter(I)
            self.S = nn.Parameter(S)
            self.B = nn.Parameter(B)
            self.fc = fc
            self.x0 = nn.Parameter(x0)

            self.J_zero_grad_mask = None
            self.I_zero_grad_mask = None
            self.recalc_masks_from_weights()
        else:
            # Masks ---------------------------
            # (num_neurons, num_neurons)
            # Set to 1 for pairs which represent sparse connections
            # There are two areas of the matrix with sparse connections:
            #   module 1->2 and module 2->3
            # A '1' is a connection, a '0' is a non-connection
            sparse_mask = torch.zeros((numn, numn))

            # list of (neuron_idx, neuron_idx)
            # Memory inefficient approach, but easy to understand,
            #  and we have a small model...
            possible_connections = []
            for i in range(npm):
                for j in range(npm):
                    possible_connections.append((i, j))

            for c12 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c12[0] + npm, c12[1]] = 1

            for c23 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c23[0] + (npm * 2), c23[1] + npm] = 1

            for c21 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c21[0], c21[1] + npm] = 1

            for c32 in random.sample(possible_connections, int(sparsity * npm ** 2)):
                sparse_mask[c32[0] + npm, c32[1] + (npm * 2)] = 1

            # Zero grad mask: used to 0 out gradients during training,
            # to account for sparsity
            # (num_neurons, num_neurons)
            self.J_zero_grad_mask = torch.zeros((numn, numn))

            # Don't zero the sparse connections
            self.J_zero_grad_mask += sparse_mask

            # Don't zero the FC connections
            fc_mask = torch.ones((npm, npm))
            for i in range(num_modules):
                start = i * npm
                self.J_zero_grad_mask[
                    start : start + npm, start : start + npm
                ] = fc_mask

            self.I_zero_grad_mask = torch.zeros((self.num_neurons, self.num_neurons))
            self.I_zero_grad_mask[-1 * npm :, -1 * npm :] = 1.0

            with torch.no_grad():
                # Params --------------------------
                # Bias: (num_neurons, 1)
                self.B = nn.Parameter(torch.empty((numn, 1)))
                # See: https://github.com/JonathanAMichaels/hfopt-matlab/blob/0a18401c62b555bf799de83aa0d722bc82cf06d2/rnn/init_rnn.m#L146
                nn.init.uniform_(self.B, -1.0, 1.0)

                # Input weights: (num_neurons, num_input_features)
                self.I = nn.Parameter(torch.empty((numn, num_input_features)))
                nn.init.normal_(self.I, mean=0.0, std=(1 / np.sqrt(num_input_features)))

                # zero out those not in the input module
                self.I[-1 * npm :, :] = 0.0

                # Hidden state response to the hold signal
                # (num_neurons, 1)
                self.S = nn.Parameter(torch.empty((numn, 1)))
                nn.init.normal_(self.S, mean=0.0, std=1)

                # Recurrent weights: (num_neurons (current), num_neurons (prev))
                self.J = nn.Parameter(torch.zeros((numn, numn)))

                # Intra-module portions
                for i in range(num_modules):
                    start = i * npm
                    nn.init.normal_(
                        self.J[start : start + npm, start : start + npm],
                        mean=0.0,
                        std=(synaptic_scaling_factor / np.sqrt(npm)),
                    )

                # Inter-module portions
                #  modules 1->2, 2->3, 3->2, 2->1
                for i in (0, 1):
                    start_in = i * npm
                    start_out = start_in + npm
                    nn.init.normal_(
                        self.J[start_out : start_out + npm, start_in : start_in + npm],
                        mean=0.0,
                        std=(1 / np.sqrt(npm)),
                    )

                for i in (1, 2):
                    start_in = i * npm
                    start_out = (i - 1) * npm
                    nn.init.normal_(
                        self.J[start_out : start_out + npm, start_in : start_in + npm],
                        mean=0.0,
                        std=(1 / np.sqrt(npm)),
                    )

                self.J *= self.J_zero_grad_mask

                self.x0 = nn.Parameter(torch.empty(self.num_neurons))
                self.fc = nn.Linear(num_neurons_per_module, output_dim)

        self.lesion = lesion
        self.stimulus = stimulus

        self.reset_hidden()

    def recalc_masks_from_weights(self):
        npm = self.num_neurons_per_module
        J_zero_grad_mask = np.zeros((self.num_neurons, self.num_neurons))
        J_zero_grad_mask[self.J != 0.0] = 1.0
        J_zero_grad_mask[:npm, :npm] = 1.0
        J_zero_grad_mask[npm : npm * 2, npm : npm * 2] = 1.0
        J_zero_grad_mask[npm * 2 : npm * 3, npm * 2 : npm * 3] = 1.0
        J_zero_grad_mask = torch.tensor(J_zero_grad_mask, requires_grad=False).float()
        self.J_zero_grad_mask = J_zero_grad_mask

        I_zero_grad_mask = np.zeros((self.num_neurons, self.num_input_features))
        I_zero_grad_mask[npm:, :] = 1.0
        I_zero_grad_mask = torch.tensor(I_zero_grad_mask, requires_grad=False).float()
        self.I_zero_grad_mask = I_zero_grad_mask

    def load_weights_from_file(self, data_path):
        self.load_state_dict(torch.load(data_path))
        self.eval()
        self.recalc_masks_from_weights()

    @property
    def tau_inv(self):
        return 1.0 / self.tau

    def reset_stim(self):
        if self.stimulus is not None:
            self.stimulus.reset()

    def reset_hidden(self):
        self.reset_stim()

        self.prev_output = None

        # Internal state of the neurons. Referred to as x_i in the text
        self.x = None

        # Accumulated firing rates, for regularizer
        self.sum_fr = 0.0
        self.denom_fr = 0.0

    def set_sparse_grads(self):
        self.J.grad *= self.J_zero_grad_mask
        self.I.grad *= self.I_zero_grad_mask

    def set_lesion(self, lesion):
        """
        Args:
            mask: tensor (num_neurons,), with zeros for neurons which
                should not fire, and ones elsewhere.
                Or: pass None to reset
        """
        self.lesion = lesion

    def forward(self, data):
        """
        Args:
          data:
            - hold: vector of: double 0.0 or 1.0: (batch_size)
            - image: Tensor((num_input_features, batch_size))
        """

        image, hold = np.split(data, [self.num_input_features], axis=0)
        batch_size = image.shape[1]

        if self.prev_output is None:
            # (batch_size, num_neurons)
            self.x = torch.tile(self.x0, (batch_size, 1))
            if self.lesion is not None:
                self.x = self.lesion.lesion(self, self.x)
            self.prev_output = self.activation_func(self.x)
        elif batch_size != self.prev_output.shape[0]:
            raise RuntimeError(
                "Must have the same batch size every time step. "
                "Did you forget to reset the module between batches?"
            )

        # Double check myself; can remove this and other asserts after testing
        assert len(image.shape) == 2
        assert image.shape[0] == self.num_input_features
        assert self.prev_output.shape[1] == self.num_neurons

        # Cleared for take-off...
        # Recurrence

        recur = self.prev_output @ self.J.T
        assert recur.shape == (batch_size, self.num_neurons)

        # Input
        inp = image.T @ self.I.T + hold.T * self.S.T
        assert inp.shape == (batch_size, self.num_neurons)

        # x broadcast up from (numn,) to (batch_size, numn)
        tdx = -self.x + recur + inp + self.B.T
        if self.stimulus is not None:
            tdx += self.stimulus.get_next()

        pre_response = self.x + tdx / 10
        assert pre_response.shape == (batch_size, self.num_neurons)

        if self.lesion is not None:
            pre_response = self.lesion.lesion(self, pre_response)

        output = self.activation_func(pre_response)
        assert output.shape == (batch_size, self.num_neurons)

        self.x = pre_response
        self.prev_output = output

        # Used for firing rate (output) regularization
        self.sum_fr += torch.sum(torch.square(output))
        self.denom_fr += self.num_neurons + batch_size

        # Return only from the final module
        ret = self.fc(output[:, :self.num_neurons_per_module])
        return ret

    def observe(self, obs_model):
        outputs = []

        for midx in range(self.num_modules):
            act = self.prev_output[
                :,
                midx
                * self.num_neurons_per_module : (midx + 1)
                * self.num_neurons_per_module,
            ]
            out = obs_model(act)
            assert out.shape == (self.prev_output.shape[0], obs_model.out_dim)
            outputs.append(out)

        # 3-tuple, elements are (batch, obs.out_dim)
        return outputs

    def unroll(self, data_in):
        data = utils.fill_jagged_array(
            [
                data_in,
            ]
        )

        steps = data.shape[1]
        pred_out = torch.empty((1, steps, self.output_dim))
        for tidx in range(steps):
            cur = data[:, tidx, :]
            pred_out[0, tidx, :] = self(cur.T)
        return pred_out

    def stimulate(self, params):
        self.stimulus.add(params)

    # Firing rate regularizer
    def fr_reg(self):
        loss = self.sum_fr / self.denom_fr
        return loss

    # Input/outout weight regularizer
    def io_reg(self):
        loss = torch.sum(torch.square(self.fc.weight))
        loss += torch.sum(torch.square(self.I))

        return loss

    def calc_loss(self, preds, outputs, lf=None):
        if lf is None:
            lf = nn.MSELoss()
        loss = lf(preds, outputs)

        loss += self.sc_loss_fr * self.fr_reg()
        loss += self.sc_loss_io * self.io_reg()
        return loss


class MichaelsDataset(Dataset):
    def __init__(self, data_file_path):
        f = michaels_load.load_from_path(data_file_path)
        inps = f["inp"]
        outs = f["targ"]

        self.num_samples = inps.shape[0]
        self.sample_len = max([s.shape[1] for s in inps])

        self.data = []
        self._load_data(inps, outs)

    def __len__(self):
        return self.num_samples

    def _load_data_single(self, idx, inps, outs):
        cur_in = inps[idx]
        cur_out = outs[idx]

        din = torch.empty((self.sample_len, cur_in.shape[0]), dtype=torch.float)
        din[: cur_in.shape[1], :] = torch.tensor(cur_in.T[:, :])
        din[cur_in.shape[1] :, :] = torch.tensor(cur_in.T[-1, :])

        dout = torch.empty((self.sample_len, cur_out.shape[0]), dtype=torch.float)
        dout[: cur_out.shape[1], :] = torch.tensor(cur_out.T[:, :])
        dout[cur_out.shape[1] :, :] = torch.tensor(cur_out.T[-1, :])

        return din, dout

    def _load_data(self, inps, outs):
        for i in range(self.num_samples):
            self.data.append(self._load_data_single(i, inps, outs))

    def __getitem__(self, idx):
        return self.data[idx]


def generate(
    model_path,
    init_data_path=None,
    stimulus=None,
    lesion=None,
    recover_after_lesion=True,
    recover_train_batch_size=64,
    recover_train_pct_stop_thresh=0.003,
    recover_train_max_epochs=1000,
):

    if init_data_path is None:
        init_data_path = michaels_load.get_default_path()

    mrnn = MichaelsRNN(init_data_path=init_data_path, stimulus=stimulus, lesion=lesion)

    if recover_after_lesion:
        torch.save(mrnn.state_dict(), model_path + "_pre")

        dataset = MichaelsDataset(init_data_path)
        optimizer = Adam(mrnn.parameters(), lr=0.008)
        best_mean_loss = 1.0e55
        running_mean_loss = None

        for eidx in range(recover_train_max_epochs):
            loader = DataLoader(
                dataset, batch_size=recover_train_batch_size, shuffle=True
            )

            losses = []
            for i_batch, sampled_batch in enumerate(loader):
                mrnn.reset_hidden()
                optimizer.zero_grad()

                din, dout = sampled_batch
                batch_size = din.shape[0]
                example_len = din.shape[1]

                preds = torch.empty((batch_size, example_len, mrnn.output_dim))
                for tidx in range(example_len):
                    cur_in = din[:, tidx, :]
                    pred = mrnn(cur_in.T)
                    preds[:, tidx, :] = pred[:, :]

                loss = torch.nn.MSELoss()(preds, dout)
                losses.append(loss.item())

                loss.backward()
                mrnn.set_sparse_grads()
                optimizer.step()

            epoch_mean_loss = statistics.mean(losses)
            print(f"Mean loss: {epoch_mean_loss}")
            if running_mean_loss is None:
                running_mean_loss = epoch_mean_loss
            else:
                running_mean_loss = (running_mean_loss + epoch_mean_loss) / 2.0
            if running_mean_loss < best_mean_loss:
                if (
                    (best_mean_loss - running_mean_loss) / running_mean_loss
                ) <= recover_train_pct_stop_thresh:
                    break
                best_mean_loss = running_mean_loss

        if eidx >= recover_train_max_epochs:
            sys.stderr.write(f"Learning didn't converage after {eidx} epochs\n")

    torch.save(mrnn.state_dict(), model_path)
    return mrnn

def load_from_file(data_path, **kwargs):
    mrnn = MichaelsRNN(**kwargs)
    mrnn.load_weights_from_file(data_path)
    return mrnn