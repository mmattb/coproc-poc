import torch
from torch.optim import AdamW

import cpn_model
import experiment


class CPN_EN_CoProc(experiment.CoProc):
    def __init__(self, cfg, retain_stim_grads=False, cuda=None):
        self.cfg = cfg

        in_dim, stim_dim, out_dim, cuda = cfg.unpack()

        self.cpn = cpn_model.CPNModelLSTM(
            in_dim,
            stim_dim,
            num_neurons=in_dim,
            activation_func=torch.nn.Tanh,
            cuda=cuda,
        )

        en_in_dim = cfg.observer_instance.out_dim + stim_dim + 1
        self.en = stim_model.StimModelLSTM(
            en_in_dim,
            out_dim,
            num_neurons=en_in_dim + 50,
            activation_func=torch.nn.Tanh,
            cuda=cuda,
        )

        self.opt_cpn = AdamW(self.cpn.parameters(), lr=1e-3)
        self.opt_en = AdamW(self.en.parameters(), lr=1e-3)

        self.current_noisey_cpn = None
        self.retain_stim_grads = retain_stim_grads

        self.stims = None
        self.preds = None
        self.reset()

    def reset(self):
        self.stims = []
        self.preds = torch.tensor

    def ensure_preds(self, ...):
        pass

    def forward(self, brain_data):
        cpn_in = torch.cat(brain_data, axis=1).detach()

        new_stim = self.cpn(cpn_in)
        if self.retain_stim_grads:
            new_stim.retain_grad()
        self.stims.append(new_stim)

        # en receives (obs, stims, trial_end)
        # (detaching just in case; we don't want to backprop here)
        new_obs_en = brain_data[0].detach()
        en_in = torch.cat((new_obs_en, new_stim, obs[-1]), axis=1)
        cur_pred = en(en_in)

        preds[:, tidx, :] = cur_pred[:, :]


        raise NotImplementedError()

    def feedback(self, actuals, targets, loss_history):
        """
        Args:
            loss_history: a loss.LossHistory
        Returns:
            True to stop training; False otherwise
        """
        raise NotImplementedError()

    def finish(self):
        """
        Called between batches. User can add any cleanup logic here, e.g.
        resetting internal state of RNNs.
        """
        pass
