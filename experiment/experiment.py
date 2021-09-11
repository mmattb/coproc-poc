import logging
import os
import time

import attr
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


# Local imports
from . import class_stdevs
from . import config
from . import stats
from . import michaels_load
from . import mRNN
from . import utils


# TODO: score(loss_history)

g_logger = logging.getLogger("experiment")


@attr.s(auto_attribs=True)
class EpochResult:
    stop: bool = False
    next_is_validation: bool = False
    user_data: stats.UserData = None

    def unpack(self):
        return self.stop, self.next_is_validation, self.user_data


def get_config(cuda=None):
    """
    Args:
        - cuda (str, torch.device, or None): None for CPU, or a string like "0" specifying a GPU
                              device. This will be passed to the CUDA_VISIBLE_DEVICES env var.
                              Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    Returns:
        - in_dim (int): dimensionality of the data coming from the brain
        - stim_dim (int): dimensionality of stimulation vector, which we get
                          from the co-proc
        - out_dim (int): dimensionality of the task: muscle velocities in our
                         case
        - cuda (torch.device, or None): will be a torch.device if we are using
                                        GPU. Note your model/Module will need to
                                        use this to push any parameters to GPU
                                        if the experiment is running on GPU.
    """
    if isinstance(cuda, str):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        cuda_out = torch.device(0)
    elif cuda is None:
        cuda_out = None
    else:
        cuda_out = cuda

    cfg = config.get_default(cuda=cuda_out)

    return cfg


def get_raw_data(cuda=None, **kwargs):
    # Just a passthrough, since 'experiment' is our simple interface
    return config.get_data_raw(cuda=cuda ** kwargs)


class CoProc:
    def forward(self, brain_data, loss_history):
        """
        Args:
            brain_data: 4-tuple (M1 data, F5 data, AIP data, trial_end signal)
                        The first three are Tensors (batch_size, cfg.observer_instance.out_dim)
                        The last is (batch_size, 1)
                        The sum of second dimensions is in_dim.
        Returns:
                the stim vector: Tensor(batch_size, stim_dim)
        """
        raise NotImplementedError()

    def feedback(self, actuals, targets, loss_history):
        """
        Args:
            actuals: torch.Tensor(batch_size, out_dim)
            targets: torch.Tensor(batch_size, out_dim)
            loss_history: a stats.LossHistory
        Returns:
            True if we should update the task loss; False if we should just
                carry it over from the prior epoch
        """
        raise NotImplementedError()

    def finish(self, loss_history):
        """
        Called between batches. User can add any cleanup logic here, e.g.
        resetting internal state of RNNs.
        Returns:
            A tuple:
                True to stop training; False otherwise,
                True to update task loss log based on this epoch,
                True if this is a validation epoch; report it as such,
                A UserData)
            If it's hard to remember the order of those, you may find
            it easier to remember by instead returning a
            EpochResult (above).
        """
        pass


def stage(coproc, cfg, recover_after_lesion=False):
    return Experiment(coproc, cfg, recover_after_lesion=recover_after_lesion)


class Experiment:
    def __init__(self, coproc, cfg, recover_after_lesion=False):
        self._coproc = coproc

        self._cfg = cfg

        mike = mRNN.MichaelsRNN(
            init_data_path=michaels_load.get_default_path(),
            stimulus=cfg.stim_instance,
            cuda=cfg.cuda,
        )
        mike.set_lesion(cfg.lesion_instance)

        self.mike = mike
        self.opt_mike = AdamW(self.mike.parameters(), lr=1e-4)

        # TODO: load from cached recovered model
        if recover_after_lesion:
            self._recover_after_lesion()
            self.opt_mike = AdamW(self.mike.parameters(), lr=1e-4)

        for param in self.mike.parameters():
            param.requires_grad = False

        self.observer = cfg.observer_instance

        (
            self.comp_loss_healthy,
            self.comp_loss_lesioned,
            self.var_whole_healthy,
            self.var_within_healthy,
        ) = self._get_healthy_vs_lesioned_stats()

        self.loss_history = stats.LossHistory(
            self.comp_loss_lesioned.item(),
            self.comp_loss_healthy.item(),
            self.var_whole_healthy,
            self.var_within_healthy,
        )

    @property
    def cfg(self):
        return self._cfg

    def _recover_after_lesion(self):
        """
        This function can be used to refine a Michaels model, simulating
        some amount of recovery. In practice, we use a model we've already
        trained in the same way.
        """
        cuda = self.cfg.cuda
        loss_func = torch.nn.MSELoss()

        dset = self.cfg.dataset
        dset_size = len(dset)
        dset_samp_len = dset.sample_len
        loader = DataLoader(dset, batch_size=dset_size, shuffle=True)

        loss = 1
        prev_loss = 2

        while loss > 0.0045:
            for batch in loader:
                prev_loss = loss

                self.mike.reset()
                self.opt_mike.zero_grad()

                din, _, _, dout, _ = batch

                preds = torch.zeros((dset_size, dset_samp_len, self.cfg.out_dim))
                for tidx in range(dout.shape[1]):
                    cur_din = din[:, tidx, :].T
                    p = self.mike(cur_din)
                    preds[:, tidx, :] = p[:, :]

                loss = loss_func(preds, dout)
                loss.backward()
                self.mike.set_coadap_grads()
                self.opt_mike.step()

                g_logger.info("Brain recovery loss: %0.7f", loss.item())


    def _get_healthy_vs_lesioned_stats(self):
        cuda = self.cfg.cuda
        comp_loss = torch.nn.MSELoss()

        dset = self.cfg.dataset
        dset_size = len(dset)
        dset_samp_len = dset.sample_len

        loader_comp = DataLoader(dset, batch_size=dset_size, shuffle=True)

        for s in loader_comp:
            din, trial_end, _, dout, labels = s

        comp_preds_healthy = torch.zeros((dset_size, dset_samp_len, self.cfg.out_dim))
        if cuda is not None:
            comp_preds_healthy = comp_preds_healthy.cuda(cuda)

        self.mike.set_lesion(None)
        try:
            self.mike.reset()
            for tidx in range(dout.shape[1]):
                cur_din = din[:, tidx, :].T
                p = self.mike(cur_din)
                comp_preds_healthy[:, tidx, :] = p[:, :]
            comp_preds_healthy = utils.trunc_to_trial_end(comp_preds_healthy, trial_end)
            comp_loss_healthy = comp_loss(comp_preds_healthy, dout)
        finally:
            self.mike.set_lesion(self.cfg.lesion_instance)

        var_whole, var_within = class_stdevs.calc_class_vars(comp_preds_healthy, labels)

        comp_preds_lesioned = torch.zeros(comp_preds_healthy.shape)
        if cuda is not None:
            comp_preds_lesioned = comp_preds_lesioned.cuda(cuda)

        self.mike.reset()
        for tidx in range(dout.shape[1]):
            cur_din = din[:, tidx, :].T
            p = self.mike(cur_din)
            comp_preds_lesioned[:, tidx, :] = p[:, :]
        comp_preds_lesioned = utils.trunc_to_trial_end(comp_preds_lesioned, trial_end)
        comp_loss_lesioned = comp_loss(comp_preds_lesioned, dout)

        self.mike.reset()

        return comp_loss_healthy, comp_loss_lesioned, var_whole, var_within

    @property
    def coproc(self):
        return self._coproc

    def _coproc_forward(self, brain_data):
        stim = self.coproc.forward(brain_data, self.loss_history)

        # Un-pythonic to assert a type, but let's be strict here...
        if not isinstance(stim, torch.Tensor):
            raise TypeError("Stim vector must be a Tensor")

        expected_stim_shape = (brain_data[0].shape[0], self.cfg.stim_dim)
        if stim.shape != expected_stim_shape:
            raise ValueError(
                f"Expected stim vector to have shape {expected_stim_shape}"
            )

        return stim

    def _coproc_feedback(self, actuals, targets, trial_end, loss_history):
        return self.coproc.feedback(actuals, targets, trial_end, loss_history)

    def _coproc_finish(self, loss_history):
        return self.coproc.finish(loss_history)

    def run(self):
        is_validation = False
        while True:
            if is_validation:

                assert (
                    len(self.cfg.loader_test.dataset) == self.cfg.loader_test.batch_size
                )
                batch = next(iter(self.cfg.loader_test))
            else:
                assert (
                    len(self.cfg.loader_train.dataset)
                    == self.cfg.loader_train.batch_size
                )
                batch = next(iter(self.cfg.loader_train))

            din, trial_end, trial_len, dout, labels = batch
            batch_size = din.shape[0]
            steps = din.shape[1]
            self.mike.reset()
            self.opt_mike.zero_grad()

            actuals = []

            mike_out = self.mike(din[:, 0, :].T)

            for tidx in range(steps - 1):
                obs_raw = self.mike.observe(self.observer)
                obs = obs_raw + (trial_end[:, tidx, :],)

                stim = self._coproc_forward(obs)

                # new_stim will be cloned in here, to prevent accidentally
                # backprop-ing through the "brain", aka mike.
                self.mike.stimulate(stim)

                mike_out = self.mike(din[:, tidx + 1, :].T)
                actuals.append(mike_out.unsqueeze(dim=1))

            actuals = torch.cat(actuals, axis=1)
            actuals = utils.trunc_to_trial_end(actuals, trial_end[:, :-1, :])

            # Give the user the result
            update_task_loss = self._coproc_feedback(
                actuals, dout, trial_end, self.loss_history
            )

            # Calc losses / stats
            if is_validation:
                self.loss_history.report_val_last_result(
                    actuals,
                    dout,
                    update_task_loss=update_task_loss,
                )
            else:
                self.loss_history.report_by_result(
                    actuals,
                    dout,
                    labels,
                    update_task_loss=update_task_loss,
                )

            # Pass the loss back to the user, who can decide what to
            # do now.
            result = self._coproc_finish(self.loss_history)
            should_stop, next_is_validation, user_data = result.unpack()

            self.loss_history.report_user_data(user_data)

            if user_data is not None:
                msg = user_data.msg
            else:
                msg = None

            self.loss_history.log(g_logger, msg=msg)

            if should_stop:
                break

            is_validation = next_is_validation

        return self.loss_history
