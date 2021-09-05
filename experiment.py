import logging
import os

import torch
from torch.utils.data import DataLoader


# Local imports
import config
import michaels_load
import mRNN
import utils


# TODO: loss_history = run(co_proc, loss_history=None)
# TODO: score(loss_history)

g_logger = logging.getLogger("experiment")


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
    dataset = mRNN.MichaelsDataset(
        michaels_load.get_default_path(), cuda=cuda, with_label=True, **kwargs
    )
    return dataset


class CoProc:
    def forward(self, brain_data):
        """
        Args:
            brain_data: 4-tuple (M1 data, F5 data, AIP data, trial_end signal)
                        The first three are Tensors (batch_size, cfg.observer_instance.out_dim)
                        The last is (batch_size, 1)
                        The sum of second dimensions is in_dim.
        Returns:
            a Tensor(batch_size, stim_dim)
        """
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


def stage(coproc, cfg, cuda=None):
    return Experiment(coproc, cfg, cuda=cuda)


class Experiment:
    def __init__(self, coproc, cfg, holdout_pct=0.2):
        self._coproc = coproc

        self._cfg = cfg

        mike = mRNN.MichaelsRNN(
            init_data_path=michaels_load.get_default_path(),
            stimulus=cfg.stim_instance,
            cuda=cfg.cuda,
        )
        mike.set_lesion(cfg.lesion_instance)

        self.mike = mike

        self.dataset, self.loader_train, self.loader_test = self._get_dataset(
            holdout_pct, cfg.cuda
        )

        (
            self.comp_loss_healthy,
            self.comp_loss_lesioned,
        ) = self._get_healthy_vs_lesioned_loss(cuda=cfg.cuda)

        self.loss_history = loss_funcs.LossHistory(
            self.comp_loss_lesioned.item(), comp_loss_healthy.item()
        )

    @property
    def cfg(self):
        return self._cfg

    def _get_dataset(self, holdout_pct, cuda):
        dataset = get_raw_data(cuda=cuda)

        probs = torch.ones(len(dataset)) / float(len(dataset))
        holdout_count = int(len(dataset) * holdout_pct)
        holdout_idxs = set(
            [p.item() for p in probs.multinomial(num_samples=holdout_count)]
        )

        train = []
        test = []
        for idx in range(len(dataset)):
            if idx in holdout_idxs:
                test.append(dataset[idx])
            else:
                train.append(dataset[idx])

        # We have two loaders, since they maintain a little bit of state,
        # and we nest EN training inside CPN training
        loader_train = DataLoader(
            train,
            batch_size=len(train),
            shuffle=True,
        )
        loader_test = DataLoader(
            test,
            batch_size=len(test),
            shuffle=True,
        )

        return dataset, loader_train, loader_test

    def _get_healthy_vs_lesioned_loss(self, cuda=None):
        comp_loss = torch.nn.MSELoss()

        loader_comp = DataLoader(
            self.dataset, batch_size=len(self.dataset), shuffle=True
        )

        for s in loader_comp:
            din, trial_end, _, dout, _ = s

        comp_preds_healthy = torch.zeros(
            (len(self.dataset), self.dataset.sample_len, self.cfg.out_dim)
        )
        if cuda is not None:
            comp_preds_healthy = comp_preds_healthy.cuda(cuda)

        self.mike.set_lesion(None)
        try:
            self.mike.reset()
            for tidx in range(din.shape[1]):
                cur_din = din[:, tidx, :].T
                p = self.mike(cur_din)
                comp_preds_healthy[:, tidx, :] = p[:, :]
            comp_preds_healthy = utils.trunc_to_trial_end(comp_preds_healthy, trial_end)
            comp_loss_healthy = comp_loss(comp_preds_healthy, dout)

            comp_preds_lesioned = torch.zeros(comp_preds_healthy.shape)
            if cuda is not None:
                comp_preds_lesioned = comp_preds_lesioned.cuda(cuda)
        finally:
            self.mike.set_lesion(self.cfg.lesion_instance)

        self.mike.reset()
        for tidx in range(din.shape[1]):
            cur_din = din[:, tidx, :].T
            p = self.mike(cur_din)
            comp_preds_lesioned[:, tidx, :] = p[:, :]
        comp_preds_lesioned = utils.trunc_to_trial_end(comp_preds_lesioned, trial_end)
        comp_loss_lesioned = comp_loss(comp_preds_lesioned, dout)

        return comp_loss_healthy, comp_loss_lesioned

    @property
    def coproc(self):
        return self._coproc

    def forward(self, brain_data):
        stim = self.coproc.forward(brain_data)

        # Un-pythonic to assert a type, but let's be strict here...
        if not isinstance(stim, torch.Tensor):
            raise TypeError("Stim vector must be a Tensor")

        expected_stim_shape = (brain_data.shape[0], self._cfg.stim_instance.out_dim)
        if stim.shape != expected_stim_shape:
            raise ValueError(
                f"Expected stim vector to have shape {expected_stim_shape}"
            )

        return stim

    def run(self):
        while True:
            for batch in self.loader_train:
                din, trial_end, trial_len, dout, labels = batch
                batch_size = din.shape[0]
                steps = din.shape[1]

