import logging
import typing

import attr
import torch
import torch.nn
from torch.utils.data import DataLoader

from . import activation
from . import lesion
from . import mRNN
from . import michaels_load
from . import observer
from . import stim


LOG_FORMAT = "%(asctime)s %(message)s"
LOG_DATEFMT = "%m-%d %H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT, level=logging.INFO)


@attr.s(auto_attribs=True, frozen=True)
class Config:
    observer_instance: observer.Observer
    stim_instance: stim.Stimulus
    lesion_instance: lesion.Lesion
    en_activation: torch.nn.Module
    cpn_activation: torch.nn.Module
    cfg_str: str
    cfg_str_short: str
    out_dim: int
    dataset: torch.utils.data.Dataset
    loader_train: torch.utils.data.DataLoader
    loader_test: torch.utils.data.DataLoader
    cuda: typing.Any

    def unpack(self):
        # 3x due to 3 modules in the mRNN
        # +1 for trial_end
        return (
            self.in_dim,
            self.stim_dim,
            self.out_dim,
            self.cuda,
        )

    @property
    def stim_dim(self):
        return self.stim_instance.out_dim

    @property
    def in_dim(self):
        return 3 * self.observer_instance.out_dim + 1

    @property
    def trial_length(self):
        return self.dataset.sample_len


DEFAULT_OBSERVER_TYPE = observer.ObserverType.gaussian
DEFAULT_STIMULATION_TYPE = stim.StimulationType.gaussian_exp
DEFAULT_LESION_TYPE = lesion.LesionType.connection
DEFAULT_LESION_ARGS = (
    [
        # No F5->M1
        (1, 2, 0, 1),
        # No M1->F5
        (0, 1, 1, 2),
    ],
)
DEFAULT_ACTIVATION_TYPE = activation.ActivationType.Tanh
DEFAULT_NUM_NEURONS_PER_MODULE = 100
DEFAULT_OBS_OUT_DIM = 20
DEFAULT_OBS_SIGMA = 1.75
DEFAULT_NUM_STIM_CHANNELS = 16
DEFAULT_STIM_SIGMA = 2.175
DEFAULT_OUT_DIM = 50
DEFAULT_HOLDOUT_PCT = 0.2


def get_raw_data(cuda=None, **kwargs):
    dataset = mRNN.MichaelsDataset(
        michaels_load.get_default_path(), cuda=cuda, with_label=True, **kwargs
    )
    return dataset


def _get_dataset(holdout_pct=0.2, cuda=None):
    dataset = get_raw_data(cuda=cuda)

    probs = torch.ones(len(dataset)) / float(len(dataset))
    holdout_count = int(len(dataset) * holdout_pct)
    holdout_idxs = set([p.item() for p in probs.multinomial(num_samples=holdout_count)])

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


# TODO: At some point cfg should be kept in e.g. JSON, and
#  we should provide an interface to pass a path in. Also:
#  a path to tweak these.


def get_default(cuda=None):
    cfg = get(cuda=cuda)
    return cfg


def get(
    observer_type=DEFAULT_OBSERVER_TYPE,
    stimulation_type=DEFAULT_STIMULATION_TYPE,
    lesion_type=DEFAULT_LESION_TYPE,
    lesion_args=DEFAULT_LESION_ARGS,
    en_activation_type=DEFAULT_ACTIVATION_TYPE,
    cpn_activation_type=DEFAULT_ACTIVATION_TYPE,
    num_neurons_per_module=DEFAULT_NUM_NEURONS_PER_MODULE,
    num_stim_channels=DEFAULT_NUM_STIM_CHANNELS,
    stim_sigma=DEFAULT_STIM_SIGMA,
    stim_retain_grad=False,
    obs_out_dim=DEFAULT_OBS_OUT_DIM,
    obs_sigma=DEFAULT_OBS_SIGMA,
    out_dim=DEFAULT_OUT_DIM,
    holdout_pct=DEFAULT_HOLDOUT_PCT,
    cuda=None,
):

    if observer_type is observer.ObserverType.passthrough:
        observer_instance = observer_type.value(num_neurons_per_module)
    elif observer_type is observer.ObserverType.gaussian:
        observer_instance = observer_type.value(
            num_neurons_per_module, out_dim=obs_out_dim, sigma=obs_sigma, cuda=cuda
        )
    else:
        raise ValueError(f"Unrecognized observer type: {observer_type}")

    if stimulation_type is stim.StimulationType.one_to_one:
        stimulus = stimulation_type.value(
            num_neurons_per_module,
            num_neurons_per_module,
        )
    elif stimulation_type is stim.StimulationType.gaussian_alpha:
        if cuda:
            raise NotImplementedError()

        # NOTE: can add the num_stim_channels and sigma arg above
        stimulus = stimulation_type.value(
            num_stim_channels,
            num_neurons_per_module,
            sigma=stim_sigma,
        )
    elif stimulation_type is stim.StimulationType.gaussian_exp:
        stimulus = stimulation_type.value(
            num_stim_channels,
            num_neurons_per_module,
            batch_size=1,  # Will be reset before use
            sigma=stim_sigma,
            retain_grad=stim_retain_grad,
            cuda=cuda,
        )
    else:
        raise ValueError(f"Unrecognized stimulation type: {stimulation_type}")

    if lesion_type is lesion.LesionType.outputs:
        if cuda:
            raise NotImplementedError()

        lesion_instance = lesion_type.value(
            num_neurons_per_module,
            *lesion_args,
        )
    elif lesion_type is lesion.LesionType.connection:
        lesion_instance = lesion_type.value(
            num_neurons_per_module, *lesion_args, cuda=cuda
        )

    elif lesion_type is lesion.LesionType.none:
        lesion_instance = None
    else:
        raise ValueError(f"Unrecognized lesion type: {lesion_type}")

    en_activation = en_activation_type.value
    cpn_activation = cpn_activation_type.value

    cfg_str = "_".join(
        [
            str(observer_instance),
            str(lesion_instance),
            str(stimulus),
            f"enAct{en_activation_type.name}",
            f"cpnAct{cpn_activation_type.name}",
        ]
    )

    cfg_str_short = "_".join(
        [
            f"obs{observer_type.name}{obs_out_dim}",
            f"lesion{lesion_type.name}",
            f"stim{stimulation_type.name}{num_stim_channels}",
            f"enAct{en_activation_type.name}",
            f"cpnAct{cpn_activation_type.name}",
        ]
    )

    dataset, loader_train, loader_test = _get_dataset(
        holdout_pct=holdout_pct, cuda=cuda
    )

    cfg_out = Config(
        observer_instance,
        stimulus,
        lesion_instance,
        en_activation,
        cpn_activation,
        cfg_str,
        cfg_str_short,
        out_dim,
        dataset,
        loader_train,
        loader_test,
        cuda,
    )
    return cfg_out
