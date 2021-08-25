import enum
import logging
import sys

LOG_FORMAT = "%(asctime)s %(message)s"
LOG_DATEFMT = "%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATEFMT)


import torch
import torch.nn

import activation
import lesion
import observer
import stim
import utils

DEFAULT_OBSERVER_TYPE = observer.ObserverType.gaussian
DEFAULT_STIMULATION_TYPE = stim.StimulationType.gaussian_exp
DEFAULT_LESION_TYPE = lesion.LesionType.connection
DEFAULT_LESION_MODULE_ID = lesion.LesionModule.M1
DEFAULT_LESION_PCT = 1.0
DEFAULT_ACTIVATION_TYPE = activation.ActivationType.Tanh
DEFAULT_NUM_NEURONS_PER_MODULE = 100
DEFAULT_BATCH_SIZE = 64


def get(
    observer_type=DEFAULT_OBSERVER_TYPE,
    stimulation_type=DEFAULT_STIMULATION_TYPE,
    lesion_type=DEFAULT_LESION_TYPE,
    lesion_args=(DEFAULT_LESION_MODULE_ID, DEFAULT_LESION_PCT),
    en_activation_type=DEFAULT_ACTIVATION_TYPE,
    cpn_activation_type=DEFAULT_ACTIVATION_TYPE,
    num_neurons_per_module=DEFAULT_NUM_NEURONS_PER_MODULE,
    batch_size=DEFAULT_BATCH_SIZE,
    num_stim_channels=35,
    stim_sigma=1,
    stim_retain_grad=False,
    obs_out_dim=20,
    obs_sigma=1.75,
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
            batch_size=batch_size,
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

    run_type_str = "_".join(
        [
            str(observer_instance),
            str(lesion_instance),
            str(stimulus),
            f"enAct{en_activation_type.name}",
            f"cpnAct{cpn_activation_type.name}",
        ]
    )

    run_type_str_short = "_".join(
        [
            f"obs{observer_type.name}{obs_out_dim}",
            f"lesion{lesion_type.name}",
            f"stim{stimulation_type.name}{num_stim_channels}",
            f"enAct{en_activation_type.name}",
            f"cpnAct{cpn_activation_type.name}",
        ]
    )

    return (
        observer_instance,
        stimulus,
        lesion_instance,
        en_activation,
        cpn_activation,
        run_type_str,
        run_type_str_short,
        batch_size,
    )
