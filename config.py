import os

import torch
import torch.nn

import lesion
import observer
import stim
import utils


DEFAULT_OBSERVER_TYPE = "gaussian"
DEFAULT_STIMULATION_TYPE = "gaussian"
DEFAULT_LESION_TYPE = "outputs"
DEFAULT_LESION_MODULE_ID = "F5"
DEFAULT_LESION_PCT = 1.0
DEFAULT_ACTIVATION_TYPE = "ReLU"
DEFAULT_NUM_NEURONS_PER_MODULE = 100
DEFAULT_RECOVERY_MODE = False
DEFAULT_BATCH_SIZE = 64

def get_activation(activation_type):
    if activation_type == "ReLU":
        activation = torch.nn.ReLU
    elif activation_type == "ReTanh":
        activation = utils.ReTanh
    elif activation_type == "Tanh":
        activation = torch.nn.Tanh
    else:
        raise ValueError(f"Unrecognized activation type: {activation_type}")

    return activation


def get(
    observer_type=DEFAULT_OBSERVER_TYPE,
    stimulation_type=DEFAULT_STIMULATION_TYPE,
    lesion_type=DEFAULT_LESION_TYPE,
    lesion_args=(DEFAULT_LESION_MODULE_ID, DEFAULT_LESION_PCT),
    en_activation_type=DEFAULT_ACTIVATION_TYPE,
    cpn_activation_type=DEFAULT_ACTIVATION_TYPE,
    recovery_mode=DEFAULT_RECOVERY_MODE,
    num_neurons_per_module=DEFAULT_NUM_NEURONS_PER_MODULE,
    batch_size=DEFAULT_BATCH_SIZE,
    num_stim_channels=35,
    obs_out_dim=20,
):
    if observer_type == "passthrough":
        observer_instance = observer.ObserverPassthrough(num_neurons_per_module)
    elif observer_type == "gaussian":
        observer_instance = observer.ObserverGaussian1d(
            num_neurons_per_module, out_dim=obs_out_dim
        )
    else:
        raise ValueError(f"Unrecognized observer type: {observer_type}")

    if stimulation_type == "1to1":
        stimulus = stim.Stimulus1to1(
            num_neurons_per_module,
            num_neurons_per_module,
        )
    elif stimulation_type == "gaussianAlpha":
        # NOTE: can add the num_stim_channels and sigma arg above
        stimulus = stim.StimulusGaussian(
            num_stim_channels,
            num_neurons_per_module,
        )
    elif stimulation_type == "gaussianExp":
        stimulus = stim.StimulusGaussianExp(
            num_stim_channels,
            num_neurons_per_module,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unrecognized stimulation type: {stimulation_type}")

    if lesion_type == "outputs":
        lesion_instance = lesion.LesionOutputs(
            num_neurons_per_module, *lesion_args,
        )
    elif lesion_type == "connection":
        lesion_instance = lesion.LesionConnectionsByIdxs(
            num_neurons_per_module, *lesion_args,
        )

    elif lesion_type == "none":
        lesion_instance = None
    else:
        raise ValueError(f"Unrecognized lesion type: {lesion_type}")

    en_activation = get_activation(en_activation_type)
    cpn_activation = get_activation(cpn_activation_type)

    recovery_str = "recov" if recovery_mode else "norecov"

    run_type_str = "_".join(
        [str(observer_instance), str(lesion_instance), str(stimulus)]
    )

    return (
        observer_instance,
        stimulus,
        lesion_instance,
        en_activation,
        cpn_activation,
        recovery_mode,
        recovery_str,
        run_type_str,
        batch_size,
    )
