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
DEFAULT_RECOVERY_MODE = True
DEFAULT_BATCH_SIZE = 64


def get(
    observer_type=DEFAULT_OBSERVER_TYPE,
    stimulation_type=DEFAULT_STIMULATION_TYPE,
    lesion_type=DEFAULT_LESION_TYPE,
    lesion_module_id=DEFAULT_LESION_MODULE_ID,
    lesion_pct=DEFAULT_LESION_PCT,
    activation_type=DEFAULT_ACTIVATION_TYPE,
    recovery_mode=DEFAULT_RECOVERY_MODE,
    num_neurons_per_module=DEFAULT_NUM_NEURONS_PER_MODULE,
    batch_size=DEFAULT_BATCH_SIZE,
    obs_out_dim=20,
):
    if observer_type == "passthrough":
        observer_instance = observer.ObserverPassthrough(num_neurons_per_module)
    elif observer_type == "gaussian":
        observer_instance = observer.ObserverGaussian1d(
            num_neurons_per_module, out_dim=20
        )
    else:
        raise ValueError(f"Unrecognized observer type: {observer_type}")

    if stimulation_type == "1to1":
        stimulus = stim.Stimulus1to1(
            num_neurons_per_module,
            num_neurons_per_module,
        )
    elif stimulation_type == "gaussian":
        # NOTE: can add the num_stim_channels and sigma arg above
        stimulus = stim.StimulusGaussian(
            35,
            num_neurons_per_module,
        )
    else:
        raise ValueError(f"Unrecognized stimulation type: {stimulation_type}")

    if lesion_type == "outputs":
        lesion_instance = lesion.LesionOutputs(
            num_neurons_per_module, lesion_module_id, lesion_pct
        )
    elif lesion_type == "none":
        lesion_instance = None
    else:
        raise ValueError(f"Unrecognized lesion type: {lesion_type}")

    if activation_type == "ReLU":
        activation = torch.nn.ReLU
    elif activation_type == "ReTanh":
        activation = utils.ReTanh
    else:
        raise ValueError(f"Unrecognized activation type: {activation_type}")

    recovery_str = "recovered" if recovery_mode else "notrecovered"

    run_type_str = "_".join(
        [str(observer_instance), str(lesion_instance), str(stimulus)]
    )

    return (
        observer_instance,
        stimulus,
        lesion_instance,
        activation,
        recovery_mode,
        recovery_str,
        run_type_str,
        batch_size,
    )