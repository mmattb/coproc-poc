import random
import torch

MODULES = ["M1", "F5", "AIP"]


def module_id_to_idxs(num_neurons_per_module, module_id):
    if isinstance(module_id, str):
        mid = MODULES.index(module_id)
    elif not isinstance(module_id, int):
        raise TypeError("module_id must be an integer index")

    return num_neurons_per_module * mid, num_neurons_per_module * (mid + 1)


class Lesion(object):
    def lesion(self, network, pre_response):
        raise NotImplementedError()


class LesionOutputs(Lesion):
    def __init__(self, num_neurons_per_module, module_id, pct):
        self.lesion_mask = torch.ones((num_neurons_per_module * 3,))
        start_idx, end_idx = module_id_to_idxs(num_neurons_per_module, module_id)
        kill_idxs = random.sample(list(range(start_idx, end_idx)),
                int(pct * num_neurons_per_module))
        self.lesion_mask[kill_idxs] = 0.0

    def lesion(self, network, pre_response):
        batch_size = pre_response.shape[0]
        out = pre_response * torch.tile(self.lesion_mask, (batch_size, 1))
        return out

