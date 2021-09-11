import os

from torch.utils.data import Dataset
import scipy.io


BASE_DIR = os.path.join("michaels", "Networks")


defaults = ["M", "recttanh", "CNN", "1e-3", "1e-5", "1e-1", "1"]


def get_default_path():
    data_folder = os.path.join(BASE_DIR, defaults[0])
    network_dir = "-".join(defaults[1:])
    network_path = os.path.join(data_folder, os.path.join(network_dir, network_dir))
    return network_path


def load_from_path(network_path):
    network_data = scipy.io.loadmat(network_path, simplify_cells=True)
    return network_data


def load(
    monkey="M",
    activation_type="recttanh",
    input_name="CNN",
    FR="1e-3",
    IO="1e-5",
    sparsity="1e-1",
    repetition="1",
):
    """
    Args:
      - monkey: Options are M and Z
      - activation_type: recttanh or rectlinear
      - input_name: CNN, Feedforward, LabeledLine-In, LabeledLine-Out, Ball, Sparse
      - FR: 0, 1e-3, 1e-2, 1e-1
      - IO: 0, 1e-5, 1e-4, 1e-3
      - sparsity: 1e-2, 1e-1, 1e0
    """
    data_folder = os.path.join(BASE_DIR, monkey)

    network_dir = "-".join([activation_type, input_name, FR, IO, sparsity, repetition])
    network_path = os.path.join(data_folder, os.path.join(network_dir, network_dir))
    return load_from_path(network_path)
