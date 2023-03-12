import torch
import argparse
from torch.utils.data import Dataset
import pickle
import models
import random
import numpy as np

import models_old

models_dict = {
    'baseline': models.Baseline,
    'resnet_sd': models.ResNetSd,
    'baseline_plus': models.BaselinePlus,
    'resnet_cov': models.ResNetCov,
    'resnet_old': models_old.ResNet
}

train_data_stats = {
    'x_stats': (
        torch.tensor([
            20.2991, 20.3594, 20.5216, 20.6499, 20.6884, 20.7566, 56.7391,
            20.8051, 20.8188, 20.8106, 20.8311, 20.8382, 20.8223, 20.7754,
            20.7643, 20.6633, 40.8093, 20.7137, 20.6758, 20.6740, 20.6480,
            20.6355, 20.6321, 20.6082, 20.6096, 50.4091, 24.4606, 20.6052,
            20.5891, 20.5126, 20.3633, 20.5775, 20.8624, 21.1913, 21.3462,
            21.6417, 22.2701
        ]),
        torch.tensor([
            2.5650, 2.5722, 2.5750, 2.5867, 2.5721, 2.5775, 9.5015, 2.5714,
            2.5902, 2.5719, 2.5873, 2.6007, 2.6480, 2.6817, 2.7239, 2.7723,
            18.4062, 2.7917, 2.8240, 2.8472, 2.9029, 2.9138, 2.9358, 2.9485,
            2.9849, 16.7919, 12.1996, 3.0012, 3.0786, 3.2332, 3.3696, 3.4180,
            3.3891, 3.4233, 3.5180, 3.5676, 3.9185
        ])
    ),
    'y_stats': (
        torch.tensor([-3.5011e+00, -8.0059e-01, 4.9956e-01, 1.0195e+03]),
        torch.tensor([1.4426e+00, 8.6663e-01, 2.8851e-01, 4.2714e+02])
    ),
    'z_stats': (torch.tensor([0.0075]), torch.tensor([0.0043]))
}

filter_names = [
    70, 90, 115, 140, 150, 162, 164, 182, 187, 200, 210, 212, 250, 277, 300,
    322, 323, 335, 356, 360, 405, 410, 430, 444, 460, 466, 470, 480, 560, 770,
    1000, 1130, 1280, 1500, 1800, 2100, 2550
]

sample_points = [215768, 407206, 566926, 761167, 433044]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # TODO change
        return torch.device('cpu')
    else:
        return torch.device('cpu')


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@', description="JWST space dust trainer"
    )
    parser.add_argument(
        "--model", default='baseline', type=str, help="which model to train"
    )
    parser.add_argument(
        "--model_kwargs",
        default='{}',
        type=str,
        help="model's keyword arguments passed as a dictionary"
    )
    parser.add_argument(
        "--batch", default=64, type=int, help="batch size"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--val_frac",
        default=0.2,
        type=float,
        help="fraction of the dataset to be left as validation set"
    )
    parser.add_argument(
        "--n_epochs", default=100, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--n_workers",
        default=8,
        type=int,
        help="number of workers used in dataloader"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="if set, datasets will be normalized"
    )
    parser.add_argument(
        "--covariance",
        action="store_true",
        help="if set, training routine will assume model predicting covariance"
    )
    parser.add_argument('data_path', type=str, help="path to ECGs' directory")
    parser.add_argument('stats_path', type=str)
    parsed_args = parser.parse_args()
    # print(f'@{parsed_args.stats_path}/args.txt')
    """
    parsed_args = parser.parse_args(
        f'@./local-test-experiment/args.txt', namespace=parsed_args
    )
    """
    return parsed_args


class JWSTSpaceDustDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

    def normalize(self, x_stats=None, y_stats=None, z_stats=None):
        def normalize_feature(feature, stats):
            if stats is None:
                mean = feature.mean(dim=0)
                sd = feature.std(dim=0)
            else:
                mean, sd = stats
            return (feature - mean) / sd, mean, sd

        self.x, x_mean, x_sd = normalize_feature(self.x, x_stats)
        self.y, y_mean, y_sd = normalize_feature(self.y, y_stats)
        self.z, z_mean, z_sd = normalize_feature(self.z, z_stats)
        return (x_mean, x_sd), (y_mean, y_sd), (z_mean, z_sd)


def get_datasets_split(args):
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    x = torch.tensor(data['X'])
    y = torch.tensor(data['y'])
    z = torch.tensor(data['z'])

    data_size = x.shape[0]
    perm = torch.randperm(data_size)
    val_dataset_size = int(data_size * args.val_frac)
    train_idxs = perm[val_dataset_size:]
    val_idxs = perm[:val_dataset_size]

    train_dataset = JWSTSpaceDustDataset(
        x[train_idxs], y[train_idxs], z[train_idxs]
    )
    val_dataset = JWSTSpaceDustDataset(
        x[val_idxs], y[val_idxs], z[val_idxs]
    )

    return train_dataset, val_dataset


def get_dataset_from_file(f_path):
    with open(f_path, 'rb') as f:
        data = pickle.load(f)
    x = torch.tensor(data['X'])
    y = torch.tensor(data['y'])
    z = torch.tensor(data['z'])
    return JWSTSpaceDustDataset(x, y, z)


def fixed_train_val_split(f_path_in, dir_out, val_frac):
    def save_indexes(indexes, f_name):
        data_copy = dict()
        data_copy['wavelengths'] = data['wavelengths'][indexes]
        data_copy['spectra'] = data['spectra'][indexes]
        data_copy['X'] = data['X'][indexes]
        data_copy['y'] = data['y'][indexes]
        data_copy['z'] = data['z'][indexes]
        data_copy['zmin'] = data['zmin']
        data_copy['zmax'] = data['zmax']
        data_copy['filter_names'] = data['filter_names']
        with open(f'{dir_out}/{f_name}', "wb") as f_out:
            pickle.dump(data_copy, f_out)

    enforce_reproducibility()
    with open(f_path_in, 'rb') as f_in:
        data = pickle.load(f_in)

    data_size = data['wavelengths'].shape[0]
    val_dataset_size = int(data_size * val_frac)
    perm = torch.randperm(data_size)
    train_indexes = perm[val_dataset_size:]
    val_indexes = perm[:val_dataset_size]
    save_indexes(train_indexes, 'sampled_filters_train.pkl')
    save_indexes(val_indexes, 'sampled_filters_val.pkl')


def reconstruct_precision(l_tri_inv, d_inv):
    return torch.bmm(
        torch.bmm(l_tri_inv.transpose(1, 2), torch.diag_embed(d_inv)), l_tri_inv
    )

def reconstruct_cov(l_tri_inv, d_inv):
    d = torch.reciprocal(d_inv)
    l_tri = torch.linalg.solve_triangular(l_tri_inv, torch.eye(d.shape[1]), upper=False)
    return torch.bmm(
        torch.bmm(l_tri, torch.diag_embed(d)), l_tri.transpose(1, 2)
    )
