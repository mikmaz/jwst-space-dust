import torch
import argparse
from torch.utils.data import Dataset
import pickle
import models
import random
import numpy as np

models_dict = {'baseline': models.Baseline, 'resnet': models.ResNet}


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
        "--batch", default=128, type=int, help="batch size"
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

    def normalize(self, y_stats=None, z_stats=None):
        def normalize_feature(feature):
            mean = feature.mean(dim=0)
            sd = feature.std(dim=0)
            return (feature - mean) / sd, mean, sd

        if y_stats is None or z_stats is None:
            normalized_y, y_mean, y_sd = normalize_feature(self.y)
            self.y = normalized_y
            normalized_z, z_mean, z_sd = normalize_feature(self.z)
            self.z = normalized_z
            return (y_mean, y_sd), (z_mean, z_sd)
        else:
            self.y = (self.y - y_stats[0]) / y_stats[1]
            self.z = (self.z - z_stats[0]) / z_stats[1]


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
