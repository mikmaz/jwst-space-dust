import matplotlib.pyplot as plt
import numpy as np
import scipy
import utils
from torch.utils.data import DataLoader
from ast import literal_eval
import torch
from tqdm import tqdm
import os
import json
from sklearn.decomposition import PCA
import pickle
import matplotlib.colors as mcolors
import seaborn as sns


def load_model_and_data(args):
    # args = utils.parse_args()
    val_dataset = utils.get_dataset_from_file(
        f'{args.data_path}/sampled_filters_val.pkl'
    )
    if args.normalize:
        val_dataset.normalize(
            utils.train_data_stats['x_stats'],
            utils.train_data_stats['y_stats'],
            utils.train_data_stats['z_stats']
        )

    val_dl = DataLoader(
        val_dataset, batch_size=128, num_workers=1
    )
    model_kwargs = literal_eval(args.model_kwargs)
    model = utils.models_dict[args.model](**model_kwargs)
    model.load_state_dict(
        torch.load(
            f'{args.stats_path}/checkpoint/best_model.pt',
            map_location=torch.device('cpu')
        )
    )
    model.eval()
    return model, val_dl


def save_stat(args, name, value):
    if os.path.isfile(f'{args.stats_path}/stats.json'):
        with open(f'{args.stats_path}/stats.json') as f:
            stats = json.load(f)
    else:
        stats = {}
    stats[name] = value
    stats_json = json.dumps(stats, indent=4)

    with open(f'{args.stats_path}/stats.json', "w") as f:
        f.write(stats_json)


def eval_mse(args, normalize_pred):
    model, val_dl = load_model_and_data(args)
    model.eval()
    mses = torch.zeros(37)
    dataset_size = val_dl.dataset.x.shape[0]
    x_mean, x_sd = utils.train_data_stats['x_stats']
    x_mean = x_mean.numpy()
    x_sd = x_sd.numpy()
    with tqdm(val_dl) as pbar:
        pbar.set_description("Evaluating MSE")
        with torch.no_grad():
            for x, y, z in pbar:
                mean, _ = model(y, z)
                if normalize_pred:
                    mean -= x_mean
                    mean /= x_sd
                mses += torch.pow(mean - x, 2).sum(dim=0)
    mean_mses = mses / dataset_size
    print(mean_mses)
    save_stat(args, 'mses', mean_mses.tolist())


def eval_sharpness(args, normalize_pred):
    model, val_dl = load_model_and_data(args)
    model.eval()
    mean_interval_lens = np.zeros((4, 37))
    biases = np.zeros((4, 37))
    confs = [0.6, 0.7, 0.8, 0.9]
    dataset_size = val_dl.dataset.x.shape[0]
    x_mean, x_sd = utils.train_data_stats['x_stats']
    x_mean = x_mean.numpy()
    x_sd = x_sd.numpy()
    with tqdm(val_dl) as pbar:
        pbar.set_description("Evaluating sharpness")
        with torch.no_grad():
            for x, y, z in pbar:
                mean, sd = model(y, z)
                mean = mean.numpy()
                sd = sd.numpy()
                if normalize_pred:
                    mean -= x_mean
                    mean /= x_sd
                    sd /= x_sd
                x_np = x.numpy()
                for i in range(len(confs)):
                    left, right = scipy.stats.norm.interval(
                        confs[i], loc=mean, scale=sd
                    )
                    mean_interval_lens[i] += (right - left).sum(axis=0)
                    biases[i] += np.logical_or(
                        x_np < left, x_np > right
                    ).sum(axis=0)
    mean_interval_lens = mean_interval_lens / dataset_size
    mean_interval_lens_all = mean_interval_lens.mean(axis=0)
    mean_interval_lens_per_conf = mean_interval_lens.mean(axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    axs[0][0].bar(
        np.arange(37),
        mean_interval_lens_all,
        tick_label=utils.filter_names,
        color='tab:blue'
    )
    axs[0][0].set_xticklabels(axs[0][0].get_xticklabels(), rotation=60,
                              ha='right')
    axs[0][0].set_xlabel('filter')
    axs[0][0].set_ylabel('mean size of 70% confidence interval')
    axs[0][0].set_title('Mean size of 70% confidence interval per filter')
    axs[0][1].bar(
        np.arange(4),
        mean_interval_lens_per_conf,
        tick_label=confs,
        color='tab:orange'
    )
    axs[0][1].set_xlabel('confidence')
    axs[0][1].set_ylabel('mean size of confidence interval')
    axs[0][1].set_title('Mean size of different confidence intervals')

    biases = biases / dataset_size
    mean_biases_per_filter = biases.mean(axis=0)
    mean_biases_per_conf = biases.mean(axis=1)
    axs[1][0].bar(
        np.arange(37),
        mean_biases_per_filter,
        tick_label=utils.filter_names,
        color='tab:blue'
    )
    axs[1][0].set_xticklabels(axs[1][0].get_xticklabels(), rotation=60,
                              ha='right')
    axs[1][0].set_xlabel('filter')
    axs[1][0].set_ylabel('ratio of elements outside 70% confidence interval')
    axs[1][0].set_title(
        "Ratio of elements outside 70% confidence interval per filter"
    )
    axs[1][1].bar(
        np.arange(4),
        mean_biases_per_conf,
        tick_label=confs,
        color='tab:orange'
    )
    axs[1][1].set_xlabel('confidence')
    axs[1][1].set_ylabel('ratio of elements outside confidence interval')
    axs[1][1].set_title(
        "Ratio of elements outside confidence intervals"
    )
    fig.tight_layout()
    fig.savefig(f'{args.stats_path}/plots/sharpness.png', format='png', dpi=400)
    plt.show()


def get_predictions(args):
    model, val_dl = load_model_and_data(args)
    means = torch.zeros(val_dl.dataset.x.shape)
    sds = torch.zeros(val_dl.dataset.x.shape)
    i = 0
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for _, y, z in pbar:
                mean, sd = model(y, z)
                batch_size = mean.shape[0]
                means[i:i + batch_size] = mean
                sds[i:i + batch_size] = sd
                i += batch_size
    return means, sds


def get_pca_plot(args):
    def plot_cmap(c, f_name, title, vmax=None):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(
            y_trans[:, 0],
            y_trans[:, 1],
            s=0.1,
            c=c,
            # alpha=0.4,
            cmap='Purples',
            vmin=c.min(),
            vmax=c.max() if vmax is None else vmax
        )
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(f'{args.stats_path}/plots/{f_name}', dpi=400)

    means, sds = get_predictions(args)
    mean_sds = sds.mean(dim=1)
    max_sds = sds.max(dim=1).values
    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    y = data['y']
    x = torch.tensor(data['X'])
    nll = -torch.distributions.Normal(means, sds).log_prob(x).sum(dim=1)
    pca = PCA(n_components=2)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    y_trans = pca.fit_transform(y)

    plot_cmap(
        mean_sds,
        'pca-vis-mean-sd.png',
        "Validation dataset projected using PCA, where coloring corresponds " +
        "to the mean value of predicted SDs"
    )
    plot_cmap(
        max_sds,
        'pca-vis-max-sd.png',
        "Validation dataset projected using PCA, where coloring corresponds " +
        "to the max value of predicted SD"
    )
    plot_cmap(
        nll,
        'pca-vis-nll.png',
        "Validation dataset projected using PCA, where coloring corresponds " +
        "to samples' NLL",
        vmax=3 * nll.mean()
    )


def top_similar_predictions(args, center, predictions=None):
    if predictions is None:
        means, sds = get_predictions(args)
    else:
        means, sds = predictions
    c_mean, c_sd = means[center], sds[center]
    kl_divs = \
        torch.log(sds / c_sd) + (c_sd ** 2 + (means - c_mean) ** 2) / (
                2 * sds ** 2) - 1 / 2
    kl_divs = kl_divs.sum(dim=1)
    kl_divs_arg_sorted = torch.argsort(kl_divs)
    torch.save(
        kl_divs_arg_sorted,
        f'{args.stats_path}/stats/{center}-kl-div-arg-sorted.pt'
    )
    return kl_divs_arg_sorted


def top_similar_predictions_plot_pca(args, centers, k=1000):
    means, sds = get_predictions(args)
    fig, ax = plt.subplots(figsize=(12, 12))
    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    y = data['y']
    # y = np.concatenate((y, data['z']), axis=1)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    y_projected = PCA(n_components=2).fit_transform(y)
    ax.scatter(
        y_projected[:, 0], y_projected[:, 1], s=0.1, c='tab:gray', alpha=0.2
    )
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, center in enumerate(centers):
        kl_divs_arg_sorted = top_similar_predictions(args, center, (means, sds))
        ax.scatter(
            y_projected[kl_divs_arg_sorted[:k], 0],
            y_projected[kl_divs_arg_sorted[:k], 1],
            c=colors[i],
            s=7
        )
        ax.scatter(
            y_projected[center, 0], y_projected[center, 1], c=colors[i],
            label=f'center={center}', s=50
        )
        ax.scatter(
            y_projected[center, 0], y_projected[center, 1], c=k, marker='x',
            s=50
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        f'{args.stats_path}/plots/top-similar-preds-pca-vis.png', dpi=400
    )


def top_similar_predictions_plot_y(args, center, k=1000):
    means, sds = get_predictions(args)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    y = data['y']
    z = data['z']
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    kl_divs_arg_sorted = top_similar_predictions(args, center, (means, sds))
    z_sorted = z[kl_divs_arg_sorted]
    z_squeezed = z_sorted.squeeze(axis=1)
    z_mask = np.abs(z_squeezed - z[center]) < 0.5 * z[center]
    for i, ax in enumerate(axs.flat):
        a, b = pairs[i]
        """
        ax.scatter(
            y[kl_divs_arg_sorted[:k], a],
            y[kl_divs_arg_sorted[:k], b],
            cmap='viridis',
        )
        """
        sns.kdeplot(
            x=y[kl_divs_arg_sorted][z_mask, a][:k],
            y=y[kl_divs_arg_sorted][z_mask, b][:k],
            fill=True,
            ax=ax
        )
        ax.set_xlabel(str(a))
        ax.set_ylabel(str(b))
        ax.set_xlim(y[:, a].min(), y[:, a].max())
        ax.set_ylim(y[:, b].min(), y[:, b].max())

    # fig.tight_layout()
    fig.suptitle(f'center={center}, k={k}')
    fig.savefig(
        f'{args.stats_path}/plots/y-mapping/{center}-top-similar-preds-with-z.png',
        dpi=400
    )


def similar_y_filter_response_stats(centers, k=1000):
    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    y = data['y']
    # y = np.concatenate((y, data['z']), axis=1)
    x = data['X']
    z = data['z']
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    z = (z - z.mean(axis=0)) / z.std(axis=0)
    # print(f"Full dataset mean of X:")
    # print(x.mean(axis=0))
    print(f"Full dataset SD of X:")
    print(x.std(axis=0), '\n')
    print(f"Full dataset SD of z:")
    print(z.std(axis=0), '\n')

    for center in centers:
        y_norm = np.sqrt(((y - y[center]) ** 2).sum(axis=1))
        y_norm_arg_sorted = np.argsort(y_norm)
        # print(f"Center's {center} mean of X of the closest k={k} points:")
        # print(x[y_norm_arg_sorted[:k]].mean(axis=0))
        print(f"Center's {center} SD of X of the closest k={k} points:")
        print(x[y_norm_arg_sorted[:k]].std(axis=0))
        print(f"Center's {center} SD of z of the closest k={k} points:")
        print(z[y_norm_arg_sorted[:k]].std(axis=0), '\n')


if __name__ == "__main__":
    top_similar_predictions_plot_y(
        utils.parse_args(),
        utils.sample_points[1]
    )
