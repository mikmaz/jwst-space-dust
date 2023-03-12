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
from scipy.stats import chi2


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
        val_dataset, batch_size=128, num_workers=1, shuffle=False
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


def get_predictions_cov(args):
    model, val_dl = load_model_and_data(args)
    dataset_size = val_dl.dataset.x.shape[0]
    k = val_dl.dataset.x.shape[1]
    means = torch.zeros(val_dl.dataset.x.shape)
    l_tris = torch.zeros((dataset_size, k, k))
    ds = torch.zeros((dataset_size, k))
    i = 0
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for _, y, z in pbar:
                mean, l_tri, d = model(y, z)
                batch_size = mean.shape[0]
                means[i:i + batch_size] = mean
                l_tris[i:i + batch_size] = l_tri
                ds[i:i + batch_size] = d
                i += batch_size
    return means, l_tris, ds


def generate_and_save_predictions(args):
    res_dict = {}
    if args.covariance:
        means, l_tris, ds = get_predictions_cov(args)
        res_dict['mean'] = means
        res_dict['l_tris'] = l_tris
        res_dict['ds'] = ds
    else:
        means, sds = get_predictions(args)
        res_dict['mean'] = means
        res_dict['sd'] = sds
    torch.save(res_dict, f'{args.stats_path}/stats/val-predictions.pt')


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
            label=f'center index: {center}', s=50
        )
        ax.scatter(
            y_projected[center, 0], y_projected[center, 1], c=k, marker='x',
            s=50
        )
    ax.legend()
    ax.set_xlabel('first principal component')
    ax.set_ylabel('second principal component')
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


def similar_y_filter_response_stats(centers, path, k=500):
    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    y = data['y']
    x = data['X']
    z = np.log(data['z'])
    y = np.concatenate((y, z), axis=1)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    # z = (z - z.mean(axis=0)) / z.std(axis=0)
    # print(f"Full dataset mean of X:")
    # print(x.mean(axis=0))
    print(f"Full dataset SD of X:")
    print(x.std(axis=0), '\n')
    print(f"Full dataset SD of z:")
    print(z.std(axis=0), '\n')
    bins = np.arange(x.shape[1])
    for center in centers:
        y_norm = np.sqrt(((y - y[center]) ** 2).sum(axis=1))
        y_norm_arg_sorted = np.argsort(y_norm)
        # print(f"Center's {center} mean of X of the closest k={k} points:")
        # print(x[y_norm_arg_sorted[:k]].mean(axis=0))
        x_top_k = x[y_norm_arg_sorted[:k]]
        fig, ax = plt.subplots(figsize=(10, 5))
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        ax.plot(bins, x_mean, color='tab:blue', label='dataset mean')
        ax.fill_between(bins, x_mean - x_std, x_mean + x_std, color='tab:blue', alpha=0.3, label='dataset SD')
        ax.errorbar(
            bins,
            x_top_k.mean(axis=0),
            yerr=x_top_k.std(axis=0),
            fmt='o',
            ecolor='tab:red',
            color='tab:orange',
            label="closest points' mean and SD",
            ms=5
        )
        ax.set_xticks(np.arange(len(utils.filter_names)), utils.filter_names)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
        ax.set_xlabel('filter')
        ax.set_ylabel('magnitude')
        ax.legend()
        ax.set_title(f'Statistics of closest k={k} points')
        fig.tight_layout()
        fig.savefig(f'{path}/seed-{center}/k-{k}.png', dpi=400)

        print(f"Center's {center} SD of X of the closest k={k} points:")
        print(x[y_norm_arg_sorted[:k]].std(axis=0))
        print(f"Center's {center} SD of z of the closest k={k} points:")
        print(z[y_norm_arg_sorted[:k]].std(axis=0), '\n')


def calibration_plot(args, stats, x):
    accuracies = []
    thresholds = [i * 0.1 for i in range(1, 10)]
    if args.covariance:
        mean, l_tri, d = stats['mean'], stats['l_tris'], stats['ds']
        df = l_tri.shape[1]
        cov_inv = torch.bmm(
            torch.bmm(l_tri.transpose(1, 2), torch.diag_embed(d)), l_tri
        )
        region = torch.bmm(
            torch.bmm((x - mean).unsqueeze(1), cov_inv),
            (x - mean).unsqueeze(2)
        ).flatten()
    else:
        mean, sd = stats['mean'], stats['sd']
    for conf in thresholds:
        if args.covariance:
            accuracies.append((region <= chi2.ppf(conf, df)).float().mean())
        else:
            left, right = scipy.stats.norm.interval(conf, loc=mean, scale=sd)
            accuracies.append(
                np.logical_and(
                    x >= torch.tensor(left), x <= torch.tensor(right)
                ).float().mean()
            )
    err_bottom = []
    err_height = []
    for i in range(len(thresholds)):
        err = accuracies[i] - thresholds[i]
        err_bottom.append(thresholds[i] if err > 0 else accuracies[i])
        err_height.append(abs(err))
    fig, ax = plt.subplots()
    ax.bar(thresholds, accuracies, width=0.1, edgecolor='black',
           color='tab:blue', label='measurements')
    ax.bar(thresholds, err_height, bottom=err_bottom, width=0.1,
           color='tab:red', alpha=0.5,
           edgecolor='tab:red', label='calibration gap')
    ax.plot(thresholds, thresholds, c='black', linestyle='--')
    ax.set_xlabel('confidence')
    ax.set_ylabel('accuracy')
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.set_title('Reliability diagram')
    fig.tight_layout()
    fig.savefig(f'{args.stats_path}/plots/reliability-diagram.png', dpi=400)


def eval_sharpness(args, mean, sd, x, suptitle=None):
    mean_interval_lens = np.zeros((4, 37))
    accuracies = np.zeros((4, 37))
    confs = [0.6, 0.7, 0.8, 0.9]
    # dataset_size = val_dl.dataset.x.shape[0]
    for i in range(len(confs)):
        left, right = scipy.stats.norm.interval(confs[i], loc=mean, scale=sd)
        accuracies[i] = np.logical_and(
            x >= torch.tensor(left), x <= torch.tensor(right)
        ).float().mean(axis=0)
        mean_interval_lens[i] = (right - left).mean(axis=0)
    mean_interval_lens_all = mean_interval_lens[1]
    mean_interval_lens_per_conf = mean_interval_lens.mean(axis=1)
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    axs[0][0].bar(
        np.arange(37),
        mean_interval_lens_all,
        tick_label=utils.filter_names,
        color='tab:blue'
    )
    axs[0][0].set_xticklabels(axs[0][0].get_xticklabels(), rotation=60)
    axs[0][0].set_xlabel('filter')
    axs[0][0].set_ylabel('mean width')
    axs[0][0].set_title('Mean width of 70% confidence interval per filter')
    axs[0][1].bar(
        np.arange(4),
        mean_interval_lens_per_conf,
        tick_label=confs,
        color='tab:orange'
    )
    axs[0][1].set_xlabel('confidence')
    axs[0][1].set_ylabel('mean width')
    axs[0][1].set_title('Mean width of different confidence intervals')

    mean_accuracies_per_filter = accuracies[1]
    mean_accuracies_per_conf = accuracies.mean(axis=1)
    axs[1][0].bar(
        np.arange(37),
        mean_accuracies_per_filter,
        tick_label=utils.filter_names,
        color='tab:blue'
    )
    axs[1][0].set_xticklabels(axs[1][0].get_xticklabels(), rotation=60)
    axs[1][0].set_xlabel('filter')
    axs[1][0].set_ylabel('accuracy')
    axs[1][0].set_title(
        "Ratio of elements inside 70% confidence interval per filter"
    )
    axs[1][1].bar(
        np.arange(4),
        mean_accuracies_per_conf,
        tick_label=confs,
        color='tab:orange'
    )
    axs[1][1].set_xlabel('confidence')
    axs[1][1].set_ylabel('accuracy')
    axs[1][1].set_title(
        "Ratio of elements inside confidence intervals"
    )
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f'{args.stats_path}/plots/sharpness.png', format='png', dpi=400)
    plt.show()


def plot_eigen_stats(explained_var, eig_vectors, path, f_name):
    bins = np.arange(1, explained_var.shape[0] + 1)
    fig1, ax1 = plt.subplots()
    ax1.bar(
        bins, explained_var, alpha=0.5, label='Individual explained variance'
    )
    ax1.plot(
        bins, np.cumsum(explained_var), label='Cumulative explained variance'
    )
    ax1.legend()
    ax1.set_xlabel('principal component index')
    ax1.set_ylabel('explained variance ratio')
    ax1.set_title(
        "Explained variance of validation sample"
    )
    fig1.tight_layout()
    fig1.savefig(f'{path}/{f_name}-explained-variance.png', dpi=400)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i in range(3):
        ax2.plot(eig_vectors[i], label=f'PC{i + 1}')
    ax2.set_xticks(bins - 1, utils.filter_names)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=60)
    ax2.set_xlabel("coefficient index (labeled with filters' names)")
    ax2.set_ylabel("coefficient value")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f'{path}/{f_name}-principal-components.png', dpi=400)


def plot_eigen_stats_dataset(x, path, f_name):
    pca = PCA()
    pca.fit(x)
    eig_vals = pca.explained_variance_
    eig_vectors = pca.components_
    explained_var = eig_vals / eig_vals.sum()
    plot_eigen_stats(explained_var, eig_vectors, path, f_name)


def plot_eigen_stats_cov(cov, stats_path, f_name):
    eig_vals, eig_vectors = np.linalg.eig(cov)
    sorted_args = np.argsort(-eig_vals)
    eig_vals, eig_vectors = eig_vals[sorted_args], eig_vectors[sorted_args]
    print(eig_vals)
    explained_var = eig_vals / eig_vals.sum()
    path = stats_path + '/plots/eigendecomposition'
    plot_eigen_stats(explained_var, eig_vectors, path, f_name)


def plot_cov(cov, stats_path, f_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cov,
        ax=ax,
        xticklabels=utils.filter_names,
        yticklabels=utils.filter_names
    )
    print(cov)
    ax.set_ylabel('filter')
    ax.set_xlabel('filter')
    ax.set_title(f'Covariance uncertainty matrix of {f_name}')
    fig.tight_layout()
    path = stats_path + '/plots/eigendecomposition'
    fig.savefig(f'{path}/sample-{f_name}-uncertainty-covariance.png')


def data_err_bar(means, sds, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        np.arange(means.shape[0]),
        means,
        yerr=sds,
        fmt='o',
        ecolor='tab:red',
        color='tab:orange',
        label='mean and standard deviation',
        ms=5
    )
    ax.set_xticks(np.arange(len(utils.filter_names)), utils.filter_names)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    ax.set_xlabel('filter')
    ax.set_ylabel('magnitude')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=400)


if __name__ == "__main__":

    similar_y_filter_response_stats(
        utils.sample_points,
        'general-figures/missing-clumpiness-variation',
        100000
    )
    """
    # preds = torch.load(f'{args.stats_path}/stats/val-predictions.pt')

    args = utils.parse_args()

    with open('../data/sampled_filters_val.pkl', 'rb') as f:
        data = pickle.load(f)
    data_err_bar(
        data['X'].mean(axis=0),
        data['X'].std(axis=0),
        'experiments/val-dataset-x-stats.png',
        title='Mean and SD of validation dataset before standardization'
    )
    
    x = (
                torch.tensor(data['X']) - utils.train_data_stats['x_stats'][0]
        ) / utils.train_data_stats['x_stats'][1]
    # x = x.numpy()
    # plot_eigen_stats(None, x=x)
    
    plot_eigen_stats_dataset(x, './general-figures', 'val-dataset')

    # preds = torch.load(f'{args.stats_path}/stats/val-predictions.pt')
    # covs = utils.reconstruct_cov(preds['l_tris'], preds['ds'])
    """
    """
    for sample in utils.sample_points:
        plot_cov(covs[sample], args.stats_path, sample)
    """
    """
    for sample in utils.sample_points:
        plot_eigen_stats_cov(
            covs[sample],
            f'{args.stats_path}',
            'sample-' + str(sample)
        )
    """
    """
    mean = (preds['mean'] - utils.train_data_stats['x_stats'][0]) / \
           utils.train_data_stats['x_stats'][1]
    sd = preds['sd'] / utils.train_data_stats['x_stats'][1]
    preds['mean'] = mean
    preds['sd'] = sd
    # eval_sharpness(args, mean, sd, x)
    calibration_plot(args, preds, x)
    """

    # calibration_plot(args, preds, x)
    """
    covs = utils.reconstruct_cov(preds['l_tris'], preds['ds'])
    # plot_eigen_stats(covs[1000])
    sd = torch.sqrt(torch.diagonal(covs, dim1=1, dim2=2))
    eval_sharpness(args, preds['mean'], sd, x)
    """

    """
    eigenvecs = torch.linalg.eig(covs[100])[1]
    print(torch.linalg.eig(covs[100])[0])
    plt.plot(eigenvecs[:, 0].numpy())
    plt.plot(eigenvecs[:, 1].numpy())
    plt.plot(eigenvecs[:, 2].numpy())
    plt.show()
    """
    # print(torch.linalg.eigvals(covs[100]).float())

    # generate_and_save_predictions(utils.parse_args())
