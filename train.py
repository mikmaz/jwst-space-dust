import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import utils
from ast import literal_eval
import torch.nn.functional as F


def loss_f(mean, sd, x):
    return -torch.distributions.Normal(mean, sd).log_prob(x).sum(dim=1).mean()


def evaluate(model, dl, device):
    val_losses = []
    val_mses = []
    model.eval()
    with torch.no_grad():
        for x, y, z in dl:
            x = x.to(device=device)
            y = y.to(device=device)
            z = z.to(device=device)
            mean, sd = model(y, z)
            loss = loss_f(mean, sd, x)
            mse = F.mse_loss(mean, x)
            val_mses.append(mse.item())
            val_losses.append(loss.item())
    mean_loss = sum(val_losses[:-1]) / len(val_losses[:-1])
    mean_mse = sum(val_mses[:-1]) / len(val_mses[:-1])
    return mean_loss, mean_mse


def train(
        model,
        train_dl,
        val_dl,
        optimizer,
        n_epochs,
        device,
        stats_path
):
    losses = []
    mses = []
    with open(f'{stats_path}/train_losses.txt', 'w+') as f:
        f.write("Training losses:\n")
    with open(f'{stats_path}/val_losses.txt', 'w+') as f:
        f.write("Validation losses:\n")
    with open(f'{stats_path}/train_mses.txt', 'w+') as f:
        f.write("Training MSEs:\n")
    with open(f'{stats_path}/val_mses.txt', 'w+') as f:
        f.write("Validation MSEs:\n")
    best_val_loss = None
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            epoch_mses = []
            for x, y, z in train_dl:
                model.train()
                optimizer.zero_grad()
                x = x.to(device=device)
                y = y.to(device=device)
                z = z.to(device=device)
                mean, sd = model(y, z)
                loss = loss_f(mean, sd, x)
                with torch.no_grad():
                    mse = F.mse_loss(mean, x)
                    epoch_mses.append(mse.item())
                epoch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if (epoch + 1) % max(1, (n_epochs // 5)) == 0:
                torch.save(
                    model.state_dict(),
                    f"{stats_path}/checkpoint/model_{epoch}.pt"
                )
                torch.save(
                    optimizer.state_dict(),
                    f"{stats_path}/checkpoint/optim_{epoch}.pt"
                )

            print(f'Epoch: {epoch}, train loss: ' +
                  f'{sum(epoch_losses[:-1]) / len(epoch_losses[:-1])}')
            with open(f'{stats_path}/train_losses.txt', 'a+') as f:
                f.write(f'{sum(epoch_losses[:-1]) / len(epoch_losses[:-1])}\n')
            with open(f'{stats_path}/train_mses.txt', 'a+') as f:
                f.write(f'{sum(epoch_mses[:-1]) / len(epoch_mses[:-1])}\n')

            val_loss, val_mse = evaluate(model, val_dl, device)
            with open(f'{stats_path}/val_losses.txt', 'a+') as f:
                f.write(f'{val_loss}\n')
            with open(f'{stats_path}/val_mses.txt', 'a+') as f:
                f.write(f'{val_mse}\n')
            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"{stats_path}/checkpoint/best_model.pt"
                )
            losses.append(sum(epoch_losses) / len(epoch_losses))

    return model, losses


def main(args):
    print(args)
    device = utils.get_device()
    print("Device:", device)
    utils.enforce_reproducibility()

    train_dataset = utils.get_dataset_from_file(
        f'{args.data_path}/sampled_filters_train.pkl'
    )
    val_dataset = utils.get_dataset_from_file(
        f'{args.data_path}/sampled_filters_val.pkl'
    )
    if args.normalize:
        x_stats, y_stats, z_stats = train_dataset.normalize()
        val_dataset.normalize(x_stats, y_stats, z_stats)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers=args.n_workers,
        shuffle=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=args.batch, num_workers=args.n_workers
    )

    model_kwargs = literal_eval(args.model_kwargs)
    model = utils.models_dict[args.model](**model_kwargs).to(device)

    lr = args.lr
    optimizer = Adam(model.parameters(), lr=lr)

    epochs = args.n_epochs
    model, losses = train(
        model,
        train_dl,
        val_dl,
        optimizer,
        epochs,
        device,
        args.stats_path
    )
    torch.save(
        model.state_dict(), f"{args.stats_path}/checkpoint/final_model.pt"
    )


if __name__ == "__main__":
    main(utils.parse_args())
