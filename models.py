import torch
from torch import nn


class Baseline(nn.Module):
    def __init__(
            self,
            in_features=5,
            out_features=37,
            n_hidden=37,
            min_sd=0.1,
            dropout=False
    ):
        super(Baseline, self).__init__()
        n_hidden = 2 * n_hidden if dropout else n_hidden
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout() if dropout else nn.Identity(),
        )
        self.mean_net = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(n_hidden, out_features),
        )
        self.sd_net = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout() if dropout else nn.Identity(),
            nn.Linear(n_hidden, out_features),
            nn.Softplus(threshold=20),
        )
        self.min_sd = min_sd
        self.dropout = dropout

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x = self.shared_net(x)
        return self.mean_net(x), self.sd_net(x) + self.min_sd


class BaselinePlus(nn.Module):
    def __init__(
            self,
            in_features=5,
            out_features=37,
            n_hidden=37,
            min_sd=0.1,
            dropout=False,
            n_hidden_layers=1
    ):
        super(BaselinePlus, self).__init__()
        n_hidden = 2 * n_hidden if dropout else n_hidden
        self.shared_net = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Dropout() if dropout else nn.Identity(),
        )
        self.mean_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout() if dropout else nn.Identity(),
            )
            for _ in range(n_hidden_layers)
        ])
        self.mean_layers.append(nn.Linear(n_hidden, out_features))
        self.sd_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout() if dropout else nn.Identity(),
            )
            for _ in range(n_hidden_layers)
        ])
        self.sd_layers.append(
            nn.Sequential(
                nn.Linear(n_hidden, out_features),
                nn.Softplus(threshold=20)
            )
        )
        self.min_sd = min_sd
        self.dropout = dropout
        self.n_hidden_layers = n_hidden_layers

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x_mean = self.shared_net(x)
        x_sd = x_mean
        for mean_layer, sd_layer in zip(self.mean_layers, self.sd_layers):
            x_mean = mean_layer(x_mean)
            x_sd = sd_layer(x_sd)
        return x_mean, x_sd + self.min_sd


class BasicBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, downsampler=None):
        super(BasicBlock, self).__init__()
        block_channels = \
            in_channels if downsampler is None else in_channels // 2
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                block_channels,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(block_channels),
            nn.ReLU(),
            nn.Conv1d(
                block_channels,
                block_channels,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(block_channels)
        )
        self.final_relu = nn.ReLU()
        self.downsampler = downsampler

    def forward(self, x):
        identity = x
        x = self.net(x)
        if self.downsampler is not None:
            identity = self.downsampler(identity)
        return self.final_relu(x + identity)


class ResNet(nn.Module):
    def _make_layer(self, in_channels, kernel_size, n_blocks):
        downsampler = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // 2,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(in_channels // 2)
        )
        blocks = [BasicBlock(in_channels, kernel_size, downsampler)]
        blocks += [
            BasicBlock(in_channels // 2, kernel_size) for _ in
            range(n_blocks - 1)
        ]
        return nn.Sequential(*blocks)

    def __init__(
            self,
            in_features=5,
            out_features=37,
            up_dim_size=37,
            up_channel_size=8,
            n_blocks=2,
            n_layers=2,
            kernel_size=3,
            min_sd=0.1
    ):
        super().__init__()
        self.upsampler = nn.Sequential(
            nn.Linear(in_features, up_dim_size),
            nn.Unflatten(1, [1, up_dim_size]),
            nn.Conv1d(1, up_channel_size, 3, padding=1),
            nn.BatchNorm1d(up_channel_size),
            nn.ReLU()
        )
        self.res_layers = nn.ModuleList(
            [self._make_layer(up_channel_size // 2 ** i, kernel_size, n_blocks)
             for i in range(n_layers)]
        )
        final_size = up_dim_size * (up_channel_size // 2 ** n_layers)
        self.mean_net = nn.Linear(final_size, out_features)
        self.sd_net = nn.Sequential(
            nn.Linear(final_size, out_features),
            nn.Softplus(threshold=20)
        )
        self.min_sd = min_sd

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x = self.upsampler(x)
        for res_layer in self.res_layers:
            x = res_layer(x)
        x = x.flatten(1)
        return self.mean_net(x), self.sd_net(x) + self.min_sd
