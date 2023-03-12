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
            up_dim_size=37,
            up_channel_size=8,
            n_blocks=2,
            n_layers=2,
            kernel_size=3,
            upsample=True
    ):
        super(ResNet, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsampler = nn.Sequential(
                nn.Linear(in_features, up_dim_size),
                nn.Unflatten(1, [1, up_dim_size]),
                nn.Conv1d(1, up_channel_size, 3, padding=1),
                nn.BatchNorm1d(up_channel_size),
                nn.ReLU()
            )
        else:
            self.upsampler = None
        self.res_layers = nn.ModuleList(
            [self._make_layer(up_channel_size // 2 ** i, kernel_size, n_blocks)
             for i in range(n_layers)]
        )

    def forward(self, x):
        if self.upsample:
            x = self.upsampler(x)
        for res_layer in self.res_layers:
            x = res_layer(x)
        return x


class ResNetSd(nn.Module):
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
        super(ResNetSd, self).__init__()
        self.res_net = ResNet(
            in_features,
            up_dim_size,
            up_channel_size,
            n_blocks,
            n_layers,
            kernel_size
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
        x = self.res_net(x)
        x = x.flatten(1)
        return self.mean_net(x), self.sd_net(x) + self.min_sd


class ResNetCov(nn.Module):
    def __init__(
            self,
            in_features=5,
            out_features=37,
            common_up_dim_size=37,
            common_up_channel_size=4,
            cov_up_channel_size=4,
            n_blocks=(1, 1, 1),
            n_layers=(1, 1, 1),
            kernel_sizes=(5, 5, 5),
            min_var=0.01
    ):
        super(ResNetCov, self).__init__()
        self.common_res_net = ResNet(
            in_features=in_features,
            up_dim_size=common_up_dim_size,
            up_channel_size=common_up_channel_size,
            n_blocks=n_blocks[0],
            n_layers=n_layers[0],
            kernel_size=kernel_sizes[0]
        )
        common_n_out_channels = common_up_channel_size // (2 ** n_layers[0])
        self.mean_res_net = ResNet(
            in_features=common_up_dim_size,
            up_channel_size=common_n_out_channels,
            n_blocks=n_blocks[1],
            n_layers=n_layers[1],
            kernel_size=kernel_sizes[1],
            upsample=False
        )
        cov_up_channel_scaling_fac = (out_features + 1) // 2
        self.upsample_cov = nn.Conv1d(
            in_channels=common_n_out_channels,
            out_channels=cov_up_channel_size * cov_up_channel_scaling_fac,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2
        )
        self.cov_res_net = ResNet(
            in_features=common_up_dim_size,
            up_channel_size=cov_up_channel_size * (out_features + 1) // 2,
            n_blocks=n_blocks[2],
            n_layers=n_layers[2],
            kernel_size=kernel_sizes[2],
            upsample=False
        )
        mean_final_size = \
            common_up_dim_size * (common_n_out_channels // 2 ** n_layers[1])
        self.mean_fc_net = nn.Linear(mean_final_size, out_features)
        cov_final_size = (
                common_up_dim_size *
                cov_up_channel_scaling_fac *
                (cov_up_channel_size // 2 ** n_layers[2])
        )
        self.cov_fc_net = nn.Linear(
            cov_final_size,
            out_features * (out_features + 1) // 2
        )
        self.min_var = min_var
        self.out_features = out_features
        self.register_buffer('identity', torch.eye(out_features))
        self.register_buffer(
            'l_tri_mask',
            torch.tril(torch.ones((out_features, out_features)), -1)
        )

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x = self.common_res_net(x)
        mean = self.mean_fc_net(self.mean_res_net(x).flatten(1))

        cov_flat = self.cov_fc_net(
            self.cov_res_net(self.upsample_cov(x)).flatten(1)
        )
        l_tri_cov = torch.zeros(
            x.shape[0], self.out_features, self.out_features, device=x.device
        )

        ti = torch.tril_indices(
            self.out_features, self.out_features, 0, device=x.device
        )
        l_tri_cov[:, ti[0], ti[1]] = cov_flat

        d = torch.exp(torch.diagonal(l_tri_cov, dim1=1, dim2=2)) + self.min_var
        l_tri_cov = l_tri_cov * self.l_tri_mask + self.identity
        return mean, l_tri_cov, d
