import torch.nn as nn
import models
import torch


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
        blocks = [models.BasicBlock(in_channels, kernel_size, downsampler)]
        blocks += [
            models.BasicBlock(in_channels // 2, kernel_size) for _ in
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
