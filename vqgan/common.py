"""
https://github.com/dome272/VQGAN-pytorch/blob/main/helper.py

The file contains Swish, Group Norm, Residual & Non-Local Blocks,  Upsample and Downsample layer for VQGAN encoder and decoder blocks.
"""

# Importing Libraries
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function first introduced in the paper https://arxiv.org/abs/1710.05941v2
    It has shown to be working better in many datasets as compares to ReLU.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    """Group Normalization is a method which normalizes the activation of the layer for better results across any batch size.
    Note : Weight Standardization is also shown to given better results when added with group norm

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # num_groups is according to the official code provided by the authors,
        # eps is for numerical stability
        # i think affine here is enabling learnable param for affine trasnform on calculated mean & standard deviation
        self.group_norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-06, affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)


class ResidualBlock(nn.Module):
    """Residual Block from the paper,
    group norm -> swish -> conv -> group norm -> swish -> conv -> dropout -> conv -> skip connection

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        dropout (float): Dropout probability.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        """
        In some cases, the shortcut connection needs to be added
        to match the dimension of the input and the output for skip connection
        """
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # shortcut connection
        if self.in_channels != self.out_channels:
            return self.conv_shortcut(x) + self.block(x)
        else:
            return x + self.block(x)


class DownsampleBlock(nn.Module):
    """
    Down sample block for the encoder. pad -> conv

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
    ) -> None:
        super().__init__()

        # (0, 1, 0, 1) - pad on left, right, top, bottom, front, back with respective size
        self.pad = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), value=0)  # and fill value of 0

        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pad(x)

        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Upsample block for the decoder. interpolate -> conv

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(
        self,
        in_channels: int,
        not_temporal: bool,
        reduce_temporal: bool,
        n: int,
        at: int,
    ) -> None:
        super().__init__()

        self.scale_factor = (2.4, 2.4, 2.4)

        """
        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        padding = (1, 1, 1)
        """
        kernel_size = (3, 3, 3)
        stride = (1, 1, 1)
        if at != n - 2:
            padding = 1
        else:
            padding = 0

        """
        if not_temporal:
            self.scale_factor = (1.0, 2.0, 2.0)

        elif reduce_temporal:  # Works only with F8 48 Frames
            self.scale_factor = (4.0, 2.0, 2.0)
            if n == 3:
                self.scale_factor = (3.0, 2.0, 2.0)
        """
        layers = [
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if at == n - 2:
            layers.append(nn.AdaptiveAvgPool3d((48, 72, 72)))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)
        x = self.conv(x)

        return x


class NonLocalBlock(nn.Module):
    """Attention mechanism similar to transformers but for CNNs, paper https://arxiv.org/abs/1805.08318

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.in_channels = in_channels

        # Normalization layer
        self.norm = nn.GroupNorm(1, in_channels)

        # Query, key, and value layers using 3D convolutions
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.project_out = nn.Conv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch, channels, frames, height, width = x.size()

        # Reshape x to (batch * cameras, in_channels, frames, height, width) for normalization
        # x = x.view(batch, cameras, frames, height, width)
        x = self.norm(x)

        # Query, key, and value layers
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Resizing the output from 5D to 3D to generate attention map
        q = q.view(batch, channels, frames * height * width)
        k = k.view(batch, channels, frames * height * width)
        v = v.view(batch, channels, frames * height * width)

        # Transpose the query tensor for dot product
        q = q.permute(0, 2, 1)

        # Main attention formula
        scores = torch.bmm(q, k) * (channels**-0.5)
        weights = self.softmax(scores)
        weights = weights.permute(0, 2, 1)

        attention = torch.bmm(v, weights)

        # Resizing the output from 3D to 5D to match the input
        attention = attention.view(batch, channels, frames, height, width)
        attention = self.project_out(attention)

        # Adding the identity to the output
        return x + attention
