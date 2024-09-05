"""
https://github.com/dome272/VQGAN-pytorch/blob/main/decoder.py

Contains the decoder implementation of VQGAN.

The decoder architecture is also highly inspired by the - Denoising Diffusion Probabilistic Models - https://arxiv.org/abs/2006.11239
According to the official implementation.  
"""

# Importing Libraries
import torch
import torch.nn as nn

from vqgan.common import GroupNorm, NonLocalBlock, ResidualBlock, Swish, UpsampleBlock


class Decoder(nn.Module):
    """
    The decoder part of the VQGAN.

    The implementation is similar to the encoder but inverse, to produce an image from a latent vector.

    Args:
        img_channels (int): Number of channels in the output image.
        latent_channels (int): Number of channels in the latent vector.
        latent_size (int): Size of the latent vector.
        intermediate_channels (list): List of channels in the intermediate layers.
        num_residual_blocks (int): Number of residual blocks b/w each downsample block.
        dropout (float): Dropout probability for residual blocks.
        attention_resolution (list): tensor size ( height or width ) at which to add attention blocks
    """

    def __init__(
        self,
        img_channels: int = 3,
        latent_channels: int = 256,
        latent_size: int = 16,
        intermediate_channels: list = [128, 128, 256, 256, 512],
        upsample_layers: int = 3,
        reduce_temporal: bool = False,
        temporal_upsample_layers: int = 3,
        num_residual_blocks: int = 3,
        dropout: float = 0.0,
        attention_resolution: list = [16],
        
    ):
        super().__init__()

        # Reverse the list to get the correct order of decoder layer channels
        intermediate_channels = intermediate_channels[::-1]

        # Appends all the layers to this list
        self.layers = []

        # Adding the first conv layer to increase the input channels to the first intermediate channels
        in_channels = intermediate_channels[0]
        self.layers.extend(
            [
                nn.Conv3d(
                    latent_channels,
                    intermediate_channels[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
                NonLocalBlock(in_channels=in_channels),
                ResidualBlock(
                    in_channels=in_channels, out_channels=in_channels, dropout=dropout
                ),
            ]
        )

        # Loop over the intermediate channels
        added_upsample = 0 
        for n in range(len(intermediate_channels)):
            out_channels = intermediate_channels[n]

            # adding the residual blocks
            for _ in range(num_residual_blocks):
                self.layers.append(ResidualBlock(in_channels, out_channels, dropout=dropout))
                in_channels = out_channels

                # adding the non local block
                if latent_size in attention_resolution:
                    self.layers.append(NonLocalBlock(in_channels))

            # Due to conv in first layer, do not upsample
            if n != 0 and added_upsample < upsample_layers:

                self.layers.append(UpsampleBlock(in_channels=in_channels, 
                                                   not_temporal=n > temporal_upsample_layers,
                                                   reduce_temporal=reduce_temporal,
                                                   n=n)
                                                   )
                # self.layers.append(UpsampleBlock(in_channels=in_channels))
                latent_size = latent_size * 2  # Upsample by a factor of 2
                added_upsample += 1

        # Adding rest of the layers
        self.layers.extend(
            [
                GroupNorm(in_channels=in_channels),
                Swish(),
                nn.Conv3d(
                    in_channels, img_channels, kernel_size=3, stride=1, padding=1
                ),
            ]
        )
        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for layer in self.layers:
        #     x = layer(x)
        return self.model(x) # x
