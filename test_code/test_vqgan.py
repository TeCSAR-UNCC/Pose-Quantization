"""
Contains test functions for the VQGAN model. 
"""

# Importing Libraries
import torch

from vqgan import Encoder, Decoder, CodeBook, Discriminator


def test_encoder():

    camera_channels = 3
    frame_length = 64
    image_size = 128
    latent_channels = 256
    frame_length = 64
    attn_resolution = 16
    latent_size_frames = 8
    concat_channels = True

    image = torch.randn(1, 1, frame_length, image_size, image_size*camera_channels)

    model = Encoder(
        img_channels=camera_channels,
        image_size=image_size,
        latent_channels=latent_channels,
        attention_resolution=[attn_resolution],
        concat_channels=concat_channels
    )

    output = model(image)

    assert output.shape == (
        1,
        latent_channels,
        latent_size_frames,
        attn_resolution,
        attn_resolution,
    ), "Output of encoder does not match"


def test_decoder():

    img_channels = 128
    img_size = 128
    frame_length = 64
    latent_channels = 256
    latent_size_frames = 8
    latent_size = 16
    attn_resolution = 16

    latent = torch.randn(1, latent_channels, latent_size_frames, latent_size, latent_size)
    model = Decoder(
        img_channels=img_channels,
        latent_size=latent_size,
        latent_channels=latent_channels,
        attention_resolution=[attn_resolution],
    )

    output = model(latent)

    assert output.shape == (
        1,
        img_channels,
        frame_length,
        img_size,
        img_size,
    ), "Output of decoder does not match"


def test_codebook():

    z = torch.randn(1, 256, 8, 16, 16)

    codebok = CodeBook(num_codebook_vectors=1024, latent_dim=256)

    z_q, min_distance_indices, loss = codebok(z)

    assert z_q.shape == (1, 256, 8, 16, 16), "Output of codebook does not match"


def test_discriminator():
    camera_channels = 3
    frame_length = 64
    image_size = 256
    latent_channels = 256
    frame_length = 64

    image = torch.randn(1, camera_channels, frame_length, image_size, image_size)

    model = Discriminator()


    output = model(image)

    assert output.shape == (1, 1, 6, 30, 30), "Output of discriminator does not match"

if __name__ == '__main__':
    test_encoder()
    test_decoder()
    test_codebook()
    test_discriminator()