"""
https://github.com/dome272/VQGAN-pytorch/blob/main/training_vqgan.py
"""

# Importing Libraries
import os
import cv2
import wandb

import imageio
from torch import nn
from tqdm import tqdm
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from utils import weights_init
from vqgan import Discriminator

from utils.axu import convert_hm_to_rgb

class VQGANTrainer:
    """Trainer class for VQGAN, contains step, train methods"""

    def __init__(
        self,
        model_config: dict,
        vqgan: torch.nn.Module,
        # Training parameters
        device: str or torch.device = "cuda",
        learning_rate: float = 2.25e-05,
        beta1: float = 0.5,
        beta2: float = 0.9,
        # Loss parameters
        perceptual_loss_factor: float = 1.0,
        rec_loss_factor: float = 1.0,
        # Discriminator parameters
        disc_factor: float = 1.0,
        disc_start: int = 100,
        # Miscellaneous parameters
        experiment_dir: str = "./experiments",
        perceptual_model: str = "vgg",
        save_every: int = 10,
        model_input: str = "m2d",
        model_recon: str = "m2d",
        run=None,
    ):

        self.model_config = model_config
        self.device = device
        self.logs = {}
        # VQGAN parameters
        self.vqgan = vqgan

        self.model_input = model_input
        assert self.model_input in ["m2d", "c2d", "2d", "3d"], "Model Input is not a valid choice"
        self.model_recon = model_recon
        assert self.model_recon in ["m2d", "2d", "3d"], "Model Output is not a valid choice"

        self.run = run

        # Discriminator parameters
        self.discriminator = Discriminator(image_channels=self.vqgan.discriminator_channels).to(
            self.device
        )
        self.discriminator.apply(weights_init)

        # Loss parameters
        self.perceptual_loss = lpips.LPIPS(net=perceptual_model).to(self.device)

        # Optimizers
        self.opt_vq, self.opt_disc = self.configure_optimizers(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2
        )

        # Hyperprameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_factor = perceptual_loss_factor
        self.rec_loss_factor = rec_loss_factor

        # Save directory
        self.expriment_save_dir = os.path.join(experiment_dir, run.name)
        
        if not os.path.exists(self.expriment_save_dir) and run.name:
            os.makedirs(self.expriment_save_dir)

        # Miscellaneous
        self.global_step = 0
        self.sample_batch = None
        self.gif_images = []
        self.save_every = save_every

    def configure_optimizers(
        self, learning_rate: float = 2.25e-05, beta1: float = 0.5, beta2: float = 0.9
    ):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            eps=1e-08,
            betas=(beta1, beta2),
        )

        return opt_vq, opt_disc
    
    def save_model(self, epoch):
        file_name = f"{self.expriment_save_dir}/3dvqgan_e{epoch}_{self.run.name}.pt"
        print("-->Saving file: {}".format(file_name))
        self.vqgan.save_checkpoint(file_name)
        
        # model_artifact = wandb.Artifact(
        #     "trained-3dvqgan", type="model", metadata=dict(self.model_config)
        # )

        # model_artifact.add_file(file_name)
        # self.run.log_artifact(model_artifact)

    def extract_video(self, heatmap, D3=False):
        hm = heatmap.cpu().detach().numpy()
        channels, frames, height, width, = heatmap.shape
        if channels <  width:
            if D3:
                hm[2] = np.rot90(hm[2], k=3, axes=(1,2))
            hm_cat = np.concatenate(hm, axis=-1)
        else:
            hm = hm.transpose(1, 2, 3, 0)
            hm = [np.max(hm, axis=1), np.max(hm, axis=2), np.max(hm, axis=3)]
            hm[2] = np.rot90(hm[2], k=3, axes=(1,2))
            hm = np.array(hm)
            hm_cat = np.concatenate(hm, axis=2)

        hm_rgb = convert_hm_to_rgb(hm_cat)
        return hm_rgb
    
    def release_video(self, heatmap, name, D3=False):
        hm_rgb = self.extract_video(heatmap, D3).transpose(0, 2, 3, 1)[...,::-1]
        T, new_W, new_H, _ = hm_rgb.shape

        out = cv2.VideoWriter(
            name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (new_H, new_W)
        )

        for i in range(T):
            out.write(hm_rgb[i])

        out.release()

    def compress_3d_heatmap(heatmap):
        # Permute to change the dimensions to (batch_size, depth, height, width, channels)
        percep_gt = heatmap.permute(0, 2, 3, 4, 1)
        
        # Max across each axis
        max_x = torch.max(percep_gt, dim=2)[0]  # Max across x axis
        max_y = torch.max(percep_gt, dim=3)[0]  # Max across y axis
        max_z = torch.max(percep_gt, dim=4)[0]  # Max across z axis

        # Concatenate them together
        percep_xyz_gt = torch.stack([max_x, max_y, max_z], dim=1)  # (batch_size, 3, depth, height, width)

        return percep_xyz_gt


    def step(self, heatmap_2D: torch.Tensor, heatmap_3D: torch.Tensor) -> torch.Tensor:
        """Performs a single training step from the dataloader images batch

        For the VQGAN, it calculates the perceptual loss, reconstruction loss, and the codebook loss and does the backward pass.

        For the discriminator, it calculates lambda for the discriminator loss and does the backward pass.

        Args:
            imgs: input tensor of shape (batch_size, channel, H, W)

        Returns:
            decoded_imgs: output tensor of shape (batch_size, channel, H, W)
        """
        
        model_input = heatmap_2D if self.model_input in ["m2d", "2d", "c2d"] else heatmap_3D
        model_recon_gt = heatmap_2D if self.model_recon in ["m2d", "2d", "c2d"] else heatmap_3D

        # Getting decoder output
        decoded_images, codebook_indices, q_loss = self.vqgan(model_input)

        """
        =======================================================================================================================
        VQ Loss
        """
        debug = False
        if debug:
            self.release_video(model_input[0], f"gt_{self.model_input}.mp4")
            self.release_video(decoded_images[0], f"output_{self.model_recon}.mp4")

        percep_gt = model_recon_gt
        percep_dec = decoded_images
            

        batch, channels, frames, height, width, = percep_dec.shape
        
        if self.model_input == "c2d":
            percep_gt = percep_gt.max(dim=1, keepdim=True)[0]
        # Used to break volumes by channels to fit into perceptual_loss
        percep_gt_ft = percep_gt.view(batch * channels * frames, height, width).unsqueeze(1)
        percep_gt_rgb = percep_gt_ft.repeat(1, 3, 1, 1)

        percep_dec_ft = percep_dec.view(batch * channels * frames, height, width).unsqueeze(1)
        percep_dec_rgb = percep_dec_ft.repeat(1, 3, 1, 1)

        
        perceptual_loss = self.perceptual_loss(percep_gt_rgb, percep_dec_rgb)
        rec_loss = torch.abs(percep_gt_rgb - percep_dec_rgb)
        perceptual_rec_loss = (
            self.perceptual_loss_factor * perceptual_loss
            + self.rec_loss_factor * rec_loss
        )
        perceptual_rec_loss = perceptual_rec_loss.mean()

        """
        =======================================================================================================================
        Discriminator Loss
        """
        disc_real = self.discriminator(percep_gt)
        disc_fake = self.discriminator(percep_dec)

        disc_factor = self.vqgan.adopt_weight(
            self.disc_factor, self.global_step, threshold=self.disc_start
        )

        g_loss = -torch.mean(disc_fake)

        λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

        d_loss_real = torch.mean(F.relu(1.0 - disc_real))
        d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

        # =======================================================================================================================
        # Backpropagation

        self.opt_vq.zero_grad()
        vq_loss.backward(
            retain_graph=True
        )  # retain_graph is used to retain the computation graph for the discriminator loss

        self.opt_disc.zero_grad()
        gan_loss.backward()

        self.opt_vq.step()
        self.opt_disc.step()

        return decoded_images, vq_loss, gan_loss, perceptual_rec_loss, codebook_indices

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 10,
    ):
        """Trains the VQGAN for the given number of epochs

        Args:
            dataloader (torch.utils.data.DataLoader): dataloader to use.
            epochs (int, optional): number of epochs to train for. Defaults to 100.
        """
        import time
        for epoch in range(epochs):
            
            for index, (heatmap_2D, heatmap_3D) in tqdm(enumerate(dataloader), total=len(dataloader)):

                # Training step
                heatmap_2D = heatmap_2D.to(self.device)
                heatmap_3D = heatmap_3D.to(self.device)

                decoded_images, vq_loss, gan_loss, perceptual_rec_loss, codebook_indices = self.step(heatmap_2D, heatmap_3D)

                if self.global_step % self.save_every == 0:

                    print(
                        f"\nEpoch: {epoch+1}/{epochs} | Batch: {index}/{len(dataloader)} | VQ Loss : {vq_loss:.4f} | Discriminator Loss: {gan_loss:.4f}"
                    )
                    hm2d_rgb = np.zeros((1, 3, 128, 384))
                    if self.model_input in ["m2d", "2d"] or self.model_recon in ["m2d", "2d"]:
                        hm2d_rgb = self.extract_video(heatmap_2D[0])
                    if self.model_input == "c2d":
                        hm2d_rgb = self.extract_video(heatmap_2D[0].max(dim=0, keepdim=True)[0])
                    
                    hm3d_rgb = np.zeros((1, 3, 128, 384))
                    if self.model_input == "3d" or self.model_recon == "3d":
                        hm3d_rgb = self.extract_video(heatmap_3D[0], D3=True)
                    
                    # hm3d_rgb = 
                    D3 = self.model_recon == '3d'
                    recons_rgb = self.extract_video(decoded_images[0], D3=D3)

                    self.logs = {
                        **self.logs,
                        "perceptual_rec_loss": perceptual_rec_loss,
                        "multi-2d heatmap": wandb.Video(
                            hm2d_rgb, fps=30, caption="original 2d heatmaps"
                        ),
                        "3d-heatmap views x, y, z": wandb.Video(
                            hm3d_rgb, fps=30, caption="original 3d heatmap"
                        ),
                        "reconstructions": wandb.Video(
                            recons_rgb, fps=30, caption="reconstructions"
                        ),
                        
                    }

                if self.global_step % (self.save_every//10) == 0:
                    codebook_indices = codebook_indices.cpu().detach().numpy()
                    self.logs = {
                        **self.logs,
                        "epoch": epoch,
                        "iter": index,
                        "vq_loss": vq_loss,
                        "gan_loss": gan_loss,
                        "perceptual_rec_loss": perceptual_rec_loss,
                        "codebook_indices": wandb.Histogram(codebook_indices),
                    }
                    wandb.log(self.logs)
                    self.logs = {}

                # Updating global step
                self.global_step += 1

                
            self.save_model(epoch)
            

                
                

                    