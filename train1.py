import os
import yaml
import torch
import cv2
import dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from easydict import EasyDict as edict
from trainer import VQGANTrainer
from vqgan import VQGAN
import argparse
import wandb
import numpy as np



def save_model(vqgan, epoch, run):
    file_name = f"experiments/{run.name}/3dvqgan_e{epoch}_{run.name}.pt"
    print("-->Saving file: {}".format(file_name))
    if not os.path.exists(f"experiments/{run.name}/"):
        os.makedirs(f"experiments/{run.name}/")
    vqgan.save_checkpoint(file_name)


def main(args, config):

    run = wandb.init(
        project="VQGAN_Init_testing",
        job_type="dVAE_model",
        config=config.architecture.vqgan,
    )

    vqgan = VQGAN(**config.architecture.vqgan)
    if config.resume:
        vqgan.load_checkpoint(config.resume)

    from vqgan import CodeBook
    vqgan.codebook = CodeBook(
            num_codebook_vectors=1024, latent_dim=256
        )
    
    for param in vqgan.encoder.parameters():
        param.requires_grad = False
    
    for param in vqgan.quant_conv.parameters():
        param.requires_grad = False

    vqgan.to(args.device)

    ds = eval("dataset." + config.dataset.name)(**config.dataset, is_training=True)
    dl_train = DataLoader(
            dataset=ds,
            batch_size=4,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=(0 > 0),
        )
    
    opt_vq = torch.optim.Adam(
        list(vqgan.encoder.parameters())
        + list(vqgan.codebook.parameters())
        + list(vqgan.quant_conv.parameters()),
        lr=2.25e-05,
        eps=1e-08,
        betas=(0.5, 0.9),
    )

    global_step = 0
    save_every = 100
    epochs = 5
    logs = {}
    for epoch in range(epochs):
                
        for index, (heatmap_2D, heatmap_3D) in tqdm(enumerate(dl_train), total=len(dl_train)):

            # Training step
            heatmap_2D = heatmap_2D.to(args.device)
            heatmap_3D = heatmap_3D.to(args.device)

            model_input = heatmap_3D
            

            encoded_images = vqgan.encoder(model_input)
            quant_x = vqgan.quant_conv(encoded_images)
            _, codebook_indices, vq_loss = vqgan.codebook(quant_x)

            opt_vq.zero_grad()
            vq_loss.backward()
            opt_vq.step()

            if global_step % save_every == 0:

                print(
                    f"\nEpoch: {epoch+1}/{epochs} | Batch: {index}/{len(dl_train)} | VQ Loss : {vq_loss:.4f}"
                )
                logs = {
                    **logs                    
                }

            if global_step % (save_every//10) == 0:
                logs = {
                    **logs,
                    "epoch": epoch,
                    "iter": index,
                    "vq_loss": vq_loss,
                }
                wandb.log(logs)
                logs = {}

            # Updating global step
            global_step += 1

        save_model(vqgan, epoch, run=run)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/3d_3d_vqgan.yml",
        help="experiment configure file name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to train the model on",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=42,
        help="Seed for Reproducibility",
    )

    args = parser.parse_args()

    args = parser.parse_args()
    with open(args.cfg) as f:
        config_main = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    main(args, config_main)