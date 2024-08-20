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

def update_vqgan_config(base_config, model_config):
    """
    Update the base VQGAN config with the values from the model config.
    """
    for key, value in model_config.items():
        if key in base_config:
            base_config[key] = value
        else:
            base_config[key] = value
    return base_config

def main(args, config_main):
    results = []

    for model_name, model_config in tqdm(config_main.models.items(), desc="Testing Models", position=0, leave=True):
        model_input_id, model_recon, *_ = model_config.config_file.split('_')

        config_path = os.path.join(config_main.configs_folder, model_config.config_file)
        with open(config_path) as f:
            config = edict(yaml.load(f, Loader=yaml.FullLoader))

        config.architecture.vqgan = update_vqgan_config(config.architecture.vqgan, model_config)

        vqgan = VQGAN(**config.architecture.vqgan)

        load_path = os.path.join(config_main.experiments_folder, model_name)
        trained_models = os.listdir(load_path)
        trained_models.sort(key=lambda x: int(x.split('_')[1][1:]))

        load_path = os.path.join(load_path, trained_models[-1])
        vqgan.load_checkpoint(load_path)
        vqgan.to(args.device)
        vqgan.eval()

        config.dataset.stride = config.dataset.data_size // config.dataset.frame_interval

        ds = eval("dataset." + config.dataset.name)(**config.dataset, is_training=False)
        
        # Dummy Trainer for video output
        vqgan_trainer = VQGANTrainer(
            model_config={},
            vqgan=vqgan,
            run=edict({"name":model_name})
        ) 
        gt, output = [], []
        for index in range(args.init, args.init + args.length):
            heatmap_2D, heatmap_3D = ds.__getitem__(index)

            # Training step
            heatmap_2D = heatmap_2D.unsqueeze(0).to(args.device)
            heatmap_3D = heatmap_3D.unsqueeze(0).to(args.device)

            model_input = heatmap_2D if model_input_id in ["m2d", "2d", "c2d"] else heatmap_3D
            model_recon_gt = heatmap_2D if model_recon in ["m2d", "2d", "c2d"] else heatmap_3D

            with torch.no_grad():
                decoded_images, codebook_indices, q_loss = vqgan(model_input)

            if model_input_id == "c2d":
                model_recon_gt = model_recon_gt.max(dim=1, keepdim=True)[0]
            if model_input_id == "3d":
                model_recon_gt[0, 2] = torch.rot90(model_recon_gt[0, 2], k=3, dims=(1,2))
                decoded_images[0, 2] = torch.rot90(decoded_images[0, 2], k=3, dims=(1,2))
            
            gt.append(model_recon_gt[0].cpu())
            output.append(decoded_images[0].cpu())

        model_recon_gt = torch.cat(gt, dim=1)
        decoded_images = torch.cat(output, dim=1)

        stacked_heatmaps = torch.cat((model_recon_gt, decoded_images), dim=2)
        vqgan_trainer.release_video(stacked_heatmaps, 
                                    f"./outputs/{model_input_id}_{model_recon}_{model_config.compression}_{model_config.num_codebook_vectors}.mp4")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/test_multiple_models.yml",
        help="experiment configure file name",
    )
    parser.add_argument(
        "--init",
        type=int,
        default=0,
        help="Inital index to start at",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10,
        help="How long to make the videos",
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