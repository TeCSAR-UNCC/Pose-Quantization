import os
import yaml
import torch
import pandas as pd
import dataset
import numpy as np
from tqdm import tqdm
from utils import compute_metrics
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
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

    try:
        for model_name, model_config in tqdm(
            config_main.models.items(), desc="Testing Models", position=0, leave=True
        ):
            model_input_id, model_recon, *_ = model_config.config_file.split("_")

            config_path = os.path.join(
                config_main.configs_folder, model_config.config_file
            )
            with open(config_path) as f:
                config = edict(yaml.load(f, Loader=yaml.FullLoader))

            config.architecture.vqgan = update_vqgan_config(
                config.architecture.vqgan, model_config
            )

            vqgan = VQGAN(**config.architecture.vqgan)

            load_path = os.path.join(config_main.experiments_folder, model_name)
            trained_models = os.listdir(load_path)
            trained_models.sort(key=lambda x: int(x.split("_")[1][1:]))

            load_path = os.path.join(load_path, trained_models[-1])
            vqgan.load_checkpoint(load_path)
            vqgan.to(args.device)
            vqgan.eval()

            ds = eval("dataset." + config.dataset.name)(
                **config.dataset, is_training=False
            )
            dl_test = DataLoader(
                dataset=ds,
                batch_size=1,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=(0 > 0),
            )
            metrics_total = {}
            for index, (heatmap_2D, heatmap_3D) in tqdm(
                enumerate(dl_test),
                total=len(dl_test),
                desc=f"Processing {model_name}",
                position=1,
                leave=False,
            ):

                heatmap_2D = heatmap_2D.to(args.device)
                heatmap_3D = heatmap_3D.to(args.device)

                model_input = (
                    heatmap_2D if model_input_id in ["m2d", "2d", "c2d"] else heatmap_3D
                )
                model_recon_gt = (
                    heatmap_2D if model_recon in ["m2d", "2d", "c2d"] else heatmap_3D
                )

                with torch.no_grad():
                    decoded_images, codebook_indices, q_loss = vqgan(model_input)

                percep_gt = model_recon_gt
                percep_dec = decoded_images

                batch, channels, frames, height, width = percep_dec.shape

                if model_input_id == "c2d":
                    percep_gt = percep_gt.max(dim=1, keepdim=True)[0]

                metric = compute_metrics(percep_gt, percep_dec)
                metric["q_loss"] = q_loss.item()

                for key in metric.keys():
                    metrics_total[key] = metrics_total.get(key, [])
                    metrics_total[key].append(metric[key])

            for key in metrics_total.keys():
                metrics_total[key] = np.array(metrics_total[key])
                metrics_total[key] = metrics_total[key].mean(axis=0)

            # Store the results in the list
            results.append(
                {
                    "model_input": model_input_id,
                    "model_output": model_recon,
                    "compression": model_config.compression,
                    "vocab_size": model_config.num_codebook_vectors,
                    "SSIM": metrics_total.get("ssim", None),
                    "PSNR": metrics_total.get("psnr", None),
                    "L1": metrics_total.get("l1", None),
                    "Std": metrics_total.get("std", None),
                    "q_loss": metrics_total.get("q_loss", None),
                }
            )

        # Convert the results list into a DataFrame
        df_results = pd.DataFrame(results)

        # Print or save the DataFrame as needed
        print(df_results)
        df_results.to_csv("model_results.csv", index=False)

    except Exception as error:
        print("ERROR ->'{}'<- HAS OCCURED, SAVING DATA NOW.".format(error))
        df_results = pd.DataFrame(results)
        print(df_results)
        df_results.to_csv("model_results.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/test_multiple_models.yml",
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
