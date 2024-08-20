# Importing Libraries
import argparse
from easydict import EasyDict as edict

import yaml
import wandb

import dataset
from trainer import Trainer
from torch.utils.data import DataLoader
from vqgan import VQGAN


# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python train.py --cfg configs/m2d_m2d_vqgan.yml
def main(args, config):
     
    model_config = dict(
        num_tokens=config.architecture.vqgan,
        num_codebook_vectors=config.architecture.vqgan.num_codebook_vectors,
        num_residual_blocks_encoder=config.architecture.vqgan.num_residual_blocks_encoder,
        num_residual_blocks_decoder=config.architecture.vqgan.num_residual_blocks_decoder,
        
    )

    run = wandb.init(
        project="VQGAN_Init_testing",
        job_type="dVAE_model",
        config=model_config,
    )

    vqgan = VQGAN(**config.architecture.vqgan)
    if config.resume:
        vqgan.load_checkpoint(config.resume)


    ds = eval("dataset." + config.dataset.name)(**config.dataset, is_training=True)
    dl_train = DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=(0 > 0),
    )
    model_input, model_recon = config.architecture.input_recon.split('_')

    trainer = Trainer(
        model_config=config.architecture.vqgan,
        vqgan=vqgan,
        config=config.trainer,
        seed=args.seed,
        device=args.device,
        model_input=model_input,
        model_recon=model_recon,
        run=run,
    )

    trainer.train_vqgan(dl_train)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/c2d_2d_vqgan.yml",
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
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    main(args, config)
