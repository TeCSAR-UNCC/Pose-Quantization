# Importing Libraries
import os

import torch
import torchvision
from utils import reproducibility

from trainer import VQGANTrainer


class Trainer:
    def __init__(
        self,
        model_config: dict,
        vqgan: torch.nn.Module,
        config: dict,
        experiment_dir: str = "experiments",
        seed: int = 42,
        device: str = "cuda",
        model_input: str = "m2d", # [m2d, 3d, c2d]
        model_recon: str = "3d", # [m2d, 3d]
        run = None,
    ) -> None:

        self.vqgan = vqgan

        self.model_config = model_config
        self.config = config
        self.experiment_dir = experiment_dir
        self.seed = seed
        self.device = device
        self.run = run
        self.model_input = model_input
        self.model_recon = model_recon

        print(f"[INFO] Setting seed to {seed}")
        reproducibility(seed)

        print(f"[INFO] Results will be saved in {experiment_dir}")
        self.experiment_dir = experiment_dir

    def train_vqgan(self, dataloader: torch.utils.data.DataLoader, epochs: int = 10):

        print(f"[INFO] Training VQGAN on {self.device} for {epochs} epoch(s).")

        self.vqgan.to(self.device)

        self.vqgan_trainer = VQGANTrainer(
            model_config=self.model_config,
            vqgan=self.vqgan,
            device=self.device,
            experiment_dir=self.experiment_dir,
            model_input = self.model_input,
            model_recon = self.model_recon,
            run=self.run,
            **self.config.vqgan,
        )

        self.vqgan_trainer.train(
            dataloader=dataloader,
            epochs=epochs,
        )

        # Saving the model
        self.vqgan.save_checkpoint(
            os.path.join(self.experiment_dir, "checkpoints", "vqgan.pt")
        )
