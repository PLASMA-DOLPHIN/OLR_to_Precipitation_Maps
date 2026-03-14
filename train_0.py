# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os

from dataset import RainDataset
from models_0 import UNetGenerator, PatchDiscriminator, PerceptualLoss
from utils import save_checkpoint, load_checkpoint, ensure_dir
import config

class Trainer:
    def __init__(self):
        ensure_dir(config.CHECKPOINT_DIR)
        
        self.device = config.DEVICE
        self.G = UNetGenerator().to(self.device)
        self.D = PatchDiscriminator().to(self.device)
        
        self.optimizer_G = torch.optim.Adam(
            self.G.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )
        self.optimizer_D = torch.optim.Adam(
            self.D.parameters(), 
            lr=config.LR, 
            betas=config.BETAS
        )
        
        # Learning rate schedulers
        self.scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, 
            step_size=config.LR_DECAY_INTERVAL, 
            gamma=config.LR_DECAY_GAMMA
        )
        self.scheduler_D = lr_scheduler.StepLR(
            self.optimizer_D, 
            step_size=config.LR_DECAY_INTERVAL, 
            gamma=config.LR_DECAY_GAMMA
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        
        # Training tracking
        self.best_g_loss = float('inf')
        self.start_epoch = 0
        
    def load_checkpoint(self, checkpoint_path):
        """Load model and optimizer state from checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_g_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed from epoch {self.start_epoch}")
            return True
        return False

    def train_discriminator(self, olr, real_rain, fake_rain):
        """Train discriminator for one step"""
        self.optimizer_D.zero_grad()
        
        # Real data
        real_pred = self.D(olr, real_rain)
        real_loss = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        
        # Fake data
        fake_pred = self.D(olr, fake_rain.detach())
        fake_loss = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), config.GRAD_CLIP)
        self.optimizer_D.step()
        
        return {
            'd_total': d_loss.item(),
            'd_real': real_loss.item(),
            'd_fake': fake_loss.item()
        }

    def train_generator(self, olr, real_rain, fake_rain):
        """Train generator for one step"""
        self.optimizer_G.zero_grad()
        
        # Adversarial loss
        fake_pred = self.D(olr, fake_rain)
        g_adv = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        
        # L1 loss (reconstruction)
        g_l1 = self.l1_loss(fake_rain, real_rain)
        
        # Perceptual loss
        g_perceptual = self.perceptual_loss(fake_rain, real_rain)
        
        # Total generator loss
        g_loss = (
            g_adv + 
            config.LAMBDA_L1 * g_l1 + 
            config.LAMBDA_PERCEPTUAL * g_perceptual
        )
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), config.GRAD_CLIP)
        self.optimizer_G.step()
        
        return {
            'g_total': g_loss.item(),
            'g_adv': g_adv.item(),
            'g_l1': g_l1.item(),
            'g_perceptual': g_perceptual.item()
        }

    def train_epoch(self, loader):
        """Train for one epoch"""
        self.G.train()
        self.D.train()
        
        losses = {
            'd_total': [], 'd_real': [], 'd_fake': [],
            'g_total': [], 'g_adv': [], 'g_l1': [], 'g_perceptual': []
        }
        
        pbar = tqdm(loader, desc="Training")
        for olr, rain in pbar:
            olr = olr.to(self.device)
            rain = rain.to(self.device)
            
            # Generate fake rain
            fake_rain = self.G(olr)
            
            # Train discriminator
            d_losses = self.train_discriminator(olr, rain, fake_rain)
            for key, val in d_losses.items():
                losses[key].append(val)
            
            # Train generator
            g_losses = self.train_generator(olr, rain, fake_rain)
            for key, val in g_losses.items():
                losses[key].append(val)
            
            # Update progress bar
            pbar.set_postfix({
                'D': f"{d_losses['d_total']:.4f}",
                'G': f"{g_losses['g_total']:.4f}",
                'Perc': f"{g_losses['g_perceptual']:.4f}"
            })
        
        # Average losses
        avg_losses = {key: sum(val) / len(val) for key, val in losses.items()}
        return avg_losses

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        for epoch in range(self.start_epoch, config.EPOCHS):
            
            # Train epoch
            avg_losses = self.train_epoch(train_loader)
            
            # Learning rate decay
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
            print(f"  D Loss: {avg_losses['d_total']:.4f} "
                  f"(Real: {avg_losses['d_real']:.4f}, Fake: {avg_losses['d_fake']:.4f})")
            print(f"  G Loss: {avg_losses['g_total']:.4f}")
            print(f"    - Adversarial: {avg_losses['g_adv']:.4f}")
            print(f"    - L1: {avg_losses['g_l1']:.4f}")
            print(f"    - Perceptual: {avg_losses['g_perceptual']:.4f}")
            print(f"  LR: {self.optimizer_G.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = avg_losses['g_total'] < self.best_g_loss
            if is_best:
                self.best_g_loss = avg_losses['g_total']
            
            if (epoch + 1) % config.CHECKPOINT_INTERVAL == 0 or is_best:
                save_checkpoint(
                    self.G, self.D, 
                    self.optimizer_G, self.optimizer_D,
                    epoch, avg_losses['g_total'],
                    f"{config.CHECKPOINT_DIR}/G_epoch_{epoch+1}.pth",
                    is_best=is_best
                )

def train():
    # Create dataset and loader
    dataset = RainDataset(config.OLR_FOLDERS, config.RAIN_FOLDERS)
    loader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer()
    
    # Resume from checkpoint if available
    latest_checkpoint = os.path.join(config.CHECKPOINT_DIR, "latest.pth")
    trainer.load_checkpoint(latest_checkpoint)
    
    # Train
    trainer.train(loader)

if __name__ == "__main__":
    train()