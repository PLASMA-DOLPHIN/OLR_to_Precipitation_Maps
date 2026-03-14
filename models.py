# models.py

import torch
import torch.nn as nn
import torchvision.models as models
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------
# Perceptual Loss
# -------------------

class PerceptualLoss(nn.Module):
    """VGG19-based perceptual loss for feature matching"""
    def __init__(self, layer='relu5_1'):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        # Layer mapping for different feature extraction points
        layer_name_mapping = {
            'relu1_1': 1,
            'relu1_2': 3,
            'relu2_1': 6,
            'relu2_2': 8,
            'relu3_1': 11,
            'relu3_2': 13,
            'relu4_1': 15,
            'relu4_2': 17,
            'relu5_1': 22,
            'relu5_2': 24,
        }
        
        layer_idx = layer_name_mapping.get(layer, 22)
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:layer_idx+1])
        
        # Freeze weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Normalization values for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        """
        Calculate perceptual loss between x and y
        Input: single-channel images (B, 1, H, W)
        """
        # Convert single channel to 3-channel for VGG
        x_3ch = x.repeat(1, 3, 1, 1)
        y_3ch = y.repeat(1, 3, 1, 1)
        
        # Normalize
        x_norm = (x_3ch - self.mean) / self.std
        y_norm = (y_3ch - self.mean) / self.std
        
        # Extract features
        x_features = self.feature_extractor(x_norm)
        y_features = self.feature_extractor(y_norm)
        
        # L1 loss on features
        loss = nn.functional.l1_loss(x_features, y_features)
        return loss


# -------------------
# Generator (U-Net with improvements)
# -------------------

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_instance_norm=False):
        super().__init__()
        
        norm_layer = nn.InstanceNorm2d if use_instance_norm else nn.BatchNorm2d

        def down_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(norm_layer(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                norm_layer(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder
        self.d1 = down_block(in_channels, 64, normalize=False)
        self.d2 = down_block(64, 128)
        self.d3 = down_block(128, 256)
        self.d4 = down_block(256, 512)
        self.d5 = down_block(512, 512)

        # Decoder
        self.u1 = up_block(512, 512, dropout=True)
        self.u2 = up_block(1024, 256, dropout=True)
        self.u3 = up_block(512, 128)
        self.u4 = up_block(256, 64)
        self.u5 = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
        
        self.output_act = nn.ReLU()

    @staticmethod
    def center_crop(tensor, target_shape):
        """Center crop tensor to target spatial dimensions"""
        _, _, h, w = tensor.shape
        target_h, target_w = target_shape
        
        if h == target_h and w == target_w:
            return tensor
        
        h_start = (h - target_h) // 2
        w_start = (w - target_w) // 2
        
        return tensor[:, :, h_start:h_start+target_h, w_start:w_start+target_w]

    def forward(self, x):
        # Store input size to match output
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Encoder with skip connections
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        # Decoder with skip connections and center cropping
        u1 = self.u1(d5)
        d4_cropped = self.center_crop(d4, (u1.shape[2], u1.shape[3]))
        u2 = self.u2(torch.cat([u1, d4_cropped], dim=1))
        
        d3_cropped = self.center_crop(d3, (u2.shape[2], u2.shape[3]))
        u3 = self.u3(torch.cat([u2, d3_cropped], dim=1))
        
        d2_cropped = self.center_crop(d2, (u3.shape[2], u3.shape[3]))
        u4 = self.u4(torch.cat([u3, d2_cropped], dim=1))
        
        d1_cropped = self.center_crop(d1, (u4.shape[2], u4.shape[3]))
        out = self.u5(torch.cat([u4, d1_cropped], dim=1))
        
        # Resize output to match input size (handles non-power-of-2 dimensions)
        if out.shape[2] != input_h or out.shape[3] != input_w:
            out = torch.nn.functional.interpolate(out, size=(input_h, input_w), mode='nearest')

        return self.output_act(out)


# -------------------
# Discriminator (PatchGAN)
# -------------------

class PatchDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        
        norm = nn.utils.spectral_norm if use_spectral_norm else lambda x: x

        self.model = nn.Sequential(
            norm(nn.Conv2d(2, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            norm(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, olr, rain):
        x = torch.cat([olr, rain], dim=1)
        return self.model(x)


# -------------------
# Training Entry Point
# -------------------

def train():
    """Run training - can be called from models.py or train.py"""
    # Import here to avoid circular imports
    from dataset import RainDataset
    from train import Trainer
    import config
    from torch.utils.data import DataLoader
    
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
