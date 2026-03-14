# config.py

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
OLR_FOLDERS = ["data/processed_olr/2023", "data/processed_olr/2024"]
RAIN_FOLDERS = ["data/processed_rain/2023", "data/processed_rain/2024"]

# Training parameters
BATCH_SIZE = 4
LR = 2e-4
BETAS = (0.5, 0.999)
EPOCHS = 100

# Loss weights
LAMBDA_L1 = 100  # Reconstruction loss weight
LAMBDA_PERCEPTUAL = 10  # Perceptual loss weight (VGG-based)

# Learning rate scheduling
LR_DECAY_INTERVAL = 20  # Decay LR every N epochs
LR_DECAY_GAMMA = 0.5  # Multiply LR by this factor

# Gradient clipping
GRAD_CLIP = 1.0

# Checkpointing
CHECKPOINT_DIR = "checkpoints-1"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs

# Data loading
NUM_WORKERS = 2
PIN_MEMORY = True