# # utils.py

# import torch
# import os
# import shutil

# def save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, loss, filename, is_best=False):
#     """
#     Save training checkpoint with both generator and discriminator states
#     """
#     checkpoint = {
#         'epoch': epoch,
#         'G_state_dict': G.state_dict(),
#         'D_state_dict': D.state_dict(),
#         'optimizer_G': optimizer_G.state_dict(),
#         'optimizer_D': optimizer_D.state_dict(),
#         'best_loss': loss
#     }
    
#     torch.save(checkpoint, filename)
    
#     # Also save as latest checkpoint
#     latest_path = os.path.join(os.path.dirname(filename), 'latest.pth')
#     shutil.copy(filename, latest_path)
    
#     # Save as best checkpoint if is_best is True
#     if is_best:
#         best_path = os.path.join(os.path.dirname(filename), 'best.pth')
#         shutil.copy(filename, best_path)
#         print(f"New best model saved: {best_path}")

# def load_checkpoint(model, optimizer, filename):
#     """Load checkpoint for single model"""
#     if os.path.exists(filename):
#         checkpoint = torch.load(filename)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         epoch = checkpoint.get('epoch', 0)
#         return epoch
#     return None

# def ensure_dir(path):
#     """Create directory if it doesn't exist"""
#     if not os.path.exists(path):
#         os.makedirs(path)

# utils.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, filename)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)