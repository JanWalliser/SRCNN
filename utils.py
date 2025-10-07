import torch
import os

def save_model(model, epoch, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"srcnn_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model loaded from {checkpoint_path}")
    return model
