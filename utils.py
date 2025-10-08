import torch
import os

def save_model(model, epoch, save_dir="Checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"srcnn_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Model loaded from {checkpoint_path}")
    return model

def test_model_single_image(model, image_tensor,device='cpu'):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))

    return output.squeeze(0).to(device)