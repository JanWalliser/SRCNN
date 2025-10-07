from srcnn_model import SRCNN
from utils import load_model
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage


def super_resolve_image(model, low_res_image_path, output_path):
    model.eval()
    transform = ToTensor()
    image = Image.open(low_res_image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = ToPILImage()(output_tensor.squeeze(0).cpu())
    output_image.save(output_path)


if __name__ == "__main__":
    model = SRCNN().cuda()
    checkpoint_path = "checkpoints2/srcnn_epoch_365.pth"
    load_model(model, checkpoint_path)

    low_res_image_path = "data/low_res/1.jpg"
    output_path = "data/results/1.jpg"
    super_resolve_image(model, low_res_image_path, output_path)
