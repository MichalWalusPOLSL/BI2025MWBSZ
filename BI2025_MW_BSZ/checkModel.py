# checkModel.py
import matplotlib
import matplotlib.pyplot as plt
#print(matplotlib.get_backend())
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import sys
import matplotlib.patches as patches
import numpy as np
from skimage.color import lab2rgb
from train import SimpleCNN

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))

def denormalize_lab(lab_norm):
    l = lab_norm[0] * 100.0; a = lab_norm[1] * 255.0 - 128.0; b = lab_norm[2] * 255.0 - 128.0
    return np.array([l, a, b])

def convert_lab_to_hex(lab_color):
    rgb_0_1 = lab2rgb(np.array(lab_color).reshape(1, 1, 3)).flatten()
    rgb_0_1_clipped = np.clip(rgb_0_1, 0, 1)
    rgb_0_255 = tuple(max(0, min(255, int(round(c * 255.0)))) for c in rgb_0_1_clipped)
    return rgb_to_hex(rgb_0_255)

IMAGES_TO_PREDICT_DIR = 'Data/PhotosToPredict'
MODEL_LOAD_PATH = 'simple_cnn_color_lab_regression_model_lab.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    plt.close('all')
    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()
    image_size = 224
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_files = [f for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif'] for f in glob.glob(os.path.join(IMAGES_TO_PREDICT_DIR, ext))]
    if not image_files: exit()
    for image_path in image_files:
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = inference_transform(original_image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)
            with torch.no_grad(): pred_outputs_lab_norm = model(input_batch)
            pred_lab_norm = pred_outputs_lab_norm[0].cpu().numpy()
            pred_lab_denorm = denormalize_lab(pred_lab_norm)
            predicted_color_hex = convert_lab_to_hex(pred_lab_denorm)
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            axes[0].imshow(original_image); axes[0].set_title("Oryginał"); axes[0].axis('off')
            axes[1].set_facecolor(predicted_color_hex); axes[1].set_xticks([]); axes[1].set_yticks([])
            axes[1].set_title(f"Pred (z LAB): {predicted_color_hex}")
            plt.tight_layout(pad=2.0);
            plt.show()