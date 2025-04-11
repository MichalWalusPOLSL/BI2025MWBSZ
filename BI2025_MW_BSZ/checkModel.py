# checkModel.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import sys
import matplotlib.patches as patches
import numpy as np

try: from train import SimpleCNN
except ImportError: SimpleCNN = None #
try: from ImageDataset import ImageDataset
except ImportError: ImageDataset = None
# -----------------------------

# Funkcja hex_to_rgb potrzebna do wizualizacji koloru tekstu na patchu
def hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) != 6: return (0, 0, 0)
    try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except ValueError: return (0, 0, 0)


def get_color_mapping(images_dir_train, labels_dir_train):
     if ImageDataset is None: return None, 0
     temp_dataset = ImageDataset(images_dir=images_dir_train, labels_dir=labels_dir_train, transform=None)
     if hasattr(temp_dataset, 'idx_to_color') and temp_dataset.idx_to_color:
         return temp_dataset.idx_to_color, temp_dataset.num_classes
     return None, 0


IMAGES_TO_PREDICT_DIR = 'Data/PhotosToPredict'
MODEL_LOAD_PATH = 'simple_cnn_color_model.pth'
IMAGES_DIR_TRAIN = 'Data/PhotosColorPicker'
LABELS_DIR_TRAIN = 'Data/Res_ColorPickerCustomPicker'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    if SimpleCNN is None or ImageDataset is None: exit()

    idx_to_color_map, num_classes_saved = get_color_mapping(IMAGES_DIR_TRAIN, LABELS_DIR_TRAIN)
    if idx_to_color_map is None or num_classes_saved == 0: exit()

    model = SimpleCNN(num_classes=num_classes_saved)
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()

    image_size = 224
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_files.extend(glob.glob(os.path.join(IMAGES_TO_PREDICT_DIR, ext)))

    if not image_files: exit()

    for image_path in image_files:
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = inference_transform(original_image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad(): output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx_tensor = torch.max(probabilities, 1)
            predicted_idx = predicted_idx_tensor.item()
            predicted_confidence = confidence.item()

            predicted_color_hex = idx_to_color_map.get(predicted_idx, "#000000") # Domyślny czarny w razie błędu mapowania

            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            axes[0].imshow(original_image)
            axes[0].set_title("Oryginał")
            axes[0].axis('off')

            axes[1].set_facecolor(predicted_color_hex)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_title(f"{predicted_color_hex}\nConf: {predicted_confidence:.2f}")

            plt.tight_layout(pad=2.0)
            plt.show(block=True)