# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
# Upewnij się, że nazwa pliku jest poprawna (np. ImageDataset.py lub color_dataset.py)
from ImageDataset import ImageDataset
from tqdm import tqdm
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.color import lab2rgb

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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*28*28, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 3), nn.Sigmoid())
    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x)
        x = self.classifier(x); return x

def denormalize_img(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1); std = torch.tensor(std).view(3, 1, 1)
    if tensor.is_cuda: mean, std = mean.to(tensor.device), std.to(tensor.device)
    tensor = tensor * std + mean; return torch.clamp(tensor, 0, 1)

if __name__ == '__main__':
    IMAGES_DIR = 'Data/PhotosColorPicker'; LABELS_DIR = 'Data/Res_ColorPickerCustomPicker'
    MODEL_SAVE_PATH = 'simple_cnn_color_lab_regression_model_lab.pth'
    BATCH_SIZE = 24; LEARNING_RATE = 0.001; NUM_EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORM_MEAN = [0.485, 0.456, 0.406]; NORM_STD = [0.229, 0.224, 0.225]
    image_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])

    dataset = ImageDataset(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, transform=train_transform)

    # Usunięto sprawdzanie len(dataset) > 0 dla skrócenia
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = SimpleCNN().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train(); running_loss = 0.0; total_items = 0
        progress_bar = tqdm(train_loader, desc=f"E: {epoch+1}/{NUM_EPOCHS}", leave=False, unit="b")
        for inputs, labels_lab_norm in progress_bar:
             inputs, labels_lab_norm = inputs.to(DEVICE), labels_lab_norm.to(DEVICE)
             optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels_lab_norm)
             loss.backward(); optimizer.step()
             running_loss += loss.item() * inputs.size(0); total_items += inputs.size(0)
             progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / total_items # Usunięto sprawdzenie total_items > 0
        print(f"E: {epoch+1}/{NUM_EPOCHS}, Loss (MSE on LAB): {epoch_loss:.6f}") # Zachowano ten print

    model.eval()
    num_to_show = min(BATCH_SIZE, 9)
    vis_batch_inputs, _ = next(iter(train_loader))
    vis_batch_inputs = vis_batch_inputs.to(DEVICE)
    with torch.no_grad(): pred_outputs_lab_norm = model(vis_batch_inputs)

    rows = int(math.ceil(num_to_show / 3.0)); cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    if rows * cols == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i in range(num_to_show):
        if i >= len(vis_batch_inputs): break
        img_tensor = vis_batch_inputs[i].cpu()
        pred_lab_norm = pred_outputs_lab_norm[i].cpu().numpy()
        pred_lab_denorm = denormalize_lab(pred_lab_norm)
        predicted_color_hex = convert_lab_to_hex(pred_lab_denorm)
        img_vis = denormalize_img(img_tensor, NORM_MEAN, NORM_STD)
        img_vis_np = img_vis.permute(1, 2, 0).numpy()
        ax = axes[i]; ax.imshow(img_vis_np)
        patch_height = 20
        rect = patches.Rectangle((0, 0), image_size, patch_height, transform=ax.transData, facecolor=predicted_color_hex, edgecolor='k', lw=0.5)
        ax.add_patch(rect)
        text_color = 'white' if pred_lab_denorm[0] < 50 else 'black'
        ax.text(image_size / 2, patch_height / 2, f"{predicted_color_hex}", ha='center', va='center', color=text_color, fontsize=9, transform=ax.transData)
        ax.axis('off')

    for j in range(num_to_show, len(axes)): axes[j].axis('off')
    plt.tight_layout(pad=0.5); plt.show()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)