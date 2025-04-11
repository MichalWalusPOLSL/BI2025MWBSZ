# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ImageDataset import ImageDataset
from tqdm import tqdm
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Funkcja hex_to_rgb potrzebna do wizualizacji koloru tekstu na patchu
def hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) != 6: return (0, 0, 0) # Zwróć czarny w razie błędu
    try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except ValueError: return (0, 0, 0)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*28*28, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x)
        x = self.classifier(x); return x

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if tensor.is_cuda: mean, std = mean.to(tensor.device), std.to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

if __name__ == '__main__':
    IMAGES_DIR = 'Data/PhotosColorPicker'
    LABELS_DIR = 'Data/Res_ColorPickerCustomPicker'
    MODEL_SAVE_PATH = 'simple_cnn_color_model.pth'
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORM_MEAN = [0.485, 0.456, 0.406]; NORM_STD = [0.229, 0.224, 0.225]
    image_size = 224

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)])

    dataset = ImageDataset(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, transform=train_transform)

    if len(dataset) > 0 and dataset.num_classes > 0 and hasattr(dataset, 'idx_to_color'):
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        model = SimpleCNN(num_classes=dataset.num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            model.train(); running_loss = 0.0; correct_predictions = 0; total_predictions = 0
            progress_bar = tqdm(train_loader, desc=f"E: {epoch+1}/{NUM_EPOCHS}", leave=False, unit="b")
            for inputs, labels in progress_bar:
                 inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                 optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels)
                 loss.backward(); optimizer.step()
                 running_loss += loss.item() * inputs.size(0); _, predicted = torch.max(outputs.data, 1)
                 total_predictions += labels.size(0); correct_predictions += (predicted == labels).sum().item()
                 progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = running_loss / total_predictions
            epoch_acc = correct_predictions / total_predictions
            print(f"E: {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # --- WIZUALIZACJA PO TRENINGU ---
        model.eval()
        num_to_show = min(BATCH_SIZE, 9)
        vis_batch_inputs, _ = next(iter(train_loader)) 
        vis_batch_inputs = vis_batch_inputs.to(DEVICE)
        with torch.no_grad():
            outputs = model(vis_batch_inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)

        rows = int(math.ceil(num_to_show / 3.0)); cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        if rows * cols == 1: axes = np.array([axes])
        axes = axes.flatten()

        for i in range(num_to_show):
            if i >= len(vis_batch_inputs): break
            img_tensor = vis_batch_inputs[i].cpu()
            pred_idx = predicted_indices[i].item(); confidence = confidences[i].item()
            img_vis = denormalize(img_tensor, NORM_MEAN, NORM_STD)
            img_vis_np = img_vis.permute(1, 2, 0).numpy()
            predicted_color_hex = dataset.idx_to_color.get(pred_idx, "#000000")
            ax = axes[i]; ax.imshow(img_vis_np)
            patch_height = 20 # Wysokość paska koloru
            rect = patches.Rectangle((0, 0), image_size, patch_height, transform=ax.transData, facecolor=predicted_color_hex, edgecolor='k', lw=0.5)
            ax.add_patch(rect)
            text_color = 'white' if sum(hex_to_rgb(predicted_color_hex)) < 382 else 'black'
            ax.text(image_size / 2, patch_height / 2, f"{predicted_color_hex} ({confidence:.1f})", ha='center', va='center', color=text_color, fontsize=8, transform=ax.transData)
            ax.axis('off')

        for j in range(num_to_show, len(axes)): axes[j].axis('off')
        plt.tight_layout(pad=0.5); plt.show()
        # --- KONIEC WIZUALIZACJI ---

        torch.save(model.state_dict(), MODEL_SAVE_PATH)