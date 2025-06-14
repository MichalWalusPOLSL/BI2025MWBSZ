import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import numpy as np
import copy

from AugmentedDataset import AugmentedDataset
from ImageDataset import UnifiedColorDataset

# --- Konfiguracja ---
IMAGES_DIR      = 'Data/PhotosColorPicker'
LABELS_DIR      = 'Data/Res_ColorPickerCustomPicker'
MODEL_SAVE_PATH = 'best_model_4color_lab.pth'

BATCH_SIZE      = 16
LEARNING_RATE   = 0.0005
NUM_EPOCHS      = 15
N_SPLITS        = 2

MODE            = '4color'
NUM_OUTPUTS     = 12
NUM_CLUSTERS    = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# --- Model ---
class SimpleCNNUnified(nn.Module):
    def __init__(self):
        super(SimpleCNNUnified, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_OUTPUTS)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# --- Trening jednego folda ---
def train_one_fold(fold, train_idx, val_idx, dataset):
    train_base = Subset(dataset, train_idx)
    train_subset = AugmentedDataset(train_base)
    val_subset   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = SimpleCNNUnified().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float('inf')
    best_wts = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss_total += criterion(outputs, labels).item() * inputs.size(0)
        val_loss = val_loss_total / len(val_loader.dataset)

        print(f"Fold {fold}, Epoch {epoch}/{NUM_EPOCHS} → Train: {train_loss:.6f}, Val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_wts = copy.deepcopy(model.state_dict())

    return best_val, best_wts

# --- Główna funkcja ---
def main():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    dataset = UnifiedColorDataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        transform=transform,
        mode=MODE,
        num_clusters=NUM_CLUSTERS
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(kf.split(dataset))

    best_overall = float('inf')
    best_weights = None
    best_fold = -1

    for i, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n=== Start fold {i}/{N_SPLITS} ===")
        val_loss, weights = train_one_fold(i, train_idx, val_idx, dataset)
        if val_loss < best_overall:
            best_overall = val_loss
            best_weights = weights
            best_fold = i

    if best_weights is not None:
        print(f"\nNajlepszy model: fold {best_fold} z Val Loss = {best_overall:.6f}")
        best_model = SimpleCNNUnified().to(DEVICE)
        best_model.load_state_dict(best_weights)
        dirpath = os.path.dirname(MODEL_SAVE_PATH)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model zapisany do: {MODEL_SAVE_PATH}")
    else:
        print("Brak modelu do zapisania.")

if __name__ == '__main__':
    main()
