import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from ImageDataset import ImageDataset
from tqdm import tqdm
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.color import lab2rgb
from sklearn.model_selection import KFold
import copy

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))

def denormalize_lab(lab_norm):
    l = lab_norm[0] * 100.0
    a = lab_norm[1] * 255.0 - 128.0
    b = lab_norm[2] * 255.0 - 128.0
    return np.array([l, a, b])

def convert_lab_to_hex(lab_color):
    rgb_0_1 = lab2rgb(np.array(lab_color).reshape(1, 1, 3)).flatten()
    rgb_0_1_clipped = np.clip(rgb_0_1, 0, 1)
    rgb_0_255 = tuple(max(0, min(255, int(round(c * 255.0)))) for c in rgb_0_1_clipped)
    return rgb_to_hex(rgb_0_255)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

def denormalize_img(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    if tensor.is_cuda:
        mean, std = mean.to(tensor.device), std.to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

if __name__ == '__main__':

    IMAGES_DIR = 'Data/PhotosColorPicker'
    LABELS_DIR = 'Data/Res_ColorPickerCustomPicker'
    MODEL_SAVE_PATH = 'best_simple_cnn_color_lab_regression_model_lab.pth'
    BATCH_SIZE = 24
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15
    N_SPLITS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224


    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    dataset = ImageDataset(IMAGES_DIR, LABELS_DIR, transform)


    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    best_val_loss = float('inf')
    best_model_wts = None
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"\nStarting fold {fold}/{N_SPLITS}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0)


        model = SimpleCNN().to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for inputs, labels_lab in tqdm(train_loader, desc=f"Fold {fold} Training Epoch {epoch}", leave=False):
                inputs, labels_lab = inputs.to(DEVICE), labels_lab.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_lab)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            train_loss = running_loss / len(train_idx)


            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for inputs, labels_lab in val_loader:
                    inputs, labels_lab = inputs.to(DEVICE), labels_lab.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_lab)
                    val_loss_total += loss.item() * inputs.size(0)
            val_loss = val_loss_total / len(val_idx)

            print(f"Fold {fold}, Epoch {epoch}/{NUM_EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_fold = fold


    if best_model_wts is not None:
        print(f"\nBest model found in fold {best_fold} with Val Loss: {best_val_loss:.6f}")
        best_model = SimpleCNN().to(DEVICE)
        best_model.load_state_dict(best_model_wts)
        torch.save(best_model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved best model to {MODEL_SAVE_PATH}")
    else:
        print("No best model found to save.")
