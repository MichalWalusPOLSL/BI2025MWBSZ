import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from train import SimpleCNN
from ImageDataset import ImageDataset


def train_one_fold(model, train_loader, val_loader, device, num_epochs, lr):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)
    return val_loss / count


def cross_val_loss(batch_size, num_epochs, lr, n_splits=5, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(
        images_dir='Data/PhotosColorPicker',
        labels_dir='Data/Res_ColorPickerCustomPicker',
        transform=transform
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    fold_losses = []
    for train_idx, val_idx in kf.split(dataset):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = SimpleCNN().to(device)
        loss = train_one_fold(model, train_loader, val_loader, device, num_epochs, lr)
        fold_losses.append(loss)

    return np.mean(fold_losses)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_sizes = [4, 8, 16, 20, 24]
    epochs_list = [3, 5, 7, 10, 15, 20]
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]

    best_loss = float('inf')
    best_config = None

    for bs in batch_sizes:
        for ep in epochs_list:
            for lr in learning_rates:
                print(f"Evaluating bs={bs}, epochs={ep}, lr={lr}...")
                avg_loss = cross_val_loss(bs, ep, lr, n_splits=5, device=device)
                print(f"  -> Avg val loss: {avg_loss:.4f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_config = (bs, ep, lr)

    print("\nBest configuration:")
    print(
        f"  batch_size={best_config[0]}, epochs={best_config[1]}, learning_rate={best_config[2]}, avg_val_loss={best_loss:.4f}")

