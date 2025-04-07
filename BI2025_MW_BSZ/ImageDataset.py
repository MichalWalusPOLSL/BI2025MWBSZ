import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images_dir='/Data/PhotosColorPicker', labels_dir='/Data/Res_ColorPickerCustomPicker', transform=None):
        """
        images_dir - folder ze zdjęciami
        labels_dir - folder z wieloma plikami .txt zawierającymi etykiety
        transform - opcjonalna transformacja dla obrazu (np. z torchvision.transforms)
        """
        self.images_dir = images_dir
        self.transform = transform
        self.data = []

        txt_files = [
            file for file in glob.glob(os.path.join(labels_dir, "*.txt"))
            if not file.endswith("_Time.txt")
        ]

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    image_file = parts[0]

                    colors = [color.strip() for color in " ".join(parts[1:]).split(',')]

                    self.data.append({
                        'image_path': os.path.join(self.images_dir, image_file),
                        'colors': colors
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image = Image.open(record['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, record['colors']


if __name__ == '__main__':
    images_dir = 'Data/PhotosColorPicker'
    labels_dir = 'Data/Res_ColorPickerCustomPicker'
    dataset = ImageDataset(images_dir, labels_dir)

    for img, colors in dataset:
        print("Przetwarzany obraz:", img)
        print("Przypisane kolory:", colors)

        plt.imshow(img)
        plt.title("Kolory: " + ", ".join(colors))
        plt.axis('off')
        plt.tight_layout()
        plt.interactive(False)
        plt.show(block=True)

        break
