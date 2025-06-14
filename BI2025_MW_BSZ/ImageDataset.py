import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
import warnings

# --- Funkcje Pomocnicze ---
def hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) != 6:
        raise ValueError()
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def try_hex_to_rgb(h):
    try:
        return hex_to_rgb(h)
    except ValueError:
        return None

def convert_hex_list_to_lab_array(hex_list):
    rgb_list_0_255 = [rgb for h in hex_list if (rgb := try_hex_to_rgb(h)) is not None]
    if not rgb_list_0_255:
        return None, None
    rgb_array_0_255 = np.array(rgb_list_0_255)
    rgb_array_0_1 = rgb_array_0_255 / 255.0
    try:
        lab_array = rgb2lab(rgb_array_0_1)
        return lab_array, rgb_array_0_1
    except Exception:
        return None, None

def normalize_lab(lab_color):
    l, a, b = lab_color
    l_norm = l / 100.0
    a_norm = (a + 128.0) / 255.0
    b_norm = (b + 128.0) / 255.0
    return [np.clip(v, 0.0, 1.0) for v in [l_norm, a_norm, b_norm]]

def get_top_two_clusters_centers(lab_array, num_clusters):
    kmeans = KMeans(n_clusters=min(num_clusters, len(lab_array)), random_state=42, n_init='auto').fit(lab_array)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    counts = np.bincount(labels)
    top_two_indices = counts.argsort()[-2:][::-1]
    top_two_centers = centers[top_two_indices]
    return top_two_centers

class UnifiedColorDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, mode='1color', num_clusters=3):
        assert mode in ['1color', '2color', '3color', '4color', '5color'], \
            "mode must be '1color', '2color', '3color', '4color' or '5color'"

        self.images_dir = images_dir
        self.transform = transform
        self.mode = mode
        self.num_clusters = num_clusters
        self.data = []

        image_hex_colors = {}
        txt_files = [f for f in glob.glob(os.path.join(labels_dir, "*.txt")) if not f.endswith("_Time.txt")]

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line or ' ' not in line:
                        continue
                    image_file, color_part = line.split(' ', 1)
                    colors = [c.strip() for c in color_part.split(',') if c.strip().startswith('#')]
                    image_path = os.path.join(self.images_dir, image_file)
                    if not os.path.exists(image_path):
                        continue

                    sufficient = {
                        '1color': len(colors) == 1,
                        '2color': len(colors) == 2,
                        '3color': len(colors) >= 3,
                        '4color': len(colors) >= 4,
                        '5color': len(colors) >= 5,
                    }
                    if sufficient[self.mode]:
                        image_hex_colors.setdefault(image_path, []).extend(colors)

        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')

        for image_path, hex_list in image_hex_colors.items():
            lab_array, _ = convert_hex_list_to_lab_array(hex_list)
            if lab_array is None or len(lab_array) == 0:
                continue

            try:
                kmeans = KMeans(n_clusters=min(self.num_clusters, len(lab_array)), random_state=42, n_init='auto')
                kmeans.fit(lab_array)
                centers = kmeans.cluster_centers_
                counts = np.bincount(kmeans.labels_)
                sorted_indices = counts.argsort()[::-1]

                mode_to_n = {
                    '1color': 1,
                    '2color': 2,
                    '3color': 3,
                    '4color': 4,
                    '5color': 5
                }
                n_colors = mode_to_n[self.mode]
                if len(sorted_indices) < n_colors:
                    continue

                lab_label = []
                for idx in sorted_indices[:n_colors]:
                    lab_label.extend(normalize_lab(centers[idx]))

                self.data.append({'image_path': image_path, 'lab_label': lab_label})
            except Exception:
                continue

        warnings.filterwarnings("default", category=UserWarning, module='sklearn.cluster._kmeans')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image = Image.open(record['image_path']).convert('RGB')
        label_tensor = torch.tensor(record['lab_label'], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label_tensor



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image = Image.open(record['image_path']).convert('RGB')
        label_tensor = torch.tensor(record['lab_label'], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label_tensor
