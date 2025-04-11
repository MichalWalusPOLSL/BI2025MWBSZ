import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.cluster import KMeans
import warnings

def hex_to_rgb(h):
    h = h.lstrip('#')
    if len(h) != 6: raise ValueError(f"Invalid hex: {h}")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))

class ImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, num_clusters=3):
        self.images_dir = images_dir
        self.transform = transform
        self.num_clusters = num_clusters
        image_hex_colors = {}
        txt_files = [f for f in glob.glob(os.path.join(labels_dir, "*.txt")) if not f.endswith("_Time.txt")]

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line or ' ' not in line: continue
                    first_space_index = line.find(' ')
                    if first_space_index == -1: continue
                    image_file = line[:first_space_index]
                    colors = [c.strip() for c in line[first_space_index:].strip().split(',') if c.strip().startswith('#')]
                    if len(colors) == 1:
                        image_path = os.path.join(self.images_dir, image_file)
                        if os.path.exists(image_path):
                            image_hex_colors.setdefault(image_path, []).append(colors[0])

        final_data_list = []
        unique_colors_set = set()
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')

        for image_path, hex_list in image_hex_colors.items():
            rgb_list = []
            for h in hex_list:
                try: rgb_list.append(hex_to_rgb(h))
                except ValueError: pass

            if not rgb_list: continue
            rgb_array = np.array(rgb_list)
            representative_hex = "#000000"

            if len(rgb_array) == 1:
                representative_hex = rgb_to_hex(tuple(rgb_array[0]))
            elif 1 < len(rgb_array) <= self.num_clusters:
                representative_hex = rgb_to_hex(tuple(np.mean(rgb_array, axis=0)))
            elif len(rgb_array) > self.num_clusters:
                 try:
                     kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto').fit(rgb_array)
                     dominant_label = np.argmax(np.bincount(kmeans.labels_))
                     centroid_rgb = kmeans.cluster_centers_[dominant_label]
                     representative_hex = rgb_to_hex(tuple(centroid_rgb))
                 except Exception:
                      representative_hex = rgb_to_hex(tuple(np.mean(rgb_array, axis=0)))


            final_data_list.append({'image_path': image_path, 'color': representative_hex})
            unique_colors_set.add(representative_hex)

        warnings.filterwarnings("default", category=UserWarning, module='sklearn.cluster._kmeans')
        self.data = final_data_list
        self.sorted_colors = sorted(list(unique_colors_set))
        self.color_to_idx = {color: i for i, color in enumerate(self.sorted_colors)}
        self.idx_to_color = {i: color for color, i in self.color_to_idx.items()}
        self.num_classes = len(self.sorted_colors)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image = Image.open(record['image_path']).convert('RGB')
        label_idx = self.color_to_idx[record['color']]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label_idx, dtype=torch.long)