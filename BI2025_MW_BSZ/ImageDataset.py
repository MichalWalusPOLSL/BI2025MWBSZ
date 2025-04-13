import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.cluster import KMeans

def hex_to_rgb(h):
    h = h.lstrip('#');
    if len(h) != 6: raise ValueError()
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))

def try_hex_to_rgb(h):
    try: return hex_to_rgb(h)
    except ValueError: return None

class ImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, num_clusters=3):
        self.images_dir = images_dir; self.transform = transform; self.num_clusters = num_clusters
        image_hex_colors = {}; txt_files = [f for f in glob.glob(os.path.join(labels_dir, "*.txt")) if not f.endswith("_Time.txt")]
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip();
                    if not line or ' ' not in line: continue
                    first_space_index = line.find(' ');
                    if first_space_index == -1: continue
                    image_file = line[:first_space_index]; colors = [c.strip() for c in line[first_space_index:].strip().split(',') if c.strip().startswith('#')]
                    if len(colors) == 1:
                        image_path = os.path.join(self.images_dir, image_file)
                        if os.path.exists(image_path): image_hex_colors.setdefault(image_path, []).append(colors[0])
        final_data_list = []
        for image_path, hex_list in image_hex_colors.items():
            rgb_list = [rgb for h in hex_list if (rgb := try_hex_to_rgb(h)) is not None]
            if not rgb_list: continue
            rgb_array = np.array(rgb_list); representative_rgb = (0,0,0)
            if len(rgb_array) == 1: representative_rgb = tuple(rgb_array[0])
            elif 1 < len(rgb_array) <= self.num_clusters: representative_rgb = tuple(np.mean(rgb_array, axis=0))
            elif len(rgb_array) > self.num_clusters:
                 try:
                     kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init='auto').fit(rgb_array)
                     dominant_label = np.argmax(np.bincount(kmeans.labels_))
                     representative_rgb = tuple(kmeans.cluster_centers_[dominant_label])
                 except Exception: representative_rgb = tuple(np.mean(rgb_array, axis=0))
            normalized_rgb = [c / 255.0 for c in representative_rgb]
            final_data_list.append({'image_path': image_path, 'rgb_label': normalized_rgb})
        self.data = final_data_list

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]; image = Image.open(record['image_path']).convert('RGB')
        label_tensor = torch.tensor(record['rgb_label'], dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, label_tensor