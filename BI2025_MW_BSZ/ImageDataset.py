# ImageDataset.py
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
    h = h.lstrip('#');
    if len(h) != 6: raise ValueError()
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def try_hex_to_rgb(h):
    try: return hex_to_rgb(h)
    except ValueError: return None

def convert_hex_list_to_lab_array(hex_list):
    rgb_list_0_255 = [rgb for h in hex_list if (rgb := try_hex_to_rgb(h)) is not None]
    if not rgb_list_0_255: return None, None
    rgb_array_0_255 = np.array(rgb_list_0_255)
    rgb_array_0_1 = rgb_array_0_255 / 255.0
    try:
        if rgb_array_0_1.ndim == 2: lab_array = rgb2lab(rgb_array_0_1)
        elif rgb_array_0_1.ndim == 1 and len(rgb_array_0_1) == 3: lab_array = rgb2lab(rgb_array_0_1.reshape(1, 1, 3)).reshape(1, 3)
        else: return None, None
        return lab_array, rgb_array_0_1
    except Exception: return None, None

def normalize_lab(lab_color):
    l, a, b = lab_color; l_norm = l / 100.0; a_norm = (a + 128.0) / 255.0; b_norm = (b + 128.0) / 255.0
    return [np.clip(v, 0.0, 1.0) for v in [l_norm, a_norm, b_norm]]

def process_colors_for_lab_label(hex_list, num_clusters):
    lab_array, rgb_array_0_1 = convert_hex_list_to_lab_array(hex_list)
    if lab_array is None or len(lab_array) == 0: return None, None, None, None
    representative_lab = np.array([50.0, 0.0, 0.0]); cluster_labels = None
    if len(lab_array) == 1:
        representative_lab = lab_array[0]; cluster_labels = np.array([0])
    elif 1 < len(lab_array) <= num_clusters:
         representative_lab = np.mean(lab_array, axis=0)
    else:
         try:
             kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(lab_array)
             cluster_labels = kmeans.labels_
             dominant_label = np.argmax(np.bincount(cluster_labels))
             representative_lab = kmeans.cluster_centers_[dominant_label]
         except Exception: representative_lab = np.mean(lab_array, axis=0)
    return representative_lab, lab_array, rgb_array_0_1, cluster_labels
# --- Koniec Funkcji Pomocniczych ---

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
        self.image_color_details = {}
        final_data_list = []
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
        warnings.filterwarnings("ignore", category=FutureWarning, module='skimage.color')
        for image_path, hex_list in image_hex_colors.items():
            representative_lab, lab_array, rgb_array_0_1, cluster_labels = process_colors_for_lab_label(hex_list, self.num_clusters)
            if representative_lab is not None:
                normalized_lab_label = normalize_lab(representative_lab)
                final_data_list.append({'image_path': image_path, 'lab_label': normalized_lab_label})
                if lab_array is not None: # Zapisz szczegóły tylko jeśli istnieją
                     self.image_color_details[image_path] = {'lab_array': lab_array, 'rgb_array_0_1': rgb_array_0_1, 'cluster_labels': cluster_labels}
        warnings.filterwarnings("default", category=UserWarning, module='sklearn.cluster._kmeans')
        warnings.filterwarnings("default", category=FutureWarning, module='skimage.color')
        self.data = final_data_list

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]; image = Image.open(record['image_path']).convert('RGB')
        label_tensor = torch.tensor(record['lab_label'], dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, label_tensor

# --- Blok Wizualizacyjny (zachowany) ---
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    IMAGES_DIR_TEST = 'Data/PhotosColorPicker'
    LABELS_DIR_TEST = 'Data/Res_ColorPickerCustomPicker'
    NUM_CLUSTERS_VIS = 3

    dataset = ImageDataset(images_dir=IMAGES_DIR_TEST, labels_dir=LABELS_DIR_TEST, transform=None, num_clusters=NUM_CLUSTERS_VIS)

    if hasattr(dataset, 'image_color_details') and dataset.image_color_details:
        markers = ['o', 's', '^', 'v', 'X', 'P', '*']
        for image_path, details in dataset.image_color_details.items():
            lab_array = details.get('lab_array') # Użyj get dla bezpieczeństwa
            rgb_array_0_1 = details.get('rgb_array_0_1')
            cluster_labels = details.get('cluster_labels')

            if lab_array is None or rgb_array_0_1 is None: continue # Pomiń, jeśli brakuje danych

            original_image = Image.open(image_path).convert('RGB')
            fig = plt.figure(figsize=(13, 6))
            ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(original_image); ax1.set_title(f"{os.path.basename(image_path)}"); ax1.axis('off')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            # Sprawdź czy klastrowanie zostało wykonane (cluster_labels nie jest None i jest wystarczająco punktów)
            if cluster_labels is not None and len(lab_array) > NUM_CLUSTERS_VIS:
                unique_labels = np.unique(cluster_labels)
                for i, label in enumerate(unique_labels):
                    mask = (cluster_labels == label)
                    marker = markers[i % len(markers)]
                    # Upewnij się, że maska ma ten sam wymiar co rgb_array_0_1
                    if mask.shape[0] == rgb_array_0_1.shape[0]:
                         ax2.scatter(lab_array[mask, 1], lab_array[mask, 2], lab_array[mask, 0],
                                     c=rgb_array_0_1[mask], marker=marker, s=60,
                                     edgecolor='k', linewidth=0.5, label=f'Klaster {label}')
                    else: # Zabezpieczenie przed niezgodnością wymiarów
                         ax2.scatter(lab_array[mask, 1], lab_array[mask, 2], lab_array[mask, 0],
                                     marker=marker, s=60, edgecolor='k', linewidth=0.5, label=f'Klaster {label}')

                ax2.set_title(f"Klastry LAB (K={NUM_CLUSTERS_VIS}, {len(lab_array)} pkt.)")
                if len(unique_labels) > 1: ax2.legend(fontsize='small') # Pokaż legendę tylko jeśli jest >1 klaster
            else:
                 ax2.scatter(lab_array[:, 1], lab_array[:, 2], lab_array[:, 0], c=rgb_array_0_1, marker='o', s=50, edgecolor='k', linewidth=0.5)
                 ax2.set_title(f"Sugestie LAB ({len(lab_array)} pkt.)")

            ax2.set_xlabel('a*'); ax2.set_ylabel('b*'); ax2.set_zlabel('L*')
            ax2.set_xlim([-100, 100]); ax2.set_ylim([-100, 100]); ax2.set_zlim([0, 100])
            ax2.view_init(elev=20., azim=-65)
            plt.tight_layout(); plt.show(block=True)