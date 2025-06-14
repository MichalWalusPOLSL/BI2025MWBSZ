import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.color import lab2rgb
from ImageDataset import UnifiedColorDataset

# --- PARAMETRY ---
IMAGES_DIR = 'Data/PhotosColorPicker'
LABELS_DIR = 'Data/Res_ColorPickerCustomPicker'
MODE = '3color'  # ← ZMIANA: tryb dla 5 kolorów
NUM_CLUSTERS = 7 if MODE == '5color' else (
    6 if MODE == '4color' else (
        5 if MODE == '3color' else (
            4 if MODE == '2color' else 3
        )
    )
)

# --- FUNKCJE POMOCNICZE ---
def denormalize_lab(lab):
    L = lab[0] * 100.0
    a = lab[1] * 255.0 - 128.0
    b = lab[2] * 255.0 - 128.0
    return [L, a, b]

def lab_to_rgb(lab):
    lab = np.array(lab).reshape(1, 1, 3)
    rgb = lab2rgb(lab).reshape(3)
    return np.clip(rgb, 0, 1)

# --- WIZUALIZACJA ---
dataset = UnifiedColorDataset(
    images_dir=IMAGES_DIR,
    labels_dir=LABELS_DIR,
    transform=None,
    mode=MODE,
    num_clusters=NUM_CLUSTERS
)

for record in dataset.data:
    image_path = record['image_path']
    lab_label = record['lab_label']

    original_image = Image.open(image_path).convert('RGB')
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image)
    ax1.set_title(os.path.basename(image_path))
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    num_colors = len(lab_label) // 3
    colors_lab = [denormalize_lab(lab_label[i * 3:(i + 1) * 3]) for i in range(num_colors)]
    colors_rgb = [lab_to_rgb(lab) for lab in colors_lab]

    for i, (lab, rgb) in enumerate(zip(colors_lab, colors_rgb), 1):
        ax2.scatter(lab[1], lab[2], lab[0], color=rgb, s=100, edgecolor='k', label=f'Dominujący {i}')

    ax2.set_title(f'{num_colors} dominujących kolorów (LAB)')
    ax2.set_xlabel('a*')
    ax2.set_ylabel('b*')
    ax2.set_zlabel('L*')
    ax2.set_xlim([-100, 100])
    ax2.set_ylim([-100, 100])
    ax2.set_zlim([0, 100])
    ax2.legend(fontsize='small')
    ax2.view_init(elev=20., azim=-65)

    plt.tight_layout()
    plt.show()
