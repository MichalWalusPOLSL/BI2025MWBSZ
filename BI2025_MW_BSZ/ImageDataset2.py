# import os
# import glob
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# from sklearn.cluster import KMeans
# from skimage.color import rgb2lab
# import warnings
#
# # --- Funkcje Pomocnicze ---
# def hex_to_rgb(h):
#     h = h.lstrip('#')
#     if len(h) != 6:
#         raise ValueError()
#     return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
#
# def try_hex_to_rgb(h):
#     try:
#         return hex_to_rgb(h)
#     except ValueError:
#         return None
#
# def convert_hex_list_to_lab_array(hex_list):
#     rgb_list_0_255 = [rgb for h in hex_list if (rgb := try_hex_to_rgb(h)) is not None]
#     if not rgb_list_0_255:
#         return None, None
#     rgb_array_0_255 = np.array(rgb_list_0_255)
#     rgb_array_0_1 = rgb_array_0_255 / 255.0
#     try:
#         lab_array = rgb2lab(rgb_array_0_1)
#         return lab_array, rgb_array_0_1
#     except Exception:
#         return None, None
#
# def normalize_lab(lab_color):
#     l, a, b = lab_color
#     l_norm = l / 100.0
#     a_norm = (a + 128.0) / 255.0
#     b_norm = (b + 128.0) / 255.0
#     return [np.clip(v, 0.0, 1.0) for v in [l_norm, a_norm, b_norm]]
#
# def get_top_two_clusters_centers(lab_array, num_clusters):
#     kmeans = KMeans(n_clusters=min(num_clusters, len(lab_array)), random_state=42, n_init='auto').fit(lab_array)
#     labels = kmeans.labels_
#     centers = kmeans.cluster_centers_
#     counts = np.bincount(labels)
#     top_two_indices = counts.argsort()[-2:][::-1]  # indeksy 2 największych klastrów
#     top_two_centers = centers[top_two_indices]
#     return top_two_centers
#
# class ImageDataset2Color(Dataset):
#     def __init__(self, images_dir, labels_dir, transform=None, num_clusters=4):
#         self.images_dir = images_dir
#         self.transform = transform
#         self.num_clusters = num_clusters
#         image_hex_colors = {}
#         txt_files = [f for f in glob.glob(os.path.join(labels_dir, "*.txt")) if not f.endswith("_Time.txt")]
#
#         for txt_file in txt_files:
#             with open(txt_file, 'r', encoding='utf-8') as file:
#                 for line in file:
#                     line = line.strip()
#                     if not line or ' ' not in line:
#                         continue
#                     image_file, color_part = line.split(' ', 1)
#                     colors = [c.strip() for c in color_part.split(',') if c.strip().startswith('#')]
#                     if len(colors) == 2:  # wybieramy tylko linie z DOKLADNIE 2 kolorami
#                         image_path = os.path.join(self.images_dir, image_file)
#                         if os.path.exists(image_path):
#                             image_hex_colors.setdefault(image_path, []).extend(colors)
#
#         final_data_list = []
#         warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
#         for image_path, hex_list in image_hex_colors.items():
#             lab_array, _ = convert_hex_list_to_lab_array(hex_list)
#             if lab_array is None or len(lab_array) < 2:
#                 continue
#             try:
#                 top_two_centers = get_top_two_clusters_centers(lab_array, self.num_clusters)
#                 if len(top_two_centers) == 2:
#                     c1 = normalize_lab(top_two_centers[0])
#                     c2 = normalize_lab(top_two_centers[1])
#                     lab_label = c1 + c2  # [L1, a1, b1, L2, a2, b2]
#                     final_data_list.append({'image_path': image_path, 'lab_label': lab_label})
#             except Exception:
#                 continue
#
#         self.data = final_data_list
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         record = self.data[idx]
#         image = Image.open(record['image_path']).convert('RGB')
#         label_tensor = torch.tensor(record['lab_label'], dtype=torch.float32)
#         if self.transform:
#             image = self.transform(image)
#         return image, label_tensor
#
# if __name__ == '__main__':
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
#
#     IMAGES_DIR_TEST = 'Data/PhotosColorPicker'
#     LABELS_DIR_TEST = 'Data/Res_ColorPickerCustomPicker'
#     NUM_CLUSTERS_VIS = 4
#
#     dataset = ImageDataset2Color(images_dir=IMAGES_DIR_TEST,
#                                   labels_dir=LABELS_DIR_TEST,
#                                   transform=None,
#                                   num_clusters=NUM_CLUSTERS_VIS)
#
#     for record in dataset.data:
#         image_path = record['image_path']
#         lab_label = record['lab_label']  # [L1, a1, b1, L2, a2, b2]
#
#         original_image = Image.open(image_path).convert('RGB')
#         fig = plt.figure(figsize=(12, 6))
#         ax1 = fig.add_subplot(1, 2, 1)
#         ax1.imshow(original_image)
#         ax1.set_title(os.path.basename(image_path))
#         ax1.axis('off')
#
#         ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#
#         # Punkty LAB
#         c1 = lab_label[:3]
#         c2 = lab_label[3:]
#
#         def denormalize_lab(lab):
#             L = lab[0] * 100.0
#             a = lab[1] * 255.0 - 128.0
#             b = lab[2] * 255.0 - 128.0
#             return [L, a, b]
#
#         def lab_to_rgb_patch(lab_color):
#             lab = np.array(lab_color).reshape(1, 1, 3)
#             from skimage.color import lab2rgb
#             rgb = lab2rgb(lab).reshape(3)
#             return np.clip(rgb, 0, 1)
#
#         c1_lab = denormalize_lab(c1)
#         c2_lab = denormalize_lab(c2)
#         c1_rgb = lab_to_rgb_patch(c1_lab)
#         c2_rgb = lab_to_rgb_patch(c2_lab)
#
#         ax2.scatter(c1_lab[1], c1_lab[2], c1_lab[0], color=c1_rgb, s=100, edgecolor='k', label='Dominujący 1')
#         ax2.scatter(c2_lab[1], c2_lab[2], c2_lab[0], color=c2_rgb, s=100, edgecolor='k', label='Dominujący 2')
#
#         ax2.set_xlabel('a*')
#         ax2.set_ylabel('b*')
#         ax2.set_zlabel('L*')
#         ax2.set_xlim([-100, 100])
#         ax2.set_ylim([-100, 100])
#         ax2.set_zlim([0, 100])
#         ax2.view_init(elev=20., azim=-65)
#         ax2.set_title('Dwa dominujące kolory (LAB)')
#         ax2.legend(fontsize='small')
#
#         plt.tight_layout()
#         plt.show()
#
