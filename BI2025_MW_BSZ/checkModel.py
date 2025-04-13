# checkModel.py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
import os
import glob


from train import SimpleCNN

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(round(c)))) for c in rgb))

IMAGES_TO_PREDICT_DIR = 'Data/PhotosToPredict'
MODEL_LOAD_PATH = 'simple_cnn_color_regression_model.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if SimpleCNN is None : exit()

    model = SimpleCNN()
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()

    image_size = 224
    inference_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_files.extend(glob.glob(os.path.join(IMAGES_TO_PREDICT_DIR, ext)))

    if not image_files: exit()

    for image_path in image_files:
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = inference_transform(original_image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad(): pred_outputs_norm = model(input_batch)

            pred_rgb_norm = pred_outputs_norm[0].cpu().numpy() 
            pred_rgb_denorm = [max(0, min(255, int(round(c * 255.0)))) for c in pred_rgb_norm]
            predicted_color_hex = rgb_to_hex(tuple(pred_rgb_denorm))

            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            axes[0].imshow(original_image)
            axes[0].set_title("Oryginał")
            axes[0].axis('off')

            axes[1].set_facecolor(predicted_color_hex)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            axes[1].set_title(f"Pred: {predicted_color_hex}")

            plt.tight_layout(pad=2.0)
            plt.show(block=True)