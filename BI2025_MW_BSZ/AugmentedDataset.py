import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        angles = [0, 90, 180, 270]
        self.aug_params = [(a, False) for a in angles] + [(a, True) for a in angles]

    def __len__(self):
        return len(self.base) * len(self.aug_params)

    def __getitem__(self, idx):
        base_idx = idx // len(self.aug_params)
        aug_idx  = idx %  len(self.aug_params)
        img, label = self.base[base_idx]
        angle, do_flip = self.aug_params[aug_idx]
        if angle != 0:
            img = TF.rotate(img, angle)
        if do_flip:
            img = TF.hflip(img)

        return img, label
