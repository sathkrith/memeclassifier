from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image

class BinaryMemeDataset(Dataset):
    def __init__(self, folder_path):
        self.data = self._load_data(folder_path)

    def _load_data(self, folder_path):
        # Replace this with actual loading logic for images and labels
        return [
            {"image": "path/to/image1.png", "label": 0},
            {"image": "path/to/image2.png", "label": 1},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        label = torch.tensor(item["label"], dtype=torch.float)
        return image, label


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
    labels = torch.tensor(labels)
    return images, labels


def get_dataset(folder_path):
    return BinaryMemeDataset(folder_path)
