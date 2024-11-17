import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class MulticlassMemeDataset(Dataset):
    def __init__(self, folder_path, all_labels=None):
        """
        Initializes the dataset for multimodal classification.

        Args:
            folder_path (str): Path to the dataset (train or dev folder).
            all_labels (List[str], optional): Predefined list of all unique labels. Defaults to None.
        """
        self.folder_path = folder_path
        self.data = self._load_data()

        # Use predefined label set if provided, otherwise find all unique labels
        self.all_labels = all_labels if all_labels else self._find_all_classes()

    def _load_data(self):
        """
        Load the dataset from the folder.

        Returns:
            List[Dict]: List of data items.
        """
        labels_path = os.path.join(self.folder_path, "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _find_all_classes(self):
        """
        Extracts all unique labels from the dataset.

        Returns:
            List[str]: Sorted list of all unique labels.
        """
        label_set = set()
        for item in self.data:
            label_set.update(item["labels"])
        return sorted(list(label_set))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item by index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple: (text, image_path, labels)
        """
        item = self.data[idx]
        text = item["text"]
        image_path = os.path.join(self.folder_path, item["image"])
        labels = torch.zeros(len(self.all_labels), dtype=torch.float)

        for label in item["labels"]:
            if label in self.all_labels:
                labels[self.all_labels.index(label)] = 1

        return text, image_path, labels


def collate_fn(batch):
    """
    Custom collate function for multimodal data.

    Args:
        batch (List[Tuple[str, str, torch.Tensor]]): List of data items.

    Returns:
        Tuple: (text_input_ids, attention_masks, images, labels)
    """
    texts, image_paths, labels = zip(*batch)

    # Tokenize text
    text_tokens = tokenizer(
        list(texts), max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )
    text_input_ids = text_tokens["input_ids"]
    attention_masks = text_tokens["attention_mask"]

    # Load and preprocess images
    images = [
        transform_image(image) for image in (Image.open(path).convert("RGB") for path in image_paths)
    ]

    # Stack data
    images = torch.stack(images)
    labels = torch.stack(labels)

    return text_input_ids, attention_masks, images, labels

def transform_image(image):
    """
    Apply default image transformations.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)


def get_dataset(folder_path, all_labels=None):
    return MulticlassMemeDataset(folder_path, all_labels)
