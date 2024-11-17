import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class MulticlassTextDataset(Dataset):
    def __init__(self, folder_path):
        self.data = self._load_data(folder_path)

    def _load_data(self, folder_path):
        # Replace this with actual loading logic for text and labels
        return [
            {"text": "Sample text 1", "label": 0},
            {"text": "Sample text 2", "label": 1},
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        tokens = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return (input_ids, attention_mask), torch.tensor(label)


def collate_fn(batch):
    inputs, labels = zip(*batch)
    input_ids = torch.stack([input[0] for input in inputs])
    attention_mask = torch.stack([input[1] for input in inputs])
    labels = torch.tensor(labels)
    return (input_ids, attention_mask), labels


def get_dataset(folder_path):
    return MulticlassTextDataset(folder_path)
