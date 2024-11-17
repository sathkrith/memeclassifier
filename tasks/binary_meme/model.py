import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from .dataset import get_dataset, collate_fn


class BinaryMemeClassifier(nn.Module):
    def __init__(self):
        super(BinaryMemeClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 1)

    def forward(self, images):
        return self.resnet(images)


def get_model(num_classes):
    return BinaryMemeClassifier()


def get_loss_function():
    pass


def train(args):
    """
    Train the Binary Meme Classifier.

    Args:
        args (Namespace): Arguments containing training configurations.
    """
    # Load datasets
    train_dataset = get_dataset(args.train_folder)
    val_dataset = get_dataset(args.dev_folder)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model, optimizer, and loss function
    model = get_model()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = get_loss_function()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            images, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {train_loss / len(train_loader)}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = [b.to(device) for b in batch]
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {val_loss / len(val_loader)}")