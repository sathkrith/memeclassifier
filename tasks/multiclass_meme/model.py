import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import RobertaModel
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from .dataset import get_dataset, collate_fn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of text and image embeddings.
    """
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.attn = nn.Linear(input_dim * 2, 1)

    def forward(self, text_features, image_features):
        # Concatenate text and image features
        combined = torch.cat((text_features, image_features), dim=1)
        # Compute attention weights
        weights = torch.relu(self.attn(combined))
        # Weighted sum of text and image features
        fused = weights * text_features + (1 - weights) * image_features
        return fused


class MulticlassMemeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MulticlassMemeClassifier, self).__init__()

        # Text encoder: Pretrained BERT
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        self.text_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3)  # Add dropout
        )

        # Image encoder: Pretrained ResNet
        resnet = resnet50(weights='DEFAULT')
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
        self.image_fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3)  # Add dropout
        )

        # Attention-based fusion
        self.fusion = AttentionFusion(256)

        # Output layer
        self.output_fc = nn.Linear(256, num_classes)

    def forward(self, text_input_ids, text_attention_mask, images):
        # Text encoding
        text_features = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask).pooler_output
        text_features = self.text_fc(text_features)

        # Image encoding
        image_features = self.image_encoder(images).view(images.size(0), -1)
        image_features = self.image_fc(image_features)

        # Attention-based fusion
        fused_features = self.fusion(text_features, image_features)

        # Output layer
        output = self.output_fc(fused_features)
        return output


def get_model(num_classes):
    return MulticlassMemeClassifier(num_classes)


def get_loss_function():
    return nn.BCEWithLogitsLoss()  # Multi-label classification


def validate(model, val_loader, device):
    """
    Validate the model and compute accuracy, precision, recall, and F1 score.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        dict: Metrics including accuracy, precision, recall, and F1 score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            text_input_ids, attention_masks, images, labels = [b.to(device) for b in batch]

            # Forward pass
            outputs = model(text_input_ids, attention_masks, images)

            # Apply sigmoid and threshold
            preds = (torch.sigmoid(outputs) > 0.5).int()

            # Collect predictions and true labels
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def train(args):
    """
    Train the Multiclass Meme Classifier.

    Args:
        args (Namespace): Arguments containing training configurations.
    """
    # Load training dataset
    train_dataset = get_dataset(args.train_folder)
    
    # Load validation dataset using the same label set as the training dataset
    val_dataset = get_dataset(args.dev_folder, all_labels=train_dataset.all_labels)

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Create model, optimizer, and loss function
    num_classes = len(train_dataset.all_labels)
    model = get_model(num_classes)
    for param in model.text_encoder.parameters():
        param.requires_grad = True
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4 )
    criterion = get_loss_function()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0

        # Shuffle dataset manually each epoch
        train_loader.dataset.data = torch.utils.data.Subset(train_loader.dataset.data, torch.randperm(len(train_loader.dataset)))

        for batch in train_loader:
            text_input_ids, attention_masks, images, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(text_input_ids, attention_masks, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {train_loss / len(train_loader)}")

        # Validation
        metrics = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.num_epochs}, "
              f"Validation Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1 Score: {metrics['f1']:.4f}")
        
        metrics = validate(model, train_loader, device)
        print(f"Epoch {epoch+1}/{args.num_epochs}, "
              f"Training Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1 Score: {metrics['f1']:.4f}")
