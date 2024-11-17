import torch.nn as nn
from transformers import BertModel


class MulticlassTextClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MulticlassTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.pooler_output)


def get_model(num_classes):
    return MulticlassTextClassifier(num_classes)


def get_loss_function():
    return CrossEntropyLoss()
