import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTModel, ViTConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# USE GPU 4

class ViT(nn.Module):
    """
    Vision Transformer (ViT) with a custom classification head.
    """

    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=10, hidden_size=768, dropout_prob=0.3):
        super(ViT, self).__init__()
        self.base_model = ViTModel.from_pretrained(model_name, output_hidden_states=True)  # Pretrained ViT
        
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)  # Pre-classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)  # Final classification layer

    def forward(self, x):

        outputs = self.base_model(pixel_values=x, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :]  # [CLS] token embedding
        pre_logits = self.pre_classifier(embeddings)
        pre_logits = torch.relu(pre_logits)
        pre_logits = self.dropout(pre_logits)
        logits = self.classifier(pre_logits)

        return {"logits": logits, "hidden_states": outputs.hidden_states}
    
    def representation(self, x):
        outputs = self.base_model(pixel_values=x, output_hidden_states=True)
        representation = outputs.hidden_states[-1][:, 0, :]  # [CLS] token embedding
        return representation

    def predict(self, x):
        logits = self.forward(x)["logits"]
        return F.softmax(logits, dim=1)

def vit_model(num_classes=10, hidden_size=768, dropout_prob=0.3):
    model = ViT(num_classes=num_classes, hidden_size=hidden_size, dropout_prob=dropout_prob)
    return model


def ce_loss(model_outputs, labels,):

    logits = model_outputs["logits"]

    # Compute Cross-Entropy Loss
    ce_loss = F.cross_entropy(logits, labels)

    total_loss = ce_loss
    return total_loss


def train_one_epoch(model, data_loader, optimizer, device, alpha, temperature):

    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()

        # Move data to device
        images = batch[0].to(device)
        labels = batch[1].to(device)

        # Forward pass
        outputs = model(images)

            # Predictions
        logits = outputs["logits"]
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())

        # Compute the combined loss
        loss= ce_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    metrics = compute_metrics(all_predictions, all_labels)
    return total_loss / len(data_loader), metrics


def evaluate(model, data_loader, device, alpha, temperature):

    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # Forward pass
            outputs = model(images)

            # Compute the combined loss
            loss = ce_loss(outputs, labels)
            total_loss += loss.item()
            # Predictions
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels)
    avg_loss = total_loss / len(data_loader)

    return avg_loss,metrics


def compute_metrics(predictions, labels):

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model