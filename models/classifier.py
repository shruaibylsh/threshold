"""
Classification fine-tuning for TimeSformer encoder.
After MAE pretraining, this module adds a classification head and
fine-tunes the encoder on labeled threshold sequences.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class ThresholdClassifier(nn.Module):
    """
    TimeSformer encoder with classification head.
    Loads pretrained encoder from MAE and adds classification head
    for 8 threshold typologies.
    """
    def __init__(self, encoder, num_classes=8, hidden_size=384, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values, return_embeddings=False):
        """
        Forward pass.
        Args:
            pixel_values: (B, T, C, H, W) video frames
            return_embeddings: If True, return embeddings before classification
        Returns:
            logits: (B, num_classes) classification logits
            embeddings: (B, hidden_size) if return_embeddings=True
        """
        encoded = self.encoder(pixel_values)
        embeddings = encoded.mean(dim=1)
        logits = self.classifier(embeddings)
        if return_embeddings:
            return logits, embeddings
        return logits

def load_pretrained_encoder(checkpoint_path, config):
    """
    Load pretrained encoder from MAE checkpoint.
    Args:
        checkpoint_path: Path to timesformer_encoder_pretrained.pt
        config: Model configuration
    Returns:
        encoder: Pretrained TimeSformer encoder
    """
    from timesformer_mae import TimeSformerEncoder
    encoder = TimeSformerEncoder(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    print(f"✓ Loaded pretrained encoder from: {checkpoint_path}")
    return encoder

def compute_metrics(preds, labels, typology_names=None):
    """
    Compute classification metrics.
    Args:
        preds: (N,) predicted class indices
        labels: (N,) ground truth class indices
        typology_names: List of class names (optional)
    Returns:
        metrics: Dictionary of metrics
    """
    if typology_names is None:
        typology_names = [f't{i+1}' for i in range(8)]
    accuracy = accuracy_score(labels, preds)
    conf_matrix = confusion_matrix(labels, preds)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'per_class_accuracy': per_class_acc,
    }
    report = classification_report(
        labels, preds,
        target_names=typology_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report
    return metrics

def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings from all samples in dataloader.
    Args:
        model: ThresholdClassifier model
        dataloader: DataLoader with samples
        device: Device to run on
    Returns:
        embeddings: (N, hidden_size) numpy array
        labels: (N,) numpy array of class labels
        predictions: (N,) numpy array of predicted labels
    """
    model.eval()
    all_embeddings, all_labels, all_predictions = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            logits, embeddings = model(pixel_values, return_embeddings=True)
            preds = logits.argmax(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    return embeddings, labels, predictions

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Check if training should stop.
        Args:
            score: Current metric value (accuracy or loss)
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

def freeze_encoder(model, freeze=True):
    """
    Freeze or unfreeze encoder parameters.
    Args:
        model: ThresholdClassifier model
        freeze: If True, freeze encoder. If False, unfreeze.
    """
    for param in model.encoder.parameters():
        param.requires_grad = not freeze
    if freeze:
        print("🔒 Encoder frozen (only training classification head)")
    else:
        print("🔓 Encoder unfrozen (fine-tuning entire model)")

def get_lr_scheduler(optimizer, num_epochs, warmup_epochs=5):
    """
    Create learning rate scheduler with warmup.
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
    Returns:
        scheduler: Learning rate scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler
