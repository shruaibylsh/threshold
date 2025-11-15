"""
Configuration for Threshold Typology Classifier.

This classifier uses a pretrained MAE encoder with an MLP head for supervised classification.
"""
from dataclasses import dataclass
import os

@dataclass
class ClassifierConfig:
    # Pretrained encoder path
    pretrained_encoder_path: str = '../MAE_models/output_mae_adjusted/encoder_mae_pretrained.pt'

    # Model architecture
    num_classes: int = 8  # t1 through t8
    encoder_dim: int = 384  # MAE encoder output dimension

    # MLP head architecture
    mlp_hidden_dim: int = 128  # Hidden layer size (0 = no hidden layer)
    mlp_dropout: float = 0.2

    # Fine-tuning strategy
    freeze_encoder: bool = True  # True = frozen encoder (linear probing), False = fine-tune encoder

    # Training hyperparameters
    learning_rate: float = 1e-3  # Higher for frozen, lower (1e-4) for fine-tuning
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100

    # Data split
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15  # Remaining after train/val

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_patience: int = 10  # ReduceLROnPlateau
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping_patience: int = 20

    # Data augmentation
    use_augmentation: bool = False  # Optional: random flip, crop, etc.

    # Output paths
    output_dir: str = '../output_classifier'
    checkpoint_dir: str = '../output_classifier/checkpoints'
    vis_dir: str = '../output_classifier/visualizations'

    def __post_init__(self):
        """Create output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        print("Classifier Configuration:")
        print(f"  Pretrained encoder: {self.pretrained_encoder_path}")
        print(f"  Freeze encoder: {self.freeze_encoder}")
        print(f"  MLP architecture: {self.encoder_dim} → {self.mlp_hidden_dim} → {self.num_classes}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Train/Val/Test split: {self.train_split:.0%}/{self.val_split:.0%}/{self.test_split:.0%}")
        print(f"  Output: {self.output_dir}")


def get_classifier_config(freeze_encoder=True):
    """Get default classifier configuration."""
    config = ClassifierConfig(freeze_encoder=freeze_encoder)

    # Adjust learning rate based on strategy
    if not freeze_encoder:
        config.learning_rate = 1e-4  # Lower LR for fine-tuning
        print("\n⚠ Fine-tuning mode: Using lower learning rate (1e-4)")

    return config


def get_finetuning_config():
    """Get configuration for full fine-tuning (unfrozen encoder)."""
    config = ClassifierConfig(
        freeze_encoder=False,
        learning_rate=1e-4,  # Lower for stability
        num_epochs=150,  # More epochs for fine-tuning
    )
    return config
