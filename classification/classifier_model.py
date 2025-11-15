"""
Threshold Typology Classifier.

Combines pretrained MAE encoder with MLP head for supervised classification.
"""
import sys
import os

# Add MAE_models to path (absolute path from this file's location)
current_dir = os.path.dirname(os.path.abspath(__file__))
mae_models_path = os.path.join(current_dir, '..', 'MAE_models')
if mae_models_path not in sys.path:
    sys.path.insert(0, mae_models_path)

import torch
import torch.nn as nn
from timesformer_mae import TimeSformerMAE
from config_mae import get_mae_config


class ThresholdClassifier(nn.Module):
    """
    Classifier for threshold typologies using pretrained MAE encoder.

    Architecture:
        Input → Pretrained Encoder → Global Pool → MLP Head → Logits
    """

    def __init__(self, config):
        """
        Args:
            config: ClassifierConfig with model settings
        """
        super().__init__()
        self.config = config

        # Load pretrained MAE encoder
        print(f"Loading pretrained encoder from: {config.pretrained_encoder_path}")
        self.encoder = self._load_pretrained_encoder(config.pretrained_encoder_path)

        # Freeze encoder if specified
        if config.freeze_encoder:
            self._freeze_encoder()
            print("✓ Encoder frozen (linear probing mode)")
        else:
            print("✓ Encoder unfrozen (fine-tuning mode)")

        # MLP head
        if config.mlp_hidden_dim > 0:
            # Two-layer MLP: encoder_dim → hidden → num_classes
            self.mlp_head = nn.Sequential(
                nn.Linear(config.encoder_dim, config.mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.mlp_dropout),
                nn.Linear(config.mlp_hidden_dim, config.num_classes)
            )
            print(f"✓ MLP Head: {config.encoder_dim} → {config.mlp_hidden_dim} → {config.num_classes}")
        else:
            # Single-layer: encoder_dim → num_classes
            self.mlp_head = nn.Linear(config.encoder_dim, config.num_classes)
            print(f"✓ MLP Head: {config.encoder_dim} → {config.num_classes}")

    def _load_pretrained_encoder(self, pretrained_path):
        """Load pretrained MAE encoder components."""
        # Create MAE model structure
        mae_config = get_mae_config()
        full_mae_model = TimeSformerMAE(mae_config)

        # Load pretrained weights
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Filter to only encoder components
        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith('encoder') or k.startswith('patch_embed') or 'emb' in k:
                encoder_state[k] = v

        # Load into model
        full_mae_model.load_state_dict(encoder_state, strict=False)

        # Extract encoder function
        return full_mae_model

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for name, param in self.encoder.named_parameters():
            if 'encoder' in name or 'patch_embed' in name or 'emb' in name:
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for name, param in self.encoder.named_parameters():
            if 'encoder' in name or 'patch_embed' in name or 'emb' in name:
                param.requires_grad = True
        print("✓ Encoder unfrozen")

    def forward(self, x, return_features=False):
        """
        Forward pass.

        Args:
            x: [B, T, C, H, W] input video frames
            return_features: If True, return (logits, features)

        Returns:
            logits: [B, num_classes] class logits
            features (optional): [B, encoder_dim] pooled features
        """
        # Encode with NO masking (mask_ratio=0)
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            x_encoded, _, _ = self.encoder.forward_encoder(x, mask_ratio=0)
            # x_encoded: [B, 224, encoder_dim]

        # Global average pooling over all tokens
        features = x_encoded.mean(dim=1)  # [B, encoder_dim]

        # MLP head for classification
        logits = self.mlp_head(features)  # [B, num_classes]

        if return_features:
            return logits, features
        else:
            return logits

    def predict_proba(self, x):
        """
        Predict class probabilities.

        Args:
            x: [B, T, C, H, W] input video frames

        Returns:
            probs: [B, num_classes] class probabilities (softmax)
        """
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs

    def get_num_params(self):
        """Get number of trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def test_classifier():
    """Test classifier initialization and forward pass."""
    from config_classifier import get_classifier_config

    print("="*80)
    print("Testing Threshold Classifier")
    print("="*80)
    print()

    # Test frozen encoder
    print("1. Testing FROZEN encoder (linear probing):")
    config = get_classifier_config(freeze_encoder=True)
    model = ThresholdClassifier(config)

    params = model.get_num_params()
    print(f"   Total params: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Frozen: {params['frozen']:,}")

    # Test forward pass
    x = torch.randn(2, 7, 1, 32, 64)  # Batch of 2 sequences
    logits = model(x)
    probs = model.predict_proba(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probs shape: {probs.shape}")
    print(f"   Probs sum: {probs.sum(dim=1)}")
    print()

    # Test fine-tuning mode
    print("2. Testing UNFROZEN encoder (fine-tuning):")
    config = get_classifier_config(freeze_encoder=False)
    model = ThresholdClassifier(config)

    params = model.get_num_params()
    print(f"   Total params: {params['total']:,}")
    print(f"   Trainable: {params['trainable']:,}")
    print(f"   Frozen: {params['frozen']:,}")
    print()

    print("✓ All tests passed!")


if __name__ == "__main__":
    test_classifier()