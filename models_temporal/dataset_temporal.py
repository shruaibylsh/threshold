"""
Dataset loader for Frame Order Prediction (Temporal Learning)

Loads 7-frame sequences, shuffles them, and provides ground truth positions.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class TemporalDataset(Dataset):
    """
    Dataset for frame order prediction.

    Each sample:
    - Loads a 7-frame sequence (before → threshold → after)
    - Shuffles the frames randomly
    - Returns shuffled frames and original position of each frame
    """

    def __init__(
        self,
        metadata_path: str,
        pano_folder: str,
        shuffle_frames: bool = True,
    ):
        """
        Args:
            metadata_path: Path to threshold_windows_metadata.csv
            pano_folder: Path to processed_panoramas folder
            shuffle_frames: If True, shuffle frames (for training). If False, keep original order (for evaluation).
        """
        self.metadata = pd.read_csv(metadata_path)
        self.pano_folder = pano_folder
        self.shuffle_frames = shuffle_frames

        # Create typology mapping (t1, t2, ..., t8)
        unique_typologies = sorted(self.metadata['typology'].apply(lambda x: x.split('-')[0]).unique())
        self.typology_to_idx = {t: i for i, t in enumerate(unique_typologies)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        typology = row['typology']
        curve = row['curve']
        window_start = row['window_start']
        window_end = row['window_end']

        # Load all 7 frames
        frames = []
        for frame_idx in range(window_start, window_end + 1):
            frame_path = os.path.join(
                self.pano_folder, f"{typology}_{curve}_{frame_idx:03d}.png"
            )
            img = Image.open(frame_path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(img_array)

        # Stack frames: (7, H, W)
        frames = np.stack(frames, axis=0)

        # Convert to PyTorch tensor and add channel dimension: (7, 1, H, W)
        frames = torch.from_numpy(frames).unsqueeze(1).float()

        # Shuffle frames if training
        if self.shuffle_frames:
            # Create random permutation
            perm = torch.randperm(7)
            frames = frames[perm]  # Shuffle frames
            # Ground truth: original position of each shuffled frame
            # If perm = [3, 0, 5, 1, 6, 2, 4], then:
            # - Frame at position 0 is originally from position 3 → target[0] = 3
            # - Frame at position 1 is originally from position 0 → target[1] = 0
            # - etc.
            targets = perm.clone()  # (7,)
        else:
            # No shuffling: each frame is in its correct position
            targets = torch.arange(7)  # [0, 1, 2, 3, 4, 5, 6]

        # Typology label (for evaluation)
        typology_name = typology.split('-')[0]
        typology_label = self.typology_to_idx[typology_name]

        return {
            'frames': frames,  # (7, 1, 32, 64)
            'targets': targets,  # (7,) original positions
            'typology_label': typology_label,
            'metadata': {
                'typology': typology,
                'curve': curve,
                'window_start': window_start,
            }
        }


def create_temporal_dataloaders(
    data_root: str,
    batch_size: int = 128,
    train_split: float = 0.85,
    num_workers: int = 4,
):
    """
    Create train and validation dataloaders for temporal learning.

    Args:
        data_root: Root directory containing 'processed_panoramas' and 'threshold_windows_metadata.csv'
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader: Training dataloader (with shuffling)
        val_loader: Validation dataloader (no shuffling)
        full_dataset: Full dataset (for extracting specific samples)
        train_indices: Training indices
        val_indices: Validation indices
    """
    metadata_path = os.path.join(data_root, 'threshold_windows_metadata.csv')
    pano_folder = os.path.join(data_root, 'processed_panoramas')

    # Create full dataset with shuffling
    full_dataset_train = TemporalDataset(
        metadata_path=metadata_path,
        pano_folder=pano_folder,
        shuffle_frames=True,
    )

    # Split into train/val
    total_size = len(full_dataset_train)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size

    train_dataset, _ = random_split(
        full_dataset_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation dataset WITHOUT shuffling (for evaluation)
    full_dataset_val = TemporalDataset(
        metadata_path=metadata_path,
        pano_folder=pano_folder,
        shuffle_frames=False,  # No shuffling for validation
    )

    _, val_dataset = random_split(
        full_dataset_val,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create full dataset for evaluation (no shuffling)
    full_dataset = full_dataset_val

    # Get indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset, train_indices, val_indices
