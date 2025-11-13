"""
Dataset loader for Frame Order Prediction (Temporal Learning)

Loads 7-frame sequences, shuffles them, and provides ground truth positions.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
        metadata_csvs: list,
        pano_folder: str,
        shuffle_frames: bool = True,
        image_size: tuple = (32, 64),
    ):
        """
        Args:
            metadata_csvs: List of CSV files with threshold sequences
            pano_folder: Path to panorama images folder
            shuffle_frames: If True, shuffle frames (for training). If False, keep original order (for evaluation).
            image_size: (H, W) for images
        """
        self.pano_folder = pano_folder
        self.shuffle_frames = shuffle_frames
        self.image_size = image_size

        # Load all CSVs into single dataframe
        dfs = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        self.metadata = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.metadata)} threshold sequences from {len(metadata_csvs)} files")

        # Create typology mapping (t1, t2, ..., t8)
        self.metadata['typology_class'] = self.metadata['typology'].str.split('-').str[0]
        self.typology_to_idx = {f't{i}': i-1 for i in range(1, 9)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        typology = row['typology']
        curve = row['curve']
        window_start = row['window_start']
        window_end = row['window_end']
        typology_class = row['typology_class']
        label_idx = self.typology_to_idx[typology_class]

        # Load all 7 frames
        frames = []
        for frame_idx in range(window_start, window_end + 1):
            frame_path = os.path.join(
                self.pano_folder, f"{typology}_{curve}_{frame_idx:03d}.png"
            )
            img = Image.open(frame_path)
            img_array = np.array(img).astype(np.float32) / 255.0

            # Resize if needed
            if img_array.shape != self.image_size:
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((self.image_size[1], self.image_size[0]))
                img_array = np.array(img_pil).astype(np.float32) / 255.0

            # Keep as grayscale (1 channel)
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

        return {
            'frames': frames,  # (7, 1, 32, 64)
            'targets': targets,  # (7,) original positions
            'typology_label': label_idx,
            'metadata': {
                'typology': typology,
                'curve': curve,
                'window_start': window_start,
                'window_end': window_end,
            }
        }


def create_temporal_dataloaders(
    data_root: str,
    batch_size: int = 128,
    train_split: float = 0.85,
    num_workers: int = 4,
    random_seed: int = 42,
):
    """
    Create train and validation dataloaders for temporal learning.

    Args:
        data_root: Root directory containing 'candidates/' and 'panos/'
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits

    Returns:
        train_loader: Training dataloader (with shuffling)
        val_loader: Validation dataloader (no shuffling)
        full_dataset: Full dataset (for extracting specific samples)
        train_indices: Training indices
        val_indices: Validation indices
    """
    candidates_folder = os.path.join(data_root, 'candidates')
    pano_folder = os.path.join(data_root, 'panos')

    # Get all threshold CSV files
    csv_files = sorted(glob.glob(os.path.join(candidates_folder, '*_thresholds.csv')))
    print(f"Found {len(csv_files)} threshold CSV files")

    # Create datasets
    # Training dataset WITH shuffling
    train_dataset_full = TemporalDataset(
        metadata_csvs=csv_files,
        pano_folder=pano_folder,
        shuffle_frames=True,
    )

    # Validation dataset WITHOUT shuffling (for evaluation)
    val_dataset_full = TemporalDataset(
        metadata_csvs=csv_files,
        pano_folder=pano_folder,
        shuffle_frames=False,
    )

    # Extract typology labels for stratification
    typology_labels = []
    for i in range(len(train_dataset_full)):
        sample = train_dataset_full[i]
        typology_labels.append(sample['typology_label'])
    typology_labels = np.array(typology_labels)

    # Analyze typology distribution
    unique_typologies = np.unique(typology_labels)
    print()
    print("="*80)
    print("TYPOLOGY DISTRIBUTION (before splitting)")
    print("="*80)
    for typology_idx in unique_typologies:
        count = np.sum(typology_labels == typology_idx)
        typology_name = [k for k, v in train_dataset_full.typology_to_idx.items() if v == typology_idx][0]
        print(f"  {typology_name}: {count} samples ({count/len(typology_labels)*100:.1f}%)")
    print("="*80)
    print()

    # Stratified split: preserve proportion of each typology
    np.random.seed(random_seed)
    train_indices = []
    val_indices = []

    for typology_idx in unique_typologies:
        # Get all indices for this typology
        typology_indices = np.where(typology_labels == typology_idx)[0]

        # Shuffle
        np.random.shuffle(typology_indices)

        # Split
        n_train = int(len(typology_indices) * train_split)
        train_indices.extend(typology_indices[:n_train].tolist())
        val_indices.extend(typology_indices[n_train:].tolist())

    # Shuffle the splits (so batches have mixed typologies)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    print(f"Stratified split completed:")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    print()

    # Verify stratification worked
    print("TRAIN SET DISTRIBUTION:")
    train_typologies = typology_labels[train_indices]
    for typology_idx in unique_typologies:
        count = np.sum(train_typologies == typology_idx)
        typology_name = [k for k, v in train_dataset_full.typology_to_idx.items() if v == typology_idx][0]
        print(f"  {typology_name}: {count} samples ({count/len(train_indices)*100:.1f}%)")
    print()

    print("VAL SET DISTRIBUTION:")
    val_typologies = typology_labels[val_indices]
    for typology_idx in unique_typologies:
        count = np.sum(val_typologies == typology_idx)
        typology_name = [k for k, v in val_dataset_full.typology_to_idx.items() if v == typology_idx][0]
        print(f"  {typology_name}: {count} samples ({count/len(val_indices)*100:.1f}%)")
    print()

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)

    # Full dataset for evaluation (no shuffling)
    full_dataset = val_dataset_full

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch norm
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset, train_indices, val_indices