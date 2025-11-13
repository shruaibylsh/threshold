"""
Dataset for SimCLR contrastive learning on threshold sequences.

Each sample returns two augmented views of the same 7-frame sequence.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ThresholdSequenceDataset(Dataset):
    """
    Dataset for SimCLR contrastive learning.

    Each sample is a 7-frame threshold sequence.
    The augmentation is applied externally to create two views.
    """
    def __init__(self, metadata_csvs, pano_folder, image_size=(32, 64)):
        """
        Args:
            metadata_csvs: List of CSV files with threshold sequences
            pano_folder: Path to panorama images
            image_size: (H, W) for images
        """
        self.pano_folder = pano_folder
        self.image_size = image_size

        # Load all CSVs into single dataframe
        dfs = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        self.metadata = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.metadata)} threshold sequences from {len(metadata_csvs)} files")

        # Extract typology class for evaluation
        self.metadata['typology_class'] = self.metadata['typology'].str.split('-').str[0]
        self.typology_to_idx = {f't{i}': i-1 for i in range(1, 9)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Load a 7-frame threshold sequence.

        Returns:
            dict with keys:
                - frames: (7, 3, H, W) tensor
                - typology_label: 0-7 (for evaluation)
                - metadata: dict with typology, curve info
        """
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
            frame_path = os.path.join(self.pano_folder, f"{typology}_{curve}_{frame_idx:03d}.png")
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

        return {
            'frames': frames,  # (7, 1, 32, 64) - grayscale
            'video': frames,   # Alias for compatibility with old code
            'typology_label': label_idx,  # 0-7
            'metadata': {
                'typology': typology,
                'curve': curve,
                'window_start': window_start,
                'window_end': window_end,
            }
        }


class SimCLRDataset(Dataset):
    """
    Wrapper that applies two augmentations to create contrastive pairs.
    """
    def __init__(self, base_dataset, augmentation):
        """
        Args:
            base_dataset: ThresholdSequenceDataset
            augmentation: ClipAugmentation instance
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns two augmented views of the same sequence.

        Returns:
            dict with keys:
                - view1: (7, 1, H, W) first augmented view - grayscale
                - view2: (7, 1, H, W) second augmented view - grayscale
                - typology_label: 0-7 (for evaluation)
                - metadata: original metadata
        """
        sample = self.base_dataset[idx]
        frames = sample['frames']  # (7, 1, H, W) - grayscale

        # Apply two independent augmentations
        view1 = self.augmentation(frames)
        view2 = self.augmentation(frames)

        return {
            'view1': view1,
            'view2': view2,
            'typology_label': sample['typology_label'],
            'metadata': sample['metadata'],
        }


class StratifiedThresholdDataset(Dataset):
    """
    Dataset that uses a subset of indices from a full dataset.
    Used for stratified train/val splitting.
    """
    def __init__(self, full_dataset, indices):
        """
        Args:
            full_dataset: ThresholdSequenceDataset with all data
            indices: List of indices to use from full dataset
        """
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.full_dataset[self.indices[idx]]


def create_simclr_dataloaders(data_root, augmentation, batch_size=16,
                               train_split=0.85, num_workers=4, random_seed=42):
    """
    Create train and validation dataloaders for SimCLR with stratified splitting.

    Ensures each typology has the same proportion in train and validation sets.

    Args:
        data_root: Root directory with 'candidates' and 'panos' folders
        augmentation: ClipAugmentation instance
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        random_seed: Random seed for reproducible splits

    Returns:
        train_loader, val_loader, train_base_dataset, val_base_dataset
    """
    candidates_folder = os.path.join(data_root, 'candidates')
    pano_folder = os.path.join(data_root, 'panos')

    # Get all threshold CSV files
    csv_files = sorted(glob.glob(os.path.join(candidates_folder, '*_thresholds.csv')))
    print(f"Found {len(csv_files)} threshold CSV files")

    # Load ALL data into one dataset
    full_dataset = ThresholdSequenceDataset(csv_files, pano_folder)

    # Extract typology labels for stratification
    typology_labels = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
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
        typology_name = [k for k, v in full_dataset.typology_to_idx.items() if v == typology_idx][0]
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
        typology_name = [k for k, v in full_dataset.typology_to_idx.items() if v == typology_idx][0]
        print(f"  {typology_name}: {count} samples ({count/len(train_indices)*100:.1f}%)")
    print()

    print("VAL SET DISTRIBUTION:")
    val_typologies = typology_labels[val_indices]
    for typology_idx in unique_typologies:
        count = np.sum(val_typologies == typology_idx)
        typology_name = [k for k, v in full_dataset.typology_to_idx.items() if v == typology_idx][0]
        print(f"  {typology_name}: {count} samples ({count/len(val_indices)*100:.1f}%)")
    print()

    # Create stratified datasets
    train_base_dataset = StratifiedThresholdDataset(full_dataset, train_indices)
    val_base_dataset = StratifiedThresholdDataset(full_dataset, val_indices)

    # Wrap with SimCLR augmentation
    train_dataset = SimCLRDataset(train_base_dataset, augmentation)
    val_dataset = SimCLRDataset(val_base_dataset, augmentation)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch norm
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_base_dataset, val_base_dataset