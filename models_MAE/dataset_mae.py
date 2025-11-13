"""
Dataset loader for MAE pretraining on threshold sequences.
"""
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ThresholdSequenceDatasetMAE(Dataset):
    """
    Load 7-frame threshold sequences for MAE pretraining.
    No augmentation needed - MAE learns from masking!
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

        # Load all CSVs
        dfs = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        self.metadata = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.metadata)} threshold sequences from {len(metadata_csvs)} files")

        # Extract typology class
        self.metadata['typology_class'] = self.metadata['typology'].str.split('-').str[0]
        self.typology_to_idx = {f't{i}': i-1 for i in range(1, 9)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Load a 7-frame threshold sequence.

        Returns:
            dict with keys:
                - frames: (7, 1, H, W) tensor [0, 1]
                - typology_label: 0-7
                - metadata: dict with info
        """
        row = self.metadata.iloc[idx]

        typology = row['typology']
        curve = row['curve']
        window_start = row['window_start']
        window_end = row['window_end']
        typology_class = row['typology_class']
        label_idx = self.typology_to_idx[typology_class]

        # Load all 7 frames (from window_start to window_end)
        frames = []
        for frame_idx in range(window_start, window_end + 1):
            frame_path = os.path.join(self.pano_folder, f"{typology}_{curve}_{frame_idx:03d}.png")

            # Load and convert to grayscale
            img = Image.open(frame_path)
            img_array = np.array(img).astype(np.float32) / 255.0

            # Resize if needed
            if img_array.shape != self.image_size:
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((self.image_size[1], self.image_size[0]))
                img_array = np.array(img_pil).astype(np.float32) / 255.0

            frames.append(img_array)

        # Stack: (7, H, W)
        frames = np.stack(frames, axis=0)

        # To tensor: (7, 1, H, W)
        frames = torch.from_numpy(frames).unsqueeze(1).float()

        return {
            'frames': frames,
            'typology_label': label_idx,
            'metadata': {
                'typology': typology,
                'curve': curve,
                'window_start': window_start,
                'window_end': window_end,
            }
        }


def create_mae_dataloaders(data_root, batch_size=64, train_split=0.85, num_workers=4):
    """
    Create train/val dataloaders for MAE pretraining.

    Args:
        data_root: Path to data folder
        batch_size: Batch size
        train_split: Train split ratio
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Find all CSV files
    candidates_dir = os.path.join(data_root, 'candidates')
    csv_files = [os.path.join(candidates_dir, f) for f in os.listdir(candidates_dir) if f.endswith('.csv')]
    csv_files = sorted(csv_files)

    print(f"Found {len(csv_files)} CSV files")

    # Panorama folder
    pano_folder = os.path.join(data_root, 'panos')

    # Create full dataset
    full_dataset = ThresholdSequenceDatasetMAE(csv_files, pano_folder)

    # Get typology labels for stratified split
    typology_labels = np.array([full_dataset.metadata.iloc[i]['typology_class']
                                 for i in range(len(full_dataset))])

    # Stratified train/val split
    indices = np.arange(len(full_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_split,
        stratify=typology_labels,
        random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Val samples: {len(val_indices)}")
    print()

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset, train_indices, val_indices