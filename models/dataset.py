"""
Dataset loader for threshold sequences.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ThresholdMAEDataset(Dataset):
    """
    Dataset for loading 7-frame threshold sequences for MAE pretraining.
    """
    def __init__(self, metadata_csvs, pano_folder, transform=None, image_size=(32, 64)):
        self.pano_folder = pano_folder
        self.transform = transform
        self.image_size = image_size
        # Load all CSVs into single dataframe
        dfs = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        self.metadata = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.metadata)} threshold sequences from {len(metadata_csvs)} files")

        self.metadata['typology_class'] = self.metadata['typology'].str.split('-').str[0]
        self.typology_to_idx = {
            f't{i}': i-1 for i in range(1, 9)
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Load a 7-frame threshold sequence.
        Returns:
            dict with keys:
                - pixel_values: (7, 3, H, W) tensor
                - label: typology class index (0-7)
                - metadata: dict with typology, curve, candidate_frame info
        """
        row = self.metadata.iloc[idx]
        typology = row['typology']
        curve = row['curve']
        window_start = row['window_start']
        window_end = row['window_end']
        # Load 7 frames (window_start to window_end inclusive)
        frames = []
        for frame_idx in range(window_start, window_end + 1):
            # Panorama filename: {typology}_{curve}_{frame:03d}.png
            pano_path = os.path.join(
                self.pano_folder,
                f"{typology}_{curve}_{frame_idx:03d}.png"
            )
            # Load grayscale image
            img = Image.open(pano_path)
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            # Ensure correct size (should be 32x64 already, but verify)
            if img_array.shape != self.image_size:
                img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
                img_pil = img_pil.resize((self.image_size[1], self.image_size[0]))  # PIL uses (W, H)
                img_array = np.array(img_pil).astype(np.float32) / 255.0
            frames.append(img_array)
        # Stack frames: (7, H, W)
        frames = np.stack(frames, axis=0)
        # Convert grayscale to RGB by repeating channel: (7, H, W) -> (7, H, W, 3)
        frames = np.stack([frames, frames, frames], axis=-1)
        # Convert to PyTorch tensor and rearrange to (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        # Get typology label
        typology_class = row['typology_class']
        label = self.typology_to_idx[typology_class]
        return {
            'pixel_values': frames,  # (7, 3, 32, 64)
            'labels': label,         # 0-7
            'metadata': {
                'typology': typology,
                'curve': curve,
                'candidate_frame': row['candidate_frame'],
                'window_start': window_start,
                'window_end': window_end,
            }
        }

def create_dataloaders(data_root, batch_size=16, train_split=0.85, num_workers=4):
    """
    Create train and validation dataloaders.
    """
    candidates_folder = os.path.join(data_root, 'candidates')
    pano_folder = os.path.join(data_root, 'panos')
    # Get all threshold CSV files
    csv_files = sorted(glob.glob(os.path.join(candidates_folder, '*_thresholds.csv')))
    print(f"Found {len(csv_files)} threshold CSV files")
    # Create full dataset
    full_dataset = ThresholdMAEDataset(csv_files, pano_folder)
    # Split into train/val
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducibility
    )
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader