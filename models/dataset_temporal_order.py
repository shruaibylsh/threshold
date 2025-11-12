"""
Dataset for pairwise temporal order prediction.

Each sample is a pair of frames (frame_i, frame_j) from the same sequence,
with a binary label indicating if they are in correct temporal order.
"""
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import random


class TemporalOrderDataset(Dataset):
    """
    Dataset for temporal order prediction.

    Each sample consists of:
    - Two frames from the same threshold sequence
    - Binary label: 1 if in correct order, 0 if swapped
    """
    def __init__(self, metadata_csvs, pano_folder, image_size=(32, 64), pairs_per_sequence=3):
        """
        Args:
            metadata_csvs: List of CSV files with threshold sequences
            pano_folder: Path to panorama images
            image_size: (H, W) for images
            pairs_per_sequence: How many pairs to generate per sequence
        """
        self.pano_folder = pano_folder
        self.image_size = image_size
        self.pairs_per_sequence = pairs_per_sequence

        # Load all CSVs into single dataframe
        dfs = []
        for csv_path in metadata_csvs:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        self.metadata = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.metadata)} threshold sequences from {len(metadata_csvs)} files")

        # Extract typology class for clustering evaluation
        self.metadata['typology_class'] = self.metadata['typology'].str.split('-').str[0]
        self.typology_to_idx = {f't{i}': i-1 for i in range(1, 9)}

        # Generate all pairs upfront for faster training
        self.pairs = self._generate_pairs()
        print(f"Generated {len(self.pairs)} frame pairs ({self.pairs_per_sequence} per sequence)")

    def _generate_pairs(self):
        """
        Generate frame pairs from sequences.

        EDGE-FRAME SAMPLING STRATEGY:
        - Sample one frame from BEFORE (frames 0-1)
        - Sample one frame from AFTER (frames 5-6)
        - This forces the model to learn the before→after transformation
          that distinguishes different typologies
        """
        pairs = []

        for idx, row in self.metadata.iterrows():
            typology = row['typology']
            curve = row['curve']
            window_start = row['window_start']
            window_end = row['window_end']
            typology_class = row['typology_class']
            label_idx = self.typology_to_idx[typology_class]

            # Available frames: [window_start, ..., window_end] (7 frames total)
            # Indices: 0, 1, 2, 3, 4, 5, 6
            frame_indices = list(range(window_start, window_end + 1))

            # Generate multiple pairs per sequence
            for _ in range(self.pairs_per_sequence):
                # EDGE-FRAME SAMPLING:
                # Sample one from BEFORE (frames 0-1)
                # Sample one from AFTER (frames 5-6)
                i = random.choice([0, 1])  # Before frames
                j = random.choice([5, 6])  # After frames

                frame_i_idx = frame_indices[i]  # Before frame
                frame_j_idx = frame_indices[j]  # After frame

                # Randomly decide to swap (50% chance)
                is_correct_order = random.random() > 0.5

                if not is_correct_order:
                    # Swap frames: now after frame comes first (wrong order)
                    frame_i_idx, frame_j_idx = frame_j_idx, frame_i_idx

                pairs.append({
                    'typology': typology,
                    'curve': curve,
                    'frame_i': frame_i_idx,
                    'frame_j': frame_j_idx,
                    'label': 1 if is_correct_order else 0,
                    'typology_label': label_idx,  # For clustering evaluation
                })

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Load a frame pair.

        Returns:
            dict with keys:
                - frames: (2, 3, H, W) tensor with two frames
                - label: 0 or 1 (binary classification)
                - typology_label: 0-7 (for clustering evaluation)
                - metadata: dict with typology, curve, frame indices
        """
        pair = self.pairs[idx]

        typology = pair['typology']
        curve = pair['curve']
        frame_i_idx = pair['frame_i']
        frame_j_idx = pair['frame_j']
        label = pair['label']
        typology_label = pair['typology_label']

        # Load frame i
        frame_i_path = os.path.join(self.pano_folder, f"{typology}_{curve}_{frame_i_idx:03d}.png")
        img_i = Image.open(frame_i_path)
        img_i_array = np.array(img_i).astype(np.float32) / 255.0

        # Resize if needed
        if img_i_array.shape != self.image_size:
            img_i_pil = Image.fromarray((img_i_array * 255).astype(np.uint8))
            img_i_pil = img_i_pil.resize((self.image_size[1], self.image_size[0]))
            img_i_array = np.array(img_i_pil).astype(np.float32) / 255.0

        # Load frame j
        frame_j_path = os.path.join(self.pano_folder, f"{typology}_{curve}_{frame_j_idx:03d}.png")
        img_j = Image.open(frame_j_path)
        img_j_array = np.array(img_j).astype(np.float32) / 255.0

        # Resize if needed
        if img_j_array.shape != self.image_size:
            img_j_pil = Image.fromarray((img_j_array * 255).astype(np.uint8))
            img_j_pil = img_j_pil.resize((self.image_size[1], self.image_size[0]))
            img_j_array = np.array(img_j_pil).astype(np.float32) / 255.0

        # Convert grayscale to RGB by repeating channel
        img_i_rgb = np.stack([img_i_array, img_i_array, img_i_array], axis=-1)
        img_j_rgb = np.stack([img_j_array, img_j_array, img_j_array], axis=-1)

        # Stack frames: (2, H, W, 3)
        frames = np.stack([img_i_rgb, img_j_rgb], axis=0)

        # Convert to PyTorch tensor and rearrange to (2, 3, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

        return {
            'frames': frames,  # (2, 3, 32, 64)
            'labels': label,   # 0 or 1
            'typology_labels': typology_label,  # 0-7
            'metadata': {
                'typology': typology,
                'curve': curve,
                'frame_i': frame_i_idx,
                'frame_j': frame_j_idx,
            }
        }


def create_dataloaders(data_root, batch_size=16, train_split=0.85, num_workers=4, pairs_per_sequence=3):
    """
    Create train and validation dataloaders for temporal order prediction.

    Args:
        data_root: Root directory with 'candidates' and 'panos' folders
        batch_size: Batch size
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        pairs_per_sequence: Number of pairs to generate per sequence

    Returns:
        train_loader, val_loader
    """
    candidates_folder = os.path.join(data_root, 'candidates')
    pano_folder = os.path.join(data_root, 'panos')

    # Get all threshold CSV files
    csv_files = sorted(glob.glob(os.path.join(candidates_folder, '*_thresholds.csv')))
    print(f"Found {len(csv_files)} threshold CSV files")

    # Split CSVs into train/val
    total_files = len(csv_files)
    train_size = int(train_split * total_files)

    train_csvs = csv_files[:train_size]
    val_csvs = csv_files[train_size:]

    print(f"Train CSVs: {len(train_csvs)}, Val CSVs: {len(val_csvs)}")

    # Create datasets
    train_dataset = TemporalOrderDataset(train_csvs, pano_folder, pairs_per_sequence=pairs_per_sequence)
    val_dataset = TemporalOrderDataset(val_csvs, pano_folder, pairs_per_sequence=pairs_per_sequence)

    print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

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