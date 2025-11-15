"""
Dataset for threshold typology classification.

Reuses MAE dataset but ensures labels are returned for supervised learning.
"""
import sys
sys.path.append('../models_MAE')

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset_mae import ThresholdSequenceDatasetMAE


def create_classifier_dataloaders(
    data_root,
    batch_size=32,
    train_split=0.70,
    val_split=0.15,
    num_workers=4,
    seed=42
):
    """
    Create train/val/test dataloaders for classification.

    Args:
        data_root: Path to data folder
        batch_size: Batch size
        train_split: Training set ratio
        val_split: Validation set ratio
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader, full_dataset, train_indices, val_indices, test_indices
    """
    # Find all CSV files in candidates directory
    candidates_dir = os.path.join(data_root, 'candidates')
    csv_files = [os.path.join(candidates_dir, f) for f in os.listdir(candidates_dir)
                 if f.endswith('.csv') and not f.startswith('b')]  # Exclude building CSVs
    csv_files = sorted(csv_files)

    # Panorama folder
    pano_folder = os.path.join(data_root, 'panos')

    # Create full dataset
    full_dataset = ThresholdSequenceDatasetMAE(
        metadata_csvs=csv_files,
        pano_folder=pano_folder
    )

    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    print(f"Dataset split:")
    print(f"  Total samples: {total_size}")
    print(f"  Train: {train_size} ({train_split:.0%})")
    print(f"  Val:   {val_size} ({val_split:.0%})")
    print(f"  Test:  {test_size} ({1-train_split-val_split:.0%})")
    print()

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Get indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # Analyze label distribution
    train_labels = [full_dataset[i]['typology_label'] for i in train_indices]
    val_labels = [full_dataset[i]['typology_label'] for i in val_indices]
    test_labels = [full_dataset[i]['typology_label'] for i in test_indices]

    print("Label distribution:")
    print("  Train:")
    for label in range(8):
        count = sum(1 for l in train_labels if l == label)
        print(f"    t{label+1}: {count} samples")
    print("  Val:")
    for label in range(8):
        count = sum(1 for l in val_labels if l == label)
        print(f"    t{label+1}: {count} samples")
    print("  Test:")
    for label in range(8):
        count = sum(1 for l in test_labels if l == label)
        print(f"    t{label+1}: {count} samples")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    return (train_loader, val_loader, test_loader,
            full_dataset, train_indices, val_indices, test_indices)


class BuildingDataset(torch.utils.data.Dataset):
    """
    Dataset for real building thresholds (b1-b4).

    Each building has multiple threshold candidates.
    """

    def __init__(self, panos_dir, candidates_dir, building_ids=['b1', 'b2', 'b3', 'b4']):
        """
        Args:
            panos_dir: Path to building panos (e.g., "data/panos_buildings")
            candidates_dir: Path to threshold CSVs (e.g., "data/candidates")
            building_ids: List of building IDs to load
        """
        self.panos_dir = panos_dir
        self.candidates_dir = candidates_dir
        self.building_ids = building_ids

        # Load all thresholds
        self.samples = []
        self._load_buildings()

    def _load_buildings(self):
        """Load all threshold candidates from buildings."""
        import pandas as pd

        for building_id in self.building_ids:
            csv_path = os.path.join(self.candidates_dir, f'{building_id}_thresholds.csv')

            if not os.path.exists(csv_path):
                print(f"⚠ Warning: {csv_path} not found, skipping {building_id}")
                continue

            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"Loaded {building_id}: {len(df)} thresholds")

            # Each row is a threshold candidate
            # CSV format: typology, curve, candidate_frame, window_start, window_end
            for idx, row in df.iterrows():
                # Generate 7 frame indices from window_start to window_end
                window_start = int(row['window_start'])
                window_end = int(row['window_end'])
                frame_indices = list(range(window_start, window_end + 1))

                # Get curve name
                curve = row['curve']
                candidate_frame = int(row['candidate_frame'])

                # Create threshold ID
                threshold_id = f"{building_id}_{curve}_f{candidate_frame:02d}"

                self.samples.append({
                    'building_id': building_id,
                    'curve': curve,  # Store curve name for image loading
                    'threshold_id': threshold_id,
                    'frame_indices': frame_indices
                })

        print(f"Total building thresholds: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load 7 frames for a threshold.

        Returns:
            {
                'frames': [7, 1, 32, 64] tensor,
                'building_id': str,
                'threshold_id': str,
                'frame_indices': list of 7 ints
            }
        """
        sample = self.samples[idx]
        building_id = sample['building_id']
        curve = sample['curve']
        threshold_id = sample['threshold_id']
        frame_indices = sample['frame_indices']

        # Load frames
        from PIL import Image
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.Resize((32, 64)),
            transforms.ToTensor(),
        ])

        frames = []
        for frame_idx in frame_indices:
            # Format: b1_curve_01_011.png
            pano_path = os.path.join(
                self.panos_dir,
                f'{building_id}_{curve}_{frame_idx:03d}.png'
            )

            if os.path.exists(pano_path):
                img = Image.open(pano_path).convert('L')  # Grayscale
                img_tensor = transform(img)  # [1, 32, 64]
                frames.append(img_tensor)
            else:
                print(f"⚠ Warning: {pano_path} not found, using zeros")
                frames.append(torch.zeros(1, 32, 64))

        frames = torch.stack(frames, dim=0)  # [7, 1, 32, 64]

        return {
            'frames': frames,
            'building_id': building_id,
            'threshold_id': threshold_id,
            'frame_indices': frame_indices
        }


def create_building_dataloader(
    panos_dir,
    candidates_dir,
    building_ids=['b1', 'b2', 'b3', 'b4'],
    batch_size=16,
    num_workers=2
):
    """
    Create dataloader for real building thresholds.

    Args:
        panos_dir: Path to building panos
        candidates_dir: Path to threshold CSVs
        building_ids: List of building IDs
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        DataLoader
    """
    dataset = BuildingDataset(panos_dir, candidates_dir, building_ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader, dataset


if __name__ == "__main__":
    # Test classifier dataset
    print("="*80)
    print("Testing Classifier Dataset")
    print("="*80)
    print()

    data_root = r"C:\Users\shrua\OneDrive\Desktop\threshold project\threshold\data"

    train_loader, val_loader, test_loader, full_dataset, _, _, _ = create_classifier_dataloaders(
        data_root=data_root,
        batch_size=32,
        train_split=0.70,
        val_split=0.15
    )

    # Test batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Frames shape: {batch['frames'].shape}")
    print(f"  Labels shape: {batch['typology_label'].shape}")
    print(f"  Labels: {batch['typology_label'][:8]}")
    print()

    print("✓ Dataset test passed!")