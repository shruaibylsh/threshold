import os, re, random, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Config
# ----------------------------
@dataclass
class DataConfig:
    root: str = r"C:\Users\shrua\OneDrive\Desktop\threshold project\threshold\data\panos"
    img_h: int = 32       # height
    img_w: int = 64       # width
    frames_per_seq: int = 7
    seed: int = 42
    train_ratio: float = 0.75
    val_ratio: float = 0.10   # test will be 1 - train - val

CFG = DataConfig()

# ----------------------------
# Utilities
# ----------------------------
_F_END = re.compile(r"_f(\d+)\.(png)$", re.IGNORECASE)
_T_PREFIX = re.compile(r"^t(\d+)")   # captures class id at start (e.g., t1, t8)

def _list_images(root: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    out = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
    return out

def _prefix_and_frame(filename: str) -> Tuple[str, Optional[int]]:
    """
    Split "t1-1_curve_01_peak_011_f00.png" into:
      prefix="t1-1_curve_01_peak_011", frame=0
    """
    base = os.path.basename(filename)
    m = _F_END.search(base)
    if not m:
        return base, None
    frame_idx = int(m.group(1))
    prefix = base[:m.start()]  # everything before "_fXX"
    return prefix, frame_idx

def _class_from_base(base: str) -> Optional[int]:
    """
    Extract class id 0..7 from start of filename (t1..t8).
    """
    m = _T_PREFIX.match(base)
    if not m:
        return None
    cls_1_based = int(m.group(1))
    if not (1 <= cls_1_based <= 8):
        return None
    return cls_1_based - 1  # 0..7

def build_index(root: str, frames_per_seq: int = 7) -> List[Dict]:
    """
    Returns a list of sequences:
      {
        'prefix': <str>,
        'label': <int 0..7>,
        'paths': [7 filepaths ordered by frame],
        'id': <unique id string>
      }
    Skips incomplete sequences or those without a valid label.
    """
    files = _list_images(root)
    groups: Dict[str, Dict[int, str]] = {}
    base_to_one_example: Dict[str, str] = {}

    for fp in files:
        base = os.path.basename(fp)
        prefix, fidx = _prefix_and_frame(base)
        if fidx is None:
            continue
        groups.setdefault(prefix, {})[fidx] = fp
        base_to_one_example[prefix] = base  # keep for label parsing

    sequences = []
    for prefix, frame_map in groups.items():
        # need exactly frames_per_seq frames, typically f00..f06 (any numbering is fine if contiguous after sort)
        if len(frame_map) != frames_per_seq:
            continue
        frame_ids = sorted(frame_map.keys())
        # ensure 7 distinct indices (we don't require 0..6 specifically)
        paths = [frame_map[i] for i in frame_ids]

        # label from the very beginning of the filename (use any one example)
        any_base = base_to_one_example[prefix]
        cls_idx = _class_from_base(any_base)
        if cls_idx is None:
            continue

        sequences.append({
            "prefix": prefix,
            "label": cls_idx,
            "paths": paths,
            "id": prefix,  # good enough
        })

    return sequences

# ----------------------------
# Transforms
# ----------------------------
def make_transforms(img_h: int, img_w: int, train: bool):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),
        ]
    else:
        aug = []
    tail = [
        transforms.ToTensor(),  # -> [1, H, W], float32 in [0,1]
    ]
    return transforms.Compose(base + aug + tail)

# ----------------------------
# Dataset
# ----------------------------
class PanosSequenceDataset(Dataset):
    def __init__(self, sequences: List[Dict], transform):
        self.sequences = sequences
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        paths = item["paths"]
        frames = []
        for p in paths:
            img = Image.open(p).convert("L")  # ensure grayscale
            frames.append(self.transform(img))  # [1, H, W]
        # Stack to [T, C, H, W]
        video = torch.stack(frames, dim=0)
        label = item["label"]
        return {
            "video": video,        # [7, 1, 32, 64]
            "label": label,        # int in [0..7]
            "id": item["id"],      # sequence id (prefix)
        }

# ----------------------------
# Split (stratified by class)
# ----------------------------
def stratified_split(
    sequences: List[Dict],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(seed)
    by_class: Dict[int, List[Dict]] = {}
    for s in sequences:
        by_class.setdefault(s["label"], []).append(s)

    train, val, test = [], [], []
    for cls, items in by_class.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)  # safety
        n_val = min(n_val, max(0, n - n_train))
        cls_train = items[:n_train]
        cls_val = items[n_train:n_train + n_val]
        cls_test = items[n_train + n_val:]
        train.extend(cls_train)
        val.extend(cls_val)
        test.extend(cls_test)
    # Shuffle each split to mix classes a bit
    random.shuffle(train); random.shuffle(val); random.shuffle(test)
    return train, val, test

# ----------------------------
# DataLoader factory
# ----------------------------
def create_dataloaders(cfg: DataConfig, batch_size: int = 64, num_workers: int = 0):
    sequences = build_index(cfg.root, cfg.frames_per_seq)
    if len(sequences) == 0:
        raise RuntimeError(f"No valid 7-frame sequences found under: {cfg.root}")

    train_seqs, val_seqs, test_seqs = stratified_split(
        sequences, cfg.train_ratio, cfg.val_ratio, cfg.seed
    )

    t_train = make_transforms(cfg.img_h, cfg.img_w, train=True)
    t_eval  = make_transforms(cfg.img_h, cfg.img_w, train=False)

    ds_train = PanosSequenceDataset(train_seqs, t_train)
    ds_val   = PanosSequenceDataset(val_seqs, t_eval)
    ds_test  = PanosSequenceDataset(test_seqs, t_eval)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # quick report
    def _count_per_class(items: List[Dict]):
        d = {i: 0 for i in range(8)}
        for s in items:
            d[s["label"]] += 1
        return 
    return dl_train, dl_val, dl_test