import random
import torch
from torchvision import transforms

class RandomHorizontalRoll:
    """Circularly shift the panorama along width (yaw)."""
    def __init__(self, p=0.5, max_frac=0.25):
        self.p = p
        self.max_frac = max_frac
    def __call__(self, clip):  # clip: [T,1,H,W] tensor in [0,1]
        if random.random() > self.p:
            return clip
        T, C, H, W = clip.shape
        max_shift = int(self.max_frac * W)
        k = random.randint(-max_shift, max_shift)
        return torch.roll(clip, shifts=k, dims=-1)

class ClipTransform:
    """
    Apply the *same* spatial/photometric ops to all frames, then optional temporal jitter.
    Input / Output: [T,1,H,W] in [0,1]
    """
    def __init__(self, size=(32, 64), blur_prob=0.5, flip_prob=0.5, roll_prob=0.5,
                 brightness=0.15, contrast=0.15, temporal_jitter_prob=0.2):
        self.size = size
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob
        self.roll = RandomHorizontalRoll(p=roll_prob, max_frac=0.25)
        self.temporal_jitter_prob = temporal_jitter_prob

        self.crop = transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(1.0, 1.0))
        self.hflip = transforms.RandomHorizontalFlip(p=flip_prob)
        self.color = transforms.ColorJitter(brightness=brightness, contrast=contrast)
        self.toPIL = transforms.ToPILImage()
        self.toTensor = transforms.ToTensor()
        k = max(3, int(0.1 * min(size)));  k += (k % 2 == 0)
        self.blur = transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))

    def _frame_ops(self, frame):
        # frame: [1,H,W] -> PIL 'L' -> back to [1,H,W]
        pil = self.toPIL(frame)
        pil = self.crop(pil)
        pil = self.hflip(pil)
        pil = self.color(pil)
        if random.random() < self.blur_prob:
            pil = self.blur(pil)
        return self.toTensor(pil)

    def __call__(self, clip):
        frames = [self._frame_ops(f) for f in clip]
        out = torch.stack(frames, dim=0)
        out = self.roll(out)
        if random.random() < self.temporal_jitter_prob and out.shape[0] > 1:
            shift = random.choice([-1, 1])
            out = torch.roll(out, shifts=shift, dims=0)
        return out

class TwoClipViews:
    """Produce two independently augmented views for SimCLR."""
    def __init__(self, **kwargs):
        self.tf1 = ClipTransform(**kwargs)
        self.tf2 = ClipTransform(**kwargs)
    def __call__(self, clip):
        return self.tf1(clip), self.tf2(clip)

class SimCLRWrapperDataset(torch.utils.data.Dataset):
    """
    Wrap a base dataset (that returns dict with 'video') to output (view1, view2).
    """
    def __init__(self, base_ds, two_views: TwoClipViews):
        self.base = base_ds
        self.two_views = two_views
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        item = self.base[idx]
        clip = item["video"]        # [T,1,H,W] float in [0,1]
        v1, v2 = self.two_views(clip)
        return v1, v2

def collate_two_views(batch):
    v1 = torch.stack([b[0] for b in batch], dim=0)  # [B,T,1,H,W]
    v2 = torch.stack([b[1] for b in batch], dim=0)
    return v1, v2

def make_simclr_transforms():
    # keep size consistent with your dataset (H,W)=(32,64)
    return TwoClipViews(
        size=(32, 64),
        blur_prob=0.5,
        flip_prob=0,
        roll_prob=0,
        brightness=0.15,
        contrast=0,
        temporal_jitter_prob=0.2
    )
