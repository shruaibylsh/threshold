import math
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

RAY_COUNT = 2048

def fibonacci_sphere_dirs(n: int) -> np.ndarray:
    """Generate n unit vectors evenly distributed on a sphere using Fibonacci method."""
    dirs = np.zeros((n, 3), dtype=np.float64)
    offset = 2.0 / n
    inc = math.pi * (3.0 - math.sqrt(5.0))
    
    for i in range(n):
        y = ((i * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        phi = i * inc
        x = math.cos(phi) * r
        z = math.sin(phi) * r
        dirs[i, :] = (x, y, z)
    
    return dirs

def dirs_to_angles_xyz(dirs: np.ndarray) -> tuple:
    """Convert Cartesian directions (x, y, z) to azimuth and elevation angles."""
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    az = np.arctan2(z, x)
    el = np.arcsin(np.clip(y, -1.0, 1.0))
    return az, el

def angles_to_pixels(az: np.ndarray, el: np.ndarray, W: int, H: int) -> tuple:
    """Map azimuth and elevation to pixel coordinates in an equirectangular panorama."""
    u = ((az + np.pi) / (2.0 * np.pi)) * (W - 1)
    v = ((np.pi / 2.0 - el) / np.pi) * (H - 1)
    return u, v

def rasterize_frame_log(distances, u, v, W, H, max_dist=10.0):
    """Rasterize ray distances into a depth panorama with log scaling and normalization."""
    eps = 1e-3
    clipped = np.clip(distances, eps, max_dist)
    log_d = np.log(clipped)
    
    pano = np.full((H, W), np.inf)
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    lin_idx = vi * W + ui
    flat = pano.flatten()
    np.minimum.at(flat, lin_idx, log_d)
    pano = flat.reshape(H, W)
    pano[np.isinf(pano)] = np.nan
    
    known_mask = np.isfinite(pano)
    if known_mask.any():
        ys, xs = np.where(known_mask)
        vals = pano[ys, xs]
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        grid_points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
        tree = cKDTree(np.column_stack([ys, xs]))
        _, nn_idx = tree.query(grid_points, k=1)
        filled = vals[nn_idx].reshape(H, W)
        log_depth = np.where(np.isfinite(filled), filled, np.log(max_dist))
    else:
        log_depth = np.full((H, W), np.log(max_dist))
    
    lo, hi = np.nanpercentile(log_depth, [1, 99])
    norm = (log_depth - lo) / (hi - lo)
    img = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return img

def rays_to_panorama(distances, W=64, H=32, max_dist=10.0):
    """Convert a single 2048-ray isovist to a depth panorama."""
    dirs = fibonacci_sphere_dirs(RAY_COUNT)
    az, el = dirs_to_angles_xyz(dirs)
    u, v = angles_to_pixels(az, el, W, H)
    return rasterize_frame_log(distances, u, v, W, H, max_dist)