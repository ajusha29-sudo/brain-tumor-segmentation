import numpy as np


def remap_labels(seg_data):
    """
    BraTS labels are {0, 1, 2, 4} - not contiguous.
    Remap 4 -> 3 so labels become {0, 1, 2, 3}, which is required
    for standard loss functions (e.g. CrossEntropyLoss) that expect
    contiguous class indices starting at 0.
    """
    remapped = seg_data.copy()
    remapped[remapped == 4] = 3
    return remapped


def normalize_volume(volume):
    """
    Z-score normalize a single MRI volume using only non-zero
    (brain tissue) voxels, so background zeros don't skew the stats.
    """
    mask = volume > 0
    mean = volume[mask].mean()
    std = volume[mask].std()
    normalized = (volume - mean) / (std + 1e-8)  # epsilon avoids divide-by-zero
    return normalized

def get_tumor_slices(seg_data, min_tumor_pixels=50):
    """
    Returns a list of slice indices (along the depth axis) that contain
    a meaningful amount of tumor tissue.

    min_tumor_pixels filters out slices with only a tiny sliver of tumor
    (e.g. 1-2 stray pixels), which are often more noise than signal.
    """
    tumor_slice_indices = []
    for idx in range(seg_data.shape[2]):
        slice_mask = seg_data[:, :, idx]
        tumor_pixel_count = np.sum(slice_mask > 0)
        if tumor_pixel_count >= min_tumor_pixels:
            tumor_slice_indices.append(idx)
    return tumor_slice_indices
