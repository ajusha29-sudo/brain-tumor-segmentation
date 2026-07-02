import nibabel as nib
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
    Background voxels are forced back to 0 after normalization so
    they remain cheap for the model to recognize and ignore.
    """
    mask = volume > 0
    mean = volume[mask].mean()
    std = volume[mask].std()
    normalized = (volume - mean) / (std + 1e-8)
    normalized[~mask] = 0
    return normalized


def get_tumor_slices(seg_data, min_tumor_pixels=50):
    """
    Returns a list of slice indices (along the depth axis) that contain
    a meaningful amount of tumor tissue.
    """
    tumor_slice_indices = []
    for idx in range(seg_data.shape[2]):
        slice_mask = seg_data[:, :, idx]
        tumor_pixel_count = np.sum(slice_mask > 0)
        if tumor_pixel_count >= min_tumor_pixels:
            tumor_slice_indices.append(idx)
    return tumor_slice_indices


def load_patient_volumes(patient_path):
    """
    Loads T1, T1ce, T2, FLAIR, and segmentation mask for one patient.
    Applies label remapping and per-volume normalization.
    Returns a dict of numpy arrays.
    """
    patient_id = patient_path.split('/')[-1]

    # build paths
    t1_path = f'{patient_path}/{patient_id}_t1.nii.gz'
    t1ce_path = f'{patient_path}/{patient_id}_t1ce.nii.gz'
    t2_path = f'{patient_path}/{patient_id}_t2.nii.gz'
    flair_path = f'{patient_path}/{patient_id}_flair.nii.gz'
    seg_path = f'{patient_path}/{patient_id}_seg.nii.gz'

    # load each file
    t1_data = nib.load(t1_path).get_fdata()
    t1ce_data = nib.load(t1ce_path).get_fdata()
    t2_data = nib.load(t2_path).get_fdata()
    flair_data = nib.load(flair_path).get_fdata()
    seg_data = nib.load(seg_path).get_fdata()

    # normalize modalities only (NOT the segmentation mask)
    t1_norm = normalize_volume(t1_data)
    t1ce_norm = normalize_volume(t1ce_data)
    t2_norm = normalize_volume(t2_data)
    flair_norm = normalize_volume(flair_data)

    # remap segmentation labels (4 -> 3)
    seg_remapped = remap_labels(seg_data)

    return {
        't1': t1_norm,
        't1ce': t1ce_norm,
        't2': t2_norm,
        'flair': flair_norm,
        'seg': seg_remapped
    }
