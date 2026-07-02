import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import get_tumor_slices, load_patient_volumes


def augment(image, mask, p=0.5):
    """
    Applies a synchronized random horizontal flip to an image and its
    corresponding mask. The same flip decision and axis choice must be
    applied to both, otherwise the label mask will no longer line up
    with the image content.

    image: (4, H, W) tensor - 4 MRI modality channels
    mask:  (H, W) tensor    - single-channel class label map
    """
    if random.random() < p:
        image = torch.flip(image, dims=[2])  # flip width axis of image
        mask = torch.flip(mask, dims=[1])    # flip width axis of mask
    return image, mask


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for 2D slice-based brain tumor segmentation on BraTS 2021.

    Each sample is a single 2D axial slice (all 4 modalities stacked as
    channels) plus its corresponding 2D segmentation mask.

    To address class imbalance (most slices contain no tumor), all
    tumor-containing slices are kept, but only a fraction (non_tumor_ratio)
    of background-only slices are kept per patient.
    """

    def __init__(self, patient_paths, non_tumor_ratio=0.2):
        self.samples = []  # list of (patient_path, slice_idx) tuples
        self._cached_patient_path = None
        self._cached_volumes = None

        for p_path in patient_paths:
            p_id = p_path.split('/')[-1]
            seg_img = nib.load(f'{p_path}/{p_id}_seg.nii.gz')
            seg_data = seg_img.get_fdata()

            tumor_slices = get_tumor_slices(seg_data)
            all_slices = list(range(seg_data.shape[2]))
            non_tumor_slices = [s for s in all_slices if s not in tumor_slices]

            num_to_keep = int(len(non_tumor_slices) * non_tumor_ratio)
            sampled_non_tumor = random.sample(non_tumor_slices, num_to_keep)

            self.samples.extend([(p_path, idx) for idx in tumor_slices])
            self.samples.extend([(p_path, idx) for idx in sampled_non_tumor])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_path, slice_idx = self.samples[idx]

        # Cache the last-loaded patient's volumes to avoid re-reading
        # the same 3D volume from disk for every slice of that patient.
        if patient_path != self._cached_patient_path:
            self._cached_volumes = load_patient_volumes(patient_path)
            self._cached_patient_path = patient_path

        volumes = self._cached_volumes

        t1_slice = volumes['t1'][:, :, slice_idx]
        t1ce_slice = volumes['t1ce'][:, :, slice_idx]
        t2_slice = volumes['t2'][:, :, slice_idx]
        flair_slice = volumes['flair'][:, :, slice_idx]

        image = np.stack([t1_slice, t1ce_slice, t2_slice, flair_slice], axis=0)
        mask = volumes['seg'][:, :, slice_idx]

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).long()

        image_tensor, mask_tensor = augment(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
