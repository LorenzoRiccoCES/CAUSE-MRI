import os
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, RandSpatialCropd, CenterSpatialCropd, RandFlipd, RandRotate90d, RandGaussianNoised,
    NormalizeIntensityd, ToTensord
)
from monai.data import DataLoader, CacheDataset, Dataset as MONAIDataset
from monai.config import KeysCollection
import nibabel as nib
import shutil
from tqdm import tqdm

class MRIPreprocessor:
    """
    Compose MONAI transforms for 3D MRI preprocessing.
    Assumes input .nii.gz files may have shape [D, H, W, 4] (multi-modal in last channel).
    Keeps all channels as input.
    """
    def __init__(self, modalities=None, crop_size=(96,96,96), spacing=(1.0,1.0,1.0), augment=False, intensity_cfg=None, augmentation_cfg=None):
        if intensity_cfg is None:
            intensity_cfg = {"a_min": 0, "a_max": 3000, "b_min": 0.0, "b_max": 1.0, "clip": True}
        if augmentation_cfg is None:
            augmentation_cfg = {"flip_prob": 0.5, "rotate90_prob": 0.5, "gaussian_noise_prob": 0.2, "gaussian_noise_std": 0.1}
        keys = ["image", "label"]
        self.modalities = modalities
        self.crop_size = crop_size
        self.spacing = spacing
        self.augment = augment
        self.keys = keys
        self.transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=intensity_cfg.get("a_min", 0), a_max=intensity_cfg.get("a_max", 3000), b_min=intensity_cfg.get("b_min", 0.0), b_max=intensity_cfg.get("b_max", 1.0), clip=intensity_cfg.get("clip", True)),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            CenterSpatialCropd(keys=["image", "label"], roi_size=crop_size),
            ToTensord(keys=["image", "label"]),
        ])
        if augment:
            self.transforms.transforms += [
                RandFlipd(keys=["image", "label"], prob=augmentation_cfg.get("flip_prob", 0.5), spatial_axis=0),
                RandRotate90d(keys=["image", "label"], prob=augmentation_cfg.get("rotate90_prob", 0.5), max_k=3),
                RandGaussianNoised(keys=["image"], prob=augmentation_cfg.get("gaussian_noise_prob", 0.2), std=augmentation_cfg.get("gaussian_noise_std", 0.1)),
            ]

    def __call__(self, data):
        return self.transforms(data)

class MRIDataLoader:
    """
    Loads 3D MRI volumes and masks, supports multi-modal MRI (e.g., FLAIR, T1w, T1gd, T2w).
    """
    def __init__(self, data_dir: str, modalities: List[str], split: str = "train", label_key: str = "label", crop_size=(96,96,96), spacing=(1.0,1.0,1.0), augment: bool = False, cache_rate: float = 0.0, num_workers: int = 4):
        self.data_dir = data_dir
        self.modalities = modalities
        self.split = split
        self.label_key = label_key
        self.crop_size = crop_size
        self.spacing = spacing
        self.augment = augment
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.data_list = self._gather_data()
        self.preprocessor = MRIPreprocessor(modalities, crop_size, spacing, augment)
        self.dataset = CacheDataset(
            data=self.data_list,
            transform=self.preprocessor,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers
        )

    def _gather_data(self) -> List[Dict]:
        # Instead of per-modality files, expect a single multi-channel file per subject
        subjects = sorted([d for d in os.listdir(os.path.join(self.data_dir, self.split)) if os.path.isdir(os.path.join(self.data_dir, self.split, d))])
        data_list = []
        for subject in subjects:
            img_path = os.path.join(self.data_dir, self.split, subject, "image.nii.gz")
            label_path = os.path.join(self.data_dir, self.split, subject, "label.nii.gz")
            data_list.append({"image": img_path, "label": label_path})
        return data_list

    def get_loader(self, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)

class MRIDataset(Dataset):
    """
    PyTorch Dataset wrapper for MONAI preprocessed MRI data.
    """
    def __init__(self, data_list: List[Dict], transform):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return self.transform(data)

# Example usage (for integration/testing):
# modalities = ["flair", "t1", "t1ce", "t2"]
# loader = MRIDataLoader(data_dir="/path/to/Task01_BrainTumour", modalities=modalities, split="train")
# train_loader = loader.get_loader(batch_size=2)
# for batch in train_loader:
#     img = batch['image'][0].numpy()  # [4, D, H, W] if batch size is 1
#     labels = batch["label"]
#     ... 

if __name__ == "__main__":
    imagesTr = 'data/raw/imagesTr'
    labelsTr = 'data/raw/labelsTr'
    output_dir = 'data/raw/train'

    os.makedirs(output_dir, exist_ok=True)

    for img_file in os.listdir(imagesTr):
        if not img_file.endswith('.nii.gz'):
            continue
        subject_id = img_file.replace('.nii.gz', '')
        subject_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        shutil.copy(os.path.join(imagesTr, img_file), os.path.join(subject_dir, 'image.nii.gz'))
        label_file = os.path.join(labelsTr, img_file)
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(subject_dir, 'label.nii.gz'))

    main()

    
