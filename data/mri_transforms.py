from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, CenterSpatialCropd, RandFlipd, RandRotate90d, RandGaussianNoised,
    NormalizeIntensityd, ToTensord, RandSpatialCropd
)

def basic_mri_transforms(modalities, crop_size=(96,96,96), spacing=(1.0,1.0,1.0)):
    keys_img = [f"image_{mod}" for mod in modalities]
    all_keys = keys_img + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Spacingd(keys=all_keys, pixdim=spacing, mode=("bilinear",)*len(keys_img)+("nearest",)),
        Orientationd(keys=all_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=keys_img, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=all_keys, source_key=keys_img[0]),
        CenterSpatialCropd(keys=all_keys, roi_size=crop_size),
        NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),
        ToTensord(keys=all_keys),
    ])

def augmented_mri_transforms(modalities, crop_size=(96,96,96), spacing=(1.0,1.0,1.0)):
    keys_img = [f"image_{mod}" for mod in modalities]
    all_keys = keys_img + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Spacingd(keys=all_keys, pixdim=spacing, mode=("bilinear",)*len(keys_img)+("nearest",)),
        Orientationd(keys=all_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=keys_img, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=all_keys, source_key=keys_img[0]),
        CenterSpatialCropd(keys=all_keys, roi_size=crop_size),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=all_keys, prob=0.5, spatial_axis=[2]),
        RandRotate90d(keys=all_keys, prob=0.5, max_k=3),
        RandGaussianNoised(keys=keys_img, prob=0.2, mean=0.0, std=0.1),
        NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),
        ToTensord(keys=all_keys),
    ])

def sliding_window_mri_transforms(modalities, window_size=(96,96,96), spacing=(1.0,1.0,1.0)):
    keys_img = [f"image_{mod}" for mod in modalities]
    all_keys = keys_img + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        Spacingd(keys=all_keys, pixdim=spacing, mode=("bilinear",)*len(keys_img)+("nearest",)),
        Orientationd(keys=all_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=keys_img, a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=all_keys, source_key=keys_img[0]),
        RandSpatialCropd(keys=all_keys, roi_size=window_size, random_size=False),
        NormalizeIntensityd(keys=keys_img, nonzero=True, channel_wise=True),
        ToTensord(keys=all_keys),
    ]) 