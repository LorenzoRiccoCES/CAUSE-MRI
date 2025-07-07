import os
import yaml
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from data.mri_data_handler import MRIDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/mri_preprocessing.yaml')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    modalities = cfg['modalities']
    crop_size = tuple(cfg['crop_size'])
    spacing = tuple(cfg['spacing'])
    augment = False  # For deterministic cropping
    loader = MRIDataLoader(
        data_dir=args.data_dir,
        modalities=modalities,
        split=args.split,
        crop_size=crop_size,
        spacing=spacing,
        augment=augment,
        cache_rate=cfg.get('cache_rate', 0.0),
        num_workers=cfg.get('num_workers', 4)
    )
    dataloader = loader.get_loader(batch_size=1, shuffle=False)

    img_out_dir = os.path.join(args.output_dir, 'img', args.split)
    label_out_dir = os.path.join(args.output_dir, 'label', args.split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    counter = 0
    for batch in tqdm(dataloader):
        # Use the new multi-channel format: batch['image'] is [B, 4, D, H, W]
        img = batch['image'][0].numpy()  # [4, D, H, W] for batch size 1
        label = batch['label'][0].numpy()  # [D, H, W] or [1, D, H, W]
        # Save as NIfTI
        img_path = os.path.join(img_out_dir, f"{counter}.nii.gz")
        label_path = os.path.join(label_out_dir, f"{counter}.nii.gz")
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(img, affine), img_path)
        nib.save(nib.Nifti1Image(label, affine), label_path)
        counter += 1

if __name__ == "__main__":
    main() 