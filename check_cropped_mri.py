import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def check_cropped_data(img_dir, label_dir, num_samples=3):
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))

    print(f"Found {len(img_files)} images and {len(label_files)} labels.")
    assert len(img_files) == len(label_files), "Mismatch in number of images and labels!"

    for idx, (img_path, label_path) in enumerate(zip(img_files, label_files)):
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        print(f"\nSample {idx}:")
        print(f"  Image: {img_path}, shape: {img.shape}, min/max: {img.min():.3f}/{img.max():.3f}")
        print(f"  Label: {label_path}, shape: {label.shape}, unique: {np.unique(label)}")

        # Check image shape
        assert img.shape[0] == 4, f"Expected 4 channels, got {img.shape[0]}"
        assert img.shape[1:] == label.shape[-3:], f"Image/label shape mismatch: {img.shape} vs {label.shape}"

        # Visualize the first modality and label (middle slice)
        if idx < num_samples:
            mid_slice = img.shape[3] // 2
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(img[0, :, :, mid_slice], cmap='gray')
            plt.title('First modality, middle slice')
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(np.squeeze(label)[:, :, mid_slice])
            plt.title('Label, middle slice')
            plt.axis('off')
            plt.show()

    print("\nAll checks passed!")

if __name__ == "__main__":
    # Update these paths as needed
    img_dir = "data/brain_mri/cropped/img/train"
    label_dir = "data/brain_mri/cropped/label/train"
    check_cropped_data(img_dir, label_dir) 