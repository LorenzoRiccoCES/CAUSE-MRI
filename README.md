
# ***Self-Supervised Vision Transformer and Causal Inference for MRI semantic segmentation***

This is pytorch implementation code for *CAusal Unsupervised Semantic sEgmentation (CAUSE)* to improve performance of unsupervised semantic segmentation, adapted in order to handle 3D MR images using Swin UNETR architecture from MONAI. 

This code is further developed by three baseline codes of [HP: Leveraging Hidden Positives for Unsupervised Semantic Segmentation](https://github.com/hynnsk/HP) accepted in [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Seong_Leveraging_Hidden_Positives_for_Unsupervised_Semantic_Segmentation_CVPR_2023_paper.pdf)
and [STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences](https://github.com/mhamilton723/STEGO) accepted in [ICLR 2022](https://iclr.cc/virtual/2022/poster/6068).
and [CAUSE: Causal Unsupervised Semantic Segmentation](https://github.com/ByungKwanLee/Causal-Unsupervised-Segmentation)(https://arxiv.org/abs/2310.07379)

Credit to BK Lee (https://github.com/ByungKwanLee?tab=repositories) for the original implementation!

---

## 🤖 CAUSE Framework (Top-Level File Directory Layout) 
    .
    ├── loader
    │   ├── netloader.py                # Self-Supervised Pretrained Model Loader & Segmentation Head Loader
    │   └── dataloader.py               # Dataloader Thanks to STEGO [ICLR 2022]
    │
    ├── data
    │   ├── mri_transforms.py           # Transformation for MR Images 
    │   └── mri_data_handler.py         # Dataloader adapted to 2D and 3D images
    |
    ├── models                          # Model Design of Self-Supervised Pretrained: [DINO/DINOv2/iBOT/MAE/MSN/Swin-UNETR]
    │   ├── dinomaevit.py               # ViT Structure of DINO and MAE
    │   ├── dinov2vit.py                # ViT Structure of DINOv2
    │   ├── ibotvit.py                  # ViT Structure of iBOT
    │   └── msnvit.py                   # ViT Structure of MSN
    │   └── swin_unetr.py               # ViT Structure of Swin-UNETR
    |   
    │
    ├── modules                         # Segmentation Head and Its Necessary Function
    │   └── segment_module.py           # [Including Tools with Generating Concept Book and Contrastive Learning
    │   └── segment.py                  # [MLP & TR] Including Tools with Generating Concept Book and Contrastive Learning
    │
    ├── utils
    │   └── utils.py                    # Utility for auxiliary tools
    │
    ├── crop_mri_dataset.py             # Crop of the dataset
    |
    ├── train_modularity.py             # (STEP 1) [MLP & TR] Generating Concept Cluster Book as a Mediator
    │
    ├── train_front_door_mlp.py         # (STEP 2) [MLP] Frontdoor Adjustment through Unsupervised Semantic Segmentation
    ├── fine_tuning_mlp.py              # (STEP 3) [MLP] Fine-Tuning Cluster Probe
    │
    ├── train_front_door_tr.py          # (STEP 2) [TR] Frontdoor Adjustment through Unsupervised Semantic Segmentation
    ├── fine_tuning_tr.py               # (STEP 3) [TR] Fine-Tuning Cluster Probe
    │
    ├── test_mlp.py                     # [MLP] Evaluating Unsupervised Semantic Segmantation Performance (Post-Processing)
    ├── test_tr.py                      # [TR] Evaluating Unsupervised Semantic Segmantation Performance (Post-Processing)
    │
    ├── requirements.txt
    └── README.md

---

## How to Run CAUSE?

For the first, we should generate the cropped dataset, then run the script

---

### 1. Training CAUSE

### (STEP 1): Generating Mediator based on Modularity

```shell script
python train_mediator.py # DINO/DINOv2/iBOT/MSN/MAE
```

### (STEP 2): Frontdoor Adjustment through Contrastive Learning

```shell script
python train_front_door_mlp.py # CAUSE-MLP

# or

python train_front_door_tr.py # CAUSE-TR
```

### (STEP 3):  *Technical STEP: Fine-Tuning Cluster Probe*

```shell script
python fine_tuning_mlp.py # CAUSE-MLP

# or

python fine_tuning_tr.py # CAUSE-TR
```

---

### 2. Testing CAUSE

```shell script
python test_mlp.py # CAUSE-MLP

# or

python test_tr.py # CAUSE-TR
```

---

## 💡 Environment Settings

* Creating Virtual Environment by Anaconda
> conda create -y -n neurips python=3.9

* Installing [PyTorch]((https://pytorch.org/)) Package in Virtual Envrionment
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

* Installing Pip Package
> pip install -r requirements.txt

* [Optional] Removing Conda and PIP Cache if Conda and PIP have been locked by unknown reasons
> conda clean -a && pip cache purge

---

### Usage
- Place your self-supervised pretrained Swin UNETR weights locally (e.g., `checkpoint/swinunetr_base.pth`).
- Name your checkpoint file with the prefix `swinunetr_` (e.g., `swinunetr_base.pth`) so the loader can recognize it.
- The loader will automatically use the Swin UNETR model for 3D MRI volumes if the checkpoint name starts with `swinunetr`.

#### Example
```python
from loader.netloader import load_model
net = load_model('checkpoint/swinunetr_base.pth')
```

- The model expects 3D MRI volumes as input (shape: `[B, C, D, H, W]`).
- The output is a 3D segmentation mask.

#### Customization
You can customize Swin UNETR parameters by editing the factory function in `models/swin_unetr.py`.

---



