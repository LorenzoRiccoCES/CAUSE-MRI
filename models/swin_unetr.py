import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

class SwinUNETRWrapper(nn.Module):
    """
    Wrapper for MONAI SwinUNETR. Expects 4-channel input for multi-modal MRI (e.g., [B, 4, D, H, W]).
    """
    def __init__(
        self,
        in_channels=4,  # Set default to 4 for multi-modal MRI
        out_channels=3,
        feature_size=48,
        patch_size=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        qkv_bias=True,
        mlp_ratio=4.0,
        norm_name='instance',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=3,
        downsample='merging',
        use_v2=False,
        pretrained_weights=None,
        **kwargs
    ):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
            **kwargs
        )
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x, return_features=False, feature_level=4):
        """
        If return_features=True, returns encoder features at the specified level, flattened as [B, N, C].
        Otherwise, returns the segmentation output.
        """
        if return_features:
            # Get encoder features at the desired level (e.g., encoder4 for deepest features)
            # encoder4: [B, C, D, H, W]
            feats = self.model.encoder4(x)
            B, C, D, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, N, C]
            return feats
        else:
            return self.model(x)


def swin_unetr_base(pretrained_weights=None, **kwargs):
    return SwinUNETRWrapper(pretrained_weights=pretrained_weights, **kwargs) 