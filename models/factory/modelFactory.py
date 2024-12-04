import torch
import ml_collections
from pathlib import Path
from models.encoder.multiTemporalViewEncoder import ThreeViewSwinTransformer
from models.modules.swinTransformer import SwinTransformer


def load_model_weights(model, path, strict=False):
    """
    Helper function to load model weights from a specified path.
    """
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    return model


def create_view_config(hidden_sizes, patches_size, depths, num_heads, mlp_dim, num_frames, input_resolution,
                       temporal_dim, temporal_ratio=None):
    """
    Helper function to generate a view configuration.
    """
    return ml_collections.ConfigDict({
        'hidden_size': hidden_sizes,
        'patches': {'size': patches_size},
        'window_size': 7,
        'depths': depths,
        'num_heads': num_heads,
        'mlp_dim': mlp_dim,
        'num_frames': num_frames,
        'input_resolution': input_resolution,
        'temporal_dim': temporal_dim,
        'temporal_ratio': temporal_ratio or [1] * len(depths)
    })


def create_multiswin():
    # Define the view configurations
    view_configs = [
        create_view_config([96, 192, 384, 768], (4, 4, 3), [2, 2, 6, 2], [3, 6, 12, 24], 768, 1,
                           [(56, 56), (28, 28), (14, 14), (7, 7)], 1, [1, 1]),
        create_view_config([96, 192, 384, 768], (4, 4, 2), [2, 2, 18, 2], [3, 6, 12, 24], 1536, 1,
                           [(56, 56), (28, 28), (14, 14), (7, 7)], 1, [1, 3]),
        create_view_config([128, 256, 512, 1024], (4, 4, 1), [2, 2, 18, 2], [4, 8, 16, 32], 3072, 3,
                           [(56, 56), (28, 28), (14, 14), (7, 7)], 3),
    ]

    # Cross view fusion configuration
    cross_view_fusion = [5, 11]
    input_token_temporal_dims = [1, 1, 3]
    temporal_encoding_config = ml_collections.ConfigDict({
        'method': '3d_conv',
        'kernel_init_method': 'central_frame_initializer',
    })

    global_encoder_config = ml_collections.ConfigDict({
        'num_heads': 12,
        'mlp_dim': 3072,
        'num_layers': 12,
        'hidden_size': 768,
        'merge_axis': 'channel',
        'num_frames': 3
    })

    # Initialize the model
    model = ThreeViewSwinTransformer(view_configs=view_configs,
                                     input_token_temporal_dims=input_token_temporal_dims,
                                     global_encoder_config=global_encoder_config)

    # Load model weights
    path = "../weights/weight.pth"
    model = load_model_weights(model, path, strict=False)

    return model, view_configs


def create_baseline():
    # Define the baseline view configuration
    view_config = create_view_config([128, 256, 512, 1024], (4, 4, 3), [2, 2, 18, 2], [4, 8, 16, 32], 3072, 3,
                                     [(56, 56), (28, 28), (14, 14), (7, 7)], 3)

    # Initialize the baseline model
    model = SwinTransformer(view_config, img_size=224, patch_size=4, in_chans=3, num_classes=0,
                            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                            norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                            use_checkpoint=False, fused_window_process=False)

    # Load model weights
    path = "../weughts/weight.pth"
    model = load_model_weights(model, path, strict=True)

    return model
