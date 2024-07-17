from pathlib import Path
import torch
import ml_collections
from models.encoder.multi_view_swin_encoder import ThreeViewSwinTransformer
from models.modules.swin_transformer import SwinTransformer


def create_multiswin():
    view_configs = [
        ml_collections.ConfigDict({
            'hidden_size': [96, 192, 384, 768],
            'patches': {
                'size': (4, 4, 3)
            },
            'window_size': 7,
            'depths' : [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'mlp_dim': 768,
            'num_frames' : 1,
            'input_resolution':[(56, 56), (28, 28), (14, 14), (7, 7)],
            'temporal_dim' : 1
        }),
        ml_collections.ConfigDict({
            'hidden_size': [96, 192, 384, 768],
            'patches': {
                'size': (4, 4, 2)
            },
            'window_size': 7,
            'depths' : [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'mlp_dim': 1536,
            'num_frames' : 1,
            'input_resolution':[(56, 56), (28, 28), (14, 14), (7, 7)],
            'temporal_dim' : 1
        }),
        ml_collections.ConfigDict({
            'hidden_size': [128, 256, 512, 1024],
            'patches': {
                'size': (4, 4, 1)
            },
            'window_size': 7,
            'depths' : [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'mlp_dim': 3072,
            'num_frames' : 3,
            'input_resolution':[(56, 56), (28, 28), (14, 14), (7, 7)],
            'temporal_dim' : 3
        })
    ]
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
        'num_frames' : 3
    })
    model = ThreeViewSwinTransformer(view_configs=view_configs, 
                                     input_token_temporal_dims = input_token_temporal_dims, 
                                     global_encoder_config=global_encoder_config)
    path = "/media/zhangying/Datas/gitCode/VIDNet/src/model_parameters/swin_ori.pth"
    #path = "/media/zhangying/Datas/gitCode/VIDNet/src/model_parameters/swin_freq.pth"
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    return model, view_configs

def create_baseline():
    # we use the base swin-transformer to construct the baseline model
    view_configs = ml_collections.ConfigDict({
            'hidden_size': [128, 256, 512, 1024],
            'patches': {
                'size': (4, 4, 3)
            },
            'window_size': 7,
            'depths' : [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'mlp_dim': 3072,
            'num_frames' : 3,
            'input_resolution':[(56, 56), (28, 28), (14, 14), (7, 7)],
            'temporal_dim' : 3
        })
     
    model = SwinTransformer(view_configs, img_size=224, patch_size=4, in_chans=3, num_classes=0,
                 embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False)
    
    path = "/media/zhangying/Datas/gitCode/VIDNet/src/model_parameters/baseline_b.pth"
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    return model
