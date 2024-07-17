from torch import nn
from einops import rearrange
from models.factory.factory import create_multiswin, create_baseline


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.base, self.configs = create_multiswin()
        
    def forward(self, x, return_attention=False, layer_id=1):
        ws = self.configs[0]["window_size"]
        out_channels = self.configs[1]["hidden_size"][-1] * 3

        final_x, view_x, dct_x = self.base(x)
        final_x = final_x if return_attention else rearrange(final_x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=1, p2=1, h=ws, w=ws, c=out_channels)
        return final_x, view_x, dct_x
    
# baseline: single view + progressive decoder
class BaselineEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = create_baseline()
    
    def forward(self, x):
        x = self.base(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=7)
        return x
    
class ThreeViewSpatialEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.base, self.configs = create_multiswin()
        
    def forward(self, x, return_attention=False, layer_id=1):
        ws = self.configs[0]["window_size"]
        out_channels = self.configs[1]["hidden_size"][-1] * 3

        final_x = self.base(x)
        final_x = final_x if return_attention else rearrange(final_x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=1, p2=1, h=ws, w=ws, c=out_channels)
        return final_x
    

