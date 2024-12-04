import torch
import torch.nn as nn
from einops import rearrange


class SEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        x1, x2 = x
        return x1*self.upsample(self.conv(x2))
    

class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

class Decoder(nn.Module):
    
    def merge_views_along_channel_axis(self, tokens, height):
        """Merges tokens from each view along the channel axis."""
        max_temporal_dim = max(self.input_token_temporal_dims)
        xs = []
        for idx, x in enumerate(tokens):
            bs, time, n, c = x.shape
            x = x.reshape(bs, self.input_token_temporal_dims[idx], (time * n) // self.input_token_temporal_dims[idx], c)
            xs.append(x.repeat((1, max_temporal_dim // x.shape[1], 1, 1)))
        out = torch.concatenate(xs, axis=-1)# out : b*maxt*n*c'
        out = rearrange(out, 'b t (h w) c -> b c t h w', h=height)
        return out
    
    def __init__(self, in_channels=2304, out_channels=1, 
                 kernel_size=7, num_classes=32, dap_k=2, 
                 features=[256, 256, 256, 256, 256], 
                 input_token_temporal_dims=[1, 1, 3],
                 rgb_features=[320, 640, 1280, 2560],
                 shape=[56, 28, 14, 7]):
        super().__init__()

        self.input_token_temporal_dims = input_token_temporal_dims
        max_temporal_dim = max(self.input_token_temporal_dims)
        self.shape = shape

        self.decoder_2 = nn.Sequential(
                            nn.Conv2d(num_classes, num_classes * dap_k**2, 3, padding=1),
                            #nn.BatchNorm2d(num_classes * dap_k**2),
                            nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
                
        self.decoder_3 = nn.Sequential(
                            nn.Conv2d(num_classes * dap_k**2, num_classes * dap_k**2, 3, padding=1),
                            #nn.BatchNorm2d( num_classes * dap_k**2),
                            nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))

        self.decoder_4 = nn.Sequential(
                            nn.Conv2d(num_classes * dap_k**2, num_classes * dap_k**2, 3, padding=1),
                            #nn.BatchNorm2d(num_classes * dap_k**2),
                            nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))

        self.decoder_5 = nn.Sequential(
                            nn.Conv2d(num_classes * dap_k**2, num_classes * dap_k**2, 3, padding=1),
                            #nn.BatchNorm2d(num_classes * dap_k**2),
                            nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        
        self.final_out = nn.Conv2d(num_classes, out_channels, 3, padding=1)
        
        # ---------------- RGB feature ------------------- #
        self.rgb_decoder_1 = nn.Sequential(
                            nn.Conv3d(rgb_features[0], features[0], kernel_size=(max_temporal_dim, 1, 1), padding=0, stride=(max_temporal_dim, 1, 1)),
                            #nn.BatchNorm3d(rgb_features[0]),
                            nn.GroupNorm(num_groups=16, num_channels=features[0]),
                            nn.ReLU(inplace=True))

        self.rgb_decoder_2 = nn.Sequential(
                            nn.Conv3d(rgb_features[1], features[1], kernel_size=(max_temporal_dim, 1, 1), padding=0, stride=(max_temporal_dim, 1, 1)),
                            #nn.BatchNorm3d(rgb_features[1]),
                            nn.GroupNorm(num_groups=16, num_channels=features[1]),
                            nn.ReLU(inplace=True))
                
        self.rgb_decoder_3 = nn.Sequential(
                            nn.Conv3d(rgb_features[2], features[2], kernel_size=(max_temporal_dim, 1, 1), padding=0, stride=(max_temporal_dim, 1, 1)),
                            #nn.BatchNorm3d(rgb_features[2]),
                            nn.GroupNorm(num_groups=16, num_channels=features[2]),
                            nn.ReLU(inplace=True))

        self.rgb_decoder_4 = nn.Sequential(
                            nn.Conv3d(rgb_features[3], features[3], kernel_size=(max_temporal_dim, 1, 1), padding=0, stride=(max_temporal_dim, 1, 1)),
                            #nn.BatchNorm3d(rgb_features[3]),
                            nn.GroupNorm(num_groups=16, num_channels=features[3]),
                            nn.ReLU(inplace=True))
        # gcm
        self.gcm1 = _GlobalConvModule(features[-1]+in_channels, 1 * num_classes * 4, (kernel_size, kernel_size))
        self.gcm2 = _GlobalConvModule(features[-2], 1 * num_classes, (kernel_size, kernel_size))
        self.gcm3 = _GlobalConvModule(features[-3], 1 * num_classes * dap_k**2, (kernel_size, kernel_size))
        self.gcm4 = _GlobalConvModule(features[-4], 1 * num_classes * dap_k**2, (kernel_size, kernel_size))
        
        # ecre
        self.ecre = nn.PixelShuffle(2)

        # seb
        self.seb1 = SEB(features[-1], features[-2])
        self.seb2 = SEB(features[-2]+features[-1], features[-3])
        self.seb3 = SEB(features[-3]+features[-2]+features[-1], features[-4])

        # upsample
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

        # dap
        self.DAP = nn.Sequential(
            nn.PixelShuffle(dap_k),
            nn.AvgPool2d((dap_k, dap_k))
        )

        # freq-aware
        # 224 -> 112
        self.decoder_frequency_0 = nn.Sequential(
                                nn.AvgPool2d(2, stride=2),
                                nn.Conv2d(9, num_classes * dap_k**2, 3, padding=1),
                                #nn.BatchNorm2d(num_classes * dap_k**2),
                                nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                                nn.Sigmoid())
                                
        # 112 -> 56
        self.decoder_frequency_1 = nn.Sequential(
                                nn.AvgPool2d(2, stride=2),
                                nn.Conv2d(num_classes * dap_k**2, num_classes * dap_k**2, 3, padding=1),
                                #nn.BatchNorm2d(num_classes * dap_k**2),  
                                nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                                nn.Sigmoid())
        # 56 -> 28
        self.decoder_frequency_2 = nn.Sequential(
                                nn.AvgPool2d(2, stride=2),
                                nn.Conv2d(num_classes * dap_k**2, num_classes * dap_k**2, 3, padding=1),
                                #nn.BatchNorm2d(num_classes * dap_k**2),
                                nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                                nn.Sigmoid())
        # 28 -> 14
        self.decoder_frequency_3 = nn.Sequential(
                                nn.AvgPool2d(2, stride=2),
                                nn.Conv2d(num_classes * dap_k**2, num_classes, 3, padding=1),
                                #nn.BatchNorm2d(num_classes),
                                nn.GroupNorm(num_groups=4, num_channels=num_classes),
                                nn.Sigmoid())
        # 14 -> 7
        self.decoder_frequency_4 = nn.Sequential(
                                nn.AvgPool2d(2, stride=2),
                                nn.Conv2d(num_classes, num_classes * dap_k**2, 3, padding=1),
                                #nn.BatchNorm2d(num_classes * 4),
                                nn.GroupNorm(num_groups=8, num_channels=num_classes * dap_k**2),
                                nn.Sigmoid())
        
    def forward(self, x, view_x, ffinfo):
        ###### reshape temporal dim ######
        merged_x = []
        for idx, stage in enumerate(view_x):
            mergedx = self.merge_views_along_channel_axis(stage, self.shape[idx]) # B *maxT*n*c'
            merged_x.append(mergedx)

        # rgb
        rgb1 = self.rgb_decoder_1(merged_x[0]).squeeze(-3)
        rgb2 = self.rgb_decoder_2(merged_x[1]).squeeze(-3)
        rgb3 = self.rgb_decoder_3(merged_x[2]).squeeze(-3)
        rgb4 = self.rgb_decoder_4(merged_x[3]).squeeze(-3)
        
        # freq
        freq0 = self.decoder_frequency_0(ffinfo) # 112
        freq1 = self.decoder_frequency_1(freq0) # 56
        freq2 = self.decoder_frequency_2(freq1) # 28
        freq3 = self.decoder_frequency_3(freq2) # 14
        freq4 = self.decoder_frequency_4(freq3) # 7

        ### start ###
        gcn0 = self.gcm1(torch.cat([rgb4, x], dim=-3))  
        out1 = self.ecre(gcn0*freq4) # channel : num_classes

        seb1 = self.seb1([rgb3, rgb4])
        gcn1 = self.gcm2(seb1) # channel : num_classes

        seb2 = self.seb2([rgb2, torch.cat([rgb3, self.upsample2(rgb4)], dim=1)])
        gcn2 = self.gcm3(seb2) # channel : num_classes * dap_k**2

        seb3 = self.seb3([rgb1, torch.cat([rgb2, self.upsample2(rgb3), self.upsample4(rgb4)], dim=1)])
        gcn3 = self.gcm4(seb3) # channel : num_classes * dap_k**2
        ### end ###

        #x = self.decoder_1(x + rgb4) # 2304-> features[0]
        x = self.decoder_2(gcn1*freq3 + out1) # features[0] -> features[1]
        x = self.decoder_3(x + gcn2*freq2) # features[1] -> features[2]
        x = self.decoder_4(x + gcn3*freq1) # features[2] -> features[3]
        x = self.decoder_5(x*freq0) # features[3] -> features[4]
        x_feats = self.DAP(x)
        binary_mask = self.final_out(x_feats)

        return binary_mask, x_feats
    

class BaselineDecoder(nn.Module):
    
    def __init__(self, in_channels = 2304, out_channels = 1, features=[256, 256, 256, 256, 256]):

        super().__init__()
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    #nn.BatchNorm2d(features[0]),
                    nn.GroupNorm(32, features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) 
                )

        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    #nn.BatchNorm2d(features[1]),
                    nn.GroupNorm(32, features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
                
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            #nn.BatchNorm2d(features[2]),
            nn.GroupNorm(32, features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            #nn.BatchNorm2d(features[3]),
            nn.GroupNorm(32, features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.decoder_5 = nn.Sequential(
            nn.Conv2d(features[3], features[4], 3, padding=1),
            #nn.BatchNorm2d(features[4]),
            nn.GroupNorm(32, features[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

        
    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x) 
        x = self.decoder_5(x)
        x = self.final_out(x) 

        return x

