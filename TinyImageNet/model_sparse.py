import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from neuron import MultiStepNegIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from wavelet_layers_sparse import (
    Haar2DForwardLevel1, Haar2DForwardLevel2, Haar2DInverseSparse
)

__all__ = ['swformer_sparse']

import math

class MLP(nn.Module):
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.c_hidden = hidden_features
        self.c_output = out_features
        

    def forward(self, x):
        T, B, C, H, W = x.shape
        
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1).contiguous())
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0,1).contiguous())
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        
        return x

class FATMSparse(nn.Module):

    def __init__(self, dim, FL_blocks=16, 
                 vth_low_l1=0.5, vth_high_l1=0.5,
                 vth_ll=0.5, vth_hl=0.5, vth_lh=0.5, vth_hh=0.5,

                 l2_sparsity_mode='channel',
                 l2_tau_ll=0.01, l2_tau_hl=0.02, l2_tau_lh=0.02, l2_tau_hh=0.05,

                 ener_gate_low=0.4, ener_gate_high=0.6, ener_sigma=0.2, ener_tau_E=1.0):
        super(FATMSparse, self).__init__()

        self.hidden_size = dim
        self.num_blocks = FL_blocks
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        
        self.haar_matrix_built = False
        self.l2_sparsity_mode = l2_sparsity_mode

        self.x_neuron = MultiStepLIFNode(tau=2.0, detach_reset=True)
        

        self.haar_forward_l1 = Haar2DForwardLevel1(
            channels=dim, 
            vth_low=vth_low_l1, 
            vth_high=vth_high_l1
        )
        
        self.haar_forward_l2 = Haar2DForwardLevel2(
            channels=dim,
            vth_ll=vth_ll, vth_hl=vth_hl, vth_lh=vth_lh, vth_hh=vth_hh,
            sparsity_mode=l2_sparsity_mode,
            tau_ll=l2_tau_ll, tau_hl=l2_tau_hl, 
            tau_lh=l2_tau_lh, tau_hh=l2_tau_hh,
            ener_gate_low=ener_gate_low, ener_gate_high=ener_gate_high,
            ener_sigma=ener_sigma, ener_tau_E=ener_tau_E
        )
        
        self.haar_inverse = Haar2DInverseSparse()
        
        self.haar_forward_bn = nn.BatchNorm2d(dim * 2)
        self.haar_multiply_bn = nn.BatchNorm2d(dim * 4)
        self.haar_inverse_bn = nn.BatchNorm2d(dim)
        
        self.scale = 0.02
        self.haar_weight = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size))
        
        self.conv1 = nn.Conv2d(self.block_size, self.block_size, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(self.block_size, self.block_size, kernel_size=3, padding=1)

        self.conv1_bn = nn.BatchNorm2d(dim)
        self.conv2_bn = nn.BatchNorm2d(dim)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)
    
    def get_stats(self):
        stats = {}
        
        if hasattr(self.haar_forward_l1, 'coeff_sparsity'):
            coeff_sparse = self.haar_forward_l1.coeff_sparsity
            stats['coeff_sparsity_low'] = coeff_sparse.sparsity_rate_low.item()
            stats['coeff_sparsity_high'] = coeff_sparse.sparsity_rate_high.item()
            stats['coeff_sparsity_avg'] = (coeff_sparse.sparsity_rate_low + 
                                            coeff_sparse.sparsity_rate_high).item() / 2
        
        if hasattr(self.haar_forward_l1, 'spk'):
            spk_l1 = self.haar_forward_l1.spk
            stats['spike_activity_l1_low'] = spk_l1.spike_activity_rates[0].item()
            stats['spike_activity_l1_high'] = spk_l1.spike_activity_rates[1].item()
            stats['spike_activity_l1_avg'] = spk_l1.spike_activity_rates.mean().item()
        
        stats['l2_sparsity_mode'] = self.l2_sparsity_mode
        
        if self.l2_sparsity_mode in ['channel', 'ener_ch']:

            if hasattr(self.haar_forward_l2, 'sparsity_module'):
                sparsity_module = self.haar_forward_l2.sparsity_module
                if hasattr(sparsity_module, 'channel_keep_ratios'):
                    channel_keep = sparsity_module.channel_keep_ratios
                    stats['channel_keep_ll'] = channel_keep[0].item()
                    stats['channel_keep_hl'] = channel_keep[1].item()
                    stats['channel_keep_lh'] = channel_keep[2].item()
                    stats['channel_keep_hh'] = channel_keep[3].item()
                    stats['channel_keep_avg'] = channel_keep.mean().item()
                    stats['channel_sparsity_avg'] = 1 - channel_keep.mean().item()
        else:

            if hasattr(self.haar_forward_l2, 'sparsity_module'):
                coeff_sparse = self.haar_forward_l2.sparsity_module
                if hasattr(coeff_sparse, 'sparsity_rate_ll'):

                    stats['coeff_sparsity_l2_ll'] = coeff_sparse.sparsity_rate_ll.item()
                    stats['coeff_sparsity_l2_hl'] = coeff_sparse.sparsity_rate_hl.item()
                    stats['coeff_sparsity_l2_lh'] = coeff_sparse.sparsity_rate_lh.item()
                    stats['coeff_sparsity_l2_hh'] = coeff_sparse.sparsity_rate_hh.item()

                    stats['coeff_sparsity_l2_avg'] = (
                        coeff_sparse.sparsity_rate_ll + 
                        coeff_sparse.sparsity_rate_hl + 
                        coeff_sparse.sparsity_rate_lh + 
                        coeff_sparse.sparsity_rate_hh
                    ).item() / 4
        
        if hasattr(self.haar_forward_l2, 'spk'):
            spk_l2 = self.haar_forward_l2.spk
            stats['spike_activity_l2_ll'] = spk_l2.spike_activity_rates[0].item()
            stats['spike_activity_l2_hl'] = spk_l2.spike_activity_rates[1].item()
            stats['spike_activity_l2_lh'] = spk_l2.spike_activity_rates[2].item()
            stats['spike_activity_l2_hh'] = spk_l2.spike_activity_rates[3].item()
            stats['spike_activity_l2_avg'] = spk_l2.spike_activity_rates.mean().item()
        
        if 'spike_activity_l1_avg' in stats and 'spike_activity_l2_avg' in stats:

            stats['spike_activity_total'] = (stats['spike_activity_l1_avg'] * 2 + 
                                              stats['spike_activity_l2_avg'] * 4) / 6
        
        return stats
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        
        if not self.haar_matrix_built:
            self.haar_forward_l1.build(H, x.device)
            self.haar_forward_l2.build(H, x.device)
            self.haar_inverse.build(H, x.device)
            self.haar_matrix_built = True

        x = self.x_neuron(x)

        haar_l1 = self.haar_forward_l1(x)
        haar_l1 = self.haar_forward_bn(haar_l1.flatten(0, 1).contiguous()).reshape(T, B, C*2, H, W//2).contiguous()
        
        haar_l2 = self.haar_forward_l2(haar_l1)
        haar_l2 = self.haar_multiply_bn(haar_l2.flatten(0, 1).contiguous()).reshape(T, B, C*4, H//2, W//2).contiguous()
        
        TB = T * B
        H_half, W_half = H // 2, W // 2
        
        haar = haar_l2.reshape(TB, self.num_blocks, self.block_size * 4, H_half, W_half)
        haar = haar.permute(0, 3, 4, 1, 2).contiguous()
        
        d = self.block_size
        LL = haar[:, :, :, :, :d]
        HL = haar[:, :, :, :, d:2*d]
        LH = haar[:, :, :, :, 2*d:3*d]
        HH = haar[:, :, :, :, 3*d:]
        
        LL_w = self.multiply(LL, self.haar_weight)
        HL_w = self.multiply(HL, self.haar_weight)
        LH_w = self.multiply(LH, self.haar_weight)
        HH_w = self.multiply(HH, self.haar_weight)
        
        haar = torch.cat([LL_w, HL_w, LH_w, HH_w], dim=-1)
        
        haar = haar.reshape(TB, H_half, W_half, C * 4).permute(0, 3, 1, 2).contiguous()
        haar = haar.reshape(T, B, C * 4, H_half, W_half)
        
        haar = self.haar_inverse(haar)
        haar = self.haar_inverse_bn(haar.flatten(0, 1).contiguous()).reshape(T, B, C, H, W).contiguous()

        x_reshaped = x.reshape(T*B*self.num_blocks, self.block_size, H, W).contiguous()
        
        conv_1 = self.conv1(x_reshaped)
        conv_1 = self.conv1_bn(conv_1.reshape(T*B, -1, H, W)).reshape(T, B, -1, H, W).contiguous()
        
        conv_2 = self.conv2(x_reshaped)
        conv_2 = self.conv2_bn(conv_2.reshape(T*B, -1, H, W)).reshape(T, B, -1, H, W).contiguous()
        
        out = haar + conv_1 + conv_2
        
        return out

class BlockSparse(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., FL_blocks=16,
                 vth_low_l1=0.5, vth_high_l1=0.5,
                 vth_ll=0.5, vth_hl=0.5, vth_lh=0.5, vth_hh=0.5,

                 l2_sparsity_mode='channel',
                 l2_tau_ll=0.01, l2_tau_hl=0.02, 
                 l2_tau_lh=0.02, l2_tau_hh=0.05,

                 ener_gate_low=0.4, ener_gate_high=0.6, 
                 ener_sigma=0.2, ener_tau_E=1.0):
        super().__init__()
        self.mixer = FATMSparse(
            dim, FL_blocks=FL_blocks,
            vth_low_l1=vth_low_l1, vth_high_l1=vth_high_l1,
            vth_ll=vth_ll, vth_hl=vth_hl, vth_lh=vth_lh, vth_hh=vth_hh,
            l2_sparsity_mode=l2_sparsity_mode,
            l2_tau_ll=l2_tau_ll, l2_tau_hl=l2_tau_hl,
            l2_tau_lh=l2_tau_lh, l2_tau_hh=l2_tau_hh,
            ener_gate_low=ener_gate_low, ener_gate_high=ener_gate_high,
            ener_sigma=ener_sigma, ener_tau_E=ener_tau_E
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.mixer(x)
        x = x + self.mlp(x)
        return x

class SPS(nn.Module):

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)


    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1).contiguous())
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x)

        x = self.proj_conv1(x.flatten(0, 1).contiguous())
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x)

        x = self.proj_conv2(x.flatten(0, 1).contiguous())
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x)
        x = self.maxpool2(x.flatten(0, 1).contiguous())

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.maxpool3(x)

        x_feat = x
        x = self.proj_lif3(x.reshape(T, B, -1, H//4, W//4)).contiguous()
        x = self.rpe_conv(x.flatten(0, 1).contiguous())
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H//4, W//4).contiguous()
        return x

class SWformerSparse(nn.Module):

    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=384, mlp_ratios=4, drop_rate=0., FL_blocks=16, 
                 depths=[6, 8, 6], T=4,

                 vth_low_l1=0.5, vth_high_l1=0.6,
                 vth_ll=0.5, vth_hl=0.6, vth_lh=0.6, vth_hh=0.7,

                 l2_sparsity_mode='channel',
                 l2_tau_ll=0.01, l2_tau_hl=0.02, 
                 l2_tau_lh=0.02, l2_tau_hh=0.05,

                 ener_gate_low=0.4, ener_gate_high=0.6,
                 ener_sigma=0.2, ener_tau_E=1.0
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        self.l2_sparsity_mode = l2_sparsity_mode

        patch_embed = SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims
        )

        self.FL_blocks = FL_blocks

        total_depths = depths if isinstance(depths, int) else sum(depths)
        block = nn.ModuleList([BlockSparse(
            dim=embed_dims, mlp_ratio=mlp_ratios,
            drop=drop_rate, FL_blocks=self.FL_blocks,
            vth_low_l1=vth_low_l1, vth_high_l1=vth_high_l1,
            vth_ll=vth_ll, vth_hl=vth_hl, vth_lh=vth_lh, vth_hh=vth_hh,
            l2_sparsity_mode=l2_sparsity_mode,
            l2_tau_ll=l2_tau_ll, l2_tau_hl=l2_tau_hl,
            l2_tau_lh=l2_tau_lh, l2_tau_hh=l2_tau_hh,
            ener_gate_low=ener_gate_low, ener_gate_high=ener_gate_high,
            ener_sigma=ener_sigma, ener_tau_E=ener_tau_E)
            for j in range(total_depths)
        ])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x = patch_embed(x)

        for blk in block:
            x = blk(x)
        
        return x.flatten(-2).mean(3)
    
    def get_model_stats(self):
        block = getattr(self, f"block")
        
        all_stats = []
        for i, blk in enumerate(block):
            if hasattr(blk.mixer, 'get_stats'):
                layer_stats = blk.mixer.get_stats()
                layer_stats['layer'] = i
                all_stats.append(layer_stats)
        
        if not all_stats:
            return {}
        
        result = {
            'num_layers': len(all_stats),
            'layer_stats': all_stats,
        }
        
        return result

    def forward(self, x):
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)
        x = self.head(x.mean(0))

        return x

@register_model
def swformer_sparse(pretrained=False, pretrained_cfg=None, **kwargs):
    model = SWformerSparse(**kwargs)
    model.default_cfg = _cfg()
    return model

if __name__ == '__main__':

    model = SWformerSparse(
        patch_size=4, embed_dims=256, mlp_ratios=4,
        in_channels=3, num_classes=10, depths=2,
        img_size_h=32, img_size_w=32,
        vth_low_l1=0.5, vth_high_l1=0.6,
        vth_ll=0.5, vth_hl=0.6, vth_lh=0.6, vth_hh=0.7
    ).cuda()

    input = torch.randn(4, 3, 32, 32).cuda()
    output = model(input)
    print(f"Output shape: {output.shape}")
    
    print("\nLearnable threshold parameters:")
    for name, param in model.named_parameters():
        if 'vth' in name or 'tau' in name or 'gates' in name:
            print(f"{name}: {param.data}")
