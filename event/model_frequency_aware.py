
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from typing import Optional, Dict

from model_sparse import SPS, MLP

from frequency_modules import FrequencyCompensatoryMLP

__all__ = ['swformer_freq_aware']

class BlockFrqAware(nn.Module):
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 drop: float = 0.0, FL_blocks: int = 16,
                 vth_low_l1: float = 0.5, vth_high_l1: float = 0.6,
                 vth_ll: float = 0.5, vth_hl: float = 0.6,
                 vth_lh: float = 0.6, vth_hh: float = 0.7,

                 use_freq_compensatory_mlp: bool = True,
                 skip_base_mlp: bool = False,

                 freq_experts_ratio: float = 0.5,

                 l2_sparsity_mode: str = 'channel',
                 l2_tau_ll: float = 0.01, l2_tau_hl: float = 0.02,
                 l2_tau_lh: float = 0.02, l2_tau_hh: float = 0.05,

                 ener_gate_low: float = 0.4, ener_gate_high: float = 0.6,
                 ener_sigma: float = 0.2, ener_tau_E: float = 1.0):
        super().__init__()
        
        self.use_freq_compensatory_mlp = use_freq_compensatory_mlp
        
        from model_sparse import FATMSparse
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
        
        hidden_dim = int(dim * mlp_ratio)
        if use_freq_compensatory_mlp:
            FreqCompMLP = FrequencyCompensatoryMLP
            
            self.mlp = FreqCompMLP(
                dim=dim, hidden_dim=hidden_dim,
                use_freq_compensation=True,
                freq_bands=4,
                freq_experts_ratio=freq_experts_ratio,
                skip_base_mlp=skip_base_mlp
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(x)
        
        if self.use_freq_compensatory_mlp:
            x = x + self.mlp(x)
        else:
            x = x + self.mlp(x)
        
        return x
    
    def get_stats(self) -> Dict:
        stats = {}
        
        if hasattr(self.mixer, 'get_stats'):
            mixer_stats = self.mixer.get_stats()
            stats.update(mixer_stats)
        
        if self.use_freq_compensatory_mlp and hasattr(self.mlp, 'get_stats'):
            mlp_stats = self.mlp.get_stats()
            if mlp_stats:
                stats['mlp'] = mlp_stats
        
        return stats

class SWformerFrequencyAware(nn.Module):
    
    def __init__(self, 
                 img_size_h: int = 128, img_size_w: int = 128,
                 patch_size: int = 16, in_channels: int = 2,
                 num_classes: int = 11,
                 embed_dims: int = 384, mlp_ratios: float = 4.0,
                 drop_rate: float = 0.0, FL_blocks: int = 16,
                 depths: int = 4, T: int = 4,

                 vth_low_l1: float = 0.5, vth_high_l1: float = 0.6,
                 vth_ll: float = 0.5, vth_hl: float = 0.6,
                 vth_lh: float = 0.6, vth_hh: float = 0.7,

                 use_freq_compensatory_mlp: bool = True,
                 skip_base_mlp: bool = False,

                 freq_experts_ratio: float = 0.5,

                 l2_sparsity_mode: str = 'channel',
                 l2_tau_ll: float = 0.01, l2_tau_hl: float = 0.02,
                 l2_tau_lh: float = 0.02, l2_tau_hh: float = 0.05,

                 ener_gate_low: float = 0.4, ener_gate_high: float = 0.6,
                 ener_sigma: float = 0.2, ener_tau_E: float = 1.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        
        self.config = {
            'use_freq_compensatory_mlp': use_freq_compensatory_mlp,
            'skip_base_mlp': skip_base_mlp,
            'l2_sparsity_mode': l2_sparsity_mode,
            'ener_gate_low': ener_gate_low,
            'ener_gate_high': ener_gate_high,
        }
        
        self.patch_embed = SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims
        )
        
        self.freq_monitor = None
        
        self.blocks = nn.ModuleList([
            BlockFrqAware(
                dim=embed_dims,
                mlp_ratio=mlp_ratios,
                drop=drop_rate,
                FL_blocks=FL_blocks,
                vth_low_l1=vth_low_l1, vth_high_l1=vth_high_l1,
                vth_ll=vth_ll, vth_hl=vth_hl, vth_lh=vth_lh, vth_hh=vth_hh,
                use_freq_compensatory_mlp=use_freq_compensatory_mlp,
                skip_base_mlp=skip_base_mlp,
                freq_experts_ratio=freq_experts_ratio,
                l2_sparsity_mode=l2_sparsity_mode,
                l2_tau_ll=l2_tau_ll, l2_tau_hl=l2_tau_hl,
                l2_tau_lh=l2_tau_lh, l2_tau_hh=l2_tau_hh,
                ener_gate_low=ener_gate_low, ener_gate_high=ener_gate_high,
                ener_sigma=ener_sigma, ener_tau_E=ener_tau_E
            ) for _ in range(depths)
        ])
        
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        
        self.FL_blocks = FL_blocks
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        
        for layer_idx, block in enumerate(self.blocks):
            x = block(x)
        
        return x.flatten(-2).mean(-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()
        
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        
        return x
    
    def get_model_stats(self) -> Dict:
        stats = {
            'config': self.config,
            'num_layers': len(self.blocks),
        }
        
        layer_stats = []
        for i, block in enumerate(self.blocks):
            blk_stats = block.get_stats()
            blk_stats['layer'] = i
            layer_stats.append(blk_stats)
        
        stats['layer_stats'] = layer_stats
        
        return stats
    
@register_model
def swformer_freq_aware(pretrained: bool = False, pretrained_cfg: Optional[Dict] = None, **kwargs):
    model = SWformerFrequencyAware(**kwargs)
    model.default_cfg = _cfg()
    return model
