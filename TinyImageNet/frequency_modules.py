
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from typing import Optional, List, Dict, Tuple
import math

class FrequencyCompensatoryMLP(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, 
                 use_freq_compensation: bool = True,
                 freq_bands: int = 4,
                 freq_experts_ratio: float = 0.25,
                 compensate_strength: float = 0.5,
                 dropout: float = 0.0,
                 skip_base_mlp: bool = False,
                 use_compensate_gate: bool = True):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.use_freq_compensation = use_freq_compensation
        self.freq_bands = freq_bands
        self.skip_base_mlp = skip_base_mlp
        self.use_compensate_gate = use_compensate_gate
        
        if not skip_base_mlp:
            self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True)
            
            self.fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(dim)
            self.lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True)
        else:

            self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)
            self.output_bn = nn.BatchNorm2d(dim)
        
        if use_freq_compensation:

            assert dim % freq_bands == 0, f"dim={dim} must be divisible by freq_bands={freq_bands}"
            self.band_dim = dim // freq_bands
            self.band_hidden = int(hidden_dim * freq_experts_ratio / freq_bands)
            
            self.freq_experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.band_dim, self.band_hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.band_hidden),
                ) for _ in range(freq_bands)
            ])
            
            if use_compensate_gate:
                self.compensate_gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dim, freq_bands, kernel_size=1),
                    nn.BatchNorm2d(freq_bands),
                )
            else:
                self.compensate_gate = None
            
            self.compensate_scale = nn.Parameter(torch.tensor(compensate_strength))
            
            if skip_base_mlp:

                self.band_fusion = nn.Sequential(
                    nn.Conv2d(self.band_hidden * freq_bands, dim, kernel_size=1),
                    nn.BatchNorm2d(dim),
                )
            else:

                self.band_fusion = nn.Sequential(
                    nn.Conv2d(self.band_hidden * freq_bands, hidden_dim, kernel_size=1),
                    nn.BatchNorm2d(hidden_dim),
                )
            
            self.fcp_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.2, detach_reset=True)
            
            self.register_buffer('avg_compensate_weights', torch.zeros(freq_bands))
            self.register_buffer('compensation_activity', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        
        if self.skip_base_mlp:

            if self.use_freq_compensation:
                x_out = self.frequency_compensation_path(x)

                x_out = self.output_proj(x_out.flatten(0, 1).contiguous())
                x_out = self.output_bn(x_out).reshape(T, B, C, H, W).contiguous()
                return x_out
            else:

                return x
        else:

            x_base = self.lif1(x)
            x_base = self.fc1(x_base.flatten(0, 1).contiguous())
            x_base = self.bn1(x_base).reshape(T, B, self.hidden_dim, H, W).contiguous()
            
            if self.use_freq_compensation:
                x_comp = self.frequency_compensation_path(x)

                x_base = x_base + x_comp * torch.sigmoid(self.compensate_scale)
            
            x_out = self.lif2(x_base)
            x_out = self.fc2(x_out.flatten(0, 1).contiguous())
            x_out = self.bn2(x_out).reshape(T, B, C, H, W).contiguous()
            
            return x_out
    
    def frequency_compensation_path(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, H, W = x.shape
        TB = T * B
        
        x_flat = x.reshape(TB, C, H, W)
        if self.use_compensate_gate and self.compensate_gate is not None:
            gate_logits = self.compensate_gate(x_flat)
            compensate_weights = torch.sigmoid(gate_logits).squeeze(-1).squeeze(-1)
            
            compensate_weights = torch.clamp(compensate_weights, 0, 1)
        else:
            compensate_weights = torch.ones(TB, self.freq_bands, device=x.device)
        
        with torch.no_grad():
            self.avg_compensate_weights = compensate_weights.mean(dim=0)
            self.compensation_activity = (compensate_weights > 0.5).float().mean()
        
        band_outputs = []
        for i in range(self.freq_bands):

            band = x[:, :, i*self.band_dim:(i+1)*self.band_dim, :, :]
            band_flat = band.reshape(TB, self.band_dim, H, W)
            
            band_processed = self.freq_experts[i](band_flat)
            
            weight = compensate_weights[:, i].view(TB, 1, 1, 1)
            band_outputs.append(band_processed * weight)
        
        x_comp = torch.cat(band_outputs, dim=1)
        x_comp = self.band_fusion(x_comp)
        
        x_comp = x_comp.reshape(T, B, -1, H, W).contiguous()
        x_comp = self.fcp_lif(x_comp)
        
        return x_comp
    
    def get_stats(self) -> Dict[str, float]:
        stats = {}
        if self.use_freq_compensation:
            stats['compensate_weights'] = self.avg_compensate_weights.cpu().tolist()
            stats['compensation_activity'] = self.compensation_activity.item()
            stats['compensate_scale'] = torch.sigmoid(self.compensate_scale).item()
        return stats
