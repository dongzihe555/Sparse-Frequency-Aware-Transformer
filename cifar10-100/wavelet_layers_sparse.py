
import torch
from torch import nn
import numpy as np
from typing import Tuple

backend = 'torch'
import torch._dynamo
torch._dynamo.config.disable = True

def normalize_haar_matrix(H, device):
    norms = torch.linalg.norm(H, axis=1, keepdims=True)
    H = H / norms
    return H.to(device).to(torch.float32)

def haar_matrix(N, device):
    H = torch.tensor(haar_1d_matrix(N)).to(torch.float32)
    return normalize_haar_matrix(H, device)

def haar_1d_matrix(n):
    if np.log2(n) % 1 > 0:
        raise ValueError("n must be a power of 2")
    
    if n == 1:
        return np.array([[1]])
    
    else:
        H_next = haar_1d_matrix(n // 2)
        upper = np.kron(H_next, [1, 1])
        lower = np.kron(np.eye(len(H_next)), [1, -1])
        H = np.vstack((upper, lower))
        return H

class LearnableSoftThreshold(nn.Module):
    def __init__(self, channels, init_tau_low=0.01, init_tau_high=0.05):
        super().__init__()

        self.tau_low = nn.Parameter(torch.tensor(init_tau_low))

        self.tau_high = nn.Parameter(torch.tensor(init_tau_high))
        self.channels = channels

        self.register_buffer('sparsity_rate_low', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_high', torch.tensor(0.0))
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        d = self.channels
        
        x_low = x[:, :, :d, :, :]
        x_high = x[:, :, d:, :, :]
        
        x_low_sparse = torch.sign(x_low) * torch.relu(torch.abs(x_low) - self.tau_low)
        
        x_high_sparse = torch.sign(x_high) * torch.relu(torch.abs(x_high) - self.tau_high)
        
        with torch.no_grad():
            self.sparsity_rate_low = (x_low_sparse.abs() < 1e-6).float().mean()
            self.sparsity_rate_high = (x_high_sparse.abs() < 1e-6).float().mean()
        
        x_sparse = torch.cat([x_low_sparse, x_high_sparse], dim=2)
        
        return x_sparse

class SubbandAwareSpk(nn.Module):
    def __init__(self, vth_low=0.5, vth_high=0.5, num_bands=2):
        super().__init__()
        self.num_bands = num_bands
        
        if num_bands == 2:

            self.vth_low = nn.Parameter(torch.tensor(vth_low))
            self.vth_high = nn.Parameter(torch.tensor(vth_high))
        elif num_bands == 4:

            self.vth_ll = nn.Parameter(torch.tensor(vth_low))
            self.vth_hl = nn.Parameter(torch.tensor(vth_high * 0.8))
            self.vth_lh = nn.Parameter(torch.tensor(vth_high * 0.8))
            self.vth_hh = nn.Parameter(torch.tensor(vth_high))
        
        self.register_buffer('spike_activity_rates', torch.zeros(num_bands))
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        
        if self.num_bands == 2:
            d = C // 2

            x_low = x[:, :, :d, :, :]
            x_high = x[:, :, d:, :, :]
            
            spike_low = torch.where(x_low >= self.vth_low, torch.ones_like(x_low),
                                   torch.where(x_low <= -self.vth_low, -torch.ones_like(x_low), 
                                              torch.zeros_like(x_low)))
            
            spike_high = torch.where(x_high >= self.vth_high, torch.ones_like(x_high),
                                    torch.where(x_high <= -self.vth_high, -torch.ones_like(x_high),
                                               torch.zeros_like(x_high)))
            
            spike = torch.cat([spike_low, spike_high], dim=2)
            
            with torch.no_grad():
                activity_low = (spike_low != 0).float().mean()
                activity_high = (spike_high != 0).float().mean()
                self.spike_activity_rates = torch.stack([activity_low, activity_high])
            
        elif self.num_bands == 4:
            d = C // 4

            x_ll = x[:, :, :d, :, :]
            x_hl = x[:, :, d:2*d, :, :]
            x_lh = x[:, :, 2*d:3*d, :, :]
            x_hh = x[:, :, 3*d:, :, :]
            
            spike_ll = torch.where(x_ll >= self.vth_ll, torch.ones_like(x_ll),
                                  torch.where(x_ll <= -self.vth_ll, -torch.ones_like(x_ll),
                                             torch.zeros_like(x_ll)))
            
            spike_hl = torch.where(x_hl >= self.vth_hl, torch.ones_like(x_hl),
                                  torch.where(x_hl <= -self.vth_hl, -torch.ones_like(x_hl),
                                             torch.zeros_like(x_hl)))
            
            spike_lh = torch.where(x_lh >= self.vth_lh, torch.ones_like(x_lh),
                                  torch.where(x_lh <= -self.vth_lh, -torch.ones_like(x_lh),
                                             torch.zeros_like(x_lh)))
            
            spike_hh = torch.where(x_hh >= self.vth_hh, torch.ones_like(x_hh),
                                  torch.where(x_hh <= -self.vth_hh, -torch.ones_like(x_hh),
                                             torch.zeros_like(x_hh)))
            
            spike = torch.cat([spike_ll, spike_hl, spike_lh, spike_hh], dim=2)
            
            with torch.no_grad():
                activity_ll = (spike_ll != 0).float().mean()
                activity_hl = (spike_hl != 0).float().mean()
                activity_lh = (spike_lh != 0).float().mean()
                activity_hh = (spike_hh != 0).float().mean()
                self.spike_activity_rates = torch.stack([activity_ll, activity_hl, activity_lh, activity_hh])
        
        return spike

class ChannelSparsity(nn.Module):
    def __init__(self, channels_per_subband, num_subbands=4):
        super().__init__()
        self.channels_per_subband = channels_per_subband
        self.num_subbands = num_subbands
        
        self.gates = nn.ParameterList([
            nn.Parameter(torch.randn(channels_per_subband))
            for _ in range(num_subbands)
        ])

        self.register_buffer('channel_keep_ratios', torch.zeros(num_subbands))
        
    def forward(self, x, training=True):
        T, B, C, H, W = x.shape
        d = self.channels_per_subband
        
        outputs = []
        avg_gate_weights_list = []
        
        for i in range(self.num_subbands):
            subband = x[:, :, i*d:(i+1)*d, :, :]
            
            gate_weights = torch.sigmoid(self.gates[i])
            
            gate_weights_expanded = gate_weights.view(1, 1, d, 1, 1)
            
            subband_sparse = subband * gate_weights_expanded
            outputs.append(subband_sparse)
            
            with torch.no_grad():
                avg_gate_weight = gate_weights.mean()
                avg_gate_weights_list.append(avg_gate_weight)
        
        with torch.no_grad():
            self.channel_keep_ratios = torch.stack(avg_gate_weights_list)
        
        return torch.cat(outputs, dim=2)

class LearnableSoftThreshold4(nn.Module):
    def __init__(self, channels, 
                 init_tau_ll=0.01, 
                 init_tau_hl=0.02, 
                 init_tau_lh=0.02, 
                 init_tau_hh=0.05):
        super().__init__()
        self.channels = channels
        
        self.tau_ll = nn.Parameter(torch.tensor(init_tau_ll))
        self.tau_hl = nn.Parameter(torch.tensor(init_tau_hl))
        self.tau_lh = nn.Parameter(torch.tensor(init_tau_lh))
        self.tau_hh = nn.Parameter(torch.tensor(init_tau_hh))
        
        self.register_buffer('sparsity_rate_ll', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_hl', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_lh', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_hh', torch.tensor(0.0))
        
    def forward(self, x):
        T, B, C4, H, W = x.shape
        C = self.channels
        
        x_ll = x[:, :, :C, :, :]
        x_hl = x[:, :, C:2*C, :, :]
        x_lh = x[:, :, 2*C:3*C, :, :]
        x_hh = x[:, :, 3*C:4*C, :, :]
        
        x_ll_sparse = torch.sign(x_ll) * torch.relu(torch.abs(x_ll) - self.tau_ll)
        x_hl_sparse = torch.sign(x_hl) * torch.relu(torch.abs(x_hl) - self.tau_hl)
        x_lh_sparse = torch.sign(x_lh) * torch.relu(torch.abs(x_lh) - self.tau_lh)
        x_hh_sparse = torch.sign(x_hh) * torch.relu(torch.abs(x_hh) - self.tau_hh)
        
        with torch.no_grad():
            self.sparsity_rate_ll = (x_ll_sparse.abs() < 1e-6).float().mean()
            self.sparsity_rate_hl = (x_hl_sparse.abs() < 1e-6).float().mean()
            self.sparsity_rate_lh = (x_lh_sparse.abs() < 1e-6).float().mean()
            self.sparsity_rate_hh = (x_hh_sparse.abs() < 1e-6).float().mean()
        
        x_sparse = torch.cat([x_ll_sparse, x_hl_sparse, x_lh_sparse, x_hh_sparse], dim=2)
        
        return x_sparse

class LearnableChSparsity(nn.Module):
    def __init__(self, channels_per_subband, num_subbands=4,
                 init_tau_ll=0.01, 
                 init_tau_hl=0.02, 
                 init_tau_lh=0.02, 
                 init_tau_hh=0.05):
        super().__init__()
        self.channels_per_subband = channels_per_subband
        self.num_subbands = num_subbands
        
        tau_ll = torch.full((channels_per_subband,), init_tau_ll)

        tau_hl = torch.full((channels_per_subband,), init_tau_hl)

        tau_lh = torch.full((channels_per_subband,), init_tau_lh)

        tau_hh = torch.full((channels_per_subband,), init_tau_hh)
        
        all_taus = torch.cat([tau_ll, tau_hl, tau_lh, tau_hh])
        self.thresholds = nn.Parameter(all_taus)
        
        self.register_buffer('sparsity_rate_ll', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_hl', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_lh', torch.tensor(0.0))
        self.register_buffer('sparsity_rate_hh', torch.tensor(0.0))
        
    def forward(self, x):
        T, B, C4, H, W = x.shape
        C = self.channels_per_subband
        
        thresholds = self.thresholds.view(1, 1, C4, 1, 1)
        
        x_sparse = torch.sign(x) * torch.relu(torch.abs(x) - thresholds)
        
        with torch.no_grad():

            x_ll = x_sparse[:, :, :C, :, :]
            x_hl = x_sparse[:, :, C:2*C, :, :]
            x_lh = x_sparse[:, :, 2*C:3*C, :, :]
            x_hh = x_sparse[:, :, 3*C:4*C, :, :]
            
            self.sparsity_rate_ll = (x_ll.abs() < 1e-6).float().mean()
            self.sparsity_rate_hl = (x_hl.abs() < 1e-6).float().mean()
            self.sparsity_rate_lh = (x_lh.abs() < 1e-6).float().mean()
            self.sparsity_rate_hh = (x_hh.abs() < 1e-6).float().mean()
        
        return x_sparse

class EnerChSparsity(nn.Module):
    
    def __init__(self, channels_per_subband, num_subbands=4,
                 gate_low=0.4, gate_high=0.6, sigma=0.2, tau_E=1.0):
        super().__init__()
        self.channels_per_subband = channels_per_subband
        self.num_subbands = num_subbands
        
        assert 0 < gate_low < gate_high < 1, "gate thresholds must be in (0,1) and gate_low < gate_high"
        self.gate_low = gate_low
        self.gate_high = gate_high
        self.sigma = sigma
        self.tau_E = tau_E
        
        self.register_buffer('mu_low', torch.log(torch.tensor(gate_low / (1 - gate_low))))
        self.register_buffer('mu_high', torch.log(torch.tensor(gate_high / (1 - gate_high))))
        
        self.register_buffer('E_center', torch.tensor(0.0))
        self.EMA_decay = 0.9
        
        self.register_buffer('channel_keep_ratios', torch.zeros(num_subbands))
        
    def compute_channel_energy(self, x_sub):

        energy = x_sub.pow(2).mean(dim=[0, 1, 3, 4])
        return energy
    
    def sample_from_distribution(self, batch_size, device):

        if self.training:
            z_low = torch.randn(batch_size, device=device) * self.sigma + self.mu_low
            z_high = torch.randn(batch_size, device=device) * self.sigma + self.mu_high
        else:
            z_low = self.mu_low.expand(batch_size).clone()
            z_high = self.mu_high.expand(batch_size).clone()
        
        return z_low, z_high
    
    def forward(self, x):
        T, B, C4, H, W = x.shape
        C = self.channels_per_subband
        device = x.device
        
        outputs = []
        avg_gate_weights_list = []
        
        for i in range(self.num_subbands):

            subband = x[:, :, i*C:(i+1)*C, :, :]
            
            energy = self.compute_channel_energy(subband)
            
            with torch.no_grad():
                E_mean = energy.mean()
                if self.E_center.item() == 0:
                    self.E_center = E_mean
                else:
                    self.E_center = self.EMA_decay * self.E_center + (1 - self.EMA_decay) * E_mean
            
            alpha = torch.sigmoid((energy - self.E_center) / self.tau_E)
            
            z_low, z_high = self.sample_from_distribution(C, device)
            
            z = alpha * z_high + (1 - alpha) * z_low
            
            gate_weights = torch.sigmoid(z)
            
            gate_weights_expanded = gate_weights.view(1, 1, C, 1, 1)
            subband_sparse = subband * gate_weights_expanded
            outputs.append(subband_sparse)
            
            with torch.no_grad():
                avg_gate_weight = gate_weights.mean()
                avg_gate_weights_list.append(avg_gate_weight)
        
        with torch.no_grad():
            self.channel_keep_ratios = torch.stack(avg_gate_weights_list)
        
        return torch.cat(outputs, dim=2)

class Haar2DForwardLevel1(nn.Module):
    def __init__(self, channels, vth_low=0.5, vth_high=0.5, use_coeff_sparsity=True):
        super().__init__()
        self.channels = channels
        self.haar_matrix_built = False
        self.use_coeff_sparsity = use_coeff_sparsity
        
        if use_coeff_sparsity:
            self.coeff_sparsity = LearnableSoftThreshold(channels, init_tau_low=0.01, init_tau_high=0.05)
        else:
            self.coeff_sparsity = None
        
        self.spk = SubbandAwareSpk(vth_low=vth_low, vth_high=vth_high, num_bands=2)
        
        
    def build(self, N, device):
        self.register_buffer('H', haar_matrix(N, device))
        self.haar_matrix_built = True
        
    def forward(self, x):
        T, B, C, H, W = x.shape
        
        if not self.haar_matrix_built:
            self.build(W, x.device)
        
        x_flat = x.reshape(T*B*C, H, W)
        
        u = torch.matmul(x_flat, self.H.T)
        
        L = u[:, :, :W//2]
        H_coeff = u[:, :, W//2:]
        
        L = L.reshape(T, B, C, H, W//2)
        H_coeff = H_coeff.reshape(T, B, C, H, W//2)
        S1 = torch.cat([L, H_coeff], dim=2)
        
        if self.use_coeff_sparsity and self.coeff_sparsity is not None:
            S1_sparse = self.coeff_sparsity(S1)
        else:
            S1_sparse = S1
        
        S1_spk = self.spk(S1_sparse)
        
        
        return S1_spk

class Haar2DForwardLevel2(nn.Module):
    def __init__(self, channels, vth_ll=0.5, vth_hl=0.5, vth_lh=0.5, vth_hh=0.5,
                 sparsity_mode='channel',

                 tau_ll=0.01, tau_hl=0.02, tau_lh=0.02, tau_hh=0.05,

                 ener_gate_low=0.4, ener_gate_high=0.6, ener_sigma=0.2, ener_tau_E=1.0,
                 use_sparsity=True):
        super().__init__()
        self.channels = channels
        self.haar_matrix_built = False
        self.sparsity_mode = sparsity_mode
        self.use_sparsity = use_sparsity
        
        if use_sparsity:
            if sparsity_mode == 'channel':

                self.sparsity_module = ChannelSparsity(channels, num_subbands=4)
            elif sparsity_mode == 'coeff4':

                self.sparsity_module = LearnableSoftThreshold4(
                    channels, 
                    init_tau_ll=tau_ll,
                    init_tau_hl=tau_hl,
                    init_tau_lh=tau_lh,
                    init_tau_hh=tau_hh
                )
            elif sparsity_mode == 'ch_individual':

                self.sparsity_module = LearnableChSparsity(
                    channels,
                    num_subbands=4,
                    init_tau_ll=tau_ll,
                    init_tau_hl=tau_hl,
                    init_tau_lh=tau_lh,
                    init_tau_hh=tau_hh
                )
            elif sparsity_mode == 'ener_ch':

                self.sparsity_module = EnerChSparsity(
                    channels,
                    num_subbands=4,
                    gate_low=ener_gate_low,
                    gate_high=ener_gate_high,
                    sigma=ener_sigma,
                    tau_E=ener_tau_E
                )
            else:
                raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}. "
                               f"Must be one of ['channel', 'coeff4', 'ch_individual', 'ener_ch']")
        else:
            self.sparsity_module = None
        
        self.spk = SubbandAwareSpk(vth_low=vth_ll, vth_high=vth_hh, num_bands=4)

        self.spk.vth_ll = nn.Parameter(torch.tensor(vth_ll))
        self.spk.vth_hl = nn.Parameter(torch.tensor(vth_hl))
        self.spk.vth_lh = nn.Parameter(torch.tensor(vth_lh))
        self.spk.vth_hh = nn.Parameter(torch.tensor(vth_hh))
        
        
    def build(self, N, device):
        self.register_buffer('H', haar_matrix(N, device))
        self.haar_matrix_built = True
        
    def forward(self, x):
        T, B, C2, H, W_half = x.shape
        C = self.channels
        
        if not self.haar_matrix_built:
            self.build(H, x.device)
        
        x_flat = x.reshape(T*B*C2, H, W_half)
        
        u = torch.matmul(self.H, x_flat)
        
        u_low = u[:, :H//2, :]
        u_high = u[:, H//2:, :]
        
        u_low = u_low.reshape(T, B, C2, H//2, W_half)
        u_high = u_high.reshape(T, B, C2, H//2, W_half)
        
        LL = u_low[:, :, :C, :, :]

        HL = u_low[:, :, C:, :, :]

        LH = u_high[:, :, :C, :, :]

        HH = u_high[:, :, C:, :, :]
        
        S2 = torch.cat([LL, HL, LH, HH], dim=2)
        
        if self.use_sparsity and self.sparsity_module is not None:
            if self.sparsity_mode == 'channel':

                S2_sparse = self.sparsity_module(S2, training=self.training)
            else:

                S2_sparse = self.sparsity_module(S2)
        else:
            S2_sparse = S2
        
        S2_spk = self.spk(S2_sparse)
        
        
        return S2_spk

class Haar2DInverseSparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.haar_matrix_built = False
        
    def build(self, N, device):
        self.register_buffer('H', haar_matrix(N, device))
        self.haar_matrix_built = True
        
    def forward(self, x):
        T, B, C4, H_half, W_half = x.shape
        C = C4 // 4
        H, W = H_half * 2, W_half * 2
        
        if not self.haar_matrix_built:
            self.build(H, x.device)

        LL = x[:, :, :C, :, :]
        HL = x[:, :, C:2*C, :, :]
        LH = x[:, :, 2*C:3*C, :, :]
        HH = x[:, :, 3*C:, :, :]
        
        u_low = torch.cat([LL, HL], dim=2)
        u_high = torch.cat([LH, HH], dim=2)
        
        u_low_flat = u_low.reshape(T*B*2*C, H_half, W_half)
        u_high_flat = u_high.reshape(T*B*2*C, H_half, W_half)
        
        u_vstack = torch.cat([u_low_flat, u_high_flat], dim=1)
        
        u_inv_vert = torch.matmul(self.H.T, u_vstack)
        u_inv_vert = u_inv_vert.reshape(T, B, 2*C, H, W_half)
        
        L = u_inv_vert[:, :, :C, :, :]
        H_coeff = u_inv_vert[:, :, C:, :, :]
        
        h_stack = torch.cat([L, H_coeff], dim=4)
        h_stack_flat = h_stack.reshape(T*B*C, H, W)
        
        x_inv = torch.matmul(h_stack_flat, self.H)
        x_inv = x_inv.reshape(T, B, C, H, W)
        
        return x_inv
