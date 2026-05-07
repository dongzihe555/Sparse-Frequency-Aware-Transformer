"""
Source: https://github.com/bic-L/Spiking-Wavelet-Transformer
This file is from the Spiking Wavelet Transformer repository and serves as the
baseline Haar wavelet transform implementation for the non-sparse SWformer model.
"""


import torch
from torch import nn
import numpy as np
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
class Haar1DForward(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.haar_neuron = neuron_type(v_threshold = vth)
    def build(self, N):
        self.H = haar_matrix(N)
    @torch.compile
    def haar_1d(self, x):
        return torch.matmul(self.H, x)
    @torch.compile
    def forward(self, x):
        haar = self.haar_1d(x)
        return self.haar_neuron(haar)
    
class Haar1DInverse(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()

        self.haar_inv_neu = neuron_type(v_threshold = vth)
        
    def build(self, N):
        self.H = haar_matrix(N)  
        
    def haar_1d_inverse(self, x):
        return torch.matmul(self.H.T, x )
    @torch.compile
    def forward(self, x):
        haar_inverse = self.haar_1d_inverse(x)
        return self.haar_inv_neu(haar_inverse)
class Haar2DForward(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()

        self.col_haar_neuron = neuron_type(v_threshold=vth, backend=backend)
        
        self.register_buffer('spike_activity_rate', torch.tensor(0.0))
    
    def build(self, N, device):
        self.H = haar_matrix(N, device)
    
    @torch.compile
    def forward(self, x):

        x = torch.matmul(x, self.H.T)
        
        x = self.col_haar_neuron(x)
        
        with torch.no_grad():
            self.spike_activity_rate = (x != 0).float().mean()
        
        x = torch.matmul(self.H, x)
        return x
class Haar2DInverse(nn.Module):
    def __init__(self, neuron_type, vth=1.0):
        super().__init__()
        self.row_haar_neuron = neuron_type(v_threshold=vth)
        self.col_haar_neuron = neuron_type(v_threshold=vth)
        
        self.register_buffer('spike_activity_rate_row', torch.tensor(0.0))
        self.register_buffer('spike_activity_rate_col', torch.tensor(0.0))
    
    def build(self, N, device):
        self.H = haar_matrix(N, device)
    
    @torch.compile
    def forward(self, x):

        x = self.row_haar_neuron(x)
        
        with torch.no_grad():
            self.spike_activity_rate_row = (x != 0).float().mean()
        
        x = torch.matmul(x, self.H)

        x = self.col_haar_neuron(x)
        
        with torch.no_grad():
            self.spike_activity_rate_col = (x != 0).float().mean()
        
        x = torch.matmul(self.H.T, x)
        return x
