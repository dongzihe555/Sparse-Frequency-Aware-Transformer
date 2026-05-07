
from abc import abstractmethod
from typing import Callable, overload
import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, base, lava_exchange
from spikingjelly import configure
import math
import numpy as np
import logging
try:
    import cupy
    from spikingjelly.clock_driven import neuron_kernel, cu_kernel_opt
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cu_kernel_opt = None

try:
    import lava.lib.dl.slayer as slayer

except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.neuron: {e}')
    slayer = None

def check_backend(backend: str):
    if backend == 'torch':
        return
    elif backend == 'cupy':
        assert cupy is not None, 'CuPy is not installed! You can install it from "https://github.com/cupy/cupy".'
    elif backend == 'lava':
        assert slayer is not None, 'Lava-DL is not installed! You can install it from "https://github.com/lava-nc/lava-dl".'
    else:
        raise NotImplementedError(backend)

class NegBaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(neg_v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('neg_v_threshold', neg_v_threshold)
        self.register_memory('v_reset', v_reset)

        self.timestep = torch.tensor(0.).cuda()
        self.firing_rate = torch.tensor(0.).cuda()

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.register_memory('inhibit', 1e-3)

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):

        pos_spike = self.surrogate_function(self.v - self.v_threshold)

        neg_spike = self.surrogate_function(-(self.v + self.neg_v_threshold  + self.inhibit))

        return pos_spike - neg_spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        
        pos_spike = spike_d > 0
        neg_spike = spike_d < 0

        spike_d_vth = pos_spike * self.v_threshold + neg_spike * self.neg_v_threshold

        if self.v_reset is None:

            self.v = self.v - spike_d_vth

        else:

            self.v = (1. - spike_d.abs()) * self.v + spike_d * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, neg_v_threshold={self.neg_v_thresholdv_threshold},v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        return spike

class NegIFNode(NegBaseNode):
    def __init__(self, v_threshold: float = 1., neg_v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, cupy_fp32_inference=False):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

        if cupy_fp32_inference:
            check_backend('cupy')
        self.cupy_fp32_inference = cupy_fp32_inference
        self.momentum = 0.1

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def forward(self, x: torch.Tensor):
        if self.cupy_fp32_inference and cupy is not None and not self.training and x.dtype == torch.float32:

            device_id = x.get_device()
            if device_id < 0:
                return super().forward(x)

            if isinstance(self.v, float):
                v = torch.zeros_like(x)
                if self.v != 0.:
                    torch.fill_(v, self.v)
                self.v = v

            if self.v_reset is None:
                hard_reset = False
            else:
                hard_reset = True

            code = rf'''
                extern "C" __global__
                void IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward(
                const float * x, const float & v_threshold, const float & neg_v_threshold, {'const float & v_reset,' if hard_reset else ''}
                float * spike, float * v,
                const int & numel)
            '''

            code += r'''
                {
                    const int index = blockIdx.x * blockDim.x + threadIdx.x;
                    if (index < numel)
                    {
                        v[index] += x[index];

                        float pos_spike = v[index] >= v_threshold;
                        float neg_spike = v[index] <= -( neg_v_threshold  + 1E-3);

                        spike[index] = pos_spike - neg_spike;

                        float spike_d_vth = pos_spike * v_threshold + neg_spike * neg_v_threshold;

            '''

            code += rf'''
                        {' v[index] = (1.0f - abs(spike_d[index])) * v[index] + spike_d[index] * v_reset;;' if hard_reset else 'v[index] -= spike_d_vth;'}
            '''

            code += r'''
                    }
                }
            '''
            if hasattr(self, 'cp_kernel'):
                if self.cp_kernel.code != code:

                    del self.cp_kernel
                    self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)
            else:
                self.cp_kernel = cupy.RawKernel(code, f"IFNode_{'hard' if hard_reset else 'soft'}_reset_inference_forward", options=configure.cuda_compiler_options, backend=configure.cuda_compiler_backend)

            with cu_kernel_opt.DeviceEnvironment(device_id):
                numel = x.numel()
                threads = configure.cuda_threads
                blocks = cu_kernel_opt.cal_blocks(numel)
                cp_numel = cupy.asarray(numel)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                cp_v_threshold = cupy.asarray(self.v_threshold, dtype=np.float32)
                if hard_reset:
                    cp_v_reset = cupy.asarray(self.v_reset, dtype=np.float32)

                spike = torch.zeros_like(x)
                if hard_reset:
                    x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, cp_v_reset, spike, self.v, cp_numel]
                else:
                    x, cp_v_threshold, spike, self.v, cp_numel = cu_kernel_opt.get_contiguous(x, cp_v_threshold, spike, self.v, cp_numel)
                    kernel_args = [x, cp_v_threshold, spike, self.v, cp_numel]
                self.cp_kernel(
                    (blocks,), (threads,),
                    cu_kernel_opt.wrap_args_to_raw_kernel(
                        device_id,
                        *kernel_args
                    )
                )
                return spike
        else:
            return super().forward(x)

class MultiStepNegIFNode(NegIFNode):
    def __init__(self, v_threshold: float = 0.5, neg_v_threshold: float = 0.5, v_reset: float = 0., momentum: float = 0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, backend='torch', lava_s_cale=1 << 6):
        super().__init__(v_threshold, neg_v_threshold, v_reset, surrogate_function, detach_reset)

        self.momentum = momentum
        
        self.register_memory('v_seq', None)

        check_backend(backend)

        self.backend = backend

        self.lava_s_cale = lava_s_cale

        if backend == 'lava':
            self.lava_neuron = self.to_lava()
        else:
            self.lava_neuron = None

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1

        if self.backend == 'torch':
            spike_seq = []
            self.v_seq = []
            for t in range(x_seq.shape[0]):
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
                self.v_seq.append(self.v.unsqueeze(0))

            spike_seq = torch.cat(spike_seq, 0)
            self.v_seq = torch.cat(self.v_seq, 0)
            
            return spike_seq

        elif self.backend == 'cupy':
            if isinstance(self.v, float):
                v_init = self.v
                self.v = torch.zeros_like(x_seq[0].data)
                if v_init != 0.:
                    torch.fill_(self.v, v_init)

            spike_seq, self.v_seq = neuron_kernel.MultiStepNegIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.neg_v_threshold, self.v_reset, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            self.v_seq = self.v_seq.reshape(x_seq.shape)

            self.v = self.v_seq[-1].clone()

            return spike_seq

        elif self.backend == 'lava':
            if self.lava_neuron is None:
                self.lava_neuron = self.to_lava()

            spike, self.v = lava_exchange.lava_neuron_forward(self.lava_neuron, x_seq, self.v)
  
            return spike

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def to_lava(self):
        return lava_exchange.to_lava_neuron(self)

    def reset(self):
        super().reset()
        if self.lava_neuron is not None:
            self.lava_neuron.current_state.zero_()
            self.lava_neuron.voltage_state.zero_()
