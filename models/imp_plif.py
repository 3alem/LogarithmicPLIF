import torch.nn as nn
import torch
import math
from spikingjelly.activation_based import neuron, layer, surrogate

class ParametricLIFNodeImprovedTau(neuron.BaseNode):
    def __init__(self, init_tau=2.0, tau_min=1e-2, tau_max=1e2):
        super().__init__(v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan())
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.log_tau_raw = nn.Parameter(torch.tensor(math.log(init_tau)))
        
    @property
    def tau(self):
        return torch.exp(self.log_tau_raw).clamp(self.tau_min, self.tau_max)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

class SNN_ImprovedParametric(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            
            layer.Linear(2 * 34 * 34, 64, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(64, 32, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(32, 16, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(16, 10), 
            
            ParametricLIFNodeImprovedTau(init_tau=2.0)  # Custom implementation
        )

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        return self.layer(x).view(batch_size, seq_len, -1).mean(dim=1)
