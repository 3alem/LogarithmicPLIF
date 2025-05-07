import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate



class SNN_Parametric(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            # Input layer
            layer.Flatten(),
            
            layer.Linear(2 * 34 * 34, 64, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(64, 32, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(32, 16, bias=False),  
            nn.Dropout(0.3),
            layer.Linear(16, 10), 

            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())

        )

    def forward(self, x):
        batch_size, seq_len, _, _, _ = x.shape
        x = x.view(batch_size * seq_len, -1)
        return self.layer(x).view(batch_size, seq_len, -1).mean(dim=1)

