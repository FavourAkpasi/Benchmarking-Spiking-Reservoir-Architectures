import torch
import torch.nn as nn
import snntorch as snn
from src.utils import get_legendre_matrices


# Model B: Random Spiking Reservoir (LSM)
class RandomLSM(nn.Module):
    """
    Random Spiking Reservoir (LSM)
    """
    def __init__(self, input_size=1, hidden_size=64, num_classes=2, beta=0.95, input_scale=50.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # The Reservoir Weights (Random and Fixed)
        self.input_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.recurrent_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize weights randomly
        nn.init.normal_(self.input_layer.weight, std=0.1 * input_scale) # Scale input to drive reservoir
        nn.init.normal_(self.recurrent_layer.weight, std=0.1)
        
        # Freeze weights
        for param in self.input_layer.parameters(): param.requires_grad = False
        for param in self.recurrent_layer.parameters(): param.requires_grad = False
            
        # The Neuron (Leaky Integrate and Fire)
        self.lif = snn.Leaky(beta=beta, reset_mechanism='zero')
        
        # The Readout Layer (Trainable)
        self.readout = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [Batch, Time, Input]
        batch_size, time_steps, _ = x.size()
        mem = torch.zeros(batch_size, self.hidden_size).to(x.device)
        spk_rec = []
        
        for step in range(time_steps):
            xt = x[:, step, :]
            # Current = Input + Recurrent Echo
            cur = self.input_layer(xt) + self.recurrent_layer(mem)
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            
        spk_rec = torch.stack(spk_rec, dim=1)
        
        # Classification on Mean Firing Rate
        rate = spk_rec.mean(dim=1)
        out = self.readout(rate)
        return out, spk_rec

# Model C: Structured LSNN (Math Feature Extractor + Spiking MLP)
class LegendreFeatureLayer(nn.Module):
    """
    Non-spiking layer that implements: x(t) = A*x(t-1) + B*u(t) using the Legendre matrices
    """
    def __init__(self, d, theta):
        super().__init__()
        self.d = d
        # Get the Legendre matrices
        A, B = get_legendre_matrices(d, theta)
        # Register as buffers (so they move to GPU but aren't trained)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        
    def forward(self, x):
        # x: [Batch, Time, 1]
        batch_size, time_steps, _ = x.size()
        
        # Initial state x(0) = 0
        state = torch.zeros(batch_size, self.d).to(x.device)
        features = []
        
        for step in range(time_steps):
            u_t = x[:, step, :] # [Batch, 1]
            
            # x(t) = A * x(t-1) + B * u(t)
            # Shapes: (d,d)*(B,d).T  + (d,1)*(B,1).T
            # We use Linear logic: state @ A.T + input @ B.T
            state = torch.matmul(state, self.A.t()) + torch.matmul(u_t, self.B.t())
            features.append(state)
            
        return torch.stack(features, dim=1) # [Batch, Time, d]

class StructuredLSNN(nn.Module):
    """
    Structured LSNN (Math Feature Extractor + Spiking MLP)
    """
    def __init__(self, input_size=1, d=24, theta=140, num_classes=2, gain=1.0):
        super().__init__()
        
        # The Math Layer (Fixed Feature Extractor)
        self.ldn = LegendreFeatureLayer(d=d, theta=theta)
        
        # The Spiking Network (Trainable)
        # The paper uses: Encoder (Linear) -> Hidden (Spiking) -> Output
        hidden_dim = 3 * d # Paper suggests 3x expansion
        
        self.encoder = nn.Linear(d, hidden_dim) # Learns how to interpret Legendre features
        self.lif = snn.Leaky(beta=0.9, reset_mechanism='zero') # Spiking nonlinearity
        self.readout = nn.Linear(hidden_dim, num_classes)
        
        # Initialize encoder gain (Sensitivity)
        nn.init.normal_(self.encoder.weight, std=gain)

    def forward(self, x):
        # 1. Extract Clean Math Features
        # x: [Batch, Time, 1] -> leg_features: [Batch, Time, d]
        leg_features = self.ldn(x)
        
        # 2. Process with Spiking Network
        batch_size, time_steps, _ = x.size()
        mem = self.lif.init_leaky()
        spk_rec = []
        
        for step in range(time_steps):
            ft = leg_features[:, step, :]
            
            # Encoder -> Hidden Spiking Layer
            cur = self.encoder(ft)
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            
        spk_rec = torch.stack(spk_rec, dim=1)
        
        # Classification
        rate = spk_rec.mean(dim=1)
        out = self.readout(rate)
        return out, spk_rec