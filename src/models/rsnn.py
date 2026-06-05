"""
rsnn_model.py
─────────────────────────────────────────────────────────────────────────────
Learned Recurrent Spiking Neural Network

Architecture
    Input [B, T, 1]
        ↓  W_in  (learned)
    Reservoir: 64 LIF neurons  +  W_rec (learned, recurrent)
        ↓  spike matrix [B, T, 64]
    Output: 2 LIF neurons  ←  W_out (learned)
        ↓  spike accumulation over T steps
    Spike rates [B, 2]  →  CrossEntropyLoss  →  WTA class decision
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.surrogate as surrogate


class LearnedRSNN(nn.Module):
    """
    Learned Recurrent Spiking Neural Network.

    Parameters
    ----------
    input_size    : int   — input feature dimension (1 for ECG scalar)
    hidden_size   : int   — number of reservoir LIF neurons (64)
    num_classes   : int   — number of output LIF neurons / classes (2)
    beta          : float — LIF membrane decay rate (e.g. 0.9, 0.95)
    surrogate_type: str   — 'fast_sigmoid' | 'sigmoid'
    sharpness     : float — slope/steepness of the surrogate gradient
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_classes: int = 2,
        beta: float = 0.9,
        surrogate_type: str = "fast_sigmoid",
        sharpness: float = 5.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # ── Surrogate gradient function ───────────────────────────────────
        if surrogate_type == "fast_sigmoid":
            # Piecewise linear approximation — tight gradient near threshold
            spike_grad = surrogate.fast_sigmoid(slope=sharpness)
        elif surrogate_type == "sigmoid":
            # Smooth approximation — wider gradient spread
            spike_grad = surrogate.sigmoid(slope=sharpness)
        else:
            raise ValueError(
                f"Unknown surrogate_type '{surrogate_type}'. "
                "Choose 'fast_sigmoid' or 'sigmoid'."
            )

        # ── Learned weight matrices  ────────
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, num_classes, bias=False)

        # ── LIF neurons (same surrogate applied to both layers) ───────────
        self.lif_res = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # ── Weight initialisation ─────────────────────────────────────────
        # Xavier uniform gives a sensible starting scale for both layers
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_rec.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, k: int = 40) -> torch.Tensor:
        """
        Forward pass with Truncated BPTT.

        Parameters
        ----------
        x : Tensor [B, T, input_size]
            Input sequence (batch-first).
        k : int
            TBPTT truncation window.
            Hidden states are detached every k steps so gradients only
            flow back through the most recent k timesteps.
            k = T (sequence length) → full BPTT.

        Returns
        -------
        spike_rates : Tensor [B, num_classes]
            Mean firing rate of each output neuron over T steps.
            Used directly as logits for CrossEntropyLoss.
            WTA prediction: spike_rates.argmax(dim=1)
        """
        B, T, _ = x.shape
        device = x.device

        # ── Initialise hidden states ──────────────────────────────────────
        mem_res = torch.zeros(B, self.hidden_size, device=device)
        mem_out = torch.zeros(B, self.num_classes, device=device)
        spk_res = torch.zeros(B, self.hidden_size, device=device)

        # Accumulate output spikes across all T steps
        spike_counts = torch.zeros(B, self.num_classes, device=device)

        # ── Temporal loop ─────────────────────────────────────────────────
        for t in range(T):

            # TBPTT: detach hidden states at every chunk boundary
            # This cuts the gradient graph beyond k steps while still
            # carrying the forward dynamics (membrane potential, spikes)
            if t > 0 and t % k == 0:
                mem_res = mem_res.detach()
                mem_out = mem_out.detach()
                spk_res = spk_res.detach()

            # ── Reservoir step ────────────────────────────────────────────
            # Input current + recurrent feedback from previous spikes
            i_t = self.W_in(x[:, t, :]) + self.W_rec(spk_res)
            spk_res, mem_res = self.lif_res(i_t, mem_res)

            # ── Output step ───────────────────────────────────────────────
            # Reservoir spikes drive 2 output LIF neurons
            o_t = self.W_out(spk_res)
            spk_out, mem_out = self.lif_out(o_t, mem_out)

            # Accumulate — use + not += to keep gradient graph intact
            spike_counts = spike_counts + spk_out

        # Normalise to firing rates for stable cross-entropy logit scale
        return spike_counts / T  # [B, 2]  values in [0, 1]

    # ─────────────────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"LearnedRSNN("
            f"hidden={self.hidden_size}, "
            f"classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
