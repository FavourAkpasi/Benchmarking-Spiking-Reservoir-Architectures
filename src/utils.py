import torch
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def train_model(model, train_loader, test_loader, device='cpu', epochs=10, lr=0.001):
    """
    Generic training function for both LSTM and SNN.
    Returns: trained_model, history_dict
    """
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'test_acc': [], 'train_time': 0}
    
    print(f"Starting training on {device}...")
    start_train = time.time()
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)

            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take just the class predictions, ignore spikes

            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Logging
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Quick validation
        test_acc = evaluate_accuracy(model, test_loader, device)
        history['test_acc'].append(test_acc)
        
        tqdm.write(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
    end_train = time.time()
    history['train_time'] = end_train - start_train
    tqdm.write(f"Training finished in {history['train_time']:.2f} seconds.")
    
    return model, history

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Evaluating", leave=False):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take just the class predictions, ignore spikes
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def measure_inference_latency(model, loader, device):
    """
    Measures how long it takes to process ONE sample (in milliseconds).
    Used for the LSTM efficiency metric.
    """
    model = model.to(device)
    model.eval()
    
    # Warm up GPU/CPU
    dummy_input, _ = next(iter(loader))
    dummy_input = dummy_input.to(device)
    _ = model(dummy_input)
    
    start_time = time.time()
    total_samples = 0
    
    with torch.no_grad():
        for data, _ in tqdm(loader, desc="Measuring latency", leave=False):
            data = data.to(device)
            _ = model(data)
            total_samples += data.size(0)
            
    end_time = time.time()
    total_time = end_time - start_time
    
    latency_ms = (total_time / total_samples) * 1000
    return latency_ms

def get_legendre_matrices(d, theta, dt=1.0):
    """
    Generates the discrete A and B matrices for the Legendre Delay Network (LDN).
    Implements the 'Neural LTI' approximation used in the SLRC/LSNN papers.
    """
    # 1. Construct Continuous LTI Matrices (A and B)
    # based on Voelker et al. (2019)
    A = np.zeros((d, d))
    B = np.zeros((d, 1))

    for i in range(d):
        B[i] = (2 * i + 1) * (-1)**i
        for j in range(d):
            if j < i:
                A[i, j] = (2 * i + 1) * (-1)**(i - j + 1)
            elif j == i:
                A[i, j] = -(i + 1)

    # 2. Discretize (Euler Approximation)
    # The paper uses: x(t) = x(t-1) + (dt/theta) * (Ax + Bu)
    # So: A_discrete = I + (dt/theta)*A
    #     B_discrete = (dt/theta)*B
    
    time_scale = dt / theta
    I = np.eye(d)
    
    A_discrete = I + (A * time_scale)
    B_discrete = B * time_scale
    
    return torch.FloatTensor(A_discrete), torch.FloatTensor(B_discrete)

def logs_to_df(model_logs):
    """model_logs: list of {'name': str, 'history': {'train_loss': [...], 'test_acc': [...], 'train_time': ...}}"""
    rows = []
    for m in model_logs:
        name = m["name"]
        hist = m["history"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        for e, loss, acc in zip(epochs, hist["train_loss"], hist["test_acc"]):
            rows.append({"model": name, "epoch": e, "train_loss": loss, "test_acc": acc})
    return pd.DataFrame(rows)

def print_train_times(model_logs):
    for m in model_logs:
        print(f"{m['name']}: {m['history'].get('train_time', 'N/A')} s")

def plot_curves(model_logs):
    """
    Accepts either:
      - model_logs (list of dicts), or
      - a DataFrame from logs_to_df with columns: model, epoch, train_loss, test_acc
    Plots loss + accuracy side-by-side per model.
    """
    if isinstance(model_logs, list):
        df = logs_to_df(model_logs)
    else:
        df = model_logs.copy()

    models = df["model"].unique().tolist()
    fig, axes = plt.subplots(nrows=len(models), ncols=2, figsize=(12, 3.2 * len(models)), sharex=False)

    if len(models) == 1:
        axes = [axes]  # normalize shape: [[ax_loss, ax_acc]]

    for i, model in enumerate(models):
        d = df[df["model"] == model].sort_values("epoch")

        ax_loss = axes[i][0]
        ax_acc = axes[i][1]

        ax_loss.plot(d["epoch"], d["train_loss"], c='red')
        ax_loss.set_title(f"{model} - Train Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.3)

        ax_acc.plot(d["epoch"], d["test_acc"], c='blue')
        ax_acc.set_title(f"{model} - Test Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def measure_single_sample_latency_ms(
    model,
    sample,
    device="cpu",
    repeats=200,
    warmup=20,
):
    """
    Measures average inference latency per *single sample* in milliseconds.

    Works with:
      - models returning logits (e.g., BaselineLSTM)
      - models returning (logits, extra) (e.g., RandomLSM / StructuredLSNN / SpikingReservoir)

    Args:
      model: trained torch.nn.Module
      sample: a single input sample shaped [T, F] or [T, F=1] (no batch dim),
              or already batched [1, T, F]
      device: "cpu" or "cuda"
      repeats: how many timed forward passes
      warmup: untimed passes to stabilize

    Returns:
      avg_ms (float)
    """
    model = model.to(device)
    model.eval()

    x = sample
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    # ensure batch dim: [1, T, F]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device)

    def _sync(dev: str):
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]

    # timing
    _sync(device)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            _sync(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return float(np.mean(times))

def compare_models_latency(model_dict, sample, device="cpu", repeats=200, warmup=20):
    """
    model_dict: {"name": model, ...}
    Prints and returns sorted list [(name, ms), ...]
    """
    results = []
    for name, model in model_dict.items():
        ms = measure_single_sample_latency_ms(
            model, sample, device=device, repeats=repeats, warmup=warmup
        )
        results.append((name, ms))

    results.sort(key=lambda x: x[1])
    for name, ms in results:
        print(f"{name}: {ms:.3f} ms / sample")
    print(f"Fastest: {results[0][0]}")
    return results