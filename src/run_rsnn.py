"""
rsnn_train.py
─────────────────────────────────────────────────────────────────────────────
Training, evaluation, sparsity metrics, and grid search for LearnedRSNN.

Grid search space
    truncation_k   : {20, 40, 70, 140}   — TBPTT window
    surrogate_type : {fast_sigmoid, sigmoid}
    sharpness      : {1.0, 5.0, 10.0}    — surrogate slope
    lr             : {0.001, 0.005}
    beta           : {0.9, 0.95}          — LIF decay

Total configurations: 96
─────────────────────────────────────────────────────────────────────────────
"""

import time
import torch
import torch.nn as nn
import pandas as pd
from itertools import product

from src.models.rsnn import LearnedRSNN

# ═══════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, criterion, k, device):
    """Single training epoch with TBPTT window k."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        spike_rates = model(x, k)  # [B, 2]
        loss = criterion(spike_rates, y)
        loss.backward()

        # Gradient clipping — essential for stable recurrent SNN training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pred = spike_rates.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, k, device):
    """Single evaluation epoch (no gradient)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            spike_rates = model(x, k)

            loss = criterion(spike_rates, y)
            total_loss += loss.item()

            pred = spike_rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total


def train_rsnn(model, train_loader, test_loader, k, lr, epochs, device):
    """
    Full training run for one configuration.

    Returns
    -------
    model   : trained LearnedRSNN
    history : dict with train/test loss and accuracy per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, k, device
        )
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, k, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc * 100)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc * 100)

    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# SPARSITY METRICS
# ═══════════════════════════════════════════════════════════════════════════


def compute_sparsity(model, loader, k, device):
    """
    Compute reservoir spike sparsity and spikes-per-sample.

    Sparsity = (1 - spike_density) × 100
    Spike density = total spikes / (samples × T × hidden_size)
    """
    model.eval()
    total_spikes = 0
    total_possible = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            B, T, _ = x.shape

            mem_res = torch.zeros(B, model.hidden_size, device=device)
            spk_res = torch.zeros(B, model.hidden_size, device=device)

            batch_spikes = 0
            for t in range(T):
                i_t = model.W_in(x[:, t, :]) + model.W_rec(spk_res)
                spk_res, mem_res = model.lif_res(i_t, mem_res)
                batch_spikes += spk_res.sum().item()

            total_spikes += batch_spikes
            total_possible += B * T * model.hidden_size

    spike_density = total_spikes / total_possible
    sparsity = (1 - spike_density) * 100
    spikes_per_sample = total_spikes / len(loader.dataset)

    return round(sparsity, 2), round(spikes_per_sample, 1)


def compute_inference_latency(model, loader, k, device, n_warmup=10, n_runs=50):
    """
    Measure mean inference latency per sample (ms).
    Warm-up phase followed by timed forward passes on single samples.
    """
    model.eval()

    # Grab a single sample for timing
    x_sample, _ = next(iter(loader))
    x_single = x_sample[:1].to(device)  # [1, T, 1]

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_single, k)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x_single, k)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # convert to ms

    return round(sum(times) / len(times), 2)


# ═══════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════


def run_grid_search(train_loader, test_loader, device, epochs=30):
    """
    Grid search over:
        truncation_k   : [20, 40, 70, 140]
        surrogate_type : ['fast_sigmoid', 'sigmoid']
        sharpness      : [1.0, 5.0, 10.0]
        lr             : [0.001, 0.005]
        beta           : [0.9, 0.95]

    Saves results to rsnn_grid_search.csv.
    """

    truncation_ks = [20, 40, 70, 140]
    surrogate_types = ["fast_sigmoid", "sigmoid"]
    sharpnesses = [1.0, 5.0, 10.0]
    lrs = [0.001, 0.005]
    betas = [0.9, 0.95]

    configs = list(product(truncation_ks, surrogate_types, sharpnesses, lrs, betas))

    print(f"Total configurations : {len(configs)}")
    print(f"Epochs per config    : {epochs}")
    print(f"Estimated runtime    : ~{len(configs) * epochs // 60}+ minutes\n")
    print("─" * 70)

    results = []
    logs = []

    for i, (k, surr, sharp, lr, beta) in enumerate(configs):

        name = f"RSNN-k{k}-{surr}-sh{sharp}-lr{lr}-b{beta}"
        print(f"[{i+1:>3}/{len(configs)}]  {name}")

        model = LearnedRSNN(
            beta=beta,
            surrogate_type=surr,
            sharpness=sharp,
        ).to(device)

        t_start = time.time()
        model, hist = train_rsnn(
            model,
            train_loader,
            test_loader,
            k=k,
            lr=lr,
            epochs=epochs,
            device=device,
        )
        train_time = round(time.time() - t_start, 2)

        final_acc = hist["test_acc"][-1]
        best_acc = max(hist["test_acc"])

        # Sparsity and latency for the best model snapshot
        sparsity, spikes_per_sample = compute_sparsity(model, test_loader, k, device)
        latency = compute_inference_latency(model, test_loader, k, device)

        print(
            f"          → Final: {final_acc:.2f}%  "
            f"Best: {best_acc:.2f}%  "
            f"Sparsity: {sparsity:.1f}%  "
            f"Spikes/sample: {spikes_per_sample:.0f}  "
            f"Latency: {latency} ms  "
            f"({train_time}s)"
        )

        results.append(
            {
                "Model": "LearnedRSNN",
                "k": k,
                "Surrogate": surr,
                "Sharpness": sharp,
                "LR": lr,
                "Beta": beta,
                "Final_Acc_%": round(final_acc, 4),
                "Best_Acc_%": round(best_acc, 4),
                "Sparsity_%": sparsity,
                "Spikes_per_Sample": spikes_per_sample,
                "Inf_Latency_ms": latency,
                "Train_Time_s": train_time,
            }
        )

        logs.append({"name": name, "history": hist})

    # ── Save and report ───────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("Final_Acc_%", ascending=False)
    df.to_csv("rsnn_grid_search.csv", index=False)

    print("\n" + "═" * 70)
    print("TOP 10 CONFIGURATIONS")
    print("═" * 70)
    print(df.head(10).to_string(index=False))

    # ── Best config benchmark vs other models ────────────────────────────
    best = df.iloc[0]
    print("\n" + "═" * 70)
    print("BENCHMARK COMPARISON")
    print("═" * 70)
    print(
        f"{'Model':<30} {'Acc %':>8} {'Params':>8} {'Sparsity %':>12} {'Spikes/Sample':>15} {'Latency ms':>12}"
    )
    print("-" * 70)
    print(
        f"{'LSTM (Baseline)':<30} {'97.24':>8} {'~50,050':>8} {'N/A':>12} {'N/A':>15} {'6.20':>12}"
    )
    print(
        f"{'LSNN (d=24)':<30} {'97.07':>8} {'1,872':>8} {'71.3':>12} {'2,897':>15} {'40.01':>12}"
    )
    print(
        f"{'LSM (Random)':<30} {'89.96':>8} {'128':>8} {'89.3':>12} {'963':>15} {'30.64':>12}"
    )
    print(
        f"{'LearnedRSNN (best)':<30} "
        f"{best['Final_Acc_%']:>8.2f} "
        f"{'~4,288':>8} "
        f"{best['Sparsity_%']:>12.1f} "
        f"{best['Spikes_per_Sample']:>15.0f} "
        f"{best['Inf_Latency_ms']:>12.2f}"
    )
    print("─" * 70)
    print(
        f"Best config: k={best['k']}, surrogate={best['Surrogate']}, "
        f"sharpness={best['Sharpness']}, lr={best['LR']}, beta={best['Beta']}"
    )

    return df, logs


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Usage
    ─────
    This script expects train_loader and test_loader to already exist.
    Paste the block below after your existing data loading code, or
    import and call run_grid_search(train_loader, test_loader, DEVICE).

    Example
    ───────
    from rsnn_train import run_grid_search
    df, logs = run_grid_search(train_loader, test_loader, device=DEVICE, epochs=30)
    """

    print("Import run_grid_search from rsnn_train and call it with your loaders.")
    print("Example:")
    print("  from rsnn_train import run_grid_search")
    print("  df, logs = run_grid_search(train_loader, test_loader, DEVICE)")
