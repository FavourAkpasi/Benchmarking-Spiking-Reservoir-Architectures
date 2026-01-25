# Benchmarking Spiking Reservoir Architectures for Efficient Univariate Time Series Classification

## 1. Project Overview
This project investigates **Spiking Neural Networks (SNNs)** and **Reservoir Computing (RC)** for detecting cardiovascular anomalies on the edge / resources constrained devices. Specifically, it benchmarks three architectures on the **ECG5000** time-series dataset to answer two key research questions:

1.  **Necessity of Training:** Can a reservoir based classifier match the performance of a conventional fully trained recurrent network?
2.  **Impact of Structure:** Does a mathematically structured memory (Legendre Polynomials) outperform a random biological memory (Liquid State Machine)?

The methodology follows the work of *Gaurav et al. (2023)*, comparing a **Structured LSNN** (Legendre Spiking Neural Network) against a **Random LSM** (Liquid State Machine) and a **Baseline LSTM**.

---

## 2. Environment Setup

### Prerequisites
* Python 3.8+
* Anaconda or `venv`

### Step 1: Create Virtual Environment
```bash
# Using Conda
conda create -n [environment_name] python=3.9
conda activate [environment_name]

# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Requirements / packages
```bash
pip install -r requirements.txt
```

## 3. Project Structure

- **data/raw**: Original ECG5000 TRAIN/TEST `.txt` files (5-class labels).
- **data/processed**: Processed binary datasets ECG5000 TRAIN/TEST `.txt` used for Normal vs Abnormal classification.
- **notebooks/01_data_exploration.ipynb**: Dataset exploration and sanity checks.
- **notebooks/02_main.ipynb**: Main experiments (training, evaluation, plots).
- **notebooks/03_grid_search.ipynb**: Hyperparameter search.
- **src/dataset.py**: `ECGDataset`, PyTorch dataloaders, and helper to create the binary dataset from raw files.
- **src/utils.py**: Training/evaluation utilities, metrics, latency measurement, plotting, and Legendre (LDN) matrices.
- **src/models/lstm.py**: Baseline LSTM model (`BaselineLSTM`).
- **src/models/reservoirs.py**: Reservoir-based spiking models (`RandomLSM`, `StructuredLSNN`).
- **requirements.txt**: Python dependencies.
