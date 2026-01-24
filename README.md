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

ECG5000_Project/
├─ data/
│  ├─ raw/                    # Original ECG5000 TRAIN/TEST txt files (5-class labels)
│  └─ processed/              # Derived binary datasets (BINARY_*.txt)
├─ notebooks/
│  ├─ 01_data_exploration.ipynb
│  └─ 02_main.ipynb           # Main experiments: training + evaluation + plots
├─ src/
│  ├─ dataset.py              # ECGDataset, dataloaders, raw→binary preprocessing helper
│  ├─ utils.py                # train/eval loop, metrics, latency, plotting, Legendre matrices
│  └─ models/
│     ├─ lstm.py              # BaselineLSTM
│     └─ reservoirs.py        # RandomLSM and StructuredLSNN (Legendre-feature + spiking readout)
├─ requirements.txt
└─ README.md
