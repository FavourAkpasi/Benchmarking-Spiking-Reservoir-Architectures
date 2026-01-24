import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class ECGDataset(Dataset):
    def __init__(self, data_path, train=True, processed=True):
        """
        Args:
            data_path (str): Path to the folder containing .txt files
            train (bool): Load training set if True, else test set
            processed (bool): If True, labels are 0 and 1. If False, labels are 1-5.
        """
        filename = 'ECG5000_TRAIN.txt' if train else 'ECG5000_TEST.txt'
        full_path = os.path.join(data_path, filename)
        
        # Load data 
        raw_data = np.loadtxt(full_path)
        
        # Column 0 is the label. If processed, labels are 0 and 1. If original, labels are 1-5.
        # If Processed, labels are 0 and 1. We do not subtract 1 from the labels.
        # If Original, labels are 1-5. Subtract 1 to get 0-4.
        if processed:
            self.labels = torch.LongTensor(raw_data[:, 0].astype(int))
        else:
            self.labels = torch.LongTensor(raw_data[:, 0].astype(int) - 1)
        
        # Columns 1 to end are the time series data
        # Shape: [Samples, TimeSteps, Features] -> [N, 140, 1]
        self.data = torch.FloatTensor(raw_data[:, 1:]).unsqueeze(2)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(data_dir, batch_size=64, processed=True):
    train_dataset = ECGDataset(data_dir, train=True, processed=processed)
    test_dataset = ECGDataset(data_dir, train=False, processed=processed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_binary_dataset(raw_dir, processed_dir):
    """
    Reads ECG5000 raw files, converts to Binary (Normal vs. Abnormal),
    and saves to processed_dir.
    
    Mapping:
    - Original Class 1 (Normal) -> New Class 0
    - Original Classes 2,3,4,5 (Abnormal) -> New Class 1
    """
    
    # create the processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    files = ['ECG5000_TRAIN.txt', 'ECG5000_TEST.txt']
    
    for filename in files:
        raw_path = os.path.join(raw_dir, filename)
        save_path = os.path.join(processed_dir, f"BINARY_{filename}")
        
        print(f"Processing {filename}...")
        
        # Load data
        try:
            data = np.loadtxt(raw_path)
        except OSError:
            print(f"Error: Could not find {raw_path}. Check your paths.")
            continue
            
        # Separate labels (col 0) and features (cols 1-end)
        labels = data[:, 0]
        features = data[:, 1:]
        
        # Create new labels array
        binary_labels = np.zeros_like(labels)
        
        # If label == 1, new_label = 0, else new_label = 1
        binary_labels = np.where(labels == 1, 0, 1)
        
        # Combine the new labels and features: [New Label, Features]
        processed_data = np.column_stack((binary_labels, features))
        
        # Save the processed data to the processed directory
        np.savetxt(save_path, processed_data, fmt='%f') # Save as float text
        
        # Print the statistics of the processed data
        n_healthy = (binary_labels == 0).sum()
        n_unhealthy = (binary_labels == 1).sum()
        print(f"  Saved to {save_path}")
        print(f"  Stats: Healthy={n_healthy}, Unhealthy={n_unhealthy}, Total={len(binary_labels)}")