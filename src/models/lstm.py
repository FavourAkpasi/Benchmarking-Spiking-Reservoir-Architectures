import torch
import torch.nn as nn

class BaselineLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=2):
        super(BaselineLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM Layer
        # Input shape: (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # The Classifier (Readout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 140, 1)
        
        # Initialize hidden and cell states (optional, defaults to 0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # take the output of the last time step
        # out shape: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        
        # Decode the hidden state of the last time step to the number of classes
        out = self.fc(out)
        return out