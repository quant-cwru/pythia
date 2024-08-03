import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.size()) == 2:
            # 2D (batch_size, features), so add a sequence dimension to make 3D
            x = x.unsqueeze(1) 
        
        # 3D (batch_size, sequence_length, features)
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Use only the last time step output
        return out.squeeze()  # Remove extra dimensions