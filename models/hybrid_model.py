import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HybridModel, self).__init__()

        self.gnn_layer = nn.Linear(input_dim, hidden_dim)
        self.transformer_layer = nn.Linear(input_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        gnn_out = torch.relu(self.gnn_layer(x))
        transformer_out = torch.relu(self.transformer_layer(x))

        combined = torch.cat((gnn_out, transformer_out), dim=1)

        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x