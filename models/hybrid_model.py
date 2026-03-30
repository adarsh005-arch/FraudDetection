import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)  # 2 classes (fraud / normal)

    def forward(self, gnn_out, transformer_out):

        # Combine both outputs
        combined = torch.cat((gnn_out, transformer_out), dim=1)

        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)

        return x