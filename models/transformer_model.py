import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, 32)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True   
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

    def forward(self, x):
        # x shape: [num_nodes, features]

        x = self.embedding(x)

        # Transformer expects: [sequence, batch, features]
        x = x.unsqueeze(1)

        x = self.transformer(x)

        x = x.squeeze(1)

        return x