import torch
import torch.nn as nn

class TimeSeriesTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, seq_length, dropout=0.1):
        super(TimeSeriesTransformerAutoencoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        z = self.transformer_encoder(x)
        x_rec = self.transformer_decoder(z, z)
        return self.fc_out(x_rec)
    
    def encode(self, x):
        x = self.embedding(x) + self.pos_encoder
        return self.transformer_encoder(x)