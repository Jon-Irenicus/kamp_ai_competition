import torch
import torch.nn as nn


class Seq2SeqLSTM(nn.Module):


    def __init__(self, input_size, future_size, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder_lstm = nn.LSTM(
            input_size=future_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, future):
        _, (hidden, cell) = self.encoder_lstm(x)

        out, _ = self.decoder_lstm(future, (hidden, cell))   
        out = torch.cat([out, future], dim=-1)           

        return self.head(out).squeeze(-1)                