import torch
import torch.nn as nn


class Seq2SeqLSTM(nn.Module):

    def __init__(self, input_size, future_size, hidden_size=128,
                 num_layers=2, dropout=0.2):
        super().__init__()

        # encoder : input을 입력한 뒤 만들어진 가중치와 output을 decoder에 전달한다.
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        # decoder : encoder를 통과한 뒤 만들어진 가중치를 초기 가중치로 사용하고,
        # output을 미래 시점에 알 수 있는 데이터와 결합하여 사용한다. 
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