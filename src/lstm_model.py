"""Seq2Seq LSTM.

encoder: 과거 구간의 [전력 + 달력] 시퀀스를 (hidden, cell) 상태로 요약
decoder: 인코더 상태에서 시작해 예측 구간의 사전 정보 시퀀스를 처리
head: 디코더 출력과 사전 정보를 결합해 구간별 전력을 출력
"""
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