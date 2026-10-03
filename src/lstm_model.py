"""첨부된 Seq2Seq LSTM 구조(LSTM.py)를 그대로 옮긴 것. 학습·데이터 처리는 src/seq.py에서 한다.

- encoder: 과거 window칸의 [전력 + 달력] 시퀀스를 읽어 (hidden, cell) 상태로 요약
- decoder: 그 상태에서 시작해, 예측일 96칸의 "미리 아는 정보" 시퀀스를 읽음
- head: 디코더 출력과 미래 정보를 이어 붙여 칸마다 전력 1개를 출력
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