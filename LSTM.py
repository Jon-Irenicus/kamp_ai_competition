import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=96, num_layers=2, dropout=0.2):
        super(Seq2SeqLSTM, self).__init__()
        
        self.output_size = output_size
        
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout, batch_first=True)
        self.decoder_lstm = nn.LSTM(1, hidden_size, num_layers, dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        _, (hidden, cell) = self.encoder_lstm(x)
        
        outputs = torch.zeros(batch_size, self.output_size, 1).to(x.device)
        
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device) 
        
        for t in range(self.output_size):
            out, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
     
            prediction = self.fc(out)
            
            outputs[:, t:t+1, :] = prediction
            
            decoder_input = prediction
            
        return outputs.squeeze(-1)