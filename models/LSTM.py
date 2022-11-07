import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_dim, num_layers, bidirectional, dropout):

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0 if num_layers < 2 else dropout)

        self.out = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_dim)

        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #print(x.shape, h0.shape, c0.shape)

        outputs, (hn, cn) = self.rnn(x, (h0, c0))

        outputs = self.dropout(outputs)

        output = self.out(outputs)
        #print(outputs.shape)

        return output
