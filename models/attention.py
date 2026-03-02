import torch
import torch.nn as nn

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs, decoder_output):
        # encoder_outputs: (batch, seq_len, hidden_dim)
        # decoder_output: (batch, 1, hidden_dim)
        z = self.w(encoder_outputs)  # (batch, seq_len, hidden_dim)
        scores = torch.bmm(decoder_output, z.transpose(1, 2))  # (batch, 1, seq_len)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)  # (batch, 1, hidden_dim)
        return weights, context