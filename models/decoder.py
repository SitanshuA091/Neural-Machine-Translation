import torch.nn as nn
import torch
from models.attention import LuongAttention

class DecoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.2):
        super(DecoderNoAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dense = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        logits = self.dense(outputs)
        return logits, hidden, cell

class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.2):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.attention = LuongAttention(hidden_dim)
        self.w_attention = nn.Linear(hidden_dim * 2, hidden_dim)
        # Note: The original had activation assignment, which is not needed; we'll apply tanh in forward.
        self.dense = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, encoder_outputs, hidden, cell):
        embedded = self.embedding(x)
        decoder_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        weights, context = self.attention(encoder_outputs, decoder_output)
        combined = torch.cat([context, decoder_output], dim=-1)
        attended = torch.tanh(self.w_attention(combined))
        logits = self.dense(attended)
        return logits, hidden, cell, weights