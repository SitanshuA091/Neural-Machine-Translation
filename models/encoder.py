import torch.nn as nn

class EncoderNoAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.2):
        super(EncoderNoAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class EncoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.2):
        super(EncoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)  # Note: removed return_sequences=True

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell