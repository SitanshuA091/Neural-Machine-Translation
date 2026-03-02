import torch
import torch.nn as nn

class Seq2SeqNoAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqNoAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encoder(src)
        logits, _, _ = self.decoder(tgt, hidden, cell)
        return logits

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encoder(src)
        decoder_logits_list = []
        # For attention, we need to iterate over time steps
        # Initialize hidden and cell from encoder
        # hidden and cell are already from encoder
        for t in range(tgt.size(1)):
            decoder_input = tgt[:, t:t+1]
            logits, hidden, cell, _ = self.decoder(decoder_input, encoder_outputs, hidden, cell)
            decoder_logits_list.append(logits)
        return torch.cat(decoder_logits_list, dim=1)