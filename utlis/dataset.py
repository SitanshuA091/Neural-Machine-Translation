import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        batch_size, seq_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        loss = self.ce_loss(logits_flat, targets_flat)
        mask = (targets_flat != 0).float()
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

def create_dataloaders(train_encoder_tensor, train_decoder_input_tensor, train_decoder_target_tensor,
                       val_encoder_tensor, val_decoder_input_tensor, val_decoder_target_tensor,
                       batch_size, shuffle=True):
    train_dataset = TensorDataset(train_encoder_tensor, train_decoder_input_tensor, train_decoder_target_tensor)
    val_dataset = TensorDataset(val_encoder_tensor, val_decoder_input_tensor, val_decoder_target_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader