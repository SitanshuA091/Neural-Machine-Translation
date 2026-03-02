import torch
import torch.optim as optim
from models.encoder import EncoderNoAttention, EncoderWithAttention
from models.decoder import DecoderNoAttention, DecoderWithAttention
from models.seq2seq import Seq2SeqNoAttention, Seq2SeqWithAttention
from utils.dataset import MaskedCrossEntropyLoss
import config

def train_no_attention(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MaskedCrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for enc_input, dec_input, dec_target in train_loader:
            enc_input, dec_input, dec_target = enc_input.to(device), dec_input.to(device), dec_target.to(device)
            optimizer.zero_grad()
            logits = model(enc_input, dec_input)
            loss = criterion(logits, dec_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_input, dec_input, dec_target in val_loader:
                enc_input, dec_input, dec_target = enc_input.to(device), dec_input.to(device), dec_target.to(device)
                logits = model(enc_input, dec_input)
                loss = criterion(logits, dec_target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def train_with_attention(model, train_loader, val_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MaskedCrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for enc_input, dec_input, dec_target in train_loader:
            enc_input, dec_input, dec_target = enc_input.to(device), dec_input.to(device), dec_target.to(device)
            optimizer.zero_grad()
            logits = model(enc_input, dec_input)
            loss = criterion(logits, dec_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_input, dec_input, dec_target in val_loader:
                enc_input, dec_input, dec_target = enc_input.to(device), dec_input.to(device), dec_target.to(device)
                logits = model(enc_input, dec_input)
                loss = criterion(logits, dec_target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def main():
    pass

if __name__ == "__main__":
    main()