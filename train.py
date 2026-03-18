import os
import json
import torch
import torch.optim as optim
import config
from utils.tokenizer import Tokenizer, pad_sequences_pytorch, generate_decoder_inputs_targets
from utils.preprocessing import preprocess_sentence, tag_target_sentences, process_dataset
from utils.dataset import MaskedCrossEntropyLoss, create_dataloaders
from models.encoder import EncoderNoAttention, EncoderWithAttention
from models.decoder import DecoderNoAttention, DecoderWithAttention
from models.seq2seq import Seq2SeqNoAttention, Seq2SeqWithAttention

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_tokenizer(tokenizer, path):
    with open(path, 'w') as f:
        json.dump(tokenizer.to_json(), f)
    print(f"Tokenizer saved to {path}")

def train_no_attention(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedCrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for enc_in, dec_in, dec_tgt in train_loader:
            enc_in, dec_in, dec_tgt = enc_in.to(device), dec_in.to(device), dec_tgt.to(device)
            optimizer.zero_grad()
            logits = model(enc_in, dec_in)
            loss = criterion(logits, dec_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_in, dec_in, dec_tgt in val_loader:
                enc_in, dec_in, dec_tgt = enc_in.to(device), dec_in.to(device), dec_tgt.to(device)
                logits = model(enc_in, dec_in)
                loss = criterion(logits, dec_tgt)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses

def train_with_attention(model, train_loader, val_loader, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedCrossEntropyLoss()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for enc_in, dec_in, dec_tgt in train_loader:
            enc_in, dec_in, dec_tgt = enc_in.to(device), dec_in.to(device), dec_tgt.to(device)
            optimizer.zero_grad()
            logits = model(enc_in, dec_in)
            loss = criterion(logits, dec_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for enc_in, dec_in, dec_tgt in val_loader:
                enc_in, dec_in, dec_tgt = enc_in.to(device), dec_in.to(device), dec_tgt.to(device)
                logits = model(enc_in, dec_in)
                loss = criterion(logits, dec_tgt)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses, val_losses

def main():
    device = config.device
    print(f"Using device: {device}")

    with open(config.train_data_path) as f:
        train = [line.rstrip() for line in f]

    SEPARATOR = '<sep>'
    train_input, train_target = map(list, zip(*[pair.split(SEPARATOR) for pair in train]))
    train_input = [preprocess_sentence(s) for s in train_input]
    train_target = [preprocess_sentence(s) for s in train_target]
    train_target_tagged = tag_target_sentences(train_target)

    # Build tokenizers
    source_tokenizer = Tokenizer(oov_token=config.UNK_TOKEN)
    source_tokenizer.fit_on_texts(train_input)
    target_tokenizer = Tokenizer(oov_token=config.UNK_TOKEN)
    target_tokenizer.fit_on_texts(train_target_tagged)

    source_vocab_size = len(source_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1
    print(f"Source vocab size: {source_vocab_size}, Target vocab size: {target_vocab_size}")

    train_enc_seqs = source_tokenizer.texts_to_sequences(train_input)
    train_dec_in, train_dec_tgt = generate_decoder_inputs_targets(train_target_tagged, target_tokenizer)

    max_enc_len = len(max(train_enc_seqs, key=len))
    max_dec_len = len(max(train_dec_in, key=len))
    print(f"Max encoding length: {max_enc_len}, Max decoding length: {max_dec_len}")
    
    #saving max_enc_len for later bleu evaluations
    with open('max_lengths.json', 'w') as f:
        json.dump({'max_enc_len': max_enc_len, 'max_dec_len': max_dec_len}, f)
    print("Max lengths saved to max_lengths.json")

    train_enc_pad = pad_sequences_pytorch(train_enc_seqs, max_enc_len)
    train_dec_in_pad = pad_sequences_pytorch(train_dec_in, max_dec_len)
    train_dec_tgt_pad = pad_sequences_pytorch(train_dec_tgt, max_dec_len)

    with open(config.val_data_path) as f:
        val = [line.rstrip() for line in f]

    val_enc_pad, val_dec_in_pad, val_dec_tgt_pad = process_dataset(
        val, source_tokenizer, target_tokenizer, max_enc_len, max_dec_len
    )

    train_enc_tensor = torch.LongTensor(train_enc_pad)
    train_dec_in_tensor = torch.LongTensor(train_dec_in_pad)
    train_dec_tgt_tensor = torch.LongTensor(train_dec_tgt_pad)
    val_enc_tensor = torch.LongTensor(val_enc_pad)
    val_dec_in_tensor = torch.LongTensor(val_dec_in_pad)
    val_dec_tgt_tensor = torch.LongTensor(val_dec_tgt_pad)

    train_loader, val_loader = create_dataloaders(
        train_enc_tensor, train_dec_in_tensor, train_dec_tgt_tensor,
        val_enc_tensor, val_dec_in_tensor, val_dec_tgt_tensor,
        batch_size=config.batch_size
    )
    print("\nTraining model WITHOUT attention...")
    encoder_no_attn = EncoderNoAttention(source_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    decoder_no_attn = DecoderNoAttention(target_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    model_no_attn = Seq2SeqNoAttention(encoder_no_attn, decoder_no_attn).to(device)

    train_losses_no_attn, val_losses_no_attn = train_no_attention(
        model_no_attn, train_loader, val_loader, config.epochs, config.learning_rate, device
    )

    save_model(model_no_attn, config.model_no_attn_path)

    print("\nTraining model WITH attention...")
    encoder_attn = EncoderWithAttention(source_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    decoder_attn = DecoderWithAttention(target_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
    model_attn = Seq2SeqWithAttention(encoder_attn, decoder_attn).to(device)
    train_losses_attn, val_losses_attn = train_with_attention(
        model_attn, train_loader, val_loader, config.epochs, config.learning_rate, device
    )

    save_model(model_attn, config.model_attn_path)
    save_tokenizer(source_tokenizer, config.source_tokenizer_path)
    save_tokenizer(target_tokenizer, config.target_tokenizer_path)

    print("\nTraining complete.")

if __name__ == "__main__":
    main()