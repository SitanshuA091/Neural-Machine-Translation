import json
import torch
import config
from utils.tokenizer import Tokenizer, pad_sequences_pytorch
from models.encoder import EncoderNoAttention, EncoderWithAttention
from models.decoder import DecoderNoAttention, DecoderWithAttention
from models.seq2seq import Seq2SeqNoAttention, Seq2SeqWithAttention

def load_model(model_class, model_path, *args, **kwargs):
    model = model_class(*args, **kwargs).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def load_tokenizer(path):
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return Tokenizer.from_json(config_dict)

def translate_no_attention(sentence, encoder, decoder, source_tokenizer, target_tokenizer, max_enc_len, max_len=30):
    encoder.eval()
    decoder.eval()
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    input_pad = torch.LongTensor(pad_sequences_pytorch(input_seq, max_enc_len)).to(config.device)

    with torch.no_grad():
        _, hidden, cell = encoder(input_pad)

    decoded = []
    current_token = config.SOS_TOKEN

    for _ in range(max_len):
        token_idx = target_tokenizer.word_index.get(current_token, 1)
        dec_input = torch.LongTensor([[token_idx]]).to(config.device)
        with torch.no_grad():
            logits, hidden, cell = decoder(dec_input, hidden, cell)
        token_idx = torch.argmax(logits[0, -1, :]).item()
        current_token = target_tokenizer.index_word.get(token_idx, config.UNK_TOKEN)
        if current_token == config.EOS_TOKEN:
            break
        decoded.append(current_token)
    return ' '.join(decoded)

def translate_with_attention(sentence, encoder, decoder, source_tokenizer, target_tokenizer, max_enc_len, max_len=30):
    encoder.eval()
    decoder.eval()
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    input_pad = torch.LongTensor(pad_sequences_pytorch(input_seq, max_enc_len)).to(config.device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = encoder(input_pad)

    decoded = []
    current_token = config.SOS_TOKEN

    for _ in range(max_len):
        token_idx = target_tokenizer.word_index.get(current_token, 1)
        dec_input = torch.LongTensor([[token_idx]]).to(config.device)
        with torch.no_grad():
            logits, hidden, cell, _ = decoder(dec_input, encoder_outputs, hidden, cell)
        token_idx = torch.argmax(logits[0, -1, :]).item()
        current_token = target_tokenizer.index_word.get(token_idx, config.UNK_TOKEN)
        if current_token == config.EOS_TOKEN:
            break
        decoded.append(current_token)
    return ' '.join(decoded)

# Load tokenizers and models once
source_tok = load_tokenizer(config.source_tokenizer_path)
target_tok = load_tokenizer(config.target_tokenizer_path)

# Get max encoding length from training data (you may want to save this during training)
max_enc_len = 20  # Replace with your actual max encoding length

# Load model without attention
encoder_no = EncoderNoAttention(len(source_tok.word_index)+1, config.embedding_dim, config.hidden_dim, config.dropout)
decoder_no = DecoderNoAttention(len(target_tok.word_index)+1, config.embedding_dim, config.hidden_dim, config.dropout)
model_no = Seq2SeqNoAttention(encoder_no, decoder_no)
model_no = load_model(Seq2SeqNoAttention, config.model_no_attn_path, encoder_no, decoder_no)

print("\nModels and tokenizers loaded. Ready for inference.")
print("To translate a sentence, use:")
print("  translate_no_attention('your french sentence here', encoder_no, decoder_no, source_tok, target_tok, max_enc_len)")
print("  translate_with_attention('your french sentence here', encoder_attn, decoder_attn, source_tok, target_tok, max_enc_len)")