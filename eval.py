import json
import argparse
import torch
import sacrebleu
from tqdm import tqdm

import config
from utils.tokenizer import Tokenizer, pad_sequences_pytorch
from utils.preprocessing import process_dataset, preprocess_sentence, tag_target_sentences
from utils.dataset import MaskedCrossEntropyLoss  # not directly used, but imported for consistency
from models.encoder import EncoderNoAttention, EncoderWithAttention
from models.decoder import DecoderNoAttention, DecoderWithAttention
from models.seq2seq import Seq2SeqNoAttention, Seq2SeqWithAttention

def load_tokenizer(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    return Tokenizer.from_json(cfg)

def load_model(model_type, model_path, source_vocab_size, target_vocab_size, device):
    """Load a saved model based on type."""
    if model_type == 'no_attention':
        encoder = EncoderNoAttention(source_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
        decoder = DecoderNoAttention(target_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
        model = Seq2SeqNoAttention(encoder, decoder).to(device)
    elif model_type == 'attention':
        encoder = EncoderWithAttention(source_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
        decoder = DecoderWithAttention(target_vocab_size, config.embedding_dim, config.hidden_dim, config.dropout).to(device)
        model = Seq2SeqWithAttention(encoder, decoder).to(device)
    else:
        raise ValueError("model_type must be 'no_attention' or 'attention'")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded {model_type} model from {model_path}")
    return model

def translate_sentence(sentence, model, source_tokenizer, target_tokenizer, max_enc_len, max_dec_len, device):
    """Translate a single sentence using the appropriate model."""
    # Preprocess input
    sentence = preprocess_sentence(sentence)
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    input_pad = torch.LongTensor(pad_sequences_pytorch(input_seq, max_enc_len)).to(device)

    with torch.no_grad():
        if isinstance(model, Seq2SeqWithAttention):
            # Attention model
            encoder_outputs, hidden, cell = model.encoder(input_pad)
            # Start decoding
            decoded = []
            current_token = config.SOS_TOKEN
            for _ in range(max_dec_len):
                token_idx = target_tokenizer.word_index.get(current_token, 1)
                dec_input = torch.LongTensor([[token_idx]]).to(device)
                logits, hidden, cell, _ = model.decoder(dec_input, encoder_outputs, hidden, cell)
                token_idx = torch.argmax(logits[0, -1, :]).item()
                current_token = target_tokenizer.index_word.get(token_idx, config.UNK_TOKEN)
                if current_token == config.EOS_TOKEN:
                    break
                decoded.append(current_token)
            return ' '.join(decoded)
        else:
            # No-attention model
            _, hidden, cell = model.encoder(input_pad)
            decoded = []
            current_token = config.SOS_TOKEN
            for _ in range(max_dec_len):
                token_idx = target_tokenizer.word_index.get(current_token, 1)
                dec_input = torch.LongTensor([[token_idx]]).to(device)
                logits, hidden, cell = model.decoder(dec_input, hidden, cell)
                token_idx = torch.argmax(logits[0, -1, :]).item()
                current_token = target_tokenizer.index_word.get(token_idx, config.UNK_TOKEN)
                if current_token == config.EOS_TOKEN:
                    break
                decoded.append(current_token)
            return ' '.join(decoded)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained NMT model with BLEU score.')
    parser.add_argument('--model_type', type=str, required=True, choices=['no_attention', 'attention'],
                        help='Type of model to evaluate.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file.')
    parser.add_argument('--data_path', type=str, default=config.val_data_path,
                        help='Path to validation/test data file (default: val_sentence_pairs.txt).')
    parser.add_argument('--max_enc_len', type=int, default=None,
                        help='Maximum encoder sequence length (if not provided, computed from validation data).')
    parser.add_argument('--max_dec_len', type=int, default=None,
                        help='Maximum decoder sequence length (if not provided, computed from validation data).')
    args = parser.parse_args()

    device = config.device
    print(f"Using device: {device}")

    # Load tokenizers
    source_tokenizer = load_tokenizer(config.source_tokenizer_path)
    target_tokenizer = load_tokenizer(config.target_tokenizer_path)

    source_vocab_size = len(source_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1

    # Load model
    model = load_model(args.model_type, args.model_path,
                       source_vocab_size, target_vocab_size, device)

    # Load validation data
    with open(args.data_path, 'r') as f:
        data = [line.rstrip() for line in f]

    # Preprocess to get source sentences and reference targets
    SEPARATOR = '<sep>'
    sources, references_raw = zip(*[pair.split(SEPARATOR) for pair in data])
    # Preprocess and tag references for tokenization (we need original untagged for BLEU)
    references_untagged = [preprocess_sentence(s) for s in references_raw]
    # For decoding we need tagged sequences to compute max length if not provided
    tagged_refs = tag_target_sentences(references_untagged)

    # Determine max lengths if not provided
    if args.max_enc_len is None or args.max_dec_len is None:
        # We need to tokenize to get lengths
        source_seqs = source_tokenizer.texts_to_sequences([preprocess_sentence(s) for s in sources])
        target_seqs = target_tokenizer.texts_to_sequences(tagged_refs)
        max_enc_len = max(len(s) for s in source_seqs) if args.max_enc_len is None else args.max_enc_len
        max_dec_len = max(len(s) for s in target_seqs) if args.max_dec_len is None else args.max_dec_len
        print(f"Computed max_enc_len = {max_enc_len}, max_dec_len = {max_dec_len}")
    else:
        max_enc_len = args.max_enc_len
        max_dec_len = args.max_dec_len

    # Generate translations
    hypotheses = []
    references = []  # list of lists of references (sacrebleu expects list of references per sentence)
    print("Translating validation sentences...")
    for src, ref in tqdm(zip(sources, references_untagged), total=len(sources)):
        # Translate
        hyp = translate_sentence(src, model, source_tokenizer, target_tokenizer,
                                 max_enc_len, max_dec_len, device)
        hypotheses.append(hyp)
        references.append([ref])  # sacrebleu wants a list of reference strings per sentence

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print(f"\nBLEU score = {bleu.score:.2f}")
    print(bleu)

if __name__ == "__main__":
    main()