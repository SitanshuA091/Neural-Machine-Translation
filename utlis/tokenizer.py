import json
import numpy as np

class Tokenizer:
    def __init__(self, oov_token='<unk>'):
        self.word_index = {}
        self.index_word = {}
        self.oov_token = oov_token
        self.oov_index = 1
        self.word_index[oov_token] = self.oov_index
        self.index_word[self.oov_index] = oov_token
        self.next_index = 2

    def fit_on_texts(self, texts):
        for text in texts:
            words = text.split()
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = self.next_index
                    self.index_word[self.next_index] = word
                    self.next_index += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = text.split()
            seq = [self.word_index.get(w, self.oov_index) for w in words]
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            words = [self.index_word.get(idx, self.oov_token) for idx in seq]
            texts.append(' '.join(words))
        return texts

    def to_json(self):
        return {
            'word_index': self.word_index,
            'index_word': {int(k): v for k, v in self.index_word.items()},
            'oov_token': self.oov_token
        }

    @staticmethod
    def from_json(config):
        tokenizer = Tokenizer(oov_token=config['oov_token'])
        tokenizer.word_index = config['word_index']
        tokenizer.index_word = {int(k): v for k, v in config['index_word'].items()}
        tokenizer.next_index = max(tokenizer.word_index.values()) + 1
        return tokenizer

def pad_sequences_pytorch(sequences, max_len, pad_value=0):
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded

def generate_decoder_inputs_targets(sentences, tokenizer):
    seqs = tokenizer.texts_to_sequences(sentences)
    decoder_inputs = [s[:-1] for s in seqs]
    decoder_targets = [s[1:] for s in seqs]
    return decoder_inputs, decoder_targets