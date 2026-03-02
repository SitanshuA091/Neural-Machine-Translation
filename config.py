import torch

# Model hyperparameters
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.2

# Training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters
embedding_dim = 128
hidden_dim = 256
dropout = 0.2
batch_size = 32
epochs = 30
learning_rate = 0.001

# Special tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'  

# Paths (can be changed later)
train_data_path = 'train_sentence_pairs.txt'
val_data_path = 'val_sentence_pairs.txt'
source_tokenizer_path = 'source_tokenizer.json'
target_tokenizer_path = 'target_tokenizer.json'
model_no_attn_path = 'model_no_attention.pth'
model_attn_path = 'model_with_attention.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
