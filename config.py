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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
