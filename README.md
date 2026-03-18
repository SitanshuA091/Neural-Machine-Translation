# Neural Machine Translation (NMT)

This repository contains a Neural Machine Translation (NMT) project implemented in Python. The project includes a Jupyter Notebook (`NMTranslation.ipynb`) that demonstrates the end-to-end process of building a French-to-English translation model using sequence modeling techniques. The model includes Luong attention and scaled dot-product attention mechanisms.

**DATASET INFO**
- Dataset used - Tatoeba - French to English sentence pairs
- No. of Sentence Pairs - 400,000 +

----

### **File Descriptions**
- `config.py` - Centralized configuration for hyperparameters, file paths, and device settings
- `train.py` - Main script that loads data, builds tokenizers, trains both models, and saves them
- `inference.py` - Provides translation functions and loads saved models for inference

**Models Directory**
- `attention.py` - Implements Luong attention mechanism for the attention-based decoder
- `encoder.py` - Contains both EncoderNoAttention and EncoderWithAttention classes
- `decoder.py` - Contains both DecoderNoAttention and DecoderWithAttention classes
- `seq2seq.py` - Wraps encoder-decoder pairs into complete Seq2Seq models

**Utils Directory**
- `tokenizer.py` - Custom tokenizer class for converting text to sequences and vice versa
- `preprocessing.py` - Functions for Unicode normalization, sentence preprocessing, and dataset processing
- `dataset.py` - Masked cross-entropy loss and DataLoader creation utilities



  

