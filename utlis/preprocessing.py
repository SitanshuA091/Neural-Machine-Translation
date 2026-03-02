import re
import unicodedata
from utlis.tokenizer import pad_sequences_pytorch
from utlis.tokenizer import generate_decoder_inputs_targets

def normalize_unicode(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(s):
    s = normalize_unicode(s)
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = s.strip()
    return s

def tag_target_sentences(sentences):
    tagged_sentences = [' '.join(['<sos>', s, '<eos>']) for s in sentences]
    return tagged_sentences

def process_dataset(dataset, source_tokenizer, target_tokenizer, max_encoding_len, max_decoding_len):
    SEPARATOR = '<sep>'
    input_data, output_data = map(list, zip(*[pair.split(SEPARATOR) for pair in dataset]))
    preprocessed_input = [preprocess_sentence(s) for s in input_data]
    preprocessed_output = [preprocess_sentence(s) for s in output_data]
    tagged_preprocessed_output = tag_target_sentences(preprocessed_output)

    encoder_inputs = source_tokenizer.texts_to_sequences(preprocessed_input)
    decoder_inputs, decoder_targets = generate_decoder_inputs_targets(
        tagged_preprocessed_output, target_tokenizer)

    padded_encoder_inputs = pad_sequences_pytorch(encoder_inputs, max_encoding_len)
    padded_decoder_inputs = pad_sequences_pytorch(decoder_inputs, max_decoding_len)
    padded_decoder_targets = pad_sequences_pytorch(decoder_targets, max_decoding_len)

    return padded_encoder_inputs, padded_decoder_inputs, padded_decoder_targets