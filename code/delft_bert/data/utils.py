import unicodedata
import json

from pytorch_transformers.tokenization_bert import BertTokenizer


def load_data(data_file):  
    data = list()
    with open(data_file) as f:
        for line in f.readlines():
            ex = json.loads(line)
            data.append(ex)

    return data



def text_tokenize(text, tokenizer, max_seq_length):
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text = tokenized_text[-(max_seq_length - 2):]
    tokens = []
    tokens.append("[CLS]")
    for token in tokenized_text:
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    
    return input_ids, input_mask
