'''
This file implements different encoding types for URL names


'''
import torch

class WordEncoder():
    def __init__(self):
        self.max_url_length = 200
        self.vocab_size = 0

    def encode_char(self, xs):
        encodings = []
        chars = sorted(list(set(''.join(xs))))        # all unique characters
        char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # reserve 0 for padding
        self.vocab_size = len(char_to_idx) + 1

        for x in xs:
            sub_encoding = [char_to_idx.get(c, 0) for i, c in enumerate(x) if i < self.max_url_length]
            if len(sub_encoding) < self.max_url_length:
                sub_encoding.extend([0 for _ in range(self.max_url_length - len(sub_encoding))])
            encodings.append(sub_encoding)       
        
        encodings = torch.tensor(encodings, dtype=torch.int64)

        return encodings
    
    def tokenize(self):

        return
    
    def struct_feature_encoding(self):

        return
    
    def vector_encoding(self):

        return