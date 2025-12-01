from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("PleIAs/Monad")
#print(len(tokenizer))  # 8192
#print(tokenizer.special_tokens_map)

# {'bos_token': '<|begin_of_text|>', 'eos_token': '<|end_of_text|>', 'unk_token': '[UNK]', 'pad_token': '[PAD]'}

class PleiasTokenizer:
    def __init__(self):
        self.tokenizer = tokenizer   # store HF tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt').squeeze(0)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def add_special_tokens(self, tokens_dict):
        self.tokenizer.add_special_tokens(tokens_dict)

    def __len__(self):
        return len(self.tokenizer)

    # let us use the object like a tokenizer directly
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    # optional: clean accessor for HF tokenizer
    @property
    def base(self):
        return self.tokenizer
