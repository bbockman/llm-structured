from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>"]
})

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
