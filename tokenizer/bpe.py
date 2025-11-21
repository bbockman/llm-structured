from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = ByteLevelBPETokenizer(
    "your_qwen2_tokenizer-qwen2-vocab.json",
    "your_qwen2_tokenizer-qwen2-merges.txt"
)

fast_tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tok.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>"})
