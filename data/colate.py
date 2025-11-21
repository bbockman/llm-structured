from datasets import load_from_disk
from transformers import AutoTokenizer

ds_stage1 = load_from_disk("/mnt/xd/datasets/synth_stage1_20B")

# can feed directly into PyTorch DataLoader
from torch.utils.data import DataLoader

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def collate_fn(batch):
    # convert list of dicts -> tensors, padding, etc.
    texts = [x["text"] for x in batch]
    # Example: tokenizer batch encode (if tokenizer defined)
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

loader = DataLoader(ds_stage1, batch_size=64, shuffle=True, collate_fn=collate_fn)
for batch in loader:
    print(batch)
    break