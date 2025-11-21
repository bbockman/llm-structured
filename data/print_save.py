import os
from datasets import load_dataset, disable_caching
from torch.utils.data import DataLoader

# Disable all caching - process in-memory only
disable_caching()

# Load the already-processed Arrow files (Stage 1 training data)
PROCESSED_BASE = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted/"

from load_shard import load_synth_shards
# Load directly - this won't create duplicates since it's already in the datasets cache format
ds = load_synth_shards(num_shards=7, base_path=PROCESSED_BASE, pattern_prefix="data", total_shards=7)

print(f"Loaded {len(ds)} examples")
print(f"Columns: {ds.column_names}")

# ---------------------------
# Iterate: Disk → RAM shard → Device batches
# ---------------------------

print("Starting iteration...")

from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace with your model name

special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",  # this creates a token, but we must also tell HF it's the pad
    "additional_special_tokens": ["<sep>", *special_langs]
})
tokenizer.pad_token = "<pad>"

collate_fn = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal LM
)

loader = DataLoader(
    ds,
    batch_size=8,
    num_workers=12,
    collate_fn=collate_fn
)

import torch

max_id = 0
for batch in loader:
    ids = batch['input_ids']
    # convert to tensor if it's still a list
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
    batch_max = ids.max().item()
    if batch_max > max_id:
        max_id = batch_max

print("Max token ID in dataset:", max_id)

for i, batch in enumerate(loader):
    print(f"Batch {i+1}: {batch['input_ids'].shape[0]} examples")
    print(f"  Sample IDs: {batch['input_ids'][0][:20]}")  # first 20 token IDs
    if i >= 10:
        break


print("\n✓ Pipeline ready")# Disable all caching - process in-memory only