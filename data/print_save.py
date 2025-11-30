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

from transformers import DataCollatorForLanguageModeling, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base

data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

loader = DataLoader(
    ds,
    batch_size=8,
    num_workers=12,
    collate_fn=data_collator
)


print("Iterating through some batches...")
for i, batch in enumerate(loader):
    print(f"Batch {i+1}: {batch['input_ids'].shape[0]} examples")
    print(f"  Sample Input IDs: {batch['input_ids'][0]}")  
    print(f"  Sample Input IDs Length: {len(batch['input_ids'][0])} tokens") 
    print(f"  Sample Attention Mask: {batch['attention_mask'][0]}")  
    print(f"  Sample Attention Mask Length: {len(batch['attention_mask'][0])} tokens")  
    print(f"  Sample Length: {batch['attention_mask'][0].sum().item()} tokens") 
    print(f"Decoded Sample: {tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)}")
    if i >= 5:
        break


print("\n✓ Pipeline ready")# Disable all caching - process in-memory only