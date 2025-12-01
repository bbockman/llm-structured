import os
from datasets import load_dataset, disable_caching
from torch.utils.data import DataLoader

# Disable all caching - process in-memory only
disable_caching()

# Load the already-processed Arrow files (Stage 1 training data)
PROCESSED_BASE = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted/"

from load_shard import load_synth_shards
# Load directly - this won't create duplicates since it's already in the datasets cache format
ds = load_synth_shards(num_shards=1, base_path=PROCESSED_BASE, pattern_prefix="data", total_shards=7)

print(f"Loaded {len(ds)} examples")
print(f"Columns: {ds.column_names}")

# ---------------------------
# Iterate: Disk → RAM shard → Device batches
# ---------------------------

print("Starting iteration...")

from torch.utils.data import DataLoader
from tokenizer.sp_tok import SPTokenizer
from data.colate import sp_pad_collator
from torch.utils.data import DataLoader

tokenizer = SPTokenizer("unigram_8k.model")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# Create a collator function that supplies pad_id
collate_fn = lambda batch: sp_pad_collator(batch, pad_id=tokenizer.pad_id)

loader = DataLoader(
    ds,
    batch_size=8,
    num_workers=12,
    collate_fn=collate_fn
)

from tokenizer.sp_tok import decode_for_display
print("Iterating through some batches...")
for i, batch in enumerate(loader):
    print(f"Batch {i+1}: {batch['input_ids'].shape[0]} examples")
    print(f"  Sample Input IDs: {batch['input_ids'][0]}")  
    print(f"  Sample Input IDs Length: {len(batch['input_ids'][0])} tokens") 
    print(f"  Sample Attention Mask: {batch['attention_mask'][0]}")  
    print(f"  Sample Attention Mask Length: {len(batch['attention_mask'][0])} tokens")  
    print(f"  Sample Length: {batch['attention_mask'][0].sum().item()} tokens") 
    print(f"Decoded Sample: {tokenizer.decode(batch['input_ids'][0])}")
    print("Decoded tokens (one per line):")
    for tok_id in batch["input_ids"][0]:
        decoded_token = tokenizer.decode([int(tok_id)])
        print(decoded_token)
    print("Decoded for display:")
    print(decode_for_display(tokenizer.decode(batch['input_ids'][0])))
    if i >= 5:
        break


print("\n✓ Pipeline ready")# Disable all caching - process in-memory only
