from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from load_shard import load_synth_shards
from datasets import disable_caching

disable_caching()

def to_stage1(example):
    if not example.get("synthetic_answer"):
        return {"text": None}  # Skip
    lang_token = f"<lang:{example['language']}>"
    text = f"<bos> {lang_token} {example['synthetic_answer'].strip()} <eos>"
    return {"text": text}

ds = load_synth_shards(num_shards=20)

# Filter for English with answers
ds = ds.filter(
    lambda x: x.get('synthetic_answer') and x.get('language') == 'en',
    num_proc=12
)

# Setup tokenizer
special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>", *special_langs]
})

# Format to stage1
ds = ds.map(
    to_stage1, 
    batched=False, 
    remove_columns=ds.column_names,
    num_proc=12
)

# Get token lengths to determine max_length
def get_length(batch):
    tokens = tokenizer(batch["text"], truncation=False)
    return {"length": [len(ids) for ids in tokens["input_ids"]]}

print("Analyzing token lengths...", flush=True)
ds_with_lengths = ds.map(get_length, batched=True, batch_size=1000, num_proc=12)

import numpy as np
lengths = ds_with_lengths["length"]
print(f"\n=== Token Length Statistics ===", flush=True)
print(f"Mean: {np.mean(lengths):.1f}", flush=True)
print(f"Median: {np.median(lengths):.1f}", flush=True)
print(f"95th percentile: {np.percentile(lengths, 95):.1f}", flush=True)
print(f"99th percentile: {np.percentile(lengths, 99):.1f}", flush=True)
print(f"Max: {np.max(lengths)}", flush=True)

# Filter by TOKEN length (not character length!)
MAX_LENGTH = 2048  # Based on your 99th percentile of 1075
ds_filtered = ds_with_lengths.filter(lambda x: x["length"] <= MAX_LENGTH, num_proc=12)
print(f"Kept {len(ds_filtered):,}/{len(ds_with_lengths):,} examples ({100*len(ds_filtered)/len(ds_with_lengths):.1f}%)", flush=True)

lengths = ds_filtered["length"]
print(f"\n=== Token Length Statistics ===", flush=True)
print(f"Max: {np.max(lengths)}", flush=True)

ds = ds_filtered.remove_columns(["length"])
del ds_with_lengths, ds_filtered, lengths

import gc
gc.collect()

# Tokenize WITHOUT padding to save memory
def tokenize(batch):
    return tokenizer(
        batch["text"],
        max_length=MAX_LENGTH,
        truncation=True
        # Remove padding - add dynamically during training
    )

fake = DataLoader(ds, batch_size=12)
for i, e in enumerate(fake):  # Use enumerate()
    print(f"Batch {i}: {e['text'][0][:20]}")
    if i >= 2:  # Just print first 3 batches
        break
    

print("Tokenizing...", flush=True)
ds = ds.map(
    tokenize,
    batched=False,
    num_proc=12,
    remove_columns=["text"],
)

output_path = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted"
ds.save_to_disk(output_path)
print(f"✓ Saved to {output_path}", flush=True)
print("✓ Pipeline ready", flush=True)