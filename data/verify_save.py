import os
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, disable_caching
from torch.utils.data import DataLoader

# Disable all caching
disable_caching()

def format_batch(batch):
    """Format batch of examples into text"""
    lang_tokens = [f"<lang:{l}>" for l in batch["language"]]
    answers = batch["synthetic_answer"]
    text = [f"<bos> {lt} {a.strip()} <eos>" for lt, a in zip(lang_tokens, answers)]
    return {"text": text}

BASE = "/mnt/xd/ml/hf/datasets/PleIAs___synth/default/0.0.0/6ebe6a97043747aa5f2232ea1182841c4a6afcb0/";
arrow_files = [f"{BASE}synth-train-{i:05d}-of-00500.arrow" for i in range(20)]

# Load Arrow files directly into memory with PyArrow
print("Loading Arrow files into memory...")
tables = []
for file in arrow_files:
    with pa.memory_map(file, 'r') as source:
        reader = ipc.open_stream(source)
        tables.append(reader.read_all())

table = pa.concat_tables(tables)
print(f"Loaded {len(table)} rows into memory")

ds = Dataset(table)

# Check what we have BEFORE filtering
print(f"\nBefore filtering: {len(ds)} examples")
print(f"Columns: {ds.column_names}")

# Sample a few to see languages
print("\nFirst 5 examples:")
for i in range(min(5, len(ds))):
    print(f"  {i}: language={ds[i].get('language')}, has_answer={bool(ds[i].get('synthetic_answer'))}")

# Count languages and missing answers
from collections import Counter
languages = Counter(ds['language'])
missing_answers = sum(1 for x in ds if not x.get('synthetic_answer'))

print(f"\nLanguage distribution:")
for lang, count in languages.most_common(10):
    print(f"  {lang}: {count}")
print(f"\nMissing synthetic_answer: {missing_answers}")

# Filter
print("\nFiltering...")
ds_filtered = ds.filter(
    lambda x: x.get('synthetic_answer') and x.get('language') == 'en',
    keep_in_memory=True
)

print(f"After filtering: {len(ds_filtered)} examples")
print(f"Filtered out: {len(table) - len(ds_filtered)} examples ({100 * (len(table) - len(ds_filtered)) / len(table):.1f}%)")

# Format
print("\nFormatting...")
ds_formatted = ds_filtered.map(
    format_batch, 
    batched=True, 
    batch_size=10000,
    remove_columns=ds_filtered.column_names,
    keep_in_memory=True
)

print(f"Formatted {len(ds_formatted)} examples")
print(f"Columns: {ds_formatted.column_names}")

# Calculate expected size
avg_text_len = sum(len(ds_formatted[i]['text']) for i in range(min(1000, len(ds_formatted)))) / min(1000, len(ds_formatted))
expected_size_gb = (len(ds_formatted) * avg_text_len) / (1024**3)
print(f"\nAverage text length: {avg_text_len:.0f} chars")
print(f"Expected uncompressed size: {expected_size_gb:.2f} GB")
print(f"Expected compressed size (50%): {expected_size_gb * 0.5:.2f} GB")

# Save to final location
output_path = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted"
print(f"\nSaving to {output_path}...")
ds_formatted.save_to_disk(output_path)

# Check the saved files
import subprocess
result = subprocess.run(['du', '-sh', output_path], capture_output=True, text=True)
saved_size = result.stdout.split()[0]
print(f"✓ Saved directory size: {saved_size}")

# Load it back and verify
print("\nVerifying saved dataset...")
from datasets import load_from_disk
ds_loaded = load_from_disk(output_path)
print(f"Loaded back: {len(ds_loaded)} examples")
print(f"Columns: {ds_loaded.column_names}")

if len(ds_loaded) != len(ds_formatted):
    print(f"⚠️ WARNING: Saved {len(ds_formatted)} but loaded {len(ds_loaded)}")
else:
    print(f"✓ Save verified - all {len(ds_loaded):,} examples present")

# Spot check a few examples
print("\nSpot checking examples...")
for i in [0, len(ds_loaded)//2, len(ds_loaded)-1]:
    print(f"  [{i}]: {ds_loaded[i]['text'][:80]}...")

# Verify with DataLoader
print("\nTesting DataLoader...")
loader = DataLoader(ds_loaded, batch_size=16, num_workers=8)

total_examples = 0
for i, batch in enumerate(loader):
    total_examples += len(batch['text'])
    if i < 3:
        print(f"Batch {i+1}: {len(batch['text'])} examples")
        print(f"  Sample: {batch['text'][0][:80]}...")
    if i >= 10:
        break

print(f"\n✓ DataLoader tested: {total_examples} examples in first 11 batches")
print(f"✓ Pipeline ready with {len(ds_loaded):,} total examples")