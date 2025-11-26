
def format_for_pretraining(example):
    """
    Handles all SYNTH exercise types according to their intended use.
    Follows Pleias' 'reasoning by design' philosophy.
    """
    query = example.get("query", "").strip()
    reasoning = example.get("synthetic_reasoning", "").strip()
    answer = example.get("synthetic_answer", "").strip()
    exercise = example.get("exercise", "")
    lang = example.get("language", "en")
    
    # Language token
    lang_token = f"<lang:{lang}>"
    
    # Exercise-specific formatting
    parts = ["<bos>", lang_token]
    
    # Query (when present)
    if query:
        parts.append(query)
    
    # Reasoning (the core of SYNTH)
    if reasoning:
        parts.append("<think>")
        parts.append(reasoning)
        parts.append("</think>")
    
    # Answer (when present and appropriate)
    if answer:
        parts.append(answer)
    
    # Only create text if we have content beyond bos + lang
    if len(parts) > 2:
        parts.append("<eos>")
        return {"text": " ".join(parts)}
    else:
        print(f"Skipping empty example for exercise: '{exercise}'")
        return {"text": None}  # Skip empty examples

def format_batch_old(batch):
    """Format batch of examples into text"""
    lang_tokens = [f"<lang:{l}>" for l in batch["language"]]
    answers = batch["synthetic_answer"]
    text = [f"<bos> {lt} {a.strip()} <eos>" for lt, a in zip(lang_tokens, answers)]
    return {"text": text}



# 30B tokens
def to_stage2(example):
    answer = example.get("synthetic_answer", "").strip()
    reasoning = example.get("synthetic_reasoning", "").strip()
    lang_token = f"<lang:{example['language']}>"
    
    if reasoning and answer:
        text = f"<bos> {lang_token} <think> {reasoning} </think> {answer} <eos>"
    elif answer:
        text = f"<bos> {lang_token} {answer} <eos>"
    else:
        return {"text": None}
    
    return {"text": text}

# 60B tokens
def to_stage3(example):
    query = example.get("query", "").strip()
    reasoning = example.get("synthetic_reasoning", "").strip()
    answer = example.get("synthetic_answer", "").strip()
    # lang_token = f"<lang:{example['language']}>"
    # exercise = example.get("exercise", "")
    
    parts = ["<bos> "]
    
    if query:
        parts.append("<query> ")
        parts.append(query)
    
    if reasoning:
        parts.append("<think> ")
        parts.append(reasoning)
    
    if answer:
        parts.append("<answer> ")
        parts.append(answer)
    
    # Include process-focused tasks (editing, memorization, constrained writing)
    if len(parts) > 1:
        parts.append("<eos>")
        return {"text": " ".join(parts)}
    else:
        return {"text": None}

# 80B tokens
# Stage 4: same as 3, but now include ALL examples including:
# - Negative queries (no answer)
# - Multilingual
# - All exercise types
# This is the format_for_pretraining function above
#  


from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from load_shard import load_synth_shards
from datasets import disable_caching

disable_caching()
#BASE = "/mnt/xd/ml/hf/datasets/PleIAs___synth/default/0.0.0/6ebe6a97043747aa5f2232ea1182841c4a6afcb0/"; 
#data_files = { "train": [f"{BASE}synth-train-{i:05d}-of-00500.arrow" for i in range(20)] } 
#ds = load_dataset("arrow", data_files=data_files, split="train")




# special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print(tokenizer.special_tokens_map)

tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": [
        "<sep>"
    ]
})

# 20B tokens
def to_stage1(example):
    if not example.get("synthetic_answer"):
        return {"text": None}  # Skip
    # lang_token = f"<lang:{example['language']}>"
    text = f"<bos> {example['synthetic_answer'].strip()} <eos>"
    return {"text": text}

def tokenize(example):
    # Single example
    tokens = tokenizer(
        example["text"],
        add_special_tokens=False,
        truncation=False,
        padding=False
    )
    tokens["length"] = len(tokens["input_ids"])
    return tokens

print("Loading dataset...")
ds = load_synth_shards(num_shards=20)

print("Filtering for English with answers...")
ds = ds.filter(
    lambda x: x.get('synthetic_answer') and x.get('language') == 'en',
    num_proc=12
)

MAX_LENGTH = 1024

print("Formatting to stage1...")
ds = ds.map(
    to_stage1, 
    batched=False, 
    remove_columns=ds.column_names,
    num_proc=12
)

print("Tokenizing...")
ds = ds.map(
    tokenize,
    batched=False,
    num_proc=12,
    remove_columns=["text"],
)

print("Filtering by token length...")
ds = ds.filter(
    lambda x: x["length"] <= MAX_LENGTH,
    num_proc=12
)

ds = ds.remove_columns(["length"])


output_path = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted"
ds.save_to_disk(output_path)
print(f"✓ Saved to {output_path}")


for example in ds.select(range(10)):
    print(f"  Sample: {tokenizer.decode(example['input_ids'], skip_special_tokens=False)}")
    print(f" ILength: {len(example['input_ids'])}")
    print(f"  IDs: {example['input_ids']}")
    print(f" MLength: {len(example['attention_mask'])}")
    print(f"  Masked: {example['attention_mask']}")


print("\n✓ Pipeline ready")