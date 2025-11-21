# test_model_realdata.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from llm.model import TinyDecoder
from data.load_shard import load_synth_shards

PROCESSED_BASE = "/mnt/xd/ml/hf/datasets/synth_stage1_formatted/"

# Create tokenizer ONCE at module level
tokenizer = AutoTokenizer.from_pretrained("gpt2")
special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>", *special_langs]
})
tokenizer.pad_token = "<pad>"

print(f"Tokenizer vocab size: {len(tokenizer)}")


def collate_fn(batch):
    """
    Dynamically pad to longest sequence in THIS batch only
    No artificial max_length - saves memory and compute!
    """
    input_ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in batch]
    attention_mask = [torch.ones(len(x['input_ids']), dtype=torch.long) for x in batch]

    # Pad to longest in THIS batch (could be 200, 300, 400, etc.)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def run_realdata_smoke(batch_size=4, max_seq_len=2048):
    # -----------------------------
    # 1. Load your tokenized dataset
    # -----------------------------
    ds = load_synth_shards(
        num_shards=7,
        base_path=PROCESSED_BASE,
        pattern_prefix="data",
        total_shards=7
    )
    print(f"Loaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    # -----------------------------
    # 2. Create DataLoader
    # -----------------------------
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_fn
    )

    # -----------------------------
    # 3. Instantiate model
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyDecoder(
        vocab_size=len(tokenizer),
        d_model=384,
        n_layers=4,
        n_heads=8,
        d_ff=1536,
        max_seq=max_seq_len  # Use parameter
    ).to(device)

    print(f"Model initialized with max_seq={max_seq_len}")

    # -----------------------------
    # 4. Run smoke test batches
    # -----------------------------
    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        print(f"\nBatch {i+1} input shape: {input_ids.shape}")

        # Forward pass
        logits = model(input_ids, attention_mask=attention_mask)
        print(f"Batch {i+1} logits shape: {logits.shape}")

        # Next-token prediction loss
        labels = input_ids.clone()
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        print(f"Loss: {loss.item():.4f}")

        # Backward pass
        loss.backward()
        print("Backward OK")

        if i >= 1:  # Just 2 batches for smoke test
            break

    print("\n✅ Smoke test passed!")


if __name__ == "__main__":
    run_realdata_smoke(batch_size=4, max_seq_len=2048)
