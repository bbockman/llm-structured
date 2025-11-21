import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm.model import TinyDecoder
from data.load_shard import load_synth_shards
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("gpt2")

special_langs = [f"<lang:{x}>" for x in ["en","es","ja","fr","zh","de"]]
tokenizer.add_special_tokens({
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<sep>", *special_langs]
})
tokenizer.pad_token = "<pad>"
PAD_ID = tokenizer.pad_token_id

print(f"Tokenizer vocab size: {len(tokenizer)}")

# ============================================================
# MEMORY TRACKING
# ============================================================
def print_gpu_memory(label, device):
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"[{label}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")

# ============================================================
# TRAINING LOOP
# ============================================================
def train_tinydecoder_lm(
    epochs=1,
    batch_size=1,
    lr=1e-4,
    max_seq_len=2048,
    save_path=None
):
    print("Loading dataset...")
    ds = load_synth_shards(
        num_shards=1,
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
        pattern_prefix="data",
        total_shards=7
    )
    print(f"Loaded {len(ds)} examples")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # ✅ Set to 0 to avoid worker memory issues
        collate_fn=data_collator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print_gpu_memory("Before model", device)
    
    model = TinyDecoder(
        vocab_size=len(tokenizer),
        d_model=384,
        n_layers=1,
        n_heads=8,
        d_ff=1536,
        max_seq=max_seq_len
    ).to(device)

    print(model)
    count_parameters(model)
    
    print_gpu_memory("After model", device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    
    print_gpu_memory("After optimizer", device)

    gpu_handle = init_gpu_monitor()
    get_gpu_stats(gpu_handle)

    print("\nStarting training...")
    best_loss = float("inf")
    best_state = model.state_dict()

    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            if step % 10 == 0:
                print_gpu_memory(f"Step {step} - Start", device)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            if step % 10 == 0:
                print(f"Batch shape: {input_ids.shape}")
                print_gpu_memory(f"Step {step} - After .to(device)", device)

            optimizer.zero_grad()

            loss = model.compute_loss(input_ids, attention_mask=attention_mask, labels=input_ids)
            
            if step % 10 == 0:
                print(f"Loss: {loss.item():.4f}")
                print_gpu_memory(f"Step {step} - After forward", device)

            loss.backward()
            
            if step % 10 == 0:
                print_gpu_memory(f"Step {step} - After backward", device)

            optimizer.step()
            
            if step % 10 == 0:
                print_gpu_memory(f"Step {step} - After optimizer", device)

            running_loss += loss.item()
            
            # ✅ CRITICAL: Delete tensors explicitly
            del input_ids, attention_mask, loss
            
            # ✅ Clear cache every 10 steps
            if step % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print_gpu_memory(f"Step {step} - After cleanup", device)
                print("-" * 80)

            if step % 50 == 0:
                print(f"Epoch {epoch+1} Step {step} Loss: {running_loss/(step+1):.4f}")
            
            # ✅ Stop after 100 steps to see the pattern
            if step >= 100:
                print("\nStopping at step 100 for analysis")
                break

        stats = get_gpu_stats(gpu_handle)
        print(f"GPU Util: {stats['gpu_util']}%, Mem: {stats['mem_used']:.1f}/{stats['mem_total']:.1f} MB")

        epoch_loss = running_loss / (step + 1)
        print(f"Epoch {epoch+1}/{epochs} LM Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

        scheduler.step()

    elapsed = time.perf_counter() - start
    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Best epoch loss = {best_loss:.4f}")

    if save_path:
        torch.save(best_state, save_path)
        print(f"Saved best model to {save_path}")

    return model, best_state, best_loss

if __name__ == "__main__":
    train_tinydecoder_lm(
        epochs=1,
        batch_size=8,
        lr=1e-4,
        save_path="tinydecoder_lm_best.pth"
    )