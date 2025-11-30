import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from llm.model import TinyDecoder
from data.load_shard import load_synth_shards

# ============================================================
# TOKENIZER
# ============================================================
from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base
PAD_ID = tokenizer.pad_token_id

print(f"Tokenizer vocab size: {len(tokenizer)}")

# ============================================================
# MEMORY PROFILER
# ============================================================
def print_memory_stats(label, device):
    """Print detailed GPU memory breakdown"""
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    free = (torch.cuda.get_device_properties(device).total_memory / 1024**3) - allocated
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Free:      {free:.2f} GB")
    print(f"{'='*60}")

def get_tensor_memory():
    """Get memory usage of all live tensors"""
    tensors = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                size = obj.element_size() * obj.nelement() / 1024**2  # MB
                dtype = str(obj.dtype)
                shape = str(tuple(obj.shape))
                key = f"{dtype}_{shape}"
                
                if key not in tensors:
                    tensors[key] = {"count": 0, "total_mb": 0}
                tensors[key]["count"] += 1
                tensors[key]["total_mb"] += size
        except:
            pass
    
    # Sort by total memory
    sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["total_mb"], reverse=True)
    
    print("\nTop 10 Tensor Types by Memory:")
    print(f"{'Type':<40} {'Count':<10} {'Total MB':<10}")
    print("-" * 60)
    for key, info in sorted_tensors[:10]:
        print(f"{key:<40} {info['count']:<10} {info['total_mb']:<10.1f}")

# ============================================================
# TRAINING WITH PROFILING
# ============================================================
def train_tinydecoder_lm(
    epochs=1,
    batch_size=8,
    lr=1e-4,
    max_seq_len=2048,
    save_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print_memory_stats("Initial Memory", device)

    # ----------------------------------
    # Load dataset
    # ----------------------------------
    print("\nLoading dataset...")
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
        shuffle=False,
        num_workers=0,  # Keep 0 for debugging
        collate_fn=data_collator
    )

    # ----------------------------------
    # Model
    # ----------------------------------
    print("\nCreating model...")
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
    
    print_memory_stats("After Model Load", device)

    # ----------------------------------
    # Optimizer
    # ----------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    
    print_memory_stats("After Optimizer Init", device)

    # ----------------------------------
    # Training loop with detailed profiling
    # ----------------------------------
    print("\nStarting training...")
    best_loss = float("inf")
    best_state = None

    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            # Profile every step for first 100 steps
            # torch.save({
            #     'input_ids': batch["input_ids"],
            #     'attention_mask': batch["attention_mask"],
            #     'step': step,
            #     'batch_shape': batch["input_ids"].shape
            # }, f'batch_step_{step}.pt')
            # print(f"✓ Saved batch to batch_step_{step}.pt")
            try:
                if step > 100 or step % 10 == 0:
                    print(f"\n{'='*80}")
                    print(f"STEP {step}")
                    print(f"{'='*80}")
                    
                    # Batch info
                    batch_shape = batch["input_ids"].shape
                    seq_len = batch_shape[1]
                    total_tokens = batch_shape[0] * batch_shape[1]
                    print(f"Batch shape: {batch_shape} ({total_tokens:,} tokens)")
                    
                    print_memory_stats(f"Before moving to GPU (Step {step})", device)
                
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                print(f"Batch {step+1}: {batch['input_ids'].shape[0]} examples")
                print(f"  Sample IDs: {batch['input_ids'][0][:20]}")  # first 20 token IDs
                if step > 100 or step % 10 == 0:
                    print_memory_stats(f"After moving to GPU (Step {step})", device)

                # Zero grad
                optimizer.zero_grad()
                
                if step > 100 or step % 10 == 0:
                    print_memory_stats(f"After zero_grad (Step {step})", device)

                # Forward pass
                loss = model.compute_loss(input_ids, attention_mask=attention_mask, labels=input_ids)
            
                if step < 100 or step % 50 == 0:
                    print(f"Loss: {loss.item():.4f}")
                    print_memory_stats(f"After loss (Step {step})", device)

                loss.backward()
                
                if step > 100 or step % 10 == 0:
                    print_memory_stats(f"After backward (Step {step})", device)

                # Step
                optimizer.step()
                
                if step > 100 or step % 50 == 0:
                    print_memory_stats(f"After optimizer step (Step {step})", device)

                running_loss += loss.item()
                
                if step > 100 or step % 50 == 0:
                    print_memory_stats(f"After deletion (Step {step})", device)
                    
                    # Show tensor breakdown
                    get_tensor_memory()
                    
                    # Force cleanup
                    torch.cuda.empty_cache()
                    print_memory_stats(f"After empty_cache (Step {step})", device)
                
                # Stop after 200 steps for analysis
                if step >= 200:
                    print(f"\n{'='*80}")
                    print("STOPPING AT STEP 200 FOR ANALYSIS")
                    print(f"{'='*80}")
                    #break
            except Exception as e:
                print(f"Error at step {step}: {e}")
                print(f"\n{'='*80}")
                print(f"CRASH DETECTED at step {step}")
                print(f"{'='*80}")
                
                # Save problematic tensors
                torch.save({
                    'input_ids': input_ids.cpu(),
                    'attention_mask': attention_mask.cpu(),

                    'step': step,
                    'batch_shape': input_ids.shape,
                    'error': str(e)
                }, f'crash_tensors_step_{step}.pt')
                
                print(f"Saved crash data to crash_tensors_step_{step}.pt")
                print(f"Error: {e}")
                
                # Also save as numpy for easier inspection
                import numpy as np
                np.savez(f'crash_data_step_{step}.npz',
                    input_ids=input_ids.cpu().numpy(),
                    attention_mask=attention_mask.cpu().numpy(),
                    logits=logits.cpu().numpy(),
                    step=step
                )
                print(f"Also saved as crash_data_step_{step}.npz")
                raise
            # Cleanup
            del input_ids, attention_mask, loss

        break  # Only 1 epoch for debugging

    elapsed = time.perf_counter() - start
    print(f"\nCompleted in {elapsed:.2f}s")

    return model, best_state, best_loss

if __name__ == "__main__":
    train_tinydecoder_lm(
        epochs=1,
        batch_size=8,
        lr=1e-4,
        save_path=None
    )