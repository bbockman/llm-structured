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
# TOKENIZER (same as smoke test)
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

import gc
def get_tensor_profile():
    """Get profile of all CUDA tensors"""
    tensors = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
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

def train_tinydecoder_lm(
    epochs=1,
    batch_size=1,
    lr=1e-4,
    max_seq_len=2048,
    save_path=None
):

    # ----------------------------------
    # Load real tokenized dataset
    # ----------------------------------
    print("Loading dataset...")
    ds = load_synth_shards(
        num_shards=1,
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
        pattern_prefix="data",
        total_shards=7
    )
    print(f"Loaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # ✅ No shuffle for repeatability
        num_workers=8,
        collate_fn=data_collator
    )

    # ----------------------------------
    # Model
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoder(
        vocab_size=len(tokenizer),
        d_model=384,
        n_layers=8,
        n_heads=8,
        d_ff=1536,
        max_seq=max_seq_len
    ).to(device)

    model.load_state_dict(torch.load("tinydecoder_lm_best.pth", map_location=device))

    print(model)
    count_parameters(model)

    # ----------------------------------
    # Optimizer + LR schedule
    # ----------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    # ----------------------------------
    # GPU monitor
    # ----------------------------------
    gpu_handle = init_gpu_monitor()
    get_gpu_stats(gpu_handle)
    
    # ----------------------------------
    # Training loop
    # ----------------------------------
    print("\nStarting training...")
    best_loss = float("inf")
    best_state = model.state_dict()

    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            loss = model.compute_loss(input_ids, attention_mask=attention_mask, labels=input_ids)
            
            if step % 50 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"Epoch {epoch+1} Step {step} Loss: {loss.item():.4f} | "
                      f"Mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

            loss.backward()
            
            # ✅ Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            
            # ✅ Explicit cleanup
            del input_ids, attention_mask, loss
            
            # ✅ Clear cache periodically
            if step % 100 == 0 and step > 0:
                torch.cuda.empty_cache()

        # ✅ End of epoch cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # GPU Stats (without tensor profile!)
        stats = get_gpu_stats(gpu_handle)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Loss: {running_loss / len(loader):.4f}")
        print(f"  GPU Util: {stats['gpu_util']}%")
        print(f"  Current Memory: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")
        print(f"  Peak Memory: {max_allocated:.2f}GB")
        print(f"{'='*80}\n")
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats(device)

        epoch_loss = running_loss / len(loader)

        # Track best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()

        scheduler.step()

    # ----------------------------------
    # Finalize
    # ----------------------------------
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