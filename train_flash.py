import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import gc

from llm import model
from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from torch.utils.data import DataLoader

from llm.model_flash import TinyDecoder
from data.load_shard import load_synth_shards
from transformers import DataCollatorWithPadding

# ============================================================
# TOKENIZER (same as smoke test)
# ============================================================
from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base

print(f"Tokenizer vocab size: {len(tokenizer)}")

def get_tensor_memory():
    import gc
    import torch
    from collections import defaultdict

    summary = defaultdict(lambda: {"count": 0, "total_mb": 0.0})
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                key = (str(obj.dtype), tuple(obj.shape), str(obj.device))
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                summary[key]["count"] += 1
                summary[key]["total_mb"] += size_mb
        except Exception:
            pass

    print("\n--- Tensor Memory Summary ---")
    for key, val in summary.items():
        dtype, shape, device = key
        print(f"{val['count']:3d}x {dtype} {shape} on {device}: {val['total_mb']:.2f} MB")
    print("--- End of Summary ---\n")

def print_param_ids(model):
    print("\n--- Model Parameter IDs ---")
    for name, param in model.named_parameters():
        print(f"{name}: id={id(param.data)}, shape={tuple(param.shape)}, device={param.device}")
    print("--- End ---\n")

def train_tinydecoder_lm(
    epochs=4,
    batch_size=1,
    lr=1e-4,
    max_seq_len=1024,
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
        total_shards=10
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
        num_workers=12,
        collate_fn=data_collator
    )

    # ----------------------------------
    # Model
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDecoder(
        vocab_size=len(tokenizer),
        d_model=1024,
        n_layers=12,
        n_heads=16,
        d_ff=4096,
        max_seq=max_seq_len
    ).to(device)

    # print_param_ids(model)

    from torch.backends.cuda import sdp_kernel
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=True,
    )


    # model.load_state_dict(torch.load("tinydecoder_lm_best.pth", map_location=device))

    print(model)
    count_parameters(model)

    # ----------------------------------
    # GPU monitor
    # ----------------------------------
    gpu_handle = init_gpu_monitor()
    get_gpu_stats(gpu_handle)
    # torch.cuda.memory_unified()
    
    # ----------------------------------
    # Training loop
    # ----------------------------------
    print("\nStarting training...")
    best_loss = float("inf")
    best_state = model.state_dict()
    batch_accum = 4

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )


    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)
    
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        get_tensor_memory() 

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if step % batch_accum == 0:
                optimizer.zero_grad()

            loss = model.compute_loss(input_ids, 
                                        pad_id=tokenizer.pad_token_id,
                                        attention_mask=attention_mask, 
                                        labels=input_ids)/batch_accum

            if step % 50 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"Epoch {epoch+1} Step {step} Loss: {loss.item() * batch_accum:.4f} | "
                      f"Mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

            loss.backward()

            # ✅ Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # only changed loss scaling since initial git commit with flash model save
            if step % batch_accum == batch_accum - 1 or step == len(loader) - 1:
                loss = loss * batch_accum / (step % batch_accum + 1)
                optimizer.step()

            running_loss += loss.item() * batch_accum 
            
            # ✅ Explicit cleanup
            del input_ids, attention_mask, loss
            
            # ✅ Clear cache periodically
            if step % 100 == 0 and step > 0:
                torch.cuda.empty_cache()

        scheduler.step()
        
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
        epochs=4,
        batch_size=8,
        lr=1e-4,
        save_path="tinydecoder_lm_best_temp.pth"
    )