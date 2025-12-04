import time
import torch
import gc

from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from torch.utils.data import DataLoader

from data.load_shard import load_synth_shards
from transformers import DataCollatorWithPadding

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

def train_tinydecoder_lm(epochs=4, batch_size=1, lr=1e-4, save_path=None, batch_accum=1, warmup=None, load_path=None):

    print("Loading dataset...")
    ds = load_synth_shards(
        num_shards=1,
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
        pattern_prefix="data",
        total_shards=10
    )
    print(f"Loaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=data_collator)

    # ----------------------------------
    # Model
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from llm.model_flash import get_current_model
    model = get_current_model(vocab_size=len(tokenizer)).to(device)

    # print_param_ids(model)  

    from torch.backends.cuda import sdp_kernel
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

    if warmup is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs-1, eta_min=lr / 10)
    else:
        scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    
    if load_path is not None:
        chkp = torch.load(load_path, map_location=device)
        missing = model.load_state_dict(chkp["model"], strict=False)
        if "optimizer" in chkp and chkp["optimizer"] is not None:
            optimizer.load_state_dict(chkp["optimizer"])
        else:
            print("No optimizer state found in checkpoint.")
        if "scheduler" in chkp and chkp["scheduler"] is not None:
            scheduler.load_state_dict(chkp["scheduler"])
        else:
            print("No scheduler state found in checkpoint.")
            
        print(f"Loaded saved params from {load_path}")
        print("Missing keys:", missing.missing_keys)
        print("Unexpected keys:", missing.unexpected_keys)

    start = time.perf_counter()
    inc_time = start

    if warmup is not None and load_path is not None:
        print("Warning: Both warmup and load_path are set. Warmup will override the scheduler from the checkpoint.")
    elif warmup is not None:
        print(f"Starting warmup for {warmup} steps...")
    elif load_path is not None:
        print(f"Resuming training from {load_path}...")
    else:
        print(f"Starting fresh training without warmup...")

    best_loss = float("inf")
    best_state = model.state_dict()
    optim_state = optimizer.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        get_tensor_memory() 

        for step, batch in enumerate(loader):
            if warmup is not None and (step >= warmup or epoch * len(loader) + step >= warmup):
                break

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
                      f"Mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved | "
                      f"Batch: {(time.perf_counter() - inc_time)*(batch_accum):.2f}s")
                inc_time = time.perf_counter()

            loss.backward()
            loss_d = loss.item()

            # ✅ Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # only changed loss scaling since initial git commit with flash model save
            if step % batch_accum == batch_accum - 1 or step == len(loader) - 1:
                loss_d = loss_d * batch_accum / (step % batch_accum + 1)
                optimizer.step()

            running_loss += loss_d 
            
            # ✅ Explicit cleanup
            del input_ids, attention_mask, loss, loss_d
            
            # ✅ Clear cache periodically
            if step % 100 == 0 and step > 0:
                torch.cuda.empty_cache()
            
            if warmup is not None:
                scheduler.step()

        if warmup is None:
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            optim_state = optimizer.state_dict()
            for state in optim_state["state"].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.detach().cpu().clone()


    # ----------------------------------
    # Finalize
    # ----------------------------------
    elapsed = time.perf_counter() - start
    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Best epoch loss = {best_loss:.4f}")


    if save_path:
        checkpoint = {
        "model": best_state,
        "optimizer": optim_state,
        #"scheduler": scheduler.state_dict() if scheduler is not None else None,
        #"epoch": epoch,
        #"step": global_step,
    }
        torch.save(checkpoint, save_path)
        print(f"Saved best model to {save_path}")

    return model, best_state, best_loss

if __name__ == "__main__":
    train_tinydecoder_lm(epochs=1,batch_size=6,batch_accum=6,lr=1e-4,save_path="warmup.pth",warmup=11_000)

    train_tinydecoder_lm(
        epochs=4,
        batch_size=6,
        batch_accum=6,
        lr=1e-4,
        load_path="warmup.pth",
        save_path="params_134postnorms.pth"
    )