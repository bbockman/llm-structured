import time
import torch
import gc

from utils import get_gpu_stats, init_gpu_monitor, count_parameters
from torch.utils.data import DataLoader

from data.load_shard import load_synth_shards
from transformers import DataCollatorWithPadding
from llm.model_flash import get_current_model
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.amp import autocast, GradScaler
from tokenizer.pleias_tok import PleiasTokenizer
from utils import get_tensor_memory
from kern.shed import cosine_with_warmup

# ============================================================

tokenizer = PleiasTokenizer().base
print(f"Tokenizer vocab size: {len(tokenizer)}")
WARMUP = 11_000

def train_tinydecoder_lm(
        epochs=4, 
        batch_size=1, 
        lr=1e-4, 
        save_path=None, 
        batch_accum=1, 
        warmup=None, 
        load_path=None,
        global_step=0,
        total_steps=1024*64*2,
        start_shard=0,
        num_shards=1,
        total_shards=10,
        round=1
    ):

    start_step = global_step

    print("Loading dataset...")
    ds = load_synth_shards(
        start_shard=start_shard,
        num_shards=num_shards,
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
        pattern_prefix="data",
        total_shards=total_shards
    )
    print(f"Loaded {len(ds)} examples")
    print(f"Columns: {ds.column_names}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=12, collate_fn=data_collator)

    # ----------------------------------
    # Model
    # ----------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_current_model(vocab_size=len(tokenizer)).to(device)

    print(model)
    count_parameters(model)

    # ----------------------------------
    # GPU monitor
    # ----------------------------------
    gpu_handle = init_gpu_monitor()
    get_gpu_stats(gpu_handle)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    if load_path is not None:
        chkp = torch.load(load_path, map_location=device)
        missing = model.load_state_dict(chkp["model"], strict=False)
        if "optimizer" in chkp and chkp["optimizer"] is not None:
            optimizer.load_state_dict(chkp["optimizer"])
        else:
            print("No optimizer state found in checkpoint.")
        global_step = chkp.get("step", 0)
        print(f"Resumed global step: {global_step}")
            
        print(f"Loaded saved params from {load_path}")
        print("Missing keys:", missing.missing_keys)
        print("Unexpected keys:", missing.unexpected_keys)
 

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    
    global_sdpa_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],set_priority=True)
    global_sdpa_ctx.__enter__()  # manual “start”
    # global_sdpa_ctx.__exit__(None, None, None)  # if you ever want to shut it off

    scaler = GradScaler(device="cuda", enabled=False)  # Disable for now

    start = time.perf_counter()
    inc_time = start

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

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=False):
                loss = model.compute_loss(input_ids, 
                                        pad_id=tokenizer.pad_token_id,
                                        attention_mask=attention_mask, 
                                        labels=input_ids)/batch_accum

            if step % 500 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"Round {round} Step {step} Loss: {loss.item() * batch_accum:.4f} | "
                      f"Mem: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved | "
                      f"Batch: {(time.perf_counter() - inc_time)*(batch_accum):.2f}s")
                inc_time = time.perf_counter()

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            running_loss += loss.item() * batch_accum

            if step % batch_accum == batch_accum - 1 or step == len(loader) - 1:
                loss = loss * batch_accum / (step % batch_accum + 1)
                lr_a = cosine_with_warmup(
                    step=global_step / batch_accum,
                    warmup_steps=WARMUP / batch_accum,
                    total_steps=total_steps / batch_accum,
                    base_lr=lr,
                    min_lr=lr / 10,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr_a
                scaler.step(optimizer)
                scaler.update()
            
            del input_ids, attention_mask, loss
            
            if step % 100 == 0 and step > 0:
                torch.cuda.empty_cache()

            global_step += 1
            
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # GPU Stats (without tensor profile!)
        stats = get_gpu_stats(gpu_handle)
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"\n{'='*80}")
        print(f"Round {round} Summary:")
        print(f"  Loss: {running_loss / len(loader):.4f}")
        print(f"  GPU Util: {stats['gpu_util']}%")
        print(f"  Current Memory: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")
        print(f"  Peak Memory: {max_allocated:.2f}GB")
        print(f"  Steps: {global_step-start_step} this round, {global_step} total")
        print(f"{'='*80}\n")
        
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats(device)

        if save_path:
            checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
        }
            torch.save(checkpoint, save_path)
            print(f"Saved best model to {save_path}")


    elapsed = time.perf_counter() - start
    print(f"\nTraining completed in {elapsed:.2f}s")

    return model, start_shard, num_shards, global_step

if __name__ == "__main__":
    
    warmup = 11_000
    rounds = 4
    epochs = 1
    batch_size = 8
    steps_per_round = 248_632 // batch_size
    batch_accum = 6
    num_shards = 1
    lr_base = 1e-4
    total_steps = steps_per_round * rounds + warmup

    train_tinydecoder_lm(epochs=1,batch_size=batch_size,batch_accum=batch_accum,lr=lr_base,warmup=WARMUP, round=0,
                         save_path="disk/models/llm-scoped/warmup.pth")

    _, start, num, _ = train_tinydecoder_lm(
        epochs=epochs,
        batch_size=batch_size,
        batch_accum=batch_accum,
        lr=lr_base,
        load_path="disk/models/llm-scoped/warmup.pth",
        save_path="disk/models/llm-scoped/params_500autocast.pth",
        total_steps=total_steps,
        start_shard=0,
        num_shards=num_shards,
        round=1
    )
    # start = 0
    # num = 1

    for shards in range(start + num, rounds):
        next_shard = start + num
        print(f"\n\n=== Starting training round {shards+2}, loading shard {next_shard} ===\n\n")
        _, start, num, _ = train_tinydecoder_lm(
            epochs=epochs,
            batch_size=batch_size,
            batch_accum=batch_accum,
            lr=lr_base,
            load_path="disk/models/llm-scoped/params_500autocast.pth",
            save_path="disk/models/llm-scoped/params_500autocast.pth",
            total_steps=total_steps,
            start_shard=next_shard,
            num_shards=num_shards,
            round=shards + 1
        )
    
    print(f"Last round completed: {rounds}, next start shard {rounds}.")