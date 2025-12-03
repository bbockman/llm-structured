# one-off debug run
from llm.model_flash import get_current_model
from tokenizer.pleias_tok import PleiasTokenizer
import torch
from torch.utils.data import DataLoader
from data.load_shard import load_synth_shards
from data.colate import sp_pad_collator

model = get_current_model(len(PleiasTokenizer().base))
ds = load_synth_shards(num_shards=1, total_shards=10, pattern_prefix="data")
loader = DataLoader(ds, batch_size=2, collate_fn=sp_pad_collator)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load("tinydecoder_lm_best_134mlp.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

with torch.no_grad():
    # enable debug only for some layers, e.g. all of them
    for blk in model.blocks:
        blk.attn.debug_attn = True
        blk.attn._attn_stats_done = False

    batch = next(iter(loader))  # small batch
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    _ = model(input_ids, attention_mask=attention_mask)

    # now read stats
    for i, blk in enumerate(model.blocks):
        stats = blk.attn
        if stats.last_head_entropy is None:
            continue
        print(f"\n[Attn stats] layer {i}")
        print("  entropy per head:", stats.last_head_entropy.numpy())
        print("  maxprob per head:", stats.last_head_maxprob.numpy())
