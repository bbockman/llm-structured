# one-off debug run
from llm.model_flash import get_current_model
from tokenizer.pleias_tok import PleiasTokenizer
import torch
from torch.utils.data import DataLoader
from data.load_shard import load_synth_shards

tokenizer = PleiasTokenizer().base
model = get_current_model(vocab_size=len(tokenizer))
ds = load_synth_shards(
        num_shards=1,
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
        pattern_prefix="data",
        total_shards=10
    )

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

loader = DataLoader(ds, batch_size=6, shuffle=False, num_workers=12, collate_fn=data_collator)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load("disk/models/llm-scoped/params_snap.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

with torch.no_grad():
    # enable debug only for some layers, e.g. all of them
    for blk in model.blocks:
        blk.debug_rms = True
        blk.debug_branch = True
        blk._debug_done = False

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model(input_ids, attention_mask=attention_mask)
        if step >= 5:
            break