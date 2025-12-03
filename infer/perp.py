# one-off debug run
from llm.model_flash import get_current_model
from tokenizer.pleias_tok import PleiasTokenizer
import torch
from torch.utils.data import DataLoader
from data.load_shard import load_synth_shards
from data.colate import sp_pad_collator

tokenizer = PleiasTokenizer().base
model = get_current_model(len(tokenizer))
ds = load_synth_shards(num_shards=1, total_shards=10, pattern_prefix="data")
loader = DataLoader(ds, batch_size=2, collate_fn=sp_pad_collator)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
checkpoint = torch.load("tinydecoder_lm_best_134mlp.pth", map_location=device)
model.load_state_dict(checkpoint)

model.eval()
total_loss, total_tokens = 0.0, 0

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss = model.compute_loss(
            input_ids,
            pad_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        # loss is per-token average over non-pad
        n_tokens = (batch["attention_mask"] > 0).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

import math
val_nll = total_loss / total_tokens
val_ppl = math.exp(val_nll)
print(f"Val NLL: {val_nll:.4f}, Val PPL: {val_ppl:.2f}")

@torch.no_grad()
def eval_continuation(model, tokenizer, examples, max_gen=64):
    model.eval()
    total_acc = 0.0
    total_tokens = 0

    for ex in examples:
        ids = torch.tensor(ex["input_ids"]).unsqueeze(0).to(device)
        prompt_len = min(50, ids.size(1) // 2)
        prompt = ids[:, :prompt_len]
        target = ids[:, prompt_len:prompt_len + max_gen]

        # greedy for now
        cur = prompt.clone()
        for _ in range(target.size(1)):
            logits = model(cur)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_token], dim=1)

        gen = cur[:, prompt_len:prompt_len + max_gen]

        same = (gen == target).sum().item()
        total_acc += same
        total_tokens += target.numel()

    print(f"Next-token exact match acc over {total_tokens} tokens: {total_acc / total_tokens:.4f}")

eval_continuation(model, tokenizer, next(iter(loader)), max_gen=64)