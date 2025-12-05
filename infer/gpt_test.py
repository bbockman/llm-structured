import torch
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tokenizer.pleias_tok import PleiasTokenizer
tokenizer = PleiasTokenizer().base
print(f"Tokenizer vocab size: {len(tokenizer)}")

from llm.model_flash import get_current_model
from data.load_shard import load_synth_shards

# Load model/checkpoint
model = get_current_model(vocab_size=len(tokenizer)).to(device)
checkpoint = torch.load("disk/models/llm-scoped/params_500autocast.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print("Model loaded successfully!")

@torch.no_grad()
def generate_from_ids(
    prompt_ids,
    max_total_tokens=256,
    temperature=0.1,
    top_k=10,
    top_p=0.98,
    ngram_size=3,
    repetition_penalty=1.92,
):
    """Generate continuation starting from a tensor of token ids."""
    input_ids = prompt_ids.unsqueeze(0).to(device)  # [1, T]
    generated_tokens = []
    history = deque(maxlen=ngram_size * 4)

    # seed history with prompt (for repetition control)
    for tok in prompt_ids.tolist():
        history.append(tok)

    max_new_tokens = max_total_tokens - input_ids.shape[1]
    if max_new_tokens <= 0:
        return input_ids[0]

    for _ in range(max_new_tokens):
        logits = model(input_ids)[0, -1, :] / max(temperature, 1e-5)

        # basic n‑gram repetition penalty
        if len(history) >= ngram_size:
            recent = list(history)
            prefix = tuple(recent[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
            for tok_id in set(recent):
                candidate = prefix + (tok_id,)
                for i in range(len(recent) - ngram_size + 1):
                    if tuple(recent[i : i + ngram_size]) == candidate:
                        logits[tok_id] /= repetition_penalty
                        break

        # top‑k
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, k)[0][..., -1, None]
            logits[logits < kth_vals] = float("-inf")

        # top‑p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")
            logits = logits.clone()
            logits[sorted_indices] = sorted_logits

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tok_id = next_token.item()

        generated_tokens.append(tok_id)
        history.append(tok_id)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        if tok_id == tokenizer.eos_token_id:
            break

    return input_ids[0]  # [T_total]

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TRAIN DATA EVAL - how well does it follow shard 0?")
    print("=" * 80)

    ds = load_synth_shards(
        num_shards=1,
        total_shards=10,
        pattern_prefix="data",
        base_path="/mnt/xd/ml/hf/datasets/synth_stage1_formatted/",
    )

    num_examples = 10          # how many training examples to inspect
    prompt_tokens = 64         # number of tokens from each example to feed as prompt
    max_total_tokens = 256     # prompt + continuation cap

    for i, example in enumerate(ds):
        if i >= num_examples:
            break

        full_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        prompt_ids = full_ids[:prompt_tokens]

        print("\n" + "-" * 80)
        print(f"Example {i}")

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        target_text = tokenizer.decode(full_ids, skip_special_tokens=False)

        print("\n--- Prompt (first 64 tokens) ---")
        print(prompt_text)

        gen_ids = generate_from_ids(
            prompt_ids,
            max_total_tokens=max_total_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

        print("\n--- Model continuation ---")
        print(gen_text)

        print("\n--- Full training target ---")
        print(target_text)

    print("\nDone!")
