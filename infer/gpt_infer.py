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
model = get_current_model(vocab_size=len(tokenizer)).to(device)

checkpoint = torch.load("disk/models/llm-scoped/params_134autocast.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
print("Model loaded successfully!")

@torch.no_grad()
def generate(
    prompt,
    max_tokens=256,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    ngram_size=3,
    repetition_penalty=1.2,
):
    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    print(f"Input IDs: {input_ids}")
    print(f"\nPrompt: {prompt}")
    print(f"Input tokens: {input_ids.shape[1]}")

    generated_tokens = []
    history = deque(maxlen=ngram_size * 4)  # track recent tokens

    for _ in range(max_tokens - input_ids.shape[1]):
        logits = model(input_ids)[0, -1, :] / max(temperature, 1e-5)

        # Basic repetition penalty on n-grams
        if len(history) >= ngram_size:
            recent = list(history)
            # last (ngram_size-1) tokens as prefix
            prefix = tuple(recent[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
            # penalize tokens that would repeat the last ngram
            for tok_id in set(recent):
                candidate = prefix + (tok_id,)
                # if this ngram already appears in history, downweight
                for i in range(len(recent) - ngram_size + 1):
                    if tuple(recent[i : i + ngram_size]) == candidate:
                        logits[tok_id] /= repetition_penalty
                        break

        # top-k
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
            logits[logits < kth_vals] = float("-inf")

        # top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)

            # mask tokens beyond nucleus
            cutoff = cumulative > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")

            # scatter back
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

    print("\nTotal generated tokens:", len(generated_tokens))
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return text

if __name__ == "__main__":
    test_prompts = [
        f"{tokenizer.bos_token} Artificial intelligence is",
        f"{tokenizer.bos_token} Geometric series converge when",
        f"{tokenizer.bos_token} In a distant future, humanity has",
        f"{tokenizer.bos_token} During World War II,",
        f"{tokenizer.bos_token} The Deep South's",
    ]

    print("\n" + "=" * 80)
    print("TESTING GENERATION (improved decoding)")
    print("=" * 80)

    for prompt in test_prompts:
        print("\n" + "-" * 80)
        out = generate(prompt)
        print(f"\nGenerated:\n{out}")

    print("\nDone!")
