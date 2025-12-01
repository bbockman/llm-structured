import torch

def sp_pad_collator(batch, pad_id):
    seqs = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    max_len = max(len(s) for s in seqs)

    padded = []
    masks  = []

    for seq in seqs:
        L = len(seq)
        pad_len = max_len - L

        padded_seq = torch.cat([seq, torch.full((pad_len,), pad_id)])
        mask       = torch.cat([torch.ones(L), torch.zeros(pad_len)])

        padded.append(padded_seq)
        masks.append(mask)

    return {
        "input_ids": torch.stack(padded),
        "attention_mask": torch.stack(masks)
    }
