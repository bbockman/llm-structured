import sentencepiece as spm
import torch
import regex as re

def normalize_for_sp(text):
    # Order matters: longest sequences first
    text = text.replace("\u00A0", " <nbs> ")
    text = text.replace("\t", " <tab> ")

    # Paragraphs (2+ newlines)
    text = re.sub(r"\n{2,}", " <par> ", text)

    # Single newlines
    text = text.replace("\n", " <new> ")

    # Collapse multiple spaces but preserve special tokens
    text = re.sub(r"[ ]{2,}", " ", text).strip()
    return text

def decode_for_display(sp_text):
    """
    Convert SP-decoded text back into readable form,
    reversing the normalization we applied before training.
    """
    txt = sp_text

    # restore whitespace structure
    txt = txt.replace("<par>", "\n\n")
    txt = txt.replace("<new>", "\n")
    txt = txt.replace("<tab>", "\t")
    txt = txt.replace("<nbsp>", "\u00A0")

    # clean double spaces (caused by training normalization)
    txt = re.sub(r"[ ]{2,}", " ", txt)
    
    return txt.strip()


class SPTokenizer:
    def __init__(self, model_path="unigram_8k.model"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def encode(self, text, add_bos=True, add_eos=True):
        return self.sp.encode(
            text,
            out_type=int,
            add_bos=add_bos,
            add_eos=add_eos
        )

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode(ids)

    # ----- HF-style call wrapper (for Dataset.map) -----
    def __call__(self, text):
        ids = self.encode(text)
        return {"input_ids": ids}

    # ----- length / special tokens -----
    def __len__(self):
        return self.sp.get_piece_size()

    @property
    def pad_id(self): return self.sp.pad_id()
    @property
    def bos_id(self): return self.sp.bos_id()
    @property
    def eos_id(self): return self.sp.eos_id()
    @property
    def unk_id(self): return self.sp.unk_id()

    @property
    def pad_token(self): return self.sp.id_to_piece(self.pad_id)
    @property
    def bos_token(self): return self.sp.id_to_piece(self.bos_id)
    @property
    def eos_token(self): return self.sp.id_to_piece(self.eos_id)
    @property
    def unk_token(self): return self.sp.id_to_piece(self.unk_id)

    @property
    def special_tokens_map(self):
        toks = []
        for i in range(len(self)):
            piece = self.sp.id_to_piece(i)
            if piece.startswith("<") and piece.endswith(">"):
                toks.append(piece)
        return toks
