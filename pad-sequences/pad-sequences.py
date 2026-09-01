import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if max_len is None:
        max_len = max(map(len, seqs), default=0)

    out = np.full((len(seqs), max_len), pad_value)

    for i, seq in enumerate(seqs):
        n = min(len(seq), max_len)
        out[i, :n] = seq[:n]

    return out