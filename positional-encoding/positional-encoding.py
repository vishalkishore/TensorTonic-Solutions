import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    PE = np.zeros((seq_len,d_model))

    pos = np.arange(seq_len)[:,None]

    de = base ** (np.arange(0,d_model,2) / d_model)

    angles = pos/de

    PE[:,0::2] = np.sin(angles)
    PE[:,1::2] = np.cos(angles[:,:d_model//2])

    return PE

    # for pos in range(seq_len):
    #     for i in range(d_model//2):
    #         angle = pos / (base ** (2*i/d_model))
    #         PE[pos,2*i] = np.sin(angle)
    #         PE[pos,2*i+1] = np.cos(angle)

    return PE