import numpy as np

def rating_normalization(matrix: list) -> list:
    arr = np.array(matrix, dtype=float)

    mask = arr != 0

    sums = arr.sum(axis=1)
    counts = mask.sum(axis=1)
    means = sums / counts

    arr = arr - means[:, None]
    arr[~mask] = 0

    return arr.tolist()