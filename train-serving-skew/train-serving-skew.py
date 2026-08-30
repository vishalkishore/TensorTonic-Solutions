import numpy as np

def detect_skew(
    train_dist: dict,
    serving_dist: dict,
    threshold: float = 0.2,
    eps: float = 1e-10
) -> dict:

    psi_dict = {}

    for col, train_values in train_dist.items():

        train_col = np.asarray(train_values, dtype=float)
        serve_col = np.asarray(serving_dist[col], dtype=float)

        train_col = np.clip(train_col, eps, None)
        serve_col = np.clip(serve_col, eps, None)

        diff = serve_col - train_col


        log_ratio = np.log(serve_col) - np.log(train_col)

        psi = float(np.sum(diff * log_ratio))

        psi_dict[col] = {
            "psi": psi,
            "skewed": bool(psi > threshold)
        }

    return psi_dict