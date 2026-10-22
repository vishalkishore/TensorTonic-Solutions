import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    real_scores_imed = sum (real_scores) / (len(real_scores))
    fake_scores_imed = sum (fake_scores) / (len(fake_scores))

    return fake_scores_imed - real_scores_imed