def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    common = len((set(recommended[:k]) & set(relevant)))
    precision_k = common / k
    recall_k = common / len(relevant)

    return [ precision_k , recall_k ]