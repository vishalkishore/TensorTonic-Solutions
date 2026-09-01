def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    
    for i in range(steps):
        g = lr*(2*a*x0 + b)
        x0 = x0 - g 

    return x0