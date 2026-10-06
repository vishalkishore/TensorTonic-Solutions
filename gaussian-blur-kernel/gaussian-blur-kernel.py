import numpy as np

def gaussian_kernel(size: int, sigma: float) -> list:
    center = (size - 1) / 2

    x = np.arange(size) - center
    y = np.arange(size) - center

    X, Y = np.meshgrid(x, y)

    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    kernel = kernel / kernel.sum()

    return kernel.tolist()