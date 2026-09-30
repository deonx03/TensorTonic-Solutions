import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    s = np.array(s)
    w = np.array(w)
    g = np.array(g)
    s = beta * s + (1 - beta) * g ** 2
    w = w - (lr / np.sqrt(s + eps)) * g
    return (w, s)