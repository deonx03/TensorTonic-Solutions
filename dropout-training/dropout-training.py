import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    rng = rng if rng is not None else np.random
    x = np.asarray(x)

    mask = rng.random(x.shape) >= p
    scale = 1 / ( 1 - p)
    dropout_pattern = mask * scale
    output = x * dropout_pattern
    
    return (output, dropout_pattern)