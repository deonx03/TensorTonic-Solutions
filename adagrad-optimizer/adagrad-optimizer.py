import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    g= np.asarray(g, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    new_g = G + g**2
    new_w = w -(lr/(np.sqrt(new_g + eps)) * g)
    return new_w, new_g