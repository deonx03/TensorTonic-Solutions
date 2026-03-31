import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.asarray(v, dtype=float)
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.where(v_mag != 0, v / v_mag, 0)
    
    return v_norm
    