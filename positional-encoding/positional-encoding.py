import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len) [:, np.newaxis]
    i = np.arange((d_model + 1) // 2) [np.newaxis,:]
    div_term = np.power(base, 2 * i / d_model)
    angles = position / div_term 
    
    n_sin = pe[:, 0::2].shape[1]
    n_cos = pe[:, 1::2].shape[1]
    
    pe[:, 0::2] = np.sin(angles[:, :n_sin])
    pe[:, 1::2] = np.cos(angles[:, :n_cos])

    return pe