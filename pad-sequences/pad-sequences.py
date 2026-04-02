import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    L = max(len(row) for row in seqs) 
    N = len(seqs)
    max_len = L if max_len is None else max_len
    
    seqs_padded = np.full((N, max_len), pad_value)
    
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        seqs_padded[i, :len(trunc)] = trunc
    return seqs_padded
        