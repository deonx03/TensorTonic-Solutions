import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    (r, c) = A.shape
    A_t= np.zeros((c, r))
    for i in range(r):
        for j in range(c):
            A_t[j][i] = A[i][j]
    return A_t