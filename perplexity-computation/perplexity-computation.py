import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions , actual_tokens= np.asarray(prob_distributions), np.asarray(actual_tokens)
    p_i = prob_distributions[np.arange(len(actual_tokens)), actual_tokens]
    n = prob_distributions.shape[0]
    H = -n**-1 * np.sum(np.log(p_i))
    return np.exp(H)
