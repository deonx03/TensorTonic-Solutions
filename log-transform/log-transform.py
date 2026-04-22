def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    X = np.asarray(values)
    return np.log1p(X)