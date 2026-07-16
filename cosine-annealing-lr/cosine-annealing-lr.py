import numpy as np
def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    progress = current_step / total_steps
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))

    return lr