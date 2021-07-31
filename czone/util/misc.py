from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike


def round_away(x: float) -> float:
    """Round to float integer away from zero--opposite of np.fix.
    
    Args:
        x (float, ArrayLike[float]): number(s) to round

    Returns:
        float, ArrayLike[float]: rounded number(s)
    """
    return np.sign(x) * np.ceil(np.abs(x))


def get_N_splits(n: int, m: int, l: int, seed: int = None) -> List[int]:
    """Get N uniform random integers in interval [M,L-M) with separation M.

    Args:
        n (int): number of indices
        m (int): minimum distance between indices and ends of list
        l (int): length of initial list
        seed (int): seed for random number generator, default None

    Returns:
        List[int]: sorted list of random indices
    """
    if (l - 2 * m < (n - 1) * m):
        raise ValueError("m is too large for number of splits requested and l")
    rng = np.random.default_rng(seed=seed)

    #seed an initial choice and create array to calculate distances in
    splits = [rng.integers(m, l - m)]
    data = np.array([x for x in range(m, l - m)])
    idx = np.ma.array(data=data, mask=np.abs(data - splits[-1]) < m)

    while (len(splits) < n):
        while (np.all(idx.mask)):
            # no options left, reseed
            splits = [rng.integers(m, l - m)]
            idx.mask = np.abs(idx.data - splits[-1]) < m

        # add new choice to list and check distance against other indices
        splits.append(rng.choice(idx.compressed()))
        idx.mask = np.logical_or(idx.mask, np.abs(idx.data - splits[-1]) < m)

    splits.sort()
    return splits


def vangle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors of same dimension in R^N.

    Args:
        v1 (np.ndarray): N-D vector
        v2 (np.ndarray): N-D vector

    Returns:
        float: angle in radians
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
