import numpy as np


def round_away(x):
    """
    Round to integer way from zero-- opposite of np.fix

    Not designed to be fast
    """
    return np.sign(x)*np.ceil(np.abs(x))

def get_N_splits(n, m, l):
    """
    n: number of indices
    m: min dist between indices
    l: length of initial list
    """
    if (l-2*m < (n-1)*m):
        raise ValueError("m is too large for number of splits requested and l")
    rng = np.random.default_rng()

    splits = [rng.integers(m, l-m)]
    data = np.array([x for x in range(m, l-m)])
    idx = np.ma.array(data=data, mask=np.abs(data-splits[-1]) < m)

    while(len(splits) < n):
        while(np.all(idx.mask)):
            # no options left, reseed
            splits = [rng.integers(m, l-m)]
            idx.mask = np.abs(idx.data-splits[-1]) < m

        splits.append(rng.choice(idx.compressed()))
        idx.mask = np.logical_or(idx.mask, np.abs(idx.data-splits[-1]) < m)

    splits.sort()
    return splits

def vector_angle(v1, v2):
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))