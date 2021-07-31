from collections.abc import Callable
from typing import List, Tuple

import numpy as np


def get_voxel_grid(dim: Tuple[int],
                   px: bool = True,
                   py: bool = True,
                   pz: bool = True):
    """Generate list of voxel neighbors for 3D voxel grid with periodic boundary conditions.

    Utility function which returns a representation of a connected 3D voxel grid
    for arbitrary periodic boundary conditions. Voxels are ordered on a 1D list
    by X, then Y, then Z. For example, a 2x2x2 voxel grid will be ordered as
    [(0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (1,1,1)]. An ordered
    list is returned which contains, for each voxel i, a list of all of its 
    neighbors on the 3D grid, as ordered in the 1D indexing scheme.

    For fully periodic boundary conditions, each voxel i will have 27 neighbors, 
    including the voxel index itself.


    Args:
        dim (Tuple[int]): size of grid in x, y, and z
        px (bool): periodicity in x
        py (bool): periodicity in y
        pz (bool): periodicity in z

    Returns:
        List[List[int]]
    """
    # get relative coordinates as 3D grid
    nn = [x for x in range(27)]
    nn_x = [(x % 3) - 1 for x in nn]
    nn_y = [((x // 3) % 3) - 1 for x in nn]
    nn_z = [(x // 9) - 1 for x in nn]

    N = np.prod(dim)
    neighbors = np.ones((N, 27)) * np.arange(N)[:, None]

    # get relative indices as 1D list
    shifts = (1, dim[0], dim[0] * dim[1])
    for i, t in enumerate(zip(nn_x, nn_y, nn_z)):
        neighbors[:, i] += np.dot(t, shifts)

    # correct list for total size of grid
    neighbors = neighbors % N

    # if fully periodic, no further corrections needed
    if px and py and pz:
        return neighbors

    # get full list of indices as array
    idx = np.array([x for x in range(N)]).astype(int)

    # get logical arrays for voxels on faces of grid
    xi_face = (idx % dim[0] == 0)[:, None]
    xf_face = (idx % dim[0] == dim[0] - 1)[:, None]

    yi_face = ((idx // dim[0]) % dim[1] == 0)[:, None]
    yf_face = ((idx // dim[0]) % dim[1] == dim[1] - 1)[:, None]

    zi_face = (idx // (dim[0] * dim[1]) == 0)[:, None]
    zf_face = (idx // (dim[0] * dim[1]) == dim[2] - 1)[:, None]

    # get local neighbors on faces as arrays so that we can use matrix
    # multiplication for logical indexing
    nn_xi = np.array([(x % 3) == 0 for x in nn])[None, :]
    nn_xf = np.array([(x % 3) == 2 for x in nn])[None, :]

    nn_yi = np.array([((x // 3) % 3) == 0 for x in nn])[None, :]
    nn_yf = np.array([((x // 3) % 3) == 2 for x in nn])[None, :]

    nn_zi = np.array([(x // 9) == 0 for x in nn])[None, :]
    nn_zf = np.array([(x // 9) == 2 for x in nn])[None, :]

    # change neighbor indices to nans if boundary is not periodic
    fx = not px
    fy = not py
    fz = not pz

    # correct just x faces
    if fx:
        neighbors[xi_face @ nn_xi] = np.nan
        neighbors[xf_face @ nn_xf] = np.nan

    # correct just y faces
    if fy:
        neighbors[yi_face @ nn_yi] = np.nan
        neighbors[yf_face @ nn_yf] = np.nan

    # correct just z faces
    if fz:
        neighbors[zi_face @ nn_zi] = np.nan
        neighbors[zf_face @ nn_zf] = np.nan

    # correct x/y edges
    if fx and fy:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
                idx_check = np.logical_and(xc, yc)
                nn_check = np.logical_and(nn_xc, nn_yc)
                neighbors[idx_check @ nn_check] = np.nan

    # correct x/z edges
    if fx and fz:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                idx_check = np.logical_and(xc, zc)
                nn_check = np.logical_and(nn_xc, nn_zc)
                neighbors[idx_check @ nn_check] = np.nan

    # correct y/z edges
    if fy and fz:
        for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
            for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                idx_check = np.logical_and(yc, zc)
                nn_check = np.logical_and(nn_yc, nn_zc)
                neighbors[idx_check @ nn_check] = np.nan

    # correct x/y/z corners
    if fx and fy and fz:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
                for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                    idx_check = np.logical_and(np.logical_and(yc, zc), xc)
                    nn_check = np.logical_and(np.logical_and(nn_yc, nn_zc),
                                              nn_xc)
                    neighbors[idx_check @ nn_check] = np.nan

    # used masked arrays to get compact lists of neighboring voxels
    mask = np.isnan(neighbors)
    neighbors_ma = np.ma.masked_array(neighbors, mask=mask).astype(int)
    neighbor_lists = [np.ma.compressed(x) for x in neighbors_ma]

    return neighbor_lists


def get_sdist_fun(dims=None, px=False, py=False, pz=False):
    """Return a squared distance function in 3D space for any PBC.

    Args:
        dims (Array[float]): length 3 array, list, or tuple for periodic
                            dimension sizes (x,y,z); must be passed if using
                            periodic boundaries
        px (bool): whether or not periodic in x
        py (bool): whether or not periodic in y
        pz (bool): whether or not periodic in z

    Returns:
        Callable[float] : squared distance function obeying periodic boundaries
        
    """
    # return standard distance function if no periodic bonadries
    if not np.any((px, py, pz)):

        def sdist(A, B):
            return np.sum((A - B)**2.0, axis=1)

    # grab relevant dimensions
    cols = [x for x, y in zip([0, 1, 2], [px, py, pz]) if y]
    sdims = np.array(dims)[cols]
    sdims = sdims[None, :, None]

    def sdist(A, B):
        dist_0 = np.abs(A - B)
        dist_1 = sdims - dist_0[:, cols, :]
        dist_0[:, cols, :] = np.min(np.stack([dist_0[:, cols, :], dist_1],
                                             axis=-1),
                                    axis=-1)

        return np.sum(dist_0 * dist_0, axis=1)

    return sdist


def calc_rdf(coords, cutoff=20.0, px=True, py=True, pz=True):
    """Calculate the 3D periodic radial distribution function of array of points.

    Args:
        coords (np.ndarray): Nx3 numpy array representing points in 3D
        cutoff (float): cutoff distance for neighbor consideration
        px (bool): whether or not coordinates are periodic in x
        py (bool): whether or not coordinates are periodic in y
        pz (bool): whether or not coordinates are periodic in z

    Returns:
        np.ndarray: radial distribution function, returned in steps of 0.1

    """
    # shift outside of negative octants
    coords -= np.min(coords, axis=0)

    # get box size and number of voxels in each direction
    dims = np.max(coords, axis=0)
    N = np.ceil(dims / cutoff).astype(int)

    Nt = np.array([1, N[0], N[0] * N[1]])

    # get voxel neighbor list and 1D voxel idx for each obj
    nn = get_voxel_grid(N, px, py, pz)
    box_idx = (np.floor(coords / cutoff) @ Nt).astype(int)

    # get periodic distance calculation
    f_sdist = get_sdist_fun(dims, px, py, pz)

    # assign coords to voxels
    parts = []
    part_ids = np.arange(coords.shape[0])
    for i in range(np.prod(N)):
        parts.append([int(x) for x in part_ids[box_idx == i]])

    # do 3D arrays so that distances are broadcasted/batched
    counts = np.zeros(int(cutoff / 0.1))

    # for each voxel, calculate distance of owned particles to all possible
    # neighbors within cutoff distance
    for i in range(np.prod(N)):
        cur_parts = coords[parts[i], :][:, :, None]

        for n in nn[box_idx[i]]:
            neighbor_parts = (coords[parts[n], :].T)[None, :, :]
            dist = np.sqrt(f_sdist(cur_parts, neighbor_parts))

            # use histogram to get counts for RDF
            tmp_counts, _ = np.histogram(dist,
                                         bins=counts.shape[0],
                                         range=(0.0, cutoff))
            counts += tmp_counts

    # correct for self interactions
    counts[0] = 0.0

    return counts
