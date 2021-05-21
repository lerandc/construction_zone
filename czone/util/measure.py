import numpy as np


def get_voxel_grid(dim, px=True, py=True, pz=True):
    """
    dim is (Nx, Ny, Nz)
    """
    nn = [x for x in range(27)]
    nn_x = [(x%3)-1 for x in nn]
    nn_y = [((x//3)%3)-1 for x in nn]
    nn_z = [(x//9)-1 for x in nn]

    N = np.prod(dim)
    neighbors = np.ones((N,27))*np.arange(N)[:,None]

    shifts = (1, dim[0], dim[0]*dim[1])
    for i, t in enumerate(zip(nn_x, nn_y, nn_z)):
        neighbors[:,i] += np.dot(t, shifts)

    neighbors = neighbors % N

    # early exit
    if px and py and pz:
        return neighbors

    # change neighbor indices to nans if boundary is not periodic

    idx = np.array([x for x in range(N)]).astype(int)

    xi_face = (idx % dim[0] == 0)[:, None]
    xf_face = (idx % dim[0] == dim[0]-1)[:, None]

    yi_face = ((idx // dim[0]) % dim[1] == 0)[:, None]
    yf_face = ((idx // dim[0]) % dim[1] == dim[1]-1)[:, None]

    zi_face = (idx//(dim[0]*dim[1]) == 0)[:, None]
    zf_face = (idx//(dim[0]*dim[1]) == dim[2]-1)[:, None]

    nn_xi = np.array([(x%3)==0 for x in nn])[None, :]
    nn_xf = np.array([(x%3)==2 for x in nn])[None, :]

    nn_yi = np.array([((x//3)%3)==0 for x in nn])[None, :]
    nn_yf = np.array([((x//3)%3)==2 for x in nn])[None, :]

    nn_zi = np.array([(x//9)==0 for x in nn])[None, :]
    nn_zf = np.array([(x//9)==2 for x in nn])[None, :]

    fx = not px
    fy = not py
    fz = not pz
    if fx:
        neighbors[xi_face @ nn_xi] = np.nan
        neighbors[xf_face @ nn_xf] = np.nan

    if fy:
        neighbors[yi_face @ nn_yi] = np.nan
        neighbors[yf_face @ nn_yf] = np.nan

    if fz:
        neighbors[zi_face @ nn_zi] = np.nan
        neighbors[zf_face @ nn_zf] = np.nan

    if fx and fy:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
                idx_check = np.logical_and(xc, yc)
                nn_check = np.logical_and(nn_xc, nn_yc)
                neighbors[idx_check @ nn_check] = np.nan
            
    if fx and fz:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                idx_check = np.logical_and(xc, zc)
                nn_check = np.logical_and(nn_xc, nn_zc)
                neighbors[idx_check @ nn_check] = np.nan

    if fy and fz:
        for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
            for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                idx_check = np.logical_and(yc, zc)
                nn_check = np.logical_and(nn_yc, nn_zc)
                neighbors[idx_check @ nn_check] = np.nan

    if fx and fy and fz:
        for xc, nn_xc in zip([xi_face, xf_face], [nn_xi, nn_xf]):
            for yc, nn_yc in zip([yi_face, yf_face], [nn_yi, nn_yf]):
                for zc, nn_zc in zip([zi_face, zf_face], [nn_zi, nn_zf]):
                    idx_check = np.logical_and(np.logical_and(yc, zc), xc)
                    nn_check = np.logical_and(np.logical_and(nn_yc, nn_zc), nn_xc)
                    neighbors[idx_check @ nn_check] = np.nan

    mask = np.isnan(neighbors)
    neighbors_ma = np.ma.masked_array(neighbors, mask=mask)
    neighbor_lists = [np.ma.compressed(x) for x in neighbors]

    return neighbor_lists

def get_sdist_fun(dims=None, px=False, py=False, pz=False):
    """
    Return a squared distance function in 3D space for any PBC
    """
    if not np.any((px,py,pz)):
        def sdist(A, B):
            return np.sum((A-B)**2.0, axis=1)

    cols = [x for x, y in zip([0,1,2], [px,py,pz]) if y]
    sdims = np.array(dims)[cols]

    def sdist(A,B):
        dist_0 = np.abs(A-B)
        dist_1 = sdims - dist_0[:,cols]
        dist_0[:,cols] = np.min(np.stack([dist_0[cols], dist_1],axis=-1), axis=-1)

        return np.sum(dist_0*dist_0, axis=1)

    return sdist

def calc_rdf(coords, cutoff=20.0, px=True, py=True, pz=True):
    # shift outside of negative octants
    coords -= np.min(coords)

    # get box size and number of voxels in each direction
    dims = np.max(coords, axis=1)
    N = np.ceil(dims/cutoff)

    # get voxel neighbor list and 1D voxel idx for each obj
    nn = get_voxel_grid(N, px, py, pz)
    box_idx = np.floor(coords/vsize) @ N

    # get periodic distance calculation
    f_sdist = get_sdist_fun(dims, px, py, pz)

    parts = []
    part_ids = np.arange(coords.shape[0])
    for i in range(np.prod(N)):
        parts.append(part_ids[box_idx==i])

    # do 3D arrays so that distances are broadcasted/batched
    counts = np.zeros(int(cutoff/0.1))
    for i in range(np.prod(N)):
        cur_parts = coords[parts[i],:][:,:,None]

        for n in nn[box_idx[i]]:
            neighbor_parts = coords[parts[n],:][None,:,:]
            dist = np.sqrt(f_sdist(cur_parts, neighbor_parts))
            tmp_counts, _ = np.histogram(dist, bins=counts.shape[0], range=(0.0, cutoff))
            counts += tmp_counts

    counts[0] = 0.0

    return counts