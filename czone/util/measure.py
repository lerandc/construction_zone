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
    neighbors = np.ones((N,27))*np.array([x for x in range(N)])[:,None]

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

    return neighbors