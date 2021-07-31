from typing import List

import numpy as np

############################
######### Utilities ########
############################


def get_p_dist(coords, new_coord, dims):
    """Calculate squared periodic distance between a coordinate and all other coordinates in a set."""

    dist_0 = np.abs(coords - new_coord)
    dist_1 = dims - dist_0
    dist_1[:, 2] = dist_0[:, 2]

    p_dist = np.sum(np.min(np.dstack([dist_0, dist_1]), axis=2)**2.0, axis=1)
    return p_dist


def get_tuples(tx, ty, tz):

    tlist = [(x, y, z) for x in tx for y in ty for z in tz]

    return tlist


############################################
######## Periodic Uniform Algorithm ########
############################################


def get_voxels(min_dist, dims):
    """Get voxel and neighbor list."""
    num_blocks = np.ceil(dims / (min_dist)).astype(int)
    voxels = []
    neighbors = []
    for i in range(num_blocks[0]):
        voxels.append([])
        neighbors.append([])
        for j in range(num_blocks[1]):
            voxels[-1].append([])
            neighbors[-1].append([])
            for k in range(num_blocks[2]):
                voxels[-1][-1].append([])

                tmp_x = [i, (i + 1) % num_blocks[0], (i - 1) % num_blocks[0]]
                tmp_y = [j, (j + 1) % num_blocks[1], (j - 1) % num_blocks[1]]
                tmp_z = [k]
                if (k > 0):
                    tmp_z.append(k - 1)
                if (k < num_blocks[2] - 1):
                    tmp_z.append(k + 1)

                tlist = get_tuples(tmp_x, tmp_y, tmp_z)

                neighbors[-1][-1].append(tlist)

    return voxels, neighbors


def gen_p_substrate(dims: List[float],
                    min_dist: float = 1.4,
                    density=.1103075,
                    print_progress=True):
    """Generate a uniformly random distributed collection of atoms with PBC.

    Given the size of a rectangular prism, a minimum bond distance, and a target
    density, generate a uniformly random collection of atoms obeying periodic
    boundary conditions in X and Y. Dimensions, minimum distance, and density
    should all be in units of angstroms but can be input in any consistent unit 
    scheme. Default values are for amorphous carbon.

    Generation algorithm loosely follows
      1. Get total number  of atoms N to generate.
      2. While substrate contains < N atoms
        a. Generate uniformly random coordinate
        b. Check distance against nearest neighbor atoms to for violation
          of bond distance
        c. If not too close to other atoms, add to substrate; else, regenerate

    Generation utilizes voxel grid in 3D space for linear scaling of distance
    calculations and therefore generation time should loosely scale linearly
    with the volume of the substrate.

    Args:
        dims (List[float]): Size of rectangular prism substrate in [x,y,z]
        min_dist (float): Minimum seperation between atoms.
        density (float): Density of substrate.

    Returns:
        np.ndarray: coordinates of atoms in periodic substrate
    """

    # get number of carbon atoms to generate
    dims = np.array(dims)
    min_dist_2 = min_dist**2.0
    dim_x = dims[0]
    dim_y = dims[1]
    dim_z = dims[2]
    sub_vol = dim_x * dim_y * dim_z

    num_c = np.round(sub_vol * density).astype(int)
    coords = np.zeros((num_c, 3))
    dims = np.array([dim_x, dim_y, dim_z])

    if print_progress:
        print("Getting neighbors")

    # get voxel grid and list of local voxel neighbors
    voxels, neighbors = get_voxels(min_dist, dims)

    coords[0, :] = np.random.random_sample((1, 3)) * dims
    if print_progress:
        print("Starting particle loop for %i particles" % num_c)

    for i in range(1, num_c):
        if print_progress and (not (i % (num_c // 5))):
            print("On %i of %i" % (i, num_c))
        new_coord = np.random.random_sample((1, 3)) * dims

        block = np.floor(new_coord / (min_dist)).astype(int)
        block = block[0]
        tlist = neighbors[block[0]][block[1]][block[2]]
        parts = []
        for t in tlist:
            parts.extend(voxels[t[0]][t[1]][t[2]])

        p_dist = get_p_dist(coords[parts, :], new_coord, dims)
        while ((len(p_dist) > 0) and (np.min(p_dist) < min_dist_2)):
            new_coord = np.random.random_sample((1, 3)) * dims
            block = np.floor(new_coord / (min_dist)).astype(int)
            block = block[0]
            tlist = neighbors[block[0]][block[1]][block[2]]
            parts = []
            for t in tlist:
                parts.extend(voxels[t[0]][t[1]][t[2]])

            p_dist = get_p_dist(coords[parts, :], new_coord, dims)

        voxels[block[0]][block[1]][block[2]].append(i)
        coords[i, :] = new_coord

    return coords
