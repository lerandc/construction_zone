from functools import reduce
from typing import List

import numpy as np

from czone.util.measure import get_voxel_grid

############################
######### Utilities ########
############################


def get_p_dist(coords, new_coord, dims):
    """Calculate squared periodic distance between a coordinate and all other coordinates in a set."""

    dist_0 = np.abs(coords - new_coord)
    dist_1 = dims - dist_0
    dist_1[:, 2] = dist_0[:, 2]

    p_dist = np.sum(np.min(np.dstack([dist_0, dist_1]), axis=2) ** 2.0, axis=1)
    return p_dist


def get_p_dist_batch(compare_coords, new_coords, dims, min_dist_2):
    dists_0 = np.abs(compare_coords - new_coords[:, None, :])
    dists_1 = dims - dists_0
    dists_1[:, :, 2] = dists_0[:, :, 2]

    ax = 0
    dists = np.sum(np.min(np.stack([dists_0, dists_1], axis=ax), axis=ax) ** 2.0, axis=2)
    return np.all(dists > min_dist_2, axis=1)


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
                if k > 0:
                    tmp_z.append(k - 1)
                if k < num_blocks[2] - 1:
                    tmp_z.append(k + 1)

                tlist = get_tuples(tmp_x, tmp_y, tmp_z)

                neighbors[-1][-1].append(tlist)

    return voxels, neighbors


def gen_p_substrate(
    dims: List[float], min_dist: float = 1.4, density=0.1103075, print_progress=False, rng=None
):
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

    rng = np.random.default_rng() if rng is None else rng

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

    if print_progress:
        print("Starting particle loop for %i particles" % num_c)

    # add first particle to lists
    for i in range(0, num_c):
        if print_progress and (not (i % (num_c // 5))):
            print("On %i of %i" % (i, num_c))
        new_coord = rng.uniform(size=(1, 3)) * dims

        block = np.floor(new_coord / (min_dist)).astype(int)
        block = block[0]
        tlist = neighbors[block[0]][block[1]][block[2]]
        parts = []
        for t in tlist:
            parts.extend(voxels[t[0]][t[1]][t[2]])

        p_dist = get_p_dist(coords[parts, :], new_coord, dims)
        while (len(p_dist) > 0) and (np.min(p_dist) < min_dist_2):
            new_coord = rng.uniform(size=(1, 3)) * dims
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


def _parse_min_distance_arg(species: List[int], min_dist: dict | np.ndarray) -> np.ndarray:
    """Parse a set of multielement pairwise minimum distances into a regular format.

    Args:
        species: Length N list of element Z numbers
        min_dist: Pairwise minimum distances.

    Returns:
        np.ndarray: Pairwise minimum distance as dict mapping

    """

    # TODO: check that only unique pairs are here, utilizing itertools combinations
    # TODO: check that species has no copies
    order = {Z: i for i, Z in enumerate(species)}

    # initialize result
    res = {}
    match min_dist:
        case dict():
            # Expects either dict of dicts, where min_dist[x] = {x:float, y:float, ...}
            # or dict of paired tuples, where min_dist[(x,y)] = float
            keys = list(min_dist.keys())
            match keys[0]:
                case tuple():
                    # check directly against combinations for pair uniqueuness
                    # for k in min_dist:
                    #     i, j = order[k[0]], order[k[1]]
                    #     res[i, j] = min_dist[k]
                    #     res[j, i] = min_dist[k]
                    for k in min_dist:
                        res[k] = min_dist[k] ** 2.0
                        res[(k[1], k[0])] = min_dist[k] ** 2.0
                case int():
                    # construct pairs and then check for uniqueness
                    for k in min_dist:
                        for kk, vv in min_dist[k].items():
                            res[order[k], order[kk]] = vv
                            res[order[kk], order[k]] = vv
                case _:
                    raise KeyError(
                        f"min_dist dictionary should have keys as int or tuple[int,int] but has invalid key format {keys[0]}"
                    )
        case np.ndarray():
            # Verify symmetry or that lower-tri - diag == 0
            raise NotImplementedError()
        case _:
            raise TypeError(
                f"Pairwise minimum distances should be dict or np.ndarray, but {type(min_dist)} was passed."
            )

    return res


def gen_multielement_random_block(
    dims: List[float],
    species,
    fractions,
    min_dist,
    density=0.1103075,
    voxel_scale=2.0,
    print_progress=False,
    rng=None,
):
    # set up pairwise distance mapping and species MC bin sampler
    pairwise_distances = _parse_min_distance_arg(species, min_dist)

    max_min_dist = np.sqrt(np.max([v for v in pairwise_distances.values()]))
    fractions = np.array(fractions)
    f_norm = np.sum(fractions)
    species_bins = np.cumsum(fractions / f_norm)

    def get_species_index(x):
        return np.min(np.where(species_bins > x))

    # initialize RNG
    rng = np.random.default_rng() if rng is None else rng

    # get number of atoms to generate
    dims = np.array(dims)
    dim_x = dims[0]
    dim_y = dims[1]
    dim_z = dims[2]
    total_volume = dim_x * dim_y * dim_z
    num_atoms = np.round(total_volume * density).astype(int)

    # initialize result arrays
    res_coords = np.zeros((num_atoms, 3))
    res_species = np.zeros((num_atoms))

    # get voxel grid and list of local voxel neighbors
    dims = np.array([dim_x, dim_y, dim_z])
    voxel_size = max_min_dist * voxel_scale
    num_blocks = np.ceil(dims / (voxel_size)).astype(int)
    tot_blocks = np.prod(num_blocks)
    Nt = np.array(
        [1, num_blocks[0], num_blocks[0] * num_blocks[1]]
    )  # to simply conversion of 3D -> 1D voxel indices

    voxels = [[] for i in range(tot_blocks)]
    neighbors = get_voxel_grid(num_blocks, True, True, False)

    if print_progress:
        print("Starting particle loop for %i particles" % num_atoms)

    # begin particle generation
    for i in range(num_atoms):
        # use a for loop here, so that we can respect the number fraction simply
        # sample next species
        new_species = species[get_species_index(rng.uniform())]

        successful = False
        while not successful:
            # generate trial coordinate
            new_coord = rng.uniform(size=(1, 3)) * dims

            # find its voxel and neighboring voxels
            block = ((np.floor(new_coord / voxel_size)) @ Nt).astype(int)[0]

            # grab atoms in neighboring voxels
            tlist = neighbors[block]
            parts = reduce(lambda x, y: x + y, [voxels[t] for t in tlist])

            compare_coords = res_coords[parts, :]
            compare_species = res_species[parts]

            # check pairwise minimum distances
            distance_check = True
            for Z in species:
                Z_filter = compare_species == Z

                p_dist = get_p_dist(compare_coords[Z_filter, :], new_coord, dims)
                if len(p_dist > 0):
                    distance_check = distance_check and (
                        p_dist.min() >= pairwise_distances[(new_species, Z)]
                    )
                if not distance_check:
                    break

            successful = distance_check

        # new corodinate satisfies all minimmum distance constraints
        # update voxel and result arrays
        voxels[block].append(i)
        res_coords[i, :] = new_coord
        res_species[i] = new_species

    return res_coords, res_species


def gen_p_substrate_batched(
    dims: List[float],
    min_dist: float = 1.4,
    density=0.1103075,
    print_progress=False,
    voxel_scale=2.0,
    batch_size=16,
    rng=None,
):
    rng = np.random.default_rng() if rng is None else rng

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
    voxel_size = min_dist * voxel_scale
    num_blocks = np.ceil(dims / (voxel_size)).astype(int)
    tot_blocks = np.prod(num_blocks)
    Nt = np.array([1, num_blocks[0], num_blocks[0] * num_blocks[1]])

    if print_progress:
        print("Getting neighbors")

    # get voxel grid and list of local voxel neighbors
    voxels = [[] for i in range(tot_blocks)]
    neighbors = get_voxel_grid(num_blocks, True, True, False)

    if print_progress:
        print("Starting particle loop for %i particles" % num_c)

    num_accepted = 0
    cur_iter = 0
    while num_accepted < num_c:
        # if print_progress and (not (i % (num_c // 5))):
        # print("On %i of %i" % (i, num_c))
        new_coord = rng.uniform(size=(batch_size, 3)) * dims

        # find block for each particle and make sure every particle is in unique block
        # to avoid full N^2 comparisons within batch
        block = (np.floor(new_coord / voxel_size) @ Nt).astype(int)
        __, idx = np.unique(block, return_index=True)
        block = block[idx]
        new_coord = new_coord[idx, :]

        compare_parts = []
        max_compare = 0
        for b in block:
            tlist = neighbors[b]
            parts = reduce(lambda x, y: x + y, [voxels[t] for t in tlist])
            max_compare = np.max([max_compare, len(parts)])
            compare_parts.append(coords[parts, :])

        all_compare_parts = np.ones((len(block), max_compare, 3)) * -1000  # what is this doing?
        for i, parts in enumerate(compare_parts):
            all_compare_parts[i, : parts.shape[0], :] = parts

        accept_filter = get_p_dist_batch(all_compare_parts, new_coord, dims, min_dist_2)

        # remove any particles that are in neighboring blocks
        inds = np.where(accept_filter)[0]
        for i in range(1, len(inds)):
            # check all previous accepted blocks to see if they are neighbors to current block
            for j in range(i):
                if block[inds[j]] in neighbors[block[inds[i]]]:
                    accept_filter[inds[j]] = False
                    continue

        for i, b in enumerate(block[accept_filter]):
            voxels[b].append(i + num_accepted)

        N_good = np.sum(accept_filter)
        if N_good + num_accepted > num_c:
            stop_idx = np.where(np.cumsum(accept_filter) > (num_c - num_accepted))[0][0]
            accept_filter[stop_idx:] = False

        coords[num_accepted : num_accepted + N_good, :] = new_coord[accept_filter, :]
        num_accepted += N_good
        cur_iter += 1

    return coords
