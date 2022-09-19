"""
Loose sketch of algorithm:
1) Find all atoms on surface with alpha shape
2) Filter atoms out, e.g., via chemistry or spatial filters
3) Choose an atom and find approximate surface normal 
4) Rotate target vector in molecule coordinates to align with surface normal.
    Target vector is relative to the bonding atom. Default: +Z
    4a) Offer ability to sample 
5) Rotate molecule about surface normal axis
    5a) Offer ability to sample
6) Set molecule outside of surface with specified bond length
7) Check for collisions/make sure none of the molecule is in the surface
8) Accept adsorbate
"""

import numpy as np
from functools import reduce

from .alpha_shape import alpha_shape_alg_3D, alpha_shape_alg_3D_with_sampling
from ..molecule import BaseMolecule
from ..volume import BaseVolume
from ..transform import rot_v, rot_vtv, Rotation, Translation
from pymatgen import Element
from scipy.sparse import csr_matrix

def sparse_matrix_from_tri(simplices):
    """Convert array of triangulation simplices into sparse matrix (graph).
    
    Assumes triangulation is for tetrahedra.

    Args:
        simplices (np.ndarray): Array (potentially reduced) of vertices 
                            representing simplices in Delaunay triangulation,
                            e.g., as returned by scipy.spatial.Delaunay().simplices

    Returns:
        NxN array as sparse matrix, where N is the max index in the triangulation.
    """
    N = np.max(simplices)+1
    mat = np.zeros((N,N))
    for s in simplices:
        for i in range(4):
            for j in range(i+1, 4):
                ii = s[i]
                jj = s[j]
                mat[ii,jj] = 1
                mat[jj,ii] = 1

    return csr_matrix(mat)

def find_approximate_normal(points, decay=0.99, tol=1e-5, margin=0, seed=23, max_steps=1000, **kwargs):
    """Use modifeid perception algorithm to find approximate surface normal to set of points.
    
    Assumes points come from local section of alpha-shape, i.e, they are bounded
    by a convex surface and thus all lie within a common half-space.

    Args:
        points (np.ndarray): Nx3 array of points representing local surface
        decay (float): "gradient" weight decay, used to stabilize algorithm
        tol (float): stopping criterion for difference between successive dot 
                    products of surface normal
        seed (int): seed for RNG, used to determine sequence of points chosen
                    to update guess for 
        max_steps (int): maximum number of steps to run

    Returns:
        np.ndarray: (3,) normalized vector representing orientation of surface normal
    """

    rng = np.random.default_rng(seed=seed)

    A = points # use matrix notation

    # initialize guess to 0
    w = np.ones(3)*1e-10

    # set sequence of iterates
    sequence = rng.integers(0, A.shape[0], max_steps)

    converging = True
    i = 0
    j = 0
    w_list = []
    while(converging):

        # choose random point from set of points
        x = A[sequence[i],:]

        # get new surface normal guess
        if w @ x >= margin:
            w_new = w - (decay**j)*x

            # store dot product against previous guess
            w_list.append((w_new @ w)/(np.linalg.norm(w_new)*np.linalg.norm(w)))
        
            # update and normalize guess
            w = w_new
            w = w/np.linalg.norm(w)
            j+=1

        i += 1
        if len(w_list) > 5:
            # if last five iterations are all under tolerance, break loop
            if reduce(lambda x,y: x & y, [x > (1-tol) for x in w_list[-5:]]):
                converging = False

        if i == max_steps:
            break

    return w

def get_nearest_neighbors(target_idx,
                          shape_dict,
                          N_shells=3,
                          surface_only=True, 
                          **kwargs):
    """Traverse alpha-shape triangulation to get nearest neighbors to surface atom.
    
    Args:
        target_idx (int): index of atom on surface to which adsorbate is attached
        shape_dict (dict): dictionary of information regarding computed alpha-shape 
        N_shells (int): number of nearest neighbor shells to to traverse
        surface_only (bool): whether or not to limit search to surface atoms
    Returns:
        List of indices of nearest neighbors of atom.
    
    """

    if surface_only:
        # reduce selection to atoms that are part of surface tetrahedra
        # do not reduce to solely surface atoms, because roughness is unstable
        # with alpha shape
        reduced_tri = shape_dict["tri"].simplices[shape_dict["surface_tris"],:]
    else:
        reduced_tri = shape_dict["tri"].simplices[shape_dict["a_tris"],:]

    graph = sparse_matrix_from_tri(reduced_tri)
    v = np.zeros((graph.shape[0],1))
    v[target_idx] = 1

    for i in range(N_shells):
        v = graph @ v

    v[target_idx] = 0
    return np.nonzero(v)[0]

def add_adsorbate(mol: BaseMolecule,
                  adsorbate_idx,
                  bond_length,
                  volume: BaseVolume,
                  mol_vector=np.array([0,0,1]).T,
                  mol_rotation: float=0.0,
                  probe_radius=2.5,
                  filters={},
                  debug=False,
                  use_sampling=True,
                  **kwargs):
    """Add adsorbate onto surface of a given volume.

    
    for filters, accept dictionaries where key:value pairs are:
        element: list of element names or atomic numbers 
        mask: boolean mask 
        indices: list of indices to include
        spatial: function: (Mx3) float array -> (M,) bool array
    
    Args:
        mol (BaseMolecule): molecule to add as adsorbate
        mol_vector (np.ndarray): (3,1) array representing direction in molecule frame
                                to orient in direction of surface normal
        mol_rotation (float): value in radians to rotate molecule about surface normal
        adsorbate_idx (int): index of atom in molecule which bonds to surface
        bond_length (float): length of bond between molecule and surface
        probe_radius (float):
        filters (dict):

    Returns:
        Adsorbate molecule 
    """

    mol_out = mol.from_molecule()

    ##  Find all atoms on surface with alpha shape
    ## TODO: test default probe radius from RDF measurement

    if "seed" in kwargs.keys():
        seed = kwargs["seed"]
    else:
        seed = np.random.randint(0,10000)

    rng = np.random.default_rng(seed=seed)

    if use_sampling:
        surface_ind, shape_dict = alpha_shape_alg_3D_with_sampling(points=volume.atoms, 
                                                    probe_radius=probe_radius,
                                                    N_samples=20,
                                                    seed=seed,
                                                    rng=rng,
                                                    return_alpha_shape=True)
    else:
        surface_ind, shape_dict = alpha_shape_alg_3D(points=volume.atoms, 
                                                    probe_radius=probe_radius, 
                                                    return_alpha_shape=True)

    valid_indices = np.zeros(volume.atoms.shape[0], dtype=bool)
    valid_indices[surface_ind] = True

    ##  Filter atoms out, e.g., via chemistry or spatial filters

    for key, val in filters.items():
        if "element" in key:
            elements = [Element(x).Z if isinstance(x, str) else x for x in val]
            tmp_mask = reduce(lambda x,y: np.logical_or(x,y), [volume.species == x for x in elements])
            valid_indices = np.logical_and(valid_indices, tmp_mask)
        elif "mask" in key:
            valid_indices = np.logical_and(valid_indices, val)
        elif "indices" in key:
            tmp_mask = np.zeros(volume.atoms.shape[0], dtype=bool)
            tmp_mask[val] = True
            valid_indices = np.logical_and(valid_indices, tmp_mask)
        elif "spatial" in key:
            tmp_mask = val(volume.atoms)
            valid_indices = np.logical_and(valid_indices, tmp_mask)
        else:
            print("Invalid filter key provided. Must be one of {elementXX, maskXX, indicesXX, spatialXX}.")
            print("Key provided: ", key)

    valid_indices = np.nonzero(valid_indices)[0]
    ## Choose target surface atom and find approximate surface normal 


    target_idx = rng.choice(valid_indices)

    # grab orientation vectors 
    nn_ind = get_nearest_neighbors(target_idx, shape_dict, **kwargs)
    nn_pos = volume.atoms[nn_ind,:] - volume.atoms[target_idx,:]

    # normalize so that all vectors are within the unit sphere
    # and such that closer neighbors have the largest dot products
    nn_pos_norms = np.linalg.norm(nn_pos, axis=1)[:, None]
    nn_pos_min = np.min(nn_pos_norms)
    nn_pos = nn_pos/nn_pos_min/(nn_pos_norms/nn_pos_min)**2.0

    # optimize margin on normal
    best_margin = 0
    j = 0
    while(j < 10):
        cur_seed = rng.integers(0,10000,1)[0]
        w = find_approximate_normal(nn_pos, decay=0.95, tol=1e-4, margin=best_margin, seed=cur_seed)

        margins = []
        for b in nn_pos:
            margins.append(w @ b)

        j+=1
        if np.max(margins) < best_margin:
            best_margin = np.max(margins)
            j = 0


    normal = w

    ##  Rotate target vector in molecule coordinates to align with surface normal.
    # set origin of molecule to adsorbate molecule
    mol_out.set_origin(idx=adsorbate_idx)
    m_vec = mol_out.orientation @ mol_vector

    # calculate rotation matrix and transform
    rot = Rotation(matrix=rot_vtv(m_vec, normal), origin=mol_out.origin)
    mol_out.transform(rot)

    ##  Rotate molecule about surface normal axis
    rot = Rotation(matrix=rot_v(normal, mol_rotation), origin=mol_out.origin)
    mol_out.transform(rot)

    ##  Set molecule outside of surface with specified bond length
    new_origin = volume.atoms[target_idx,:] + normal*bond_length
    mol_out.transform(Translation(new_origin-mol_out.origin))

    ##  Check for collisions/make sure none of the molecule is in the surface
    if np.all(np.logical_not(volume.checkIfInterior(mol_out.atoms))):
        if debug:
            return mol_out, target_idx, nn_ind, nn_pos
        else:
            return mol_out, True
    else:
        print("Adsorbate placement failed. Molecule landed inside volume.")
        print("Check input parameters.")
        if debug:
            return mol_out, target_idx, nn_ind, nn_pos
        else:
            return mol_out, False
