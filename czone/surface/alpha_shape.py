"""
Primitives and algorithms for determining alpha-shape of collection of points.
"""

from unittest.mock import NonCallableMagicMock
import numpy as np
from scipy.spatial import Delaunay

def tetrahedron_circumradii(points):
    """Calculates circumradii of set of tetrahedron. Vectorized code.

    vectorized version of adapted code from https://github.com/python-adaptive/adaptive
    uses determinants to calculate circumradius of tetrahedron, ref: https://mathworld.wolfram.com/Circumsphere.html
    
    Args:
        points (np.ndarray): Nx4x3 array of points, representing vertices of N tetrahedra.

    Returns:
        np.ndarray of circumradii of tetrahedra
    """
    points = np.array(points) #+ np.random.rand(points.shape[0], points.shape[1], points.shape[2])*1e-8 # add small random number to avoid divide by zero
    pts = points[:,1:] - points[:,0,None]
    
    x1 = pts[:,0,0]
    y1 = pts[:,0,1]
    z1 = pts[:,0,2]
    
    x2 = pts[:,1,0]
    y2 = pts[:,1,1]
    z2 = pts[:,1,2]
    
    x3 = pts[:,2,0]
    y3 = pts[:,2,1]
    z3 = pts[:,2,2]

    l1 = x1 * x1 + y1 * y1 + z1 * z1
    l2 = x2 * x2 + y2 * y2 + z2 * z2
    l3 = x3 * x3 + y3 * y3 + z3 * z3

    # Compute some determinants:
    dx = +l1 * (y2 * z3 - z2 * y3) - l2 * (y1 * z3 - z1 * y3) + l3 * (y1 * z2 - z1 * y2)
    dy = +l1 * (x2 * z3 - z2 * x3) - l2 * (x1 * z3 - z1 * x3) + l3 * (x1 * z2 - z1 * x2)
    dz = +l1 * (x2 * y3 - y2 * x3) - l2 * (x1 * y3 - y1 * x3) + l3 * (x1 * y2 - y1 * x2)
    aa = +x1 * (y2 * z3 - z2 * y3) - x2 * (y1 * z3 - z1 * y3) + x3 * (y1 * z2 - z1 * y2)
    a = 2 * aa

    center = np.vstack([dx / a, -dy / a, dz / a])
    radius = np.linalg.norm(center,axis=0)
    return radius

def alpha_shape_alg_3D(points, probe_radius, return_alpha_shape=False):
    """Use alpha shape algorithm to determine points on exterior of collection of points.

    Performs alpha-shape algorithm ##TODO: cite a source here

    Args:
        points (np.ndarray): Nx3 array representing coordinates of points in object
        probe_radius (float): radius of test
        return_alpha_shape (bool): return dictionary of alpha shape arrays, default False.
    Returns:
        List of indices of points on exterior of surface for given alpha-shape.
    """

    ## Get alpha-shape
    # get delaunay triangulation of points
    points = points+1e-4*np.random.rand(*points.shape)
    tri = Delaunay(points)

    # get cicrcumradii of all tetrahedron in triangulation
    circumradii = tetrahedron_circumradii((points+1e-10*np.random.rand(*points.shape))[tri.simplices, :])
    
    # check which tetrahedra in triangulation are part of alpha-shape for given probe radius
    probe_test = circumradii <= probe_radius

    ## Get outer elements of alpha-shape
    # check which tetrahedra are on outside of triangulation
    outside = np.sum(tri.neighbors >= 0,axis=1) < 4

    # check which tetrahedra are neighbors to those not in alpha-shape

    # however, only want to check neighbors that are on the outside of the triangulation
    # so that we can avoid counting tetrahedra that fail the probe test on large internal voids

    neighbor_check_0 = np.logical_not(probe_test[tri.neighbors]) # neighbors that fail probe test
    neighbor_check_1 = outside[tri.neighbors] # neighbors on the outside
    neighbor_check = np.any(np.logical_and(neighbor_check_0, neighbor_check_1), axis=1)

    # get tetrahedra that are on the surface of alpha-shape
    surface_tris = np.logical_and(probe_test, np.logical_or(outside, neighbor_check))
    
    ## Get outer points of outer elements of alpha-shape
    # determine which points are along the outer edges of tetrahedra/alpha-shape
    sub_points = tri.simplices[surface_tris,:]
    sub_neighbors = tri.neighbors[surface_tris,:]
    out_points = sub_points[np.logical_not(np.logical_or(sub_neighbors==-1, np.logical_not(probe_test[sub_neighbors])))]

    if return_alpha_shape:
        shape_dict = {}

        # return the full triangulation and corresponding surface tris
        shape_dict["tri"]= tri
        shape_dict["surface_tris"] = surface_tris

        # also return list of simplices indices for full alpha shape
        a_tris = np.logical_not(np.logical_and(outside, np.logical_not(probe_test)))
        shape_dict["a_tris"] = np.nonzero(a_tris)

        return list(set(out_points)), shape_dict
    else:
        return list(set(out_points))


def alpha_shape_alg_3D_with_sampling(points, probe_radius, N_samples, std=1e-4, rng = None, seed = None, return_alpha_shape=False):
    """Use alpha shape algorithm to determine points on exterior of collection of points.

    Performs alpha-shape algorithm ##TODO: cite a source here

    Args:
        points (np.ndarray): Nx3 array representing coordinates of points in object
        probe_radius (float): radius of test
        return_alpha_shape (bool): return dictionary of alpha shape arrays, default False.
    Returns:
        List of indices of points on exterior of surface for given alpha-shape.
    """


    if seed is None:
        seed = np.random.randint(0,10000)

    if rng is None:
        rng = np.random.default_rng(seed=seed)

    
    ## Get alpha-shape
    # get delaunay triangulation of points
    tri = Delaunay(points)

    for i in range(N_samples):
        # get cicrcumradii of all tetrahedron in triangulation
        noise = std*rng.standard_normal(size=points.shape)

        circumradii = tetrahedron_circumradii((points+noise)[tri.simplices, :])
        
        # check which tetrahedra in triangulation are part of alpha-shape for given probe radius
        if i == 0:
            probe_test = circumradii <= probe_radius
        else:
            probe_test = np.logical_or(probe_test, circumradii <= probe_radius)

    ## Get outer elements of alpha-shape
    # check which tetrahedra are on outside of triangulation
    outside = np.sum(tri.neighbors >= 0,axis=1) < 4

    # check which tetrahedra are neighbors to those not in alpha-shape

    # however, only want to check neighbors that are on the outside of the triangulation
    # so that we can avoid counting tetrahedra that fail the probe test on large internal voids

    neighbor_check_0 = np.logical_not(probe_test[tri.neighbors]) # neighbors that fail probe test
    neighbor_check_1 = outside[tri.neighbors] # neighbors on the outside
    neighbor_check = np.any(np.logical_and(neighbor_check_0, neighbor_check_1), axis=1)

    # get tetrahedra that are on the surface of alpha-shape
    surface_tris = np.logical_and(probe_test, np.logical_or(outside, neighbor_check))
    
    ## Get outer points of outer elements of alpha-shape
    # determine which points are along the outer edges of tetrahedra/alpha-shape
    sub_points = tri.simplices[surface_tris,:]
    sub_neighbors = tri.neighbors[surface_tris,:]
    out_points = sub_points[np.logical_not(np.logical_or(sub_neighbors==-1, np.logical_not(probe_test[sub_neighbors])))]

    if return_alpha_shape:
        shape_dict = {}

        # return the full triangulation and corresponding surface tris
        shape_dict["tri"]= tri
        shape_dict["surface_tris"] = surface_tris

        # also return list of simplices indices for full alpha shape
        a_tris = np.logical_not(np.logical_and(outside, np.logical_not(probe_test)))
        shape_dict["a_tris"] = np.nonzero(a_tris)

        return list(set(out_points)), shape_dict
    else:
        return list(set(out_points))