import unittest
import numpy as np
from czone.volume.volume import makeRectPrism
from czone.volume.algebraic import Sphere, Plane, get_bounding_box, convex_hull_to_planes, Cylinder
from czone.transform import Rotation
from scipy.spatial import ConvexHull, Delaunay
from functools import reduce

seed = 32135213035
rng = np.random.default_rng(seed=seed)

class Test_Sphere(unittest.TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        for _ in  range(self.N_trials):
            radius = rng.uniform(size=(1,))[0]
            center = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)

            self.assertEqual(radius, sphere.radius)
            self.assertTrue(np.array_equal(center, sphere.center))
            self.assertEqual(tol, sphere.tol)

            rt, ct = sphere.params
            self.assertEqual(radius, rt)
            self.assertTrue(np.array_equal(center, ct))

        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), 0.0)
        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), -1.0)
        self.assertRaises(ValueError, lambda x, y: Sphere(x, y), 1.0, [2,0])
        self.assertRaises(ValueError, lambda x, y: Sphere(x, y), 1.0, [[2,0,0],[1,0,0,]])

    def test_check_interior(self):
        for _ in  range(self.N_trials):
            radius = rng.uniform(1, 1000, size=(1,))[0]
            center = rng.normal(0, 10, size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)

            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.linalg.norm(test_points - center, axis=1) <= radius + tol

            self.assertTrue(np.array_equal(ref_check, sphere.checkIfInterior(test_points)))


class Test_Plane(unittest.TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        def is_collinear(a,b):
            cos_theta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            return np.isclose(cos_theta, 1.0)
        
        for _ in  range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)

            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(normal, plane.normal))
            self.assertTrue(np.array_equal(point, plane.point))
            self.assertEqual(tol, plane.tol)

            nt, pt = plane.params
            self.assertTrue(np.array_equal(point, pt))
            self.assertTrue(np.isclose(1.0, np.linalg.norm(nt)))
            self.assertTrue(is_collinear(normal, nt))

            plane.flip_orientation()
            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(-normal, plane.normal))

        self.assertRaises(ValueError, lambda x: Plane(x,(0,0,0)), (0,0,0))

    def test_check_interior(self):
        for _ in  range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.dot(test_points - point, normal/np.linalg.norm(normal)) < (tol)
    
            self.assertTrue(np.array_equal(ref_check, plane.checkIfInterior(test_points)))

            # This would fail for any points on the plane or within tolerance of the interiority check
            tol_filter = plane.dist_from_plane(test_points) > plane.tol
            test_points = test_points[tol_filter, :]

            ref_check = np.dot(test_points - point, normal/np.linalg.norm(normal)) < (tol)
            ref_arr = np.logical_not(ref_check)            
            self.assertTrue(np.array_equal(ref_arr, plane.flip_orientation().checkIfInterior(test_points)))

    def test_dist_from_plane(self):
        for _ in range(self.N_trials):
            point = rng.normal(size=(3,))
            plane = Plane([0,0,1], point)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            ref_arr = np.abs(test_points[:, 2]-point[2])
            self.assertTrue(np.allclose(plane.dist_from_plane(test_points), ref_arr))

            Rs = rng.normal(size=(3,3))
            R = Rotation(np.linalg.qr(Rs)[0], origin=point)

            r_plane = Plane.from_alg_object(plane, transformation=[R])
            r_points = R.applyTransformation(test_points)
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

            r_plane.flip_orientation()
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

    def test_project_point(self):
        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            proj_points = plane.project_point(test_points)
            proj_dist = plane.dist_from_plane(proj_points)
            self.assertTrue(np.allclose(proj_dist, np.zeros_like(proj_dist)))

    def test_convex_hull_to_planes(self):
        for _ in range(self.N_trials):
            hull_points = rng.uniform(-5, 5, size=(32, 3))
            test_points = rng.uniform(-10, 10, size=(self.N_test_points, 3))

            tri = Delaunay(hull_points)

            planes = convex_hull_to_planes(hull_points)

            ref_check = tri.find_simplex(test_points) > -1
            test_check = reduce(lambda x, y: np.logical_and(x,y), [p.checkIfInterior(test_points) for p in planes])

            self.assertTrue(np.array_equal(test_check, ref_check))

            

    def test_bounding_box(self):

        for _ in range(self.N_trials):
            hull_points = rng.uniform(-5, 5, size=(32, 3))
            test_points = rng.uniform(-10, 10, size=(self.N_test_points, 3))

            tri = Delaunay(hull_points)
            planes = convex_hull_to_planes(hull_points)

            test_points, status = get_bounding_box(planes)
            ref_points = tri.points[np.unique(tri.convex_hull),:]
            self.assertTrue(np.allclose(np.sort(test_points, axis=0), np.sort(ref_points, axis=0)))
            
        
        # If < 4 planes, should return unbounded
        _, status = get_bounding_box(planes[:4])
        self.assertEqual(status, 3)


        # If two planes mutually exclude eachother, feasible region should be null
        new_plane = Plane.from_alg_object(planes[0])
        new_plane.flip_orientation()
        new_plane.point = new_plane.point - new_plane.normal
        hs, status = get_bounding_box(planes + [new_plane])
        self.assertEqual(status, 2)

        # This raises a QHullerror, since the planes are on top of each other.
        new_plane = Plane.from_alg_object(planes[0])
        new_plane.flip_orientation()
        _, status = get_bounding_box(planes + [new_plane])
        self.assertEqual(status, 2)


class Test_Cylinder(unittest.TestCase):

    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        def is_collinear(a,b):
            cos_theta = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            return np.isclose(cos_theta, 1.0)

        for _ in range(self.N_trials):
            axis = rng.normal(size=(3,1))

            point = rng.normal(size=(3,1))
            
            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)

            cyl = Cylinder(axis, point, radius, length)

            ta, tp, tr, tl = cyl.params()
            
            self.assertEqual(radius, tr)
            self.assertEqual(length, tl)
            self.assertTrue(np.array_equal(point.reshape((3,)), tp))
            self.assertTrue(is_collinear(axis[:,0], ta))
            self.assertTrue(np.isclose(np.linalg.norm(ta), 1.0))

        # good values    
        ga = np.array([0,0,1])
        gp = np.array([0,0,0])
        gr = 1.0
        gl = 1.0

        ## errors on radius
        # type
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, 'a', gl))

        # value
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, -1.0, gl))
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, 0.5*np.finfo(float).eps, gl))
            
        ## errors on length
        # type
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, gr,'a'))

        # value
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, gr, -1.0))
        self.assertRaises(ValueError, lambda : Cylinder(ga, gp, gr, 0.5*np.finfo(float).eps))
            
        ## errors on axis
        # shape
        self.assertRaises(ValueError, lambda: Cylinder([0,0,0,1], gp, gr, gl))

        # value
        self.assertRaises(ValueError, lambda: Cylinder([0,0,0], gp, gr, gl))
            
        ## errors on point
        # shape
        self.assertRaises(ValueError, lambda: Cylinder(ga, [0,0,0,1], gr, gl))

    def test_bounding_box(self):
        def is_collinear(a,b):
            cos_theta = np.dot(a,b)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b))
            return np.all(np.logical_or(np.isclose(cos_theta, 1.0), np.isclose(cos_theta, -1.0)))

        for _ in range(self.N_trials):
            axis = rng.normal(size=(3,1))
            point = rng.normal(size=(3,1))
            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)
            cyl = Cylinder(axis, point, radius, length)
        
            bbox = cyl.get_bounding_box()

            ## Assert shape
            self.assertEqual(bbox.shape, (8,3))

            ## Box is centered correctly
            self.assertTrue(np.allclose(np.mean(bbox, axis=0), point.reshape((3,))))

            ## Box has the right length and is oriented correctly
            dvecs = bbox[:4, :] - bbox[4:, :]
            d = np.linalg.norm(dvecs, axis=1)
            print(d, length)
            self.assertTrue(np.allclose(d, length))
            self.assertTrue(is_collinear(dvecs, axis))

            ## Box has the right volume
            vol = ConvexHull(bbox).volume
            self.assertTrue(np.isclose(vol / (length*(2*radius)**2.0), 1.0))
