"""
Voxel Class
Luis Rangel DaCosta

defines a voxel in 3D cartesian space based on a linear basis set
"""
import numpy as np

class Voxel():

    def __init__(self, bases=np.identity(3), scale=np.array([1]), origin=np.array([0.0, 0.0, 0.0])):
        """
        bases is 3x3 array defining vectors that span 3D space
            [0,:] = [x_1, y_1, z_1]
            [1,:] = [x_2, y_2, z_2]
            [2,:] = [x_3, y_3, z_3]
        scale is broadcastable array defining scaling factors for each vector in bases
        """
        #initialize fields
        self._scale = []
        self._bases = []
        self._origin = []
        self.scale = scale
        self.bases = bases
        self.origin = origin

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        try:
            self.bases*np.array(scale)
        except ValueError:
            raise ValueError("Bases and scale are not broadcastable. Scale array must be of shape=(1,) (3,), (1,3), or (3,1)")
        
        self._scale = np.array(scale)

    @property
    def bases(self):
        return self._bases

    @bases.setter
    def bases(self, bases):
        bases = np.array(bases)

        assert(bases.shape == (3,3)), '''Bases must be 3x3 numpy array that defines vectors that span 3D space
            [0,:] = [x_1, y_1, z_1]
            [1,:] = [x_2, y_2, z_2]
            [2,:] = [x_3, y_3, z_3]'''
        
        #check for collinearity
        assert (not self._collinear(bases[0,:], bases[1,:])), "Bases vectors must linearly indepedent"
        assert (not self._collinear(bases[0,:], bases[2,:])), "Bases vectors must linearly indepedent"
        assert (not self._collinear(bases[1,:], bases[2,:])), "Bases vectors must linearly indepedent"
        
        self._bases = bases
        self

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        assert(origin.shape == (3,)), "Origin must have shape (3,)"
        self._origin = np.array(origin)

    @property
    def sbases(self):
        """
        scaled basis set
        """
        return self._bases*self._scale

    @property
    def corners(self):
        """
        returns 8 corners comprising all single unit linear combinations of basis set
        """
        corners = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]])
        return self.origin + np.array([np.sum(self.sbases*i,axis=0) for i in corners])

    @property
    def midpoint(self):
        """
        return midpoint of voxel defined by basis set
        """
        return self.origin + self.sbases*np.array([1,1,1])/2

    @property
    def reciprocal_bases(self):
        return np.linalg.inv(self.sbases)

    def _collinear(self, vec1, vec2):
        """
        checks if two vectors are collinear
        """
        return np.abs((np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2)))) == 1.0


    def get_extents(self, box):
        """
        determine minimum contiguous block of voxels that fully covers a space 

        solves for vector extents to corners of a simple rectangular bounding box

        box is 8x3 np array
        """
        extents = []
        box = box - self.origin
        for point in box:
            extent = np.linalg.solve(self.sbases,np.array(point))
            extents.append(extent)

        extents = np.array(extents)
        min_extent = np.floor(np.min(extents,axis=0))
        max_extent = np.ceil(np.max(extents,axis=0))

        return min_extent.astype(np.int64), max_extent.astype(np.int64)

    def rotate(self, R):
        self.bases = np.dot(R, self.bases.T).T 