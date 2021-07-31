import numpy as np


class Voxel():
    """Voxel class used to span space for generators and track transformations.

    Voxels provide an alterable view of bases and orientation of crystalline
    generators and are the actual transformed object, not Generators.
    This is in contrst to applying transformations directly to the underlying 
    pymatgen Structure object, for speed and ease of manipulation.
    Voxels also help determine how much of a "block" to a Generator needs to 
    build for the purpose of supplying atoms to a larger volume.

    Attributes:
        scale (float): Scaling factor of basis set.
        bases (np.ndarray): Basis vectors defining crystal unit cell.
        sbases (np.ndarray): Scaled basis set.
        reciprocal_bases (np.ndarray): Basis vectors defining unit cell of reciprocal lattice.
        origin (np.ndarray): Origin of Voxel grid.
    """

    def __init__(self,
                 bases: np.ndarray = np.identity(3),
                 scale: float = np.array([1]),
                 origin: np.ndarray = np.array([0.0, 0.0, 0.0])):
        self._scale = None
        self._bases = None
        self._origin = None
        self.scale = scale
        self.bases = bases
        self.origin = origin

    @property
    def scale(self):
        """Scaling factor of basis set."""
        return self._scale

    @scale.setter
    def scale(self, scale: float):
        try:
            self.bases * np.array(scale)
        except ValueError:
            raise ValueError(
                "Bases and scale are not broadcastable. Scale array must be of shape=(1,) (3,), (1,3), or (3,1)"
            )

        self._scale = np.array(scale)

    @property
    def bases(self):
        """Basis vectors defining crystal unit cell. Vectors are rows of matrix."""
        return self._bases

    @bases.setter
    def bases(self, bases: np.ndarray):
        bases = np.array(bases)

        assert (
            bases.shape == (3, 3)
        ), '''Bases must be 3x3 numpy array that defines vectors that span 3D space
            [0,:] = [x_1, y_1, z_1]
            [1,:] = [x_2, y_2, z_2]
            [2,:] = [x_3, y_3, z_3]'''

        #check for collinearity
        assert (not self._collinear(
            bases[0, :], bases[1, :])), "Bases vectors must linearly indepedent"
        assert (not self._collinear(
            bases[0, :], bases[2, :])), "Bases vectors must linearly indepedent"
        assert (not self._collinear(
            bases[1, :], bases[2, :])), "Bases vectors must linearly indepedent"

        self._bases = bases

    @property
    def origin(self):
        """Origin of Voxel grid in space."""
        return self._origin

    @origin.setter
    def origin(self, origin: np.ndarray):
        assert (origin.shape == (3,)), "Origin must have shape (3,)"
        self._origin = np.array(origin)

    @property
    def sbases(self):
        """Basis vectors defining crystal unit cell, scaled by scaling factor."""
        return self._bases * self._scale

    @property
    def reciprocal_bases(self):
        """Basis vectors defining unit cell of reciprocal lattice."""
        return np.linalg.inv(self.sbases)

    def _collinear(self, vec1: np.ndarray, vec2: np.ndarray):
        """Check if two vectors are collinear. Used for determining validity of basis set.

        Args:
            vec1 (np.ndarray): first vector
            vec2 (np.ndarray): second vector

        Returns:
            bool indicating whether vectors are collinear.
        """
        return np.abs((np.dot(vec1, vec2) /
                       (np.linalg.norm(vec1) * np.linalg.norm(vec2)))) == 1.0

    def get_extents(self, box: np.ndarray):
        """Determine minimum contiguous block of voxels that fully covers a space.

        Args:
            box (np.ndarray): Set of points defining extremities of space.

        Returns:
            Tuple of minimum extents and maximum extents indicating how many
            voxels to tile in space, and where, to span a given region.
        """
        extents = []
        box = box - self.origin
        for point in box:
            extent = np.linalg.solve(self.sbases, np.array(point))
            extents.append(extent)

        extents = np.array(extents)
        min_extent = np.floor(np.min(extents, axis=0))
        max_extent = np.ceil(np.max(extents, axis=0))

        return min_extent.astype(np.int64), max_extent.astype(np.int64)
