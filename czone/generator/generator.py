"""
Generator Class
Luis Rangel DaCosta
"""
from pymatgen import Structure, Lattice
from ..volume.voxel import Voxel
import numpy as np

class Generator():

    def __init__(self):
        self._structure = None
        self._lattice = None
        self._species = None
        self._coords = None
        self._orientation = None
        self._origin = None
        self._transformations = [] #Sequence of transformations 
        self._voxel = None

    """
    Properties
    """
    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value

    @property
    def lattice(self):
        return self._lattice
    
    @lattice.setter
    def lattice(self, value):
        self._lattice = value

    @property
    def species(self):
        return self._species
    
    @species.setter
    def species(self, value):
        self._species = value

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = value

    def orientation(self, type):
        return self._orientation

    @property
    def voxel(self):
        return self._voxel

    @voxel.setter
    def voxel(self, voxel):
        if not isinstance(voxel, Voxel):
            raise TypeError("Supplied voxel is not of Voxel() class")

        self._voxel = voxel

    """ 
    Operations
    """
    def supply_atoms(self, bbox):
        """
        Add atoms to bounding box region
        """
        #generation method for pymatgen lattice structure
        if(isinstance(self.structure, Structure)):
            min_extent, max_extent = self.voxel.get_extents(bbox)
            fcoords = np.copy(self.structure.frac_coords) #grab fractional in case bases are rotated

            coords = []
            species = []
            for i in range(min_extent[0], max_extent[0]+1):
                for j in range(min_extent[1], max_extent[1]+1):
                    for k in range(min_extent[2], max_extent[2]+1):
                        new_coords = np.matmul(self.voxel.sbases, np.array([i,j,k])) \
                                    + np.dot(self.voxel.sbases, fcoords.T).T
                        coords.append(new_coords)
                        species.append(self.species)

            return np.squeeze(np.array(coords)), np.squeeze(np.array(species))

    def rotate(self,R):
        #rotate basis of voxel for equivalence
        self.voxel.rotate(R)
                        



def BasicStructure(Z=[1], coords=[[0.0, 0.0, 0.0]], cellDims=[2.5, 2.5, 2.5], cellAngs=[90, 90, 90]):
    """
    Define simple translating unit
    """
    tmp = Generator()
    tmp.coords = coords
    tmp.species = Z
    tmp.lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                            alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp.structure = Structure(tmp.lattice, Z, coords)
    tmp.voxel = Voxel(scale=cellDims[0])

    return tmp

def from_generator(orig, **kwargs):
    """
    Construct a generator from another generator
    **kwargs encodes relationship
    """

    if not isinstance(orig, Generator):
        raise TypeError("Supplied generator is not of Generator() class")

    new_generator = copy.deepcopy(orig)
    if "species" in kwargs.keys():
        new_generator.species = kwargs["species"]

    if "translate" in kwargs.keys():
        new_generator

    return new_generator