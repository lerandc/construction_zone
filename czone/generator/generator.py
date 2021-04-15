"""
Generator Class
Luis Rangel DaCosta
"""
import numpy as np
from ..volume.voxel import Voxel
from pymatgen import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number

#####################################
########## Generator Classes ########
#####################################

class Generator():
    """
    Generator class for crystal systems
    Utilizes pymatgen Lattice and Structure to supply relevant methods
    """
    def __init__(self, origin=None, structure=None):
        self._structure = None
        self._lattice = None
        self._species = None
        self._coords = None
        self._origin = None
        self._orientation = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self._voxel = None

    """
    Properties
    """
    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        assert(isinstance(structure, Structure)), "Structure must be pymatgen Structure object"
        self._structure = structure

    @property
    def lattice(self):
        return self.structure.lattice
    
    @property
    def species(self):
        return self.structure.atomic_numbers
    
    @property
    def coords(self):
        return self.structure.frac_coords

    @property
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
        min_extent, max_extent = self.voxel.get_extents(bbox)
        fcoords = np.copy(self.coords) #grab fractional in case bases are rotated

        coords = []
        species = []
        for i in range(min_extent[0], max_extent[0]+1):
            for j in range(min_extent[1], max_extent[1]+1):
                for k in range(min_extent[2], max_extent[2]+1):
                    new_coords = np.matmul(self.voxel.sbases, np.array([i,j,k])) \
                                + np.dot(self.voxel.sbases, fcoords.T).T \
                                + self.voxel.origin
                    coords.append(new_coords)
                    species.append(self.species)

        return np.squeeze(np.array(coords)), np.squeeze(np.array(species))

    def rotate(self,R):
        #rotate basis of voxel for equivalence
        self.voxel.rotate(R)


#####################################
######### Utility routines ##########
#####################################

def BasicStructure(Z=[1], coords=[[0.0, 0.0, 0.0]], cellDims=[2.5, 2.5, 2.5], cellAngs=[90, 90, 90]):
    """
    Define simple translating unit
    """
    tmp = Generator()
    tmp.species = Z
    tmp.lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                            alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp.structure = Structure(tmp.lattice, Z, coords)
    tmp.voxel = Voxel(scale=cellDims[0])

    return tmp

def from_spacegroup(Z, coords, cellDims, cellAngs, sgn=None, sym=None):
    if sym is None:
        sg = SpaceGroup(int_symbol=sg_symbol_from_int_number(sgn))
    else:
        sg = SpaceGroup(int_symbol=sym)

    test_lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                        alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp = Generator()
    tmp.structure = Structure.from_spacegroup(sg.int_number, lattice=b.lattice, species=Z, coords=coords)
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