"""
Generator Class
Luis Rangel DaCosta
"""
from pymatgen import Structure, Lattice

class Generator():

    def __init__(self):
        self._structure = None
        self._lattice = None
        self._species = None
        self._coords = None
        self._orientation = None
        self._transformations = [] #Sequence of transformations 

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

    """ 
    Operations
    """


def BasicStructure(Z=[1], coords=[[0.0, 0.0, 0.0]], cellDims=[2.5, 2.5, 2.5], cellAngs=[90, 90, 90]):
    """
    Define simple translating unit
    """
    tmp = Generator()
    tmp.coords = coords
    tmp.species = Z
    tmp.lattice = Lattice.from_parameters(a=cellDims[0], b=cellDims[1], c=cellDims[2],
                                            alpha=cellAngs[0], beta=cellAngs[1], gamma=cellAngs[2])

    tmp.structure = Structure(self.lattice, Z, coords)

    return 