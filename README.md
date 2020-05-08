# construction_zone
Modules for generating nanoscale+ atomic scenes, primarily using pymatgen as generators with S/TEM image simulation in mind


Current TODOs:
-Add scene class
-Add convex hull collision between two volumes
    -currently attribute access errors with 
    -add midpoint generation for interiority check
    -alter interior/planar check to return logical list instead of returning
-Add complex volume class
    -needs operators for geometry union/intersection/subtraction etc.
-Add atomic generator class