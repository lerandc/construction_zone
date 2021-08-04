Construction Zone
=============================================

Construction Zone is an open-source Python package designed as a tool to help
build and generate complex nanoscale atomic models. The package was designed with
simulation workflows, like TEM and MD simulations, and machine learning pipelines in mind.
Its interface is flexible and allows for easy development of progammatic constructions
that are suitable for applications in materials science, and much of the bookkeeping
typically required when making complex atomic objects is abstracted from the user.

The basis design principle of Construction Zone is to a sculpting approach to creating atomic objects and scenes.
Construction Zone implements several core classes which help accomplish this task.
Generator objects are additive elements that provide the sculpting blocks--
they are classes which fill a given space with atoms, whether that is as a crystalline 
lattice or an amorphous collection of points. Volume objects are subtractive elements
that define the boundaries of an nanoscale structure. Together, Volumes and Generators
can be joined together and treated as semantic objects, like a nanoparticle or a substrate.
Multiple objects in space can interact with eachother in Scenes, which take care
of conflict resolutions like object intersetion and atom overlapping. 
Volumes and Generators can also be transformed with Transformation objects, which
can apply arbitrary transformations to the structures at hand, like symmetry operations
or strain fields. 

If you use Construction Zone in your own work, we kindly ask that you cite the following:
Rangel DaCosta, Luis, & Scott, Mary. (2021). Construction Zone (v2021.08.04). Zenodo. https://doi.org/10.5281/zenodo.5161161

.. toctree::
   :maxdepth: 4
   :caption: Contents

   installation
   examples
   modules
   references
   license
   acknowledgement