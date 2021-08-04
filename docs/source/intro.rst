Intro
=================================

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