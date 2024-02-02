## Testing roadmap for Construction Zone:

Test goals below are listed in relative priority/dependence order, where relevant.

High Priority:
- Finish unit tests for algebraic volumes, excluding Miller plane utility
- Unit tests for Volumes and Multivolumes
- Unit tests for core Generator objects
- Write direct unit tests for Transform core
- Unit tests for Scenes
- Integration tests for Volumes-Generators-Transforms, comparing against Molecules to ensure structure generation consistency
- Unit tests for Amorphous Generators and associated algorithms, utils
- Parial unit tests for Prefab core (not including example defect classes)

Medium priority:
- Unit tests for Prefab core defect classes
- Unit tests for Wulff particles
- Unit tests for Alpha shapes
- Unit tests for Adsorbate algorithms and utilities
- Unit tests for Post transformations and Strain utilities

Low priority:
- Unit tests for Misc. Utils (where not tested directly elsewhere)
- Unit tests for Voxel module (or, deprecation and refactoring into Generator)
- Unit tests and/or deprecation of Viz utilities

Other goals (to be included as part of a feature roadmap):
- Expose and unify handling of RNGs 
- Write doc strings where possible
- Implement __repr__ methods where missing

As tests are completed, we can expect significant refactoring and clarification of API, where possible, as well as implementation of
better error handling. 

### Manual Coverage Summaray (LRD, 2 Feb. 2024):
(x = total, + = partial (direct),  - = partial (indirect), blank = none)

[ ] Generator \
 | [ ] Amorphous Algorithms \
 | [ ] Generator Core
 
 [x] Molecule

 [ ] Prefab \
 | [ ] Prefab Core \
 | [ ] Wulff

 [ ] Scene

 [ ] Surface \
 | [ ] Adsorbate \
 | [ ] Alpha shapes

 [-] Transform \
 | [ ] Post \
 | [ ] Strain \
 | [-] Transform Core

 [ ] Util \
 | [ ] Measure \
 | [ ] Misc.

 [ ] Viz 

 [+] Volume \
 | [+] Algebraic \
 | [ ] Volume Core \
 | [ ] Voxel 


### Detailed coverage summary, via Coverage Report + pytest (LRD, 2 Feb. 2024):

```
Name                                                                         Stmts   Miss  Cover   Missing
----------------------------------------------------------------------------------------------------------
czone/__init__.py                             8      0   100%
czone/generator/__init__.py                   2      0   100%
czone/generator/amorphous_algorithms.py     137    127     7%   15-20, 23-29, 33-35, 45-69, 107-158, 168-245
czone/generator/generator.py                185    105    43%   44, 54-60, 91-104, 113, 117-126, 132, 137, 142, 147, 152,
                                                                156-160, 165, 169-172, 177, 181-183, 188, 192-194, -236,
                                                                244-252, 280-297, 316-321, 348-360, 369, 373-375, 380,
                                                                384, 389, 393-394, 399, 403-404, 409, 416-422, 436, 439
czone/molecule/__init__.py                    1      0   100%
czone/molecule/molecule.py                  156      1    99%   229
czone/prefab/__init__.py                      2      0   100%
czone/prefab/prefab.py                      262    178    32%   32, 65-81, 86, 90, 95, 99-102, 107, 111, 116, 120-122, 
                                                                127, 131-133, 138, 142-145, 151-218, 244, 252, 256, 281, 
                                                                289, 293, 325-337, 342, 346, 351, 355-358, 363, 368, 
                                                                372-374, 379, 383-386, 393-467, 489-495, 499, 503-507, 
                                                                511, 515-522, 526-540
czone/prefab/wulff.py                       170    106    38%   44, 53, 57-61, 66, 71, 75-78, 83, 87-91, 107, 122, 135, 
                                                                144-147, 160-162, 165-169, 172, 191-195, 208-210, 223-225,
                                                                238-240, 253-255, 317-320, 325, 329-330, 333-369, 398-399,
                                                                413-417, 420-459
czone/scene/__init__.py                       1      0   100%
czone/scene/scene.py                         85     55    35%   25-43, 48, 52-54, 59, 68-72, 77, 81, 86, 92, 98-99, 109, 
                                                                122-133, 147-164, 177-187
czone/surface/__init__.py                     2      0   100%
czone/surface/adsorbate.py                  118    106    10%   39-49, 70-110, 129-145, 181-292
czone/surface/alpha_shape.py                 75     69     8%   21-49, 66-110, 127-185
czone/transform/__init__.py                   3      0   100%
czone/transform/post.py                      47     28    40%   20, 34, 39-42, 46-55, 58, 63, 66, 74-76, 84-92, 96-99
czone/transform/strain.py                   113     71    37%   26, 38, 46-50, 55, 59-60, 64, 68, 73, 77-78, 85, 102-117,
                                                                126, 130-144, 152-164, 184-198, 207, 211-214, 219, 
                                                                223-232, 240-258
czone/transform/transform.py                228    104    54%   44, 56, 69, 75, 80, 91, 111-118, 122, 126-129, 133, 136,
                                                                139, 142-148, 204, 216, 221-226, 234, 251, 297-303, 307, 
                                                                312, 316-321, 325, 328-337, 352-355, 360, 364, 369-370, 
                                                                375-381, 389-398, 401-403, 406-408, 411-413, 431-434, 
                                                                447-472, 486, 512-519, 546-560
czone/util/__init__.py                        2      0   100%
czone/util/measure.py                       118    112     5%   33-156, 174-193, 211-253
czone/util/misc.py                           25     18    28%   16, 31-54, 67
czone/viz/__init__.py                         1      0   100%
czone/viz/viz.py                            171    159     7%   5, 14-66, 70-149, 153-232, 236-254, 261-279
czone/volume/__init__.py                      3      0   100%
czone/volume/algebraic.py                   226     72    68%   42, 47, 246-250, 254, 259, 263-265, 270, 274-276, 281, 
                                                                285-289, 294, 298-302, 305-313, 319-339, 364, 441-492
czone/volume/volume.py                      246    175    29%   41, 46, 51, 56, 61, 65-68, 77, 82, 91, 100, 138-165, 174,
                                                                178-182, 187, 195-198, 203, 208, 213, 218, 222-223, 
                                                                226-235, 244-247, 255-274, 278-291, 294-312, 320-344, 
                                                                348-353, 366-378, 404-410, 415, 423-431, 444-455, 458-462,
                                                                465-477, 481-502, 517-521, 543-554
czone/volume/voxel.py                        55     34    38%   26-31, 36, 40-47, 52, 56-73, 78, 82-83, 88, 93, 105, 
                                                                118-128
test_molecule.py                                                               176      0   100%
test_volume.py                                                                 130      0   100%
----------------------------------------------------------------------------------------------------------
TOTAL                                                                         2748   1520    45%
```