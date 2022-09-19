# Construction Zone
[![DOI](https://zenodo.org/badge/261777347.svg)](https://zenodo.org/badge/latestdoi/261777347)

Modules for generating nanoscale+ atomic scenes, primarily using pymatgen as generators with S/TEM image simulation in mind.

Full documentation can be found here: [https://construction-zone.readthedocs.io/](https://construction-zone.readthedocs.io/).

If you use Construction Zone in your own work, we kindly ask that you cite the following:
Rangel DaCosta, Luis, & Scott, Mary. (2021). Construction Zone (v2021.08.04). Zenodo. https://doi.org/10.5281/zenodo.5161161



## Installation

Construction Zone can be easily installed with pip:
```
pip install czone
```

We strongly recommend utilizing an environment manager such as Anaconda, and 
installing Construction Zone into your environment of choice.  

To install Construction Zone into a clean environment with Anaconda, you could
do the following:
```
conda create -n environment_name python=3.7
conda activate environment_name
pip install czone
```

In this example, we manually set the target Python version to v3.7. 
Construction Zone has been tested only for Python 3.7 and above; earlier versions
may work but are not supported.

Stable versions of Construction Zone will be passed onto PyPi. To use the current,
development version of Construction Zone, you can set up the Python package in
development mode. Again, we recommend doing so with an environment manager.

An example development installation could be achieved by the following:
```
conda create -n environment_name python=3.7
conda activate environment_name
git clone https://github.com/lerandc/construction_zone.git 
cd construction_zone
python setup.py --develop
```

Development mode installations import Python packages directly from the source
every time. You could freely edit the source code yourself, or just use the 
installation to pull fresh code from the development branch by running `git pull`
in the repository directory.

## Acknowledgment

We acknowledge support for the development of Construction Zone from the Toyota Research Institute.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, 
Office of Advanced Scientific Computing Research, Department of Energy Computational Science Graduate Fellowship 
under Award Number DE-SC0021110.

This software was prepared as an account of work sponsored by an agency of the United
States Government. Neither the United States Government nor any agency thereof, nor any of their
employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the
accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or
represents that its use would not infringe privately owned rights. Reference herein to any specific
commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or favoring by the United States
Government or any agency thereof. The views and opinions of authors expressed herein do not
necessarily state or reflect those of the United States Government or any agency thereof.


[<img src="docs/source/imgs/csgf_logo.png" width="267" height="80" />](https://www.krellinst.org/csgf/)

[![TRI Logo](docs/source/imgs/toyota_research_institute.png)](https://www.tri.global/)