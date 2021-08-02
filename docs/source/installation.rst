Installation
=================================

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
