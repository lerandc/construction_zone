from setuptools import setup, find_packages

setup(
    name='czone',
    version='0.1',
    packages=find_packages(),
    description='An open source python package for generating nanoscale+ atomic scenes',
    url='https://github.com/lerandc/construction_zone',
    author='Luis Rangel DaCosta',
    author_email='luisrd@berkeley.edu',
    python_requires='>=3.7',
    install_requires=[
        'pymatgen >= 2020.4.29',
        'numpy >= 1.16.2',
        'scipy >= 1.3.0',
        'ase >= 3.21.0']
)