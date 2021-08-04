from setuptools import setup, find_packages

setup(
    name='czone',
    version='2021.08.04',
    description='An open source python package for generating nanoscale+ atomic scenes',
    url='https://github.com/lerandc/construction_zone',
    author='Luis Rangel DaCosta',
    author_email='luisrd@berkeley.edu',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'pymatgen >= 2020.4.29',
        'numpy >= 1.16.2',
        'scipy >= 1.3.0',
        'ase >= 3.21.0',
        'wulffpack >= 1.1.0']
)