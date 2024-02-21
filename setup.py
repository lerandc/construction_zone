from setuptools import setup, find_packages

setup(
    name='czone',
    version='2022.09.20',
    description='An open source python package for generating nanoscale+ atomic scenes',
    url='https://github.com/lerandc/construction_zone',
    author='Luis Rangel DaCosta',
    author_email='luisrd@berkeley.edu',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'pymatgen == 2024.2.20',
        'numpy >= 1.16.2, <1.22.0',
        'scipy >= 1.3.0, <1.8',
        'ase >= 3.21.0',
        'wulffpack >= 1.1.0']
)
