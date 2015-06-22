import os
import sys
# from distutils.core import setup
from setuptools import setup

setup(name='fusedwind',
      version="0.3",
      description="",
      long_description="""\
""",
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
      ],
      keywords='',
      author='',
      author_email='',
      url='http://fusedwind.org',
      license='Apache License, Version 2.0',
      packages=[
          'fusedwind',
          'fusedwind.core',
          'fusedwind.util',
          'fusedwind.test',
          'fusedwind.variables',
          'fusedwind.turbine',
          'fusedwind.plant_flow',
          'fusedwind.plant_cost',
      ],
      install_requires=[
        'six', 'Sphinx', 'numpydoc', 'networkx',
      ],
      entry_points= """
      """
    )

