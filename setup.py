import os
import numpy
import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension


try:  # do we have cython?
    from Cython.Build import cythonize
    USE_CYTHON = True
except:
    USE_CYTHON = False

NAME = "helita"
PACKAGES = ["io", "obs", "sim", "utils"]
VERSION = "0.8.0"

ext = '.pyx' if USE_CYTHON else '.c'
NUMPY_INC = numpy.get_include()
EXT_PACKAGES = {   # C and Fortran extensions
    "anapyio" : ["io", [NUMPY_INC, os.path.join(NAME, "io/src")],
                 [os.path.join(NAME, "io/anapyio" + ext),
                  os.path.join(NAME, "io/src/libf0.c"),
                  os.path.join(NAME, "io/src/anacompress.c"),
                  os.path.join(NAME, "io/src/anadecompress.c")]],
    "cstagger" : ["sim", [NUMPY_INC],
                  [os.path.join(NAME, "sim/cstagger" + ext)]],
    "radtrans" : ["utils", [NUMPY_INC],
                  [os.path.join(NAME, "utils/radtrans" + ext)]],
    "utilsfast" : ["utils", [NUMPY_INC],
                   [os.path.join(NAME, "utils/utilsfast" + ext)]],
    "voigtv" : ["utils", [], [os.path.join(NAME, "utils/voigtv.f")]]
}

extensions = [
    Extension(
        name="%s.%s.%s" % (NAME, pprop[0], pname),
        include_dirs=pprop[1],
        sources=pprop[2])
    for pname, pprop in EXT_PACKAGES.items()
]

if USE_CYTHON:
    extensions = cythonize(extensions)

setup(
    name=NAME,
    version=VERSION,
    description="Solar physics python tools from ITA/UiO",
    author="Tiago M. D. Pereira et al.",
    license="BSD",
    url="http://%s.readthedocs.io" % NAME,
    keywords=['astronomy', 'astrophysics', 'solar physics', 'space', 'science'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Cython',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: Implementation :: CPython',
          'Topic :: Scientific/Engineering :: Astronomy',
          'Topic :: Scientific/Engineering :: Physics'
    ],
    packages=[NAME] + ["%s.%s" % (NAME, package) for package in PACKAGES],
    package_data={'': ['*.pyx', '*.f']},
    ext_modules=extensions,
    python_requires='>=2.7',
    use_2to3=False
)
