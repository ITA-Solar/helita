import os
import numpy
import setuptools
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from numpy.distutils import fcompiler


try:  # do we have cython?
    from Cython.Build import cythonize
    USE_CYTHON = True
except:
    USE_CYTHON = False
USE_FORTRAN = fcompiler.get_default_fcompiler()

NAME = "helita"
PACKAGES = ["data", "io", "obs", "sim", "utils", "vis"]
VERSION = "0.9.0"

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
                   [os.path.join(NAME, "utils/utilsfast" + ext)]]
}
if USE_FORTRAN:
    EXT_PACKAGES["trnslt"] = ["utils", [], [os.path.join(NAME, "utils/trnslt.f90")]]

extensions = [
    Extension(
        name="%s.%s.%s" % (NAME, pprop[0], pname),
        include_dirs=pprop[1],
        sources=pprop[2])
    for pname, pprop in EXT_PACKAGES.items()
]

if USE_CYTHON:  # Always compile for Python 3 (v2 no longer supported)
    extensions = cythonize(extensions, compiler_directives={'language_level' : "3"})

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
    package_data={'': ['*.pyx', '*.f90', 'data/*']},
    ext_modules=extensions,
    python_requires='>=2.7',
    use_2to3=False
)
