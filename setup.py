#!/usr/bin/env python
from setuptools import setup  # isort:skip
import os
from itertools import chain

import numpy
from Cython.Build import cythonize
from numpy.distutils import fcompiler
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

try:
    # Recommended for setuptools 61.0.0+
    # (though may disappear in the future)
    from setuptools.config.setupcfg import read_configuration
except ImportError:
    from setuptools.config import read_configuration

################################################################################
# Programmatically generate some extras combos.
################################################################################
extras = read_configuration("setup.cfg")['options']['extras_require']

# Dev is everything
extras['dev'] = list(chain(*extras.values()))

# All is everything but tests and docs
exclude_keys = ("tests", "docs", "dev")
ex_extras = dict(filter(lambda i: i[0] not in exclude_keys, extras.items()))
# Concatenate all the values together for 'all'
extras['all'] = list(chain.from_iterable(ex_extras.values()))

################################################################################
# Cython extensions
################################################################################
NUMPY_INC = numpy.get_include()
EXT_PACKAGES = {
    "anapyio": ["io", [NUMPY_INC, os.path.join("helita", "io/src")],
                [os.path.join("helita", "io/anapyio.pyx"),
                 os.path.join("helita", "io/src/libf0.c"),
                 os.path.join("helita", "io/src/anacompress.c"),
                 os.path.join("helita", "io/src/anadecompress.c")]],
    "radtrans": ["utils", [NUMPY_INC],
                 [os.path.join("helita", "utils/radtrans.pyx")]],
    "utilsfast": ["utils", [NUMPY_INC],
                  [os.path.join("helita", "utils/utilsfast.pyx")]]
}
extensions = [
    Extension(
        name=f"helita.{pprop[0]}.{pname}",
        include_dirs=pprop[1],
        sources=pprop[2])
    for pname, pprop in EXT_PACKAGES.items()
]
extensions = cythonize(extensions, compiler_directives={'language_level': "3"})

################################################################################
# Setup
################################################################################
setup(
    extras_require=extras,
    use_scm_version=True,
    ext_modules=extensions,
)
