"""
Set of tools to interface with output and input from simulations
and radiative transfer codes. Also includes routines for working
with synthetic spectra.
"""

import imp

try:
    imp.find_module('at_tools')
    found = True
except ImportError:
    found = False

if found:
    __all__ = ["bifrost", "multi", "multi3d", "muram", "rh", "rh15d",
            "simtools", "synobs", "ebysus"]
else:
    __all__ = ["bifrost", "multi", "multi3d", "muram", "rh", "rh15d",
            "simtools", "synobs"]


from . import bifrost
from . import multi
from . import muram
from . import rh
if found:
    from . import ebysus
