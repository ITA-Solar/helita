"""
Set of tools to interface with output and input from simulations
and radiative transfer codes. Also includes routines for working
with synthetic spectra.
"""

try:
    found = True
except ImportError:
    found = False


try:
    PYCUDA_INSTALLED = True
except ImportError:
    PYCUDA_INSTALLED = False


if found:
    __all__ = ["bifrost", "multi", "multi3d", "muram", "rh", "rh15d",
               "simtools", "synobs", "ebysus", "cipmocct", "laresav",
               "pypluto", "matsumotosav"]
else:
    __all__ = ["bifrost", "multi", "multi3d", "muram", "rh", "rh15d",
               "simtools", "synobs"]


from . import bifrost, multi, rh

if found:
    from . import muram
