# import builtins
import warnings

# import internal modules
from . import document_vars
from .load_arithmetic_quantities import do_cstagger

## import the relevant things from the internal module "units"
from .units import (
  UNI, USI, UCGS, UCONST,
  Usym, Usyms, UsymD,
  U_TUPLE,
  DIMENSIONLESS, UNITS_FACTOR_1, NO_NAME,
  UNI_length, UNI_time, UNI_mass,
  UNI_speed, UNI_rho, UNI_nr, UNI_hz
)

# import external public modules
import numpy as np

def load_fromfile_quantities(obj, quant, order='F', mode='r', panic=False, save_if_composite=False, cgsunits=None, **kwargs):
  '''loads quantities which are stored directly inside files.

  save_if_composite: False (default) or True.
    use True for bifrost; False for ebysus.
    See _get_composite_var() for more details.

  cgsunits: None or value
    None --> ignore
    value --> multiply val by this value if val was a simple var.
  '''
  __tracebackhide__ = True  # hide this func from error traceback stack.

  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'fromfile',
      ('These are the quantities which are stored directly inside the snapshot files.\n'
       'Their values are "calculated" just by reading the appropriate part of the appropriate file.\n'
       '(Except for composite_var, which is included here only because it used to be in bifrost.py.)')
                              )

  val = obj._get_simple_var(quant, order=order, mode=mode, panic=panic, **kwargs) # method of obj.
  if not (None in (val, cgsunits)):   # val and cgsunits are both not None
    val = val*cgsunits
  if val is None:
    val = _get_simple_var_xy(obj, quant, order=order, mode=mode) # method defined in this file.
  if val is None:
    val = _get_composite_var(obj, quant, save_if_composite=save_if_composite, **kwargs) # method defined in this file.
  return val


@document_vars.quant_tracking_simple('SIMPLE_XY_VAR')
def _get_simple_var_xy(obj, var, order='F', mode='r'):
  ''' Reads a given 2D variable from the _XY.aux file '''
  if var == '':
    docvar = document_vars.vars_documenter(obj, 'SIMPLE_XY_VAR', getattr(obj, 'auxxyvars', []), _get_composite_var.__doc__)
    # TODO << fill in the documentation for simple_xy_var quantities here.
    return None

  if var not in obj.auxxyvars:
    return None

  # determine the file
  fsuffix  = '_XY.aux'
  idx      = obj.auxxyvars.index(var)
  filename = obj.file_root + fsuffix

  # memmap the variable
  if not os.path.isfile(filename):
    raise FileNotFoundError('_get_simple_var_xy: variable {} should be in {} file, not found!'.format(var, filename))
  dsize = np.dtype(obj.dtype).itemsize    # size of the data type
  offset = obj.nx * obj.ny * idx * dsize  # offset in the file
  return np.memmap(filename, dtype=obj.dtype, order=order, mode=mode,
                   offset=offset, shape=(obj.nx, obj.ny))


# default
_COMPOSITE_QUANT = ('COMPOSITE_QUANT', ['ux', 'uy', 'uz', 'ee', 's'])
# get value
@document_vars.quant_tracking_simple(_COMPOSITE_QUANT[0])
def _get_composite_var(obj, var, *args, save_if_composite=False, **kwargs):
  ''' gets velocities, internal energy ('e' / 'r'), entropy.

  save_if_composite: False (default) or True.
    if True, also set obj.variables[var] = result.
    (Provided for backwards compatibility with bifrost, which
    used to call _get_composite_var then do obj.variables[var] = result.)
    (True is NOT safe for ebysus, unless proper care is taken to save _metadata to obj.variables.)

  *args and **kwargs go to get_var.
  '''
  if var == '':
      docvar = document_vars.vars_documenter(obj, *_COMPOSITE_QUANT, _get_composite_var.__doc__, nfluid=1)
      for ux in ['ux', 'uy', 'uz']:
          docvar(ux, '{x:}-component of velocity [simu. velocity units]'.format(x=ux[-1]), uni=UNI_speed)
      docvar('ee', "internal energy. get_var('e')/get_var('r').", uni_f=UNI.e/UNI.r, usi_name=Usym('J'))
      docvar('s', 'entropy (??)', uni=DIMENSIONLESS)
      return None

  if var not in _COMPOSITE_QUANT[1]:
      return None

  if var in ['ux', 'uy', 'uz']:  # velocities
      # u = p / r.
      ## r is on center of grid cell, but p is on face, 
      ## so we need to interpolate.
      ## r is at (0,0,0), ux and px are at (-0.5, 0, 0)
      ## --> to align r with px, shift by xdn
      x = var[-1]  # axis; 'x', 'y', or 'z'
      interp = x+'dn'
      p = obj.get_var('p' + x)
      r = obj.get_var('r' + interp)
      return p / r

  elif var == 'ee':   # internal energy
      return obj.get_var('e') / obj.get_var('r')

  elif var == 's':   # entropy?
      return np.log(obj.get_var('p', *args, **kwargs)) - \
          obj.params['gamma'][obj.snapInd] * np.log(
              obj.get_var('r', *args, **kwargs))