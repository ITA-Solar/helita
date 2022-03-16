"""
These quantities relate to doing manipulations.
Frequently, they are "added" to regular variable names.
Examples:
  - get_var('u2') is roughly equal to get_var('ux')**2 + get_var('uy')**2 + get_var('uz')**2
  - get_var('drdxdn') takes derivative of 'r' and pushes down in x.
  - get_var('rxup') pushes 'r' up in x, by interpolating.
In general, these are not hard coded for every variable, but rather you will add to names.
For example, you can do get_var('d'+var+'dxdn') for any var which get_var knows how to get.

Interpolation guide:
  'up' moves up by 0.5 (i.e. half a grid cell)
  'dn' moves down by 0.5 (i.e. half a grid cell)
  scalars are in center of cell, at (0,0,0).
    e.g.: density, energy
  B, p are on the faces of the cell. Example:
    Bx at ( -0.5,  0  ,  0   )
    By at (  0  , -0.5,  0   )
    Bz at (  0  ,  0  , -0.5 )
    B = magnetic field; p = momentum density.
  E, i are on the edges of the cell. Example:
    Ex at (  0  , -0.5, -0.5 )
    Ey at ( -0.5,  0  , -0.5 )
    Ez at ( -0.5, -0.5,  0   )
    E = electric field; i = current per unit area.
"""


# import built-ins
from multiprocessing.dummy import Pool as ThreadPool
import warnings

# import internal modules
from . import document_vars
try:
  from . import cstagger
except ImportError:
  warnings.warn("failed to import helita.sim.cstagger; running stagger with stagger_kind='cstagger' will crash.")
try:
  from . import stagger
except ImportError:
  warnings.warn("failed to import helita.sim.stagger; running stagger with stagger_kind='stagger' will crash.")

## import the relevant things from the internal module "units"
from .units import (
  UNI, USI, UCGS, UCONST,
  Usym, Usyms, UsymD,
  U_TUPLE,
  DIMENSIONLESS, UNITS_FACTOR_1, NO_NAME,
  UNI_length, UNI_time
)

# import external public modules
import numpy as np

# set constants
AXES      = ('x', 'y', 'z')
YZ_FROM_X = dict(x=('y', 'z'), y=('z', 'x'), z=('x', 'y'))  # right-handed coord system x,y,z given x.
EPSILON   = 1.0e-20   # small number which is added in denominators of some operations.


# we need to convert to float32 before doing cstagger.do.
## not sure why this conversion isnt done in the cstagger method, but it is a bit 
## painful to change the method itself (required re-installing helita for me) so we will
## instead just change our calls to the method here. -SE Apr 22 2021
CSTAGGER_TYPES = ['float32']  # these are the allowed types
def do_stagger(arr, operation, default_type=CSTAGGER_TYPES[0], obj=None):
  '''does stagger of arr.
  For stagger_kind='cstagger', first does some preprocessing:
    - ensure arr is the correct type, converting if necessary.
      if type conversion is necessary, convert to default_type.
    - TODO: check _can_interp here, instead of separately.
  For other stagger kinds,
    - assert that obj has been provided
    - call stagger via obj: obj.stagger.do(arr, operation)
  '''
  kind = getattr(obj,'stagger_kind', stagger.DEFAULT_STAGGER_KIND)
  if kind == 'cstagger': # use cstagger routine.
    arr = np.array(arr, copy=False)     # make numpy array, if necessary.
    if arr.dtype not in CSTAGGER_TYPES: # if necessary,
      arr = arr.astype(default_type)      # convert type
    return cstagger.do(arr, operation)  # call the original cstagger function
  else:                  # use stagger routine.
    assert obj is not None, f'obj must be provided to use stagger, with stagger_kind = {stagger_kind}.'
    return obj.stagger.do(arr, operation)

do_cstagger = do_stagger   # << alias, for backwards compatibility.

def _can_interp(obj, axis, warn=True):
  '''return whether we can interpolate. Make warning if we can't.
  must check before doing any cstagger operation.
  pythonic stagger methods (e.g. 'numba', 'numpy') make this check on their own.
  '''
  if not obj.do_stagger:  # this is True by default; if it is False we assume that someone 
    return False       # intentionally turned off interpolation. So we don't make warning.
  kind = getattr(obj,'stagger_kind', stagger.DEFAULT_STAGGER_KIND)
  if kind != 'cstagger':
    return True   # we can let the pythonic methods check _can_interp on their own.
  if not getattr(obj, 'cstagger_exists', False):
    if obj.verbose:
      warnmsg = 'interpolation requested, but cstagger not initialized, for obj={}! '.format(object.__repr__(obj)) +\
              'We will skip the interpolation, and instead return the original value.'
      warnings.warn(warnmsg) # warn user we will not be interpolating! (cstagger doesn't exist)
    return False
  if not getattr(obj, 'n'+axis, 0) >=5:
    if obj.verbose:
      warnmsg = 'requested interpolation in {x:} but obj.n{x:} < 5. '.format(x=axis) +\
              'We will skip this interpolation, and instead return the original value.'
      warnings.warn(warnmsg) # warn user we will not be interpolating! (dimension is too small)
    return False
  return True


''' --------------------- functions to load quantities --------------------- '''

def load_arithmetic_quantities(obj,quant, *args__None, **kwargs__None):
  '''load arithmetic quantities.
  *args__None and **kwargs__None go to nowhere.
  '''
  __tracebackhide__ = True  # hide this func from error traceback stack.
  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'arquantities', 'Computes arithmetic quantities')

  # tell which funcs to use for getting things. (funcs will be called in the order listed here)
  _getter_funcs = (
    get_center, get_deriv, get_interp,
    get_module, get_horizontal_average,
    get_gradients_vect, get_gradients_scalar, get_vector_product,
    get_square, get_lg, get_numop, get_ratios,
    get_projections, get_angle,
    get_stat_quant, get_fft_quant,
  )

  val = None
  # loop through the function and QUANT pairs, running the functions as appropriate.
  for getter in _getter_funcs:
    val = getter(obj, quant)
    if val is not None:
      break
  else:  # didn't break; val is still None
    return None
  # << did break; found a non-None val.
  document_vars.select_quant_selection(obj)  # (bookkeeping for obj.got_vars_tree(), obj.get_units(), etc.)
  return val


# default
_DERIV_QUANT = ('DERIV_QUANT', ['d'+x+up for up in ('up', 'dn') for x in AXES])
# get value
def get_deriv(obj,quant):
  '''
  Computes derivative of quantity.
  Example: 'drdxup' does dxup for var 'r'.
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_DERIV_QUANT, get_deriv.__doc__,
                                           uni=UNI.quant_child(0) / UNI_length)
    docvar('dxup',  'spatial derivative in the x axis with half grid up [simu units]')
    docvar('dyup',  'spatial derivative in the y axis with half grid up [simu units]')
    docvar('dzup',  'spatial derivative in the z axis with half grid up [simu units]')
    docvar('dxdn',  'spatial derivative in the x axis with half grid down [simu units]')
    docvar('dydn',  'spatial derivative in the y axis with half grid down [simu units]')
    docvar('dzdn',  'spatial derivative in the z axis with half grid down [simu units]')
    return None

  getq = quant[-4:]   # the quant we are "getting" by this function. (here: dxup, dyup, ..., or dzdn)

  if not (quant[0] == 'd' and getq in _DERIV_QUANT[1]):
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _DERIV_QUANT[0], delay=True)

  # interpret quant string
  axis = quant[-3]
  q    = quant[1:-4]  # base variable 
  var  = obj.get_var(q)

  # handle "cant interpolate" case
  if not _can_interp(obj, axis):
    if obj.verbose:
      warnings.warn("Can't interpolate; using np.gradient to take derivative, instead.")
    xidx = dict(x=0, y=1, z=2)[axis]  # axis; 0, 1, or 2.
    if var.shape[xidx] <= 1:
      return np.zeros_like(var)
    dvar = np.gradient(var, axis=xidx)  # 3D
    dx   = getattr(obj, 'd'+axis+'1d')  # 1D; needs dims to be added. add dims below.
    dx   = np.expand_dims(dx, axis=tuple(set((0,1,2)) - set([xidx])))
    dvardx = dvar / dx
    return dvardx

  # calculate derivative with interpolations
  if obj.numThreads > 1:
    if obj.verbose:
      print('Threading', whsp*8, end="\r", flush=True)
    quantlist = [quant[-4:] for numb in range(obj.numThreads)]
    def deriv_loop(var, quant):
      return do_stagger(var, 'd' + quant[0], obj=obj)
    if axis != 'z':
      return threadQuantity_z(deriv_loop, obj.numThreads, var, quantlist)
    else:
      return threadQuantity_y(deriv_loop, obj.numThreads, var, quantlist)
  else:
    if obj.lowbus:
      output = np.zeros_like(var)
      if axis != 'z':
        for iiz in range(obj.nz):
          slicer = np.s_[:, :, iiz:iiz+1]
          staggered = do_stagger(var[slicer], 'd' + quant[-4:], obj=obj)
          output[slicer] = staggered
      else:
        for iiy in range(obj.ny):
          slicer = np.s_[:, iiy:iiy+1, :]
          staggered = do_stagger(var[slicer], 'd' + quant[-4:], obj=obj)
          output[slicer] = staggered

      return output
    else:
      return do_stagger(var, 'd' + quant[-4:], obj=obj)


# default
_CENTER_QUANT = ('CENTER_QUANT', [x+'c' for x in AXES] + ['_center'])
# get value
def get_center(obj,quant, *args, **kwargs):
  '''
  Center the variable in the midle of the grid cells
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_CENTER_QUANT, get_center.__doc__, uni=UNI.quant_child(0))
    docvar('_center', 'quant_center brings quant to center via interpolation. Requires mesh_location_tracking to be enabled')
    return None

  getq = quant[-2:]  # the quant we are "getting" by this function. E.g. 'xc'.

  if getq in _CENTER_QUANT[1]:
    q = quant[:-1]  # base variable, including axis. E.g. 'efx'.
  elif quant.endswith('_center'):
    assert getattr(obj, mesh_location_tracking, False), "mesh location tracking is required for this to be enabled"
    q = quant[:-len('_center')]
  else:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _CENTER_QUANT[0], delay=True)

  # get the variable (pre-centering).
  var = obj.get_var(q, **kwargs)

  # determine which interpolations are necessary.
  if stagger.has_mesh_location(var):  # << using mesh_location_tracking >>
    transf = var.meshloc.steps_to((0,0,0))   # e.g. the list: ['xup', 'ydn', 'zdn']
    if len(transf) == 0:
      warnings.warn(f'called get_center on an already-centered variable: {q}')
  else:                               # << not using mesh_location_tracking >>
    axis = quant[-2]
    qvec = q[:-1]      # base variable, without axis. E.g. 'ef'.
    if qvec in ['i', 'e', 'j', 'ef']:     # edge-centered variable. efx is at (0, -0.5, -0.5)
      AXIS_TRANSFORM = {'x': ['yup', 'zup'],
                        'y': ['xup', 'zup'],
                        'z': ['xup', 'yup']}
    else:
      AXIS_TRANSFORM = {'x': ['xup'],
                        'y': ['yup'],
                        'z': ['zup']}
    transf = AXIS_TRANSFORM[axis]
  
  # do interpolation
  if obj.lowbus:
    # do "lowbus" version of interpolation  # not sure what is this? -SE Apr21 2021
    output = np.zeros_like(var)
    for interp in transf:
      axis = interp[0]
      if _can_interp(obj, axis):
        if axis != 'z':
          for iiz in range(obj.nz):
            slicer = np.s_[:, :, iiz:iiz+1]
            staggered = do_stagger(var[slicer], interp, obj=obj)
            output[slicer] = staggered
        else:
          for iiy in range(obj.ny):
            slicer = np.s_[:, iiy:iiy+1, :]
            staggered = do_stagger(var[slicer], interp, obj=obj)
            output[slicer] = staggered
  else:
    # do "regular" version of interpolation
    for interp in transf:
      if _can_interp(obj, interp[0]):
        var = do_stagger(var, interp, obj=obj)
  return var


# default
_INTERP_QUANT = ('INTERP_QUANT', [x+up for up in ('up', 'dn') for x in AXES])
# get value
def get_interp(obj, quant):
  '''simple interpolation. var must end in interpolation instructions.
  e.g. get_var('rxup') --> do_stagger(get_var('r'), 'xup')
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_INTERP_QUANT, get_interp.__doc__, uni=UNI.quant_child(0))
    for xup in _INTERP_QUANT[1]:
      docvar(xup, 'move half grid {up:} in the {x:} axis'.format(up=xup[1:], x=xup[0]))
    return None

  # interpret quant string
  varname, interp = quant[:-3], quant[-3:]

  if not interp in _INTERP_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, interp, _INTERP_QUANT[0], delay=True)

  val = obj.get_var(varname)      # un-interpolated value
  if _can_interp(obj, interp[0]):
    val = do_stagger(val, interp, obj=obj) # interpolated value
  return val


# default
_MODULE_QUANT = ('MODULE_QUANT', ['mod', 'h', '_mod'])
# get value
def get_module(obj,quant):
  '''
  Module or horizontal component of vectors
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_MODULE_QUANT, get_module.__doc__, uni=UNI.quant_child(0))
    docvar('mod',  'starting with mod computes the module of the vector [simu units]. sqrt(vx^2 + vy^2 + vz^2).')
    docvar('_mod', 'ending with mod computes the module of the vector [simu units]. sqrt(vx^2 + vy^2 + vz^2). ' +\
                   "This is an alias for starting with mod. E.g. 'modb' and 'b_mod' mean the same thing.")
    docvar('h',  'ending with h computes the horizontal component of the vector [simu units]. sqrt(vx^2 + vy^2).')
    return None

  # interpret quant string
  if quant.startswith('mod'):
    getq = 'mod'
    q    = quant[len('mod') : ]
  elif quant.endswith('_mod'):
    getq = 'mod'
    q    = quant[ : -len('_mod')]
  elif quant.endswith('h'):
    getq = 'h'
    q    = quant[ : -len('h')]
  else:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _MODULE_QUANT[0], delay=True)

  # actually get the quant:
  result = obj.get_var(q + 'xc') ** 2
  result += obj.get_var(q + 'yc') ** 2
  if getq == 'mod':
    result += obj.get_var(q + 'zc') ** 2

  return np.sqrt(result)


# default
_HORVAR_QUANT = ('HORVAR_QUANT', ['horvar'])
# get value
def get_horizontal_average(obj,quant):
  '''
  Computes horizontal average
  '''
  
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_HORVAR_QUANT, get_horizontal_average.__doc__)
    docvar('horvar',  'starting with horvar computes the horizontal average of a variable [simu units]')

  # interpret quant string
  getq = quant[:6]
  if not getq in _HORVAR_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _HORVAR_QUANT[0], delay=True)

  # Compares the variable with the horizontal mean
  if getq == 'horvar':
    result = np.zeros_like(obj.r)
    result += obj.get_var(quant[6:])  # base variable
    horv = np.mean(np.mean(result, 0), 0)
    for iix in range(0, getattr(obj, 'nx')):
      for iiy in range(0, getattr(obj, 'ny')):
        result[iix, iiy, :] = result[iix, iiy, :] / horv[:]
  return result


# default
_GRADVECT_QUANT = ('GRADVECT_QUANT',
                   ['div', 'divup', 'divdn',
                   'rot', 'she', 'curlcc', 'curvec',
                   'chkdiv', 'chbdiv', 'chhdiv']
                  )
# get value
def get_gradients_vect(obj,quant):
  '''
  Vectorial derivative opeartions

  for rot, she, curlcc, curvec, ensure that quant ends with axis.
  e.g. curvecbx gets the x component of curl of b.
  '''
  

  if quant=='':
    docvar = document_vars.vars_documenter(obj, *_GRADVECT_QUANT, get_gradients_vect.__doc__)
    for div in ['div', 'divup']:
      docvar(div,     'starting with, divergence [simu units], shifting up (e.g. dVARdxup) for derivatives', uni=UNI.quant_child(0))
    docvar('divdn',   'starting with, divergence [simu units], shifting down (e.g. dVARdxdn) for derivatives', uni=UNI.quant_child(0))
    docvar('rot',     'starting with, rotational (a.k.a. curl) [simu units]', uni=UNI.quant_child(0))
    docvar('she',     'starting with, shear [simu units]', uni=UNI.quant_child(0))
    docvar('curlcc',  'starting with, curl but shifted (via interpolation) back to original location on cell [simu units]', uni=UNI.quant_child(0))
    docvar('curvec',  'starting with, curl of face-centered vector (e.g. B, p) [simu units]', uni=UNI.quant_child(0))
    docvar('chkdiv',  'starting with, ratio of the divergence with the maximum of the abs of each spatial derivative [simu units]')
    docvar('chbdiv',  'starting with, ratio of the divergence with the sum of the absolute of each spatial derivative [simu units]')
    docvar('chhdiv',  'starting with, ratio of the divergence with horizontal averages of the absolute of each spatial derivative [simu units]')
    return None

  # interpret quant string
  for GVQ in _GRADVECT_QUANT[1]:
    if quant.startswith(GVQ):
      getq = GVQ                   # the quant we are "getting" via this function. (e.g. 'rot' or 'div')
      q    = quant[len(GVQ) : ]    # the "base" quant, i.e. whatever is left after pulling getq.
      break
  else:  # if we did not break, we did not match any GVQ to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _GRADVECT_QUANT[0], delay=True)

  # do calculations and return result
  if getq == 'chkdiv':
    if getattr(obj, 'nx') < 5:  # 2D or close
      varx = np.zeros_like(obj.r)
    else:
      varx = obj.get_var('d' + q + 'xdxup')

    if getattr(obj, 'ny') > 5:
      vary = obj.get_var('d' + q + 'ydyup')
    else:
      vary = np.zeros_like(varx)

    if getattr(obj, 'nz') > 5:
      varz = obj.get_var('d' + q + 'zdzup')
    else:
      varz = np.zeros_like(varx)
    return np.abs(varx + vary + varx) / (np.maximum(
        np.abs(varx), np.abs(vary), np.abs(varz)) + EPSILON)

  elif getq == 'chbdiv':
    varx = obj.get_var(q + 'x')
    vary = obj.get_var(q + 'y')
    varz = obj.get_var(q + 'z')
    if getattr(obj, 'nx') < 5:  # 2D or close
      result = np.zeros_like(varx)
    else:
      result = obj.get_var('d' + q + 'xdxup')

    if getattr(obj, 'ny') > 5:
      result += obj.get_var('d' + q + 'ydyup')

    if getattr(obj, 'nz') > 5:
      result += obj.get_var('d' + q + 'zdzup')

    return np.abs(result / (np.sqrt(
        varx * varx + vary * vary + varz * varz) + EPSILON))

  elif getq == 'chhdiv':
    varx = obj.get_var(q + 'x')
    vary = obj.get_var(q + 'y')
    varz = obj.get_var(q + 'z')
    if getattr(obj, 'nx') < 5:  # 2D or close
      result = np.zeros_like(varx)
    else:
      result = obj.get_var('d' + q + 'xdxup')

    if getattr(obj, 'ny') > 5:
      result += obj.get_var('d' + q + 'ydyup')

    if getattr(obj, 'nz') > 5:
      result += obj.get_var('d' + q + 'zdzup')

    for iiz in range(0, obj.nz):
      result[:, :, iiz] = np.abs(result[:, :, iiz]) / np.mean((
          np.sqrt(varx[:, :, iiz]**2 + vary[:, :, iiz]**2 +
                    varz[:, :, iiz]**2)))
    return result

  elif getq in ['div', 'divup', 'divdn']:  # divergence of vector quantity
    up = 'dn' if (getq == 'divdn') else 'up'
    result = 0
    for xdx in ['xdx', 'ydy', 'zdz']:
      result += obj.get_var('d' + q + xdx + up)
    return result

  elif getq == 'curlcc': # re-aligned curl
    x = q[-1]  # axis, 'x', 'y', 'z'
    q = q[:-1] # q without axis
    y,z = YZ_FROM_X[x]
    dqz_dy = obj.get_var('d' + q + z + 'd' + y + 'dn' + y + 'up')
    dqy_dz = obj.get_var('d' + q + y + 'd' + z + 'dn' + z + 'up')
    return dqz_dy - dqy_dz

  elif getq == 'curvec': # curl of vector which is originally on face of cell
    x = q[-1]  # axis, 'x', 'y', 'z'
    q = q[:-1] # q without axis
    y,z = YZ_FROM_X[x]
    # interpolation notes:
    ## qz is at (0, 0, -0.5); dqzdydn is at (0, -0.5, -0.5)
    ## qy is at (0, -0.5, 0); dqydzdn is at (0, -0.5, -0.5)
    dqz_dydn = obj.get_var('d' + q + z + 'd' + y + 'dn')
    dqy_dzdn = obj.get_var('d' + q + y + 'd' + z + 'dn')
    return dqz_dydn - dqy_dzdn

  elif getq in ['rot', 'she']:
    q = q[:-1]  # base variable
    qaxis = quant[-1]
    if qaxis == 'x':
      if getattr(obj, 'ny') < 5:  # 2D or close
        result = np.zeros_like(obj.r)
      else:
        result = obj.get_var('d' + q + 'zdyup')
      if getattr(obj, 'nz') > 5:
        if quant[:3] == 'rot':
          result -= obj.get_var('d' + q + 'ydzup')
        else:  # shear
          result += obj.get_var('d' + q + 'ydzup')
    elif qaxis == 'y':
      if getattr(obj, 'nz') < 5:  # 2D or close
        result = np.zeros_like(obj.r)
      else:
        result = obj.get_var('d' + q + 'xdzup')
      if getattr(obj, 'nx') > 5:
        if quant[:3] == 'rot':
          result -= obj.get_var('d' + q + 'zdxup')
        else:  # shear
          result += obj.get_var('d' + q + 'zdxup')
    elif qaxis == 'z':
      if getattr(obj, 'nx') < 5:  # 2D or close
        result = np.zeros_like(obj.r)
      else:
        result = obj.get_var('d' + q + 'ydxup')
      if getattr(obj, 'ny') > 5:
        if quant[:3] == 'rot':
          result -= obj.get_var('d' + q + 'xdyup')
        else:  # shear
          result += obj.get_var('d' + q + 'xdyup')
    return result


# default
_GRADSCAL_QUANT = ('GRADSCAL_QUANT', ['gra'])
# get value
def get_gradients_scalar(obj,quant):
  '''
  Gradient of a scalar
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_GRADSCAL_QUANT, get_gradients_scalar.__doc__)
    docvar('gra',  'starting with, Gradient of a scalar [simu units]. dqdx + dqdy + dqdz.' +\
                   ' Shifting up for derivatives.', uni=UNI.quant_child(0))
    return None

  getq = quant[:3]

  if not getq in _GRADSCAL_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _GRADSCAL_QUANT[0], delay=True)
  
  # do calculations and return result
  if getq == 'gra':
    q = quant[3:]  # base variable
    result = obj.get_var('d' + q + 'dxup')
    result += obj.get_var('d' + q + 'dyup')
    result += obj.get_var('d' + q + 'dzup')
  return result


# default
_SQUARE_QUANT = ('SQUARE_QUANT', ['2'])
# get value
def get_square(obj,quant):
  '''|vector| squared. Equals got product of vector with itself'''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_SQUARE_QUANT, get_square.__doc__,
                                           uni=UNI.quant_child(0)**2)
    docvar('2',  'ending with, Square of a vector [simu units].' +\
                 ' (Dot product of vector with itself.) Example: b2 --> bx^2 + by^2 + bz^2.')
    return None

  getq = quant[-1]
  if not getq in _SQUARE_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _SQUARE_QUANT[0], delay=True)

  # interpret quant string
  q = quant[:-1]  # vector name

  # do calculations and return result
  if getq == '2':
    result  = obj.get_var(q + 'xc') ** 2
    result += obj.get_var(q + 'yc') ** 2
    result += obj.get_var(q + 'zc') ** 2
    return result


# default
_LOG_QUANT = ('LOG_QUANT', ['lg', 'log_', 'ln_'])
# get value
def get_lg(obj,quant):
  '''Logarithm of a variable. E.g. log_r --> log10(r)'''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_LOG_QUANT, get_lg.__doc__, uni=DIMENSIONLESS)
    for lg in ['lg', 'log_']:
      docvar(lg,  'starting with, logarithm base 10 of a variable expressed in [simu. units].')
    docvar('ln_', 'starting with, logarithm base e  of a variable expressed in [simu. units].')
    return None

  # interpret quant string
  for LGQ in _LOG_QUANT[1]:
    if quant.startswith(LGQ):
      getq = LGQ                   # the quant we are "getting" via this function. (e.g. 'lg' or 'ln_')
      q    = quant[len(LGQ) : ]    # the "base" quant, i.e. whatever is left after pulling getq.
      break
  else:  # if we did not break, we did not match any LGQ to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _LOG_QUANT[0], delay=True)

  # do calculations and return result
  if getq in ['lg', 'log_']:
    return np.log10(obj.get_var(q))
  elif getq == 'ln_':
    return np.log(obj.get_var(q))


# default
_NUMOP_QUANT = ('NUMOP_QUANT', ['delta_', 'deltafrac_', 'abs_'])
# get value
def get_numop(obj,quant):
  '''Some numerical operation on a variable. E.g. delta_var computes (var - var.mean()).'''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_NUMOP_QUANT, get_numop.__doc__)
    docvar('delta_', 'starting with, deviation from mean. delta_v --> v - mean(v)', uni=UNI.qc(0))
    docvar('deltafrac_', 'starting with, fractional deviation from mean. deltafrac_v --> v / mean(v) - 1', uni=DIMENSIONLESS)
    docvar('abs_', 'starting with, absolute value of a scalar. abs_v --> |v|', uni=UNI.qc(0))
    return None

  # interpret quant string
  for start in _NUMOP_QUANT[1]:
    if quant.startswith(start):
      getq = start                 # the quant we are "getting" via this function. (e.g. 'lg' or 'ln_')
      base = quant[len(getq) : ]   # the "base" quant, i.e. whatever is left after pulling getq.
      break
  else:  # if we did not break, we did not match any start to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _NUMOP_QUANT[0], delay=True)

  # do calculations and return result
  v = obj.get_var(base)
  if getq == 'delta_':
    return (v - np.mean(v))
  elif getq == 'deltafrac_':
    return (v / np.mean(v)) - 1
  elif getq == 'abs_':
    return np.abs(v)

# default
_RATIO_QUANT = ('RATIO_QUANT', ['rat'])
# get value
def get_ratios(obj,quant):
  '''Ratio of two variables'''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_RATIO_QUANT, get_ratios.__doc__, uni=UNI.qc(0)/UNI.qc(1))
    docvar('rat',  'in between with, ratio of two variables [simu units]. aratb gives a/b.')
    return None

  # interpret quant string
  for RAT in _RATIO_QUANT[1]:
    qA, rat, qB = quant.partition(RAT)
    if qB != '':
      break
  else:  # if we did not break, we did not match any RAT to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, rat, _RATIO_QUANT[0], delay=True)

  # do calculations and return result
  qA_val = obj.get_var(qA)
  qB_val = obj.get_var(qB)
  return qA_val / (qB_val + EPSILON)


# default
_PROJ_QUANT = ('PROJ_QUANT', ['par', 'per'])
# get value
def get_projections(obj,quant):
  '''Projected vectors'''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_PROJ_QUANT, get_projections.__doc__, uni=UNI.quant_child(0))
    docvar('par',  'in between with, parallel component of the first vector respect to the second vector [simu units]')
    docvar('per',  'in between with, perpendicular component of the first vector respect to the second vector [simu units]')
    return None

  # interpret quant string
  for PAR in _PROJ_QUANT[1]:
    v1, par, v2 = quant.partition(PAR)
    if v2 != '':
      break
  else:  # if we did not break, we did not match any PAR to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, par, _PROJ_QUANT[0], delay=True)

  # do calculations and return result v1 onto v2
  x_a = obj.get_var(v1 + 'xc')
  y_a = obj.get_var(v1 + 'yc')
  z_a = obj.get_var(v1 + 'zc')
  x_b = obj.get_var(v2 + 'xc')
  y_b = obj.get_var(v2 + 'yc')
  z_b = obj.get_var(v2 + 'zc')
  def proj_task(x1, y1, z1, x2, y2, z2):
    '''do projecting; can be used in threadQuantity() or as is'''
    v2Mag = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
    v2x, v2y, v2z = x2 / v2Mag, y2 / v2Mag, z2 / v2Mag
    parScal = x1 * v2x + y1 * v2y + z1 * v2z
    parX, parY, parZ = parScal * v2x, parScal * v2y, parScal * v2z
    if par == 'par':
      return np.abs(parScal)
    elif par == 'per':
      perX = x1 - parX
      perY = y1 - parY
      perZ = z1 - parZ
      v1Mag = np.sqrt(perX**2 + perY**2 + perZ**2)
      return v1Mag

  if obj.numThreads > 1:
    if obj.verbose:
      print('Threading', whsp*8, end="\r", flush=True)

    return threadQuantity(proj_task, obj.numThreads,
                          x_a, y_a, z_a, x_b, y_b, z_b)
  else:
    return proj_task(x_a, y_a, z_a, x_b, y_b, z_b)


# default
_VECTOR_PRODUCT_QUANT = \
          ('VECTOR_PRODUCT_QUANT',
            ['times', '_facecross_', '_edgecross_', '_edgefacecross_',
             '_facecrosstocenter_', '_facecrosstoface_'
            ]
          )
# get value
def get_vector_product(obj,quant):
  '''cross product between two vectors.
  call via <v1><times><v2><x>.
  Example, for the x component of b cross u, you should call get_var('b_facecross_ux').
  '''
  if quant=='':
    docvar = document_vars.vars_documenter(obj, *_VECTOR_PRODUCT_QUANT, get_vector_product.__doc__,
                                           uni=UNI.quant_child(0) * UNI.quant_child(1))
    docvar('times',  '"naive" cross product between two vectors. (We do not do any interpolation.) [simu units]')
    docvar('_facecross_', ('cross product [simu units]. For two face-centered vectors, such as B, u. '
                           'result is edge-centered. E.g. result_x is at ( 0  , -0.5, -0.5).'))
    docvar('_edgecross_', ('cross product [simu units]. For two edge-centered vectors, such as E, I. '
                           'result is face-centered. E.g. result_x is at (-0.5,  0  ,  0  ).'))
    docvar('_edgefacecross_', ('cross product [simu units]. A_edgefacecross_Bx gives x-component of A x B.'
                           'A must be edge-centered (such as E, I); B must be face-centered, such as B, u.'
                           'result is face-centered. E.g. result_x is at (-0.5,  0  ,  0  ).'))
    docvar('_facecrosstocenter_', ('cross product for two face-centered vectors such as B, u. '
                           'result is fully centered. E.g. result_x is at ( 0  ,  0  ,  0  ).'
                           ' For most cases, it is better to use _facecrosstoface_'))
    docvar('_facecrosstoface_', ('cross product for two face-centered vectors such as B, u. '
                           'result is face-centered E.g. result_x is at (-0.5,  0 ,  0  ).'), uni=UNI.quant_child(0))
    return None

  # interpret quant string
  for TIMES in _VECTOR_PRODUCT_QUANT[1]:
    A, cross, q = quant.partition(TIMES)
    if q != '':
      B, x = q[:-1], q[-1]
      y, z = YZ_FROM_X[x]
      break
  else:  # if we did not break, we did not match any RAT to quant, so we return None.
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, cross, _VECTOR_PRODUCT_QUANT[0], delay=True)

  # at this point, we know quant looked like <A><times><B><x>

  if cross == 'times':
    return (obj.get_var(A + y) * obj.get_var(B + z) -
            obj.get_var(A + z) * obj.get_var(B + y))

  elif cross == '_facecross_':
    # interpolation notes, for x='x', y='y', z='z':
    ## resultx will be at (0, -0.5, -0.5)
    ## Ay, By are at (0, -0.5,  0  ).  we must shift by zdn to align with result.
    ## Az, Bz are at (0,  0  , -0.5).  we must shift by ydn to align with result.
    ydn, zdn = y+'dn', z+'dn'
    Ay = obj.get_var(A+y + zdn)
    By = obj.get_var(B+y + zdn)
    Az = obj.get_var(A+z + ydn)
    Bz = obj.get_var(B+z + ydn)
    AxB__x = Ay * Bz - By * Az   # x component of A x B. (x='x', 'y', or 'z')
    return AxB__x

  elif cross == '_edgecross_':
    # interpolation notes, for x='x', y='y', z='z':
    ## resultx will be at (-0.5, 0, 0)
    ## Ay, By are at (-0.5,  0  , -0.5).  we must shift by zup to align with result.
    ## Az, Bz are at (-0.5, -0.5,  0  ).  we must shift by yup to align with result.
    yup, zup = y+'up', z+'up'
    Ay = obj.get_var(A+y + zup)
    By = obj.get_var(B+y + zup)
    Az = obj.get_var(A+z + yup)
    Bz = obj.get_var(B+z + yup)
    AxB__x = Ay * Bz - By * Az   # x component of A x B. (x='x', 'y', or 'z')
    return AxB__x

  elif cross == '_edgefacecross_':
    # interpolation notes, for x='x', y='y', z='z':
    ## resultx will be at (-0.5, 0, 0)
    ## Ay is at (-0.5,  0  , -0.5). we must shift by   zup   to align with result.
    ## Az is at (-0.5, -0.5,  0  ). we must shift by   yup   to align with result.
    ## By is at ( 0  , -0.5,  0  ). we must shift by xdn yup to align with result.
    ## Bz is at ( 0  ,  0  , -0.5). we must shift by xdn zup to align with result.
    xdn, yup, zup = x+'dn', y+'up', z+'up'
    Ay = obj.get_var(A+y + zup)
    Az = obj.get_var(A+z + yup)
    By = obj.get_var(B+y + xdn+yup)
    Bz = obj.get_var(B+z + xdn+zup)
    AxB__x = Ay * Bz - By * Az   # x component of A x B. (x='x', 'y', or 'z')
    return AxB__x

  elif cross == '_facecrosstocenter_':
    # interpolation notes, for x='x', y='y', z='z':
    ## resultx will be at (0, 0, 0)
    ## Ay, By are at (0, -0.5,  0  ).  we must shift by yup to align with result.
    ## Az, Bz are at (0,  0  , -0.5).  we must shift by zup to align with result.
    yup, zup = y+'up', z+'up'
    Ay = obj.get_var(A+y + yup)
    By = obj.get_var(B+y + yup)
    Az = obj.get_var(A+z + zup)
    Bz = obj.get_var(B+z + zup)
    AxB__x = Ay * Bz - By * Az   # x component of A x B. (x='x', 'y', or 'z')
    return AxB__x

  elif cross == '_facecrosstoface_':
    # resultx will be at (-0.5, 0, 0).
    ## '_facecrosstocenter_' gives result at (0, 0, 0) so we shift by xdn to align.
    return obj.get_var(A+'_facecrosstocenter_'+B+x + x+'dn')


# default
_HATS = ['_hat'+x for x in AXES]
_ANGLES_XXY = ['_angle'+ xxy for xxy in ['xxy', 'yyz', 'zzx']]
_ANGLE_QUANT = _HATS + _ANGLES_XXY
_ANGLE_QUANT = ('ANGLE_QUANT', _ANGLE_QUANT)
# get value
def get_angle(obj,quant):
  '''angles. includes unit vector, and angle off an axis in a plane (xy, yz, or zx).

  Presently not very efficient, due to only being able to return one unit vector component at a time.

  call via <var><anglequant>.
  Example: b_angleyyz --> angle off of the positive y axis in the yz plane, for b (magnetic field).

  TODO: interpolation
  '''
  if quant=='':
    docvar = document_vars.vars_documenter(obj, *_ANGLE_QUANT, get_angle.__doc__)
    for x in AXES:
      docvar('_hat'+x, x+'-component of unit vector. Example: b_hat'+x+' is '+x+'-component of unit vector for b.',
                       uni=DIMENSIONLESS)
    for _angle_xxy in _ANGLES_XXY:
      x, y = _angle_xxy[-2], _angle_xxy[-1]   # _angle_xxy[-3] == _angle_xxy[-1]
      docvar(_angle_xxy, 'angle off of the positive '+x+'-axis in the '+x+y+'plane. Result in range [-pi, pi].',
                         uni_f=UNITS_FACTOR_1, uni_name=Usym('radians'))
    return None

  # interpret quant string
  var, _, command = quant.rpartition('_')
  command = '_' + command

  if command not in _ANGLE_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, command, _ANGLE_QUANT[0], delay=True)

  # do calculations and return result
  if command in _HATS:
    x = command[-1]  # axis; 'x', 'y', or 'z'
    varhatx = obj.get_var(var+x) / obj.get_var('mod'+var)
    return varhatx

  if command in _ANGLES_XXY:
    x, y = command[-2], command[-1] # _angle_xxy[-3] == _angle_xxy[-1]
    varx = obj.get_var(var + x)
    vary = obj.get_var(var + y)
    return np.arctan2(vary, varx)


#default
_STAT_QUANT = ('STAT_QUANT', ['mean_', 'variance_', 'std_'])
# get value
def get_stat_quant(obj, quant):
  '''statistics such as mean, std.

  The result will be a single value (not a 3D array).
  '''
  if quant=='':
    docvar = document_vars.vars_documenter(obj, *_STAT_QUANT, get_stat_quant.__doc__)
    docvar('mean_', 'mean_v --> np.mean(v)', uni=UNI.qc(0))
    docvar('variance_', 'variance_v --> np.var(v).', uni=UNI.qc(0)**2)
    docvar('std_', 'std_v --> np.std(v)', uni=UNI.qc(0))
    return None

  # interpret quant string
  command, _, var = quant.partition('_')
  command = command + '_'

  if command not in _STAT_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, command, _STAT_QUANT[0], delay=True)

  # do calculations and return result
  val = obj.get_var(var)
  if command == 'mean_':
    return np.mean(val)
  elif command == 'variance_':
    return np.var(val)
  elif command == 'std_':
    return np.std(val)
  else:
    raise NotImplementedError(f'command={repr(command)} in get_stat_quant')


#default
_FFT_QUANT = ('FFT_QUANT', ['fft2_', 'fftxy_', 'fftyz_', 'fftxz_'])
# get value
def get_fft_quant(obj, quant):
  '''Fourier transform, using np.fft.fft2, and shifting using np.fft.fftshift.

  result will be complex-valued. (consider get_var('abs_fft2_quant') to convert to magnitude.)

  See obj.kx, ky, kz for the corresponding coordinates in k-space.
  See obj.get_kextent for the extent to use if plotting k-space via imshow.

  Also sets obj._latest_fft_axes = ('x', 'y'), ('x', 'z') or ('y', 'z') as appropriate.

  Note that for plotting with imshow, you will likely want to transpose and use origin='lower'.
  Example, making a correctly labeled and aligned plot of FFT(r[:, 0, :]):
    dd = BifrostData(...)
    val = dd('abs_fftxz_r')[:, 0, :]    # == |FFT(dd('r')[:, 0, :])|
    extent = dd.get_kextent('xz', units='si')
    plt.imshow(val.T, origin='lower', extent=extent)
    plt.xlabel('kx [1/m]'); plt.ylabel('kz [1/m]')
    plt.xlim([0, None])   # <-- not necessary, however numpy's FFT of real-valued input
        # will be symmetric under rotation by 180 degrees, so half the spectrum is redundant.
  '''
  if quant=='':
    docvar = document_vars.vars_documenter(obj, *_FFT_QUANT, get_fft_quant.__doc__, uni=UNI.qc(0))
    shifted = ' result will be shifted so that the zero-frequency component is in the middle (via np.fft.fftshift).'
    docvar('fft2_', '2D fft. requires 2D data (i.e. x, y, or z with length 1). result will be 2D.' + shifted)
    docvar('fftxy_', '2D fft in (x, y) plane, at each z. result will be 3D.' + shifted)
    docvar('fftyz_', '2D fft in (y, z) plane, at each x. result will be 3D.' + shifted)
    docvar('fftxz_', '2D fft in (x, z) plane, at each y. result will be 3D.' + shifted)
    return None

  # interpret quant string
  command, _, var = quant.partition('_')
  command = command + '_'

  if command not in _FFT_QUANT[1]:
    return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, command, _FFT_QUANT[0], delay=True)

  # do calculations and return result
  val = obj(var)
  if command == 'fft2_':
    if np.shape(val) != obj.shape:
      raise NotImplementedError(f'fft2_ for {repr(var)} with shape {np.shape(val)} not equal to obj.shape {obj.shape}')
    if obj.xLength == 1:
      return obj(f'fftyz_{var}')[0, :, :]
    elif obj.yLength == 1:
      return obj(f'fftxz_{var}')[:, 0, :]
    elif obj.zLength == 1:
      return obj(f'fftxy_{var}')[:, :, 0]
    else:
      errmsg = f'fft2_ requires x, y, or z to have length 1, but obj.shape = {obj.shape}.' +\
                'maybe you meant to specify axes, via fftxy_, fftyz, or fftxz_?'
      raise ValueError(errmsg)
  elif command in ('fftxy_', 'fftyz_', 'fftxz_'):
    x, y = command[3:5]
    obj._latest_fft_axes = (x, y)    # <-- bookkeeping
    AX_STR_TO_I = {'x': 0, 'y': 1, 'z': 2}
    xi = AX_STR_TO_I[x]
    yi = AX_STR_TO_I[y]
    fft_unshifted = np.fft.fft2(val, axes=(xi, yi))
    return np.fft.fftshift(fft_unshifted)
  else:
    raise NotImplementedError(f'command={repr(command)} in get_fft_quant')


''' ------------- End get_quant() functions; Begin helper functions -------------  '''

def threadQuantity(task, numThreads, *args):
  # split arg arrays
  args = list(args)

  for index in range(np.shape(args)[0]):
    args[index] = np.array_split(args[index], numThreads)

  # make threadpool, task = task, with zipped args
  pool = ThreadPool(processes=numThreads)
  result = np.concatenate(pool.starmap(task, zip(*args)))
  return result


def threadQuantity_y(task, numThreads, *args):
  # split arg arrays
  args = list(args)

  for index in range(np.shape(args)[0]):
    if len(np.shape(args[index])) == 3:
      args[index] = np.array_split(args[index], numThreads, axis=1)
    else:
      args[index] = np.array_split(args[index], numThreads)
  # make threadpool, task = task, with zipped args
  pool = ThreadPool(processes=numThreads)
  result = np.concatenate(pool.starmap(task, zip(*args)), axis=1)
  return result


def threadQuantity_z(task, numThreads, *args):
  # split arg arrays
  args = list(args)

  for index in range(np.shape(args)[0]):
    if len(np.shape(args[index])) == 3:
      args[index] = np.array_split(args[index], numThreads, axis=2)
    else:
      args[index] = np.array_split(args[index], numThreads)

  # make threadpool, task = task, with zipped args
  pool = ThreadPool(processes=numThreads)
  result = np.concatenate(pool.starmap(task, zip(*args)), axis=2)
  return result
