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


TODO:
  - cleanup code in get_center using quants from get_interp.
  - be more sophisticated in learning if quant is edge-centered or face-centered;
    - e.g. get_center now just checks if quant is i, e, j, or ef.
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
DEFAULT_STAGGER_KIND = 'stagger'
AXES      = ('x', 'y', 'z')
YZ_FROM_X = dict(x=('y', 'z'), y=('z', 'x'), z=('x', 'y'))  # right-handed coord system x,y,z given x.
EPSILON   = 1.0e-20   # small number which is added in denominators of some operations.


# we need to convert to float32 before doing cstagger.do.
## not sure why this conversion isnt done in the cstagger method, but it is a bit 
## painful to change the method itself (required re-installing helita for me) so we will
## instead just change our calls to the method here. -SE Apr 22 2021
CSTAGGER_TYPES = ['float32']  # these are the allowed types
def do_cstagger(arr, operation, default_type=CSTAGGER_TYPES[0], obj=None):
  '''does cstagger, after ensuring arr is the correct type, converting if necessary.
  if type conversion is necessary, convert to default_type.
  '''
  kind = getattr(obj,'stagger_kind',DEFAULT_STAGGER_KIND)
  if kind == 'cstagger': # use cstagger routine.
    arr = np.array(arr, copy=False)     # make numpy array, if necessary.
    if arr.dtype not in CSTAGGER_TYPES: # if necessary,
      arr = arr.astype(default_type)      # convert type
    return cstagger.do(arr, operation)  # call the original cstagger function
  else:                  # use stagger routine.
    # stagger routine requires 'diff' kwarg if doing a derivative.
    if operation.startswith('dd'):
      x    = operation[2]  # get the axis. operation is like ddxup or ddxdn. x may be x, y, or z.
      xdir = operation[2:]
      diff = getattr(obj, 'd'+x+'id'+xdir)  # for debugging: if crashing here, make sure obj is not None.
    else:
      diff = None
    # deal with boundaries. (Note obj.get_param isn't defined for everyone, e.g. BifrostData, so we can't use it.)
    bdr_pad = {x: ('reflect' if obj.params['periodic_'+x][obj.snapInd] else 'wrap') for x in AXES}
    return stagger.do(arr, operation, diff=diff, DEFAULT_PAD = bdr_pad)

def _can_interp(obj, axis, warn=True):
  '''return whether we can interpolate. Make warning if we can't.'''
  if not obj.cstagop:  # this is True by default; if it is False we assume that someone 
    return False       # intentionally turned off interpolation. So we don't make warning.
  if not getattr(obj, 'cstagger_exists', False):
    warnmsg = 'interpolation requested, but cstagger not initialized, for obj={}! '.format(object.__repr__(obj)) +\
              'We will skip the interpolation, and instead return the original value.'
    warnings.warn(warnmsg) # warn user we will not be interpolating! (cstagger doesn't exist)
    return False
  if not getattr(obj, 'n'+axis, 0) >=5:
    warnmsg = 'requested interpolation in {x:} but obj.n{x:} < 5. '.format(x=axis) +\
              'We will skip this interpolation, and instead return the original value.'
    warnings.warn(warnmsg) # warn user we will not be interpolating! (dimension is too small)
    return False
  return True


''' --------------------- functions to load quantities --------------------- '''

def load_arithmetic_quantities(obj,quant, *args, **kwargs):
  __tracebackhide__ = True  # hide this func from error traceback stack.
  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'arquantities', 'Computes arithmetic quantities')

  val = get_center(obj,quant)
  if val is None:
    if obj.cstagop != False: # this is only for cstagger routines 
      val = get_deriv(obj,quant)
  if val is None:
    val = get_interp(obj,quant)
  if val is None:
    val = get_module(obj,quant)
  if val is None:
    val = get_horizontal_average(obj,quant)
  if val is None:
    val = get_gradients_vect(obj,quant)
  if val is None:
    if obj.cstagop != False:  # this is only for cstagger routines 
      val = get_gradients_scalar(obj,quant)
  if val is None:
    val = get_square(obj,quant)
  if val is None:
    val = get_lg(obj,quant)
  if val is None:
    val = get_ratios(obj,quant)
  if val is None:
    val = get_projections(obj,quant)
  if val is None:
    val = get_vector_product(obj,quant)
  if val is None:
    val = get_angle(obj, quant)

  if val is not None:                         # if got a value, use obj._quant_selection
    document_vars.select_quant_selection(obj) #           to update obj._quant_selected.
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
      return do_cstagger(var, 'd' + quant[0], obj=obj)
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
          staggered = do_cstagger(var[slicer], 'd' + quant[-4:], obj=obj)
          output[slicer] = staggered
      else:
        for iiy in range(obj.ny):
          slicer = np.s_[:, iiy:iiy+1, :]
          staggered = do_cstagger(var[slicer], 'd' + quant[-4:], obj=obj)
          output[slicer] = staggered

      return output
    else:
      return do_cstagger(var, 'd' + quant[-4:], obj=obj)


# default
_CENTER_QUANT = ('CENTER_QUANT', [x+'c' for x in AXES])
# get value
def get_center(obj,quant, *args, **kwargs):
  '''
  Center the variable in the midle of the grid cells
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_CENTER_QUANT, get_center.__doc__, uni=UNI.quant_child(0))
    return None

  getq = quant[-2:]  # the quant we are "getting" by this function.

  if not getq in _CENTER_QUANT[1]:
      return None

  # tell obj the quant we are getting by this function.
  document_vars.setattr_quant_selected(obj, getq, _CENTER_QUANT[0], delay=True)

  # interpret quant string
  axis = quant[-2]
  q    = quant[:-1]  # base variable

  if q in ['i', 'e', 'j', 'ef']:     # edge-centered variable. efx is at (0, -0.5, -0.5)
    AXIS_TRANSFORM = {'x': ['yup', 'zup'],
                      'y': ['xup', 'zup'],
                      'z': ['xup', 'yup']}
  else:
    AXIS_TRANSFORM = {'x': ['xup'],
                      'y': ['yup'],
                      'z': ['zup']}
  transf = AXIS_TRANSFORM[axis]

  var = obj.get_var(q, **kwargs)
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
            staggered = do_cstagger(var[slicer], interp, obj=obj)
            output[slicer] = staggered
        else:
          for iiy in range(obj.ny):
            slicer = np.s_[:, iiy:iiy+1, :]
            staggered = do_cstagger(var[slicer], interp, obj=obj)
            output[slicer] = staggered
  else:
    # do "regular" version of interpolation
    for interp in transf:
      if _can_interp(obj, interp[0]):
        var = do_cstagger(var, interp, obj=obj)
  return var


# default
_INTERP_QUANT = ('INTERP_QUANT', [x+up for up in ('up', 'dn') for x in AXES])
# get value
def get_interp(obj, quant):
  '''simple interpolation. var must end in interpolation instructions.
  e.g. get_var('rxup') --> do_cstagger(get_var('r'), 'xup')
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
    val = do_cstagger(val, interp, obj=obj) # interpolated value
  return val


# default
_MODULE_QUANT = ('MODULE_QUANT', ['mod', 'h'])
# get value
def get_module(obj,quant):
  '''
  Module or horizontal component of vectors
  '''
  if quant == '':
    docvar = document_vars.vars_documenter(obj, *_MODULE_QUANT, get_module.__doc__, uni=UNI.quant_child(0))
    docvar('mod',  'starting with mod computes the module of the vector [simu units]. sqrt(vx^2 + vy^2 + vz^2).')
    docvar('h',  'ending with h computes the horizontal component of the vector [simu units]. sqrt(vx^2 + vy^2).')
    return None

  # interpret quant string
  if quant.startswith('mod'):
    getq = 'mod'
    q    = quant[len('mod') : ]
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
            ['times', '_facecross_', '_edgecross_',
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
    docvar('_facecrosstocenter_', ('cross product for two face-centered vectors such as B, u. '
                           'result is fully centered. E.g. result_x is at ( 0  ,  0  ,  0  ).'
                           'for most cases, it is better to use _facecrosstoface_'))
    docvar('_facecrosstoface_', ('cross product for two face-centered vectors such as B, u. '
                           'result is face-centered E.g. result_x is at ( 0  , -0.5,  0  ).'), uni=UNI.quant_child(0))
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
