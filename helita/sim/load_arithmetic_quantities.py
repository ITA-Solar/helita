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
from . import cstagger
from . import document_vars

# import external public modules
import numpy as np

# we need to convert to float32 before doing cstagger.do.
## not sure why this conversion isnt done in the cstagger method, but it is a bit 
## painful to change the method itself (required re-installing helita for me) so we will
## instead just change our calls to the method here. -SE Apr 22 2021
CSTAGGER_TYPES = ['float32']  # these are the allowed types
def do_cstagger(arr, operation='xup', default_type=CSTAGGER_TYPES[0]):
  '''does cstagger, after ensuring arr is the correct type, converting if necessary.
  if type conversion is necessary, convert to default_type.
  '''
  arr = np.array(arr, copy=False)     # make numpy array, if necessary.
  if arr.dtype not in CSTAGGER_TYPES: # if necessary,
    arr = arr.astype(default_type)      # convert type
  return cstagger.do(arr, operation)  # call the original cstagger function


''' --------------------- functions to load quantities --------------------- '''

def load_arithmetic_quantities(obj,quant, *args, **kwargs):
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
  return val

def _can_interp(obj, axis):
  return ( obj.cstagop ) and ( getattr(obj, 'n'+axis) >=5 )

def get_deriv(obj,quant):
  '''
  Computes derivative of quantity.
  Example: 'drdxup' does dxup for var 'r'.
  '''
  quant = quant.lower()

  DERIV_QUANT = ['dxup', 'dyup', 'dzup', 'dxdn', 'dydn', 'dzdn']

  docvar = document_vars.vars_documenter(obj, 'DERIV_QUANT', DERIV_QUANT, get_deriv.__doc__)
  docvar('dxup',  'spatial derivative in the x axis with half grid up [simu units]')
  docvar('dyup',  'spatial derivative in the y axis with half grid up [simu units]')
  docvar('dzup',  'spatial derivative in the z axis with half grid up [simu units]')
  docvar('dxdn',  'spatial derivative in the x axis with half grid down [simu units]')
  docvar('dydn',  'spatial derivative in the y axis with half grid down [simu units]')
  docvar('dzdn',  'spatial derivative in the z axis with half grid down [simu units]')

  if (quant == '') or not (quant[0] == 'd' and quant[-4:] in DERIV_QUANT):
    return None

  axis = quant[-3]
  q = quant[1:-4]  # base variable 
  var = obj.get_var(q)

  def deriv_loop(var, quant):
    return do_cstagger(var, 'd' + quant[0])

  if getattr(obj, 'n' + axis) < 5:  # 2D or close
    print('(WWW) get_quantity: DERIV_QUANT: '
          'n%s < 5, derivative set to 0.0' % axis)
    return np.zeros_like(var)
  else:
    if obj.numThreads > 1:
      if obj.verbose:
        print('Threading', whsp*8, end="\r", flush=True)
      quantlist = [quant[-4:] for numb in range(obj.numThreads)]
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
            staggered = do_cstagger(var[slicer], 'd' + quant[-4:])
            output[slicer] = staggered
        else:
          for iiy in range(obj.ny):
            slicer = np.s_[:, iiy:iiy+1, :]
            staggered = do_cstagger(var[slicer], 'd' + quant[-4:])
            output[slicer] = staggered

        return output
      else:
        return do_cstagger(var, 'd' + quant[-4:])


def get_center(obj,quant, *args, **kwargs):
  '''
  Center the variable in the midle of the grid cells
  '''
  CENTER_QUANT = ['xc', 'yc', 'zc']

  docvar = document_vars.vars_documenter(obj, 'CENTER_QUANT', CENTER_QUANT, get_center.__doc__)

  if (quant == '') or not quant[-2:] in CENTER_QUANT:
      return None

  # This brings a given vector quantity to cell centres
  axis = quant[-2]
  q = quant[:-1]  # base variable

  if q[:-1] == 'i' or q == 'e':
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
            staggered = do_cstagger(var[slicer], interp)
            output[slicer] = staggered
        else:
          for iiy in range(obj.ny):
            slicer = np.s_[:, iiy:iiy+1, :]
            staggered = do_cstagger(var[slicer], interp)
            output[slicer] = staggered
  else:
    # do "regular" version of interpolation
    for interp in transf:
      if _can_interp(obj, interp[0]):
        var = do_cstagger(var, interp)
  return var

def get_interp(obj, quant):
  '''simple interpolation. var must end in interpolation instructions.
  e.g. rxup --> do_cstagger(get_var('r'), 'xup')
  '''
  INTERP_QUANT = ['xup', 'yup', 'zup',
                  'xdn', 'ydn', 'zdn']
  if quant == '':
    docvar = document_vars.vars_documenter(obj, 'INTERP_QUANT', INTERP_QUANT, get_interp.__doc__)
    for xup in INTERP_QUANT:
      docvar(xup, 'move half grid {up:} in the {x:} axis'.format(up=xup[1:], x=xup[0]))
    return None

  varname, interp = quant[:-3], quant[-3:]
  if not interp in INTERP_QUANT:
    return None
  else:
    val = obj.get_var(varname)      # un-interpolated value
    if _can_interp(obj, interp[0]):
      val = do_cstagger(val, interp) # interpolated value
    else:
      # return un-interpolated value; warn that we are not actually interpolating.
      warnings.warn('requested interpolation in {x:} but obj.n{x:} < 5 '.format(x=interp[0]) +\
                    'or obj.cstagop==False! Skipping this interpolation.')
    return val


def get_module(obj,quant):
  '''
  Module or horizontal component of vectors
  '''
  MODULE_QUANT = ['mod', 'h']  # This one must be called the last

  docvar = document_vars.vars_documenter(obj, 'MODULE_QUANT', MODULE_QUANT, get_module.__doc__)
  docvar('mod',  'starting with mod computes the module of the vector [simu units]')
  docvar('h',  'ending with h computes the horizontal component of the vector [simu units]')

  if (quant == '') or not ((quant[:3] in MODULE_QUANT) or (quant[-1] in MODULE_QUANT)):
    return None

  # Calculate module of vector quantity
  if (quant[:3] in MODULE_QUANT):
    q = quant[3:]
  else:
    q = quant[:-1]
  if q == 'b':
    if not obj.do_mhd:
      raise ValueError("No magnetic field available.")

  result = obj.get_var(q + 'xc') ** 2
  result += obj.get_var(q + 'yc') ** 2
  if not(quant[-1] in MODULE_QUANT):
    result += obj.get_var(q + 'zc') ** 2

  if (quant[:3] in MODULE_QUANT) or (quant[-1] in MODULE_QUANT):
    return np.sqrt(result)


def get_horizontal_average(obj,quant):
  '''
  Computes horizontal average
  '''
  HORVAR_QUANT = ['horvar']

  docvar = document_vars.vars_documenter(obj, 'HORVAR_QUANT', HORVAR_QUANT, get_horizontal_average.__doc__)
  docvar('horvar',  'starting with horvar computes the horizontal average of a variable [simu units]')

  if (quant == '') or not quant[:6] in HORVAR_QUANT:
    return None

  # Compares the variable with the horizontal mean
  if quant[:6] == 'horvar':
    result = np.zeros_like(obj.r)
    result += obj.get_var(quant[6:])  # base variable
    horv = np.mean(np.mean(result, 0), 0)
    for iix in range(0, getattr(obj, 'nx')):
      for iiy in range(0, getattr(obj, 'ny')):
        result[iix, iiy, :] = result[iix, iiy, :] / horv[:]
  return result


def get_gradients_vect(obj,quant):
  '''
  Vectorial derivative opeartions
  '''
  GRADVECT_QUANT = ['div', 'rot', 'she', 'chkdiv', 'chbdiv', 'chhdiv']

  docvar = document_vars.vars_documenter(obj, 'GRADVECT_QUANT', GRADVECT_QUANT, get_gradients_vect.__doc__)
  docvar('div',  'starting with, divergence [simu units]')
  docvar('rot',  'starting with, rotational [simu units]')
  docvar('she',  'starting with, shear [simu units]')
  docvar('chkdiv',  'starting with, ratio of the divergence with the maximum of the abs of each spatial derivative [simu units]')
  docvar('chbdiv',  'starting with, ratio of the divergence with the sum of the absolute of each spatial derivative [simu units]')
  docvar('chhdiv',  'starting with, ratio of the divergence with horizontal averages of the absolute of each spatial derivative [simu units]')


  if (quant == '') or not (quant[:6] in GRADVECT_QUANT or quant[:3] in GRADVECT_QUANT):
      return None

  if quant[:3] == 'chk':
    q = quant[6:]  # base variable
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
        np.abs(varx), np.abs(vary), np.abs(varz)) + 1.0e-20)

  elif quant[:3] == 'chb':
    q = quant[6:]  # base variable
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
        varx * varx + vary * vary + varz * varz) + 1.0e-20))

  elif quant[:3] == 'chh':
    q = quant[6:]  # base variable
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

  elif quant[:3] == 'div':  # divergence of vector quantity
    q = quant[3:]  # base variable
    if getattr(obj, 'nx') < 5:  # 2D or close
      result = np.zeros_like(obj.r)
    else:
      result = obj.get_var('d' + q + 'xdxup')
    if getattr(obj, 'ny') > 5:
      result += obj.get_var('d' + q + 'ydyup')
    if getattr(obj, 'nz') > 5:
      result += obj.get_var('d' + q + 'zdzup')

  elif quant[:3] == 'rot' or quant[:3] == 'she':
    q = quant[3:-1]  # base variable
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


def get_gradients_scalar(obj,quant):
  '''
  Gradient of a scalar
  '''
  GRADSCAL_QUANT = ['gra']

  docvar = document_vars.vars_documenter(obj, 'GRADSCAL_QUANT', GRADSCAL_QUANT, get_gradients_scalar.__doc__)
  docvar('gra',  'starting with, Gradient of a scalar [simu units]')

  if (quant == '') or not quant[:3] in GRADSCAL_QUANT:
    return None
  
  if quant[:3] == 'gra':
    q = quant[3:]  # base variable
    if getattr(obj, 'nx') < 5:  # 2D or close
      result = np.zeros_like(obj.r)
    else:
      result = obj.get_var('d' + q + 'dxup')
    if getattr(obj, 'ny') > 5:
      result += obj.get_var('d' + q + 'dyup')
    if getattr(obj, 'nz') > 5:
      result += obj.get_var('d' + q + 'dzup')
  return result


def get_square(obj,quant):
  '''module of a square of a vector'''
  SQUARE_QUANT = ['2']  # This one must be called the towards the last

  docvar = document_vars.vars_documenter(obj, 'SQUARE_QUANT', SQUARE_QUANT, get_square.__doc__)
  docvar('2',  'ending with, Square of a vector [simu units]')

  if (quant == '') or not quant[-1] in SQUARE_QUANT:
    return None

  try: 
    result = obj.get_var(quant[:-1] + 'xc') ** 2
    result += obj.get_var(quant[:-1] + 'yc') ** 2
    result += obj.get_var(quant[:-1] + 'zc') ** 2
    return result
  except:
    return None


def get_lg(obj,quant):
  '''Logarithmic base 10 of a variable'''
  LG_QUANT = ['lg']  

  docvar = document_vars.vars_documenter(obj, 'LG_QUANT', LG_QUANT, get_lg.__doc__)
  docvar('LG',  'starting with, Logarithmic of a variable [simu units]')

  if (quant == '') or not quant[:2] in LG_QUANT:
    return None

  try: 
    return np.log10(obj.get_var(quant[2:]))
  except:
    return None


def get_ratios(obj,quant):
  '''Ratio of two variables'''
  RATIO_QUANT = 'rat'

  docvar = document_vars.vars_documenter(obj, 'RATIO_QUANT', RATIO_QUANT, get_ratios.__doc__)
  docvar('rat',  'in between with, Ratio of two variables [simu units]')

  if (quant != '') or not RATIO_QUANT in quant:
    return None

  # Calculate module of vector quantity
  q = quant[:quant.find(RATIO_QUANT)]
  if q[0] == 'b':
    if not obj.do_mhd:
      raise ValueError("No magnetic field available.")
  result = obj.get_var(q)
  q = quant[quant.find(RATIO_QUANT) + 3:]
  if q[0] == 'b':
    if not obj.do_mhd:
      raise ValueError("No magnetic field available.")
  return result / (obj.get_var(q) + 1e-19)



def get_projections(obj,quant):
  '''Projected vectors'''
  PROJ_QUANT = ['par', 'per']

  docvar = document_vars.vars_documenter(obj, 'PROJ_QUANT', PROJ_QUANT, get_projections.__doc__)
  docvar('par',  'in between with, parallel component of the first vector respect to the second vector [simu units]')
  docvar('per',  'in between with, perpendicular component of the first vector respect to the second vector [simu units]')


  if (quant == '') or not quant[1:4] in PROJ_QUANT:
    return None

  # projects v1 onto v2
  v1 = quant[0]
  v2 = quant[4]
  x_a = obj.get_var(v1 + 'xc', obj.snap)
  y_a = obj.get_var(v1 + 'yc', obj.snap)
  z_a = obj.get_var(v1 + 'zc', obj.snap)
  x_b = obj.get_var(v2 + 'xc', obj.snap)
  y_b = obj.get_var(v2 + 'yc', obj.snap)
  z_b = obj.get_var(v2 + 'zc', obj.snap)
  # can be used for threadQuantity() or as is
  def proj_task(x1, y1, z1, x2, y2, z2):
    v2Mag = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
    v2x, v2y, v2z = x2 / v2Mag, y2 / v2Mag, z2 / v2Mag
    parScal = x1 * v2x + y1 * v2y + z1 * v2z
    parX, parY, parZ = parScal * v2x, parScal * v2y, parScal * v2z
    result = np.abs(parScal)
    if quant[1:4] == 'per':
      perX = x1 - parX
      perY = y1 - parY
      perZ = z1 - parZ
      v1Mag = np.sqrt(perX**2 + perY**2 + perZ**2)
      result = v1Mag
    return result

  if obj.numThreads > 1:
    if obj.verbose:
      print('Threading', whsp*8, end="\r", flush=True)

    return threadQuantity(proj_task, obj.numThreads,
                          x_a, y_a, z_a, x_b, y_b, z_b)
  else:
    return proj_task(x_a, y_a, z_a, x_b, y_b, z_b)


def get_vector_product(obj,quant):
  VECO_QUANT = ['times']
  
  docvar = document_vars.vars_documenter(obj, 'VECO_QUANT', VECO_QUANT, get_vector_product.__doc__)
  docvar('times',  'in between with, vectorial products or two vectors [simu units]')

  if (quant == '') or not quant[1:6] in VECO_QUANT:
    return None

  # projects v1 onto v2
  v1 = quant[0]
  v2 = quant[-2]
  axis = quant[-1]
  if axis == 'x':
    varsn = ['y', 'z']
  elif axis == 'y':
    varsn = ['z', 'y']
  elif axis == 'z':
    varsn = ['x', 'y']
  #return (obj.get_var(v1 + varsn[0] + 'c') *
  #    obj.get_var(v2 + varsn[1] + 'c') - obj.get_var(v1 + varsn[1] + 'c') *
  #    obj.get_var(v2 + varsn[0] + 'c'))
  return (obj.get_var(v1 + varsn[0]) *
      obj.get_var(v2 + varsn[1]) - obj.get_var(v1 + varsn[1]) *
      obj.get_var(v2 + varsn[0]))


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
