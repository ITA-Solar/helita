from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from . import cstagger

def load_arithmetic_quantities(obj,quant, *args, **kwargs):
  quant = quant.lower()

  if not hasattr(obj,'description'):
    obj.description = {}

  val = get_center(obj,quant)
  if np.shape(val) == ():
    if obj.cstagop != False: # this is only for cstagger routines 
      val = get_deriv(obj,quant)
  if np.shape(val) == ():
    val = get_module(obj,quant)
  if np.shape(val) == ():
    val = get_horizontal_average(obj,quant)
  if np.shape(val) == ():
    val = get_gradients_vect(obj,quant)
  if np.shape(val) == ():
    if obj.cstagop != False:  # this is only for cstagger routines 
      val = get_gradients_scalar(obj,quant)
  if np.shape(val) == ():
    val = get_square(obj,quant)
  if np.shape(val) == ():
    val = get_lg(obj,quant)
  if np.shape(val) == ():
    val = get_ratios(obj,quant)
  if np.shape(val) == ():
    val = get_projections(obj,quant)
  if np.shape(val) == ():
    val = get_vector_product(obj,quant)
  return val

def get_deriv(obj,quant):
  quant = quant.lower()

  DERIV_QUANT = ['dxup', 'dyup', 'dzup', 'dxdn', 'dydn', 'dzdn']
  obj.description['DERIV'] = ('Spatial derivative (Bifrost units). '
                               'It must start with d and end with: ' +
                               ', '.join(DERIV_QUANT))
  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n"+ obj.description['DERIV']
  else:
    obj.description['ALL'] = obj.description['DERIV']

  if (quant == ''):
    return None

  if quant[0] == 'd' and quant[-4:] in DERIV_QUANT:
    # Calculate derivative of quantity
    axis = quant[-3]
    q = quant[1:-4]  # base variable 
    var = obj.get_var(q)

    def deriv_loop(var, quant):
      return cstagger.do(var.astype('float32'), 'd' + quant[0])

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
              output[:, :, iiz] = np.reshape(
                 cstagger.do((var[:, :, iiz].reshape((
                 obj.nx, obj.ny, 1)).astype('float32')), 'd' + quant[-4:]),(obj.nx, obj.ny))
          else:
            for iiy in range(obj.ny):
              output[:, iiy, :] = np.reshape(
                 cstagger.do((var[:, iiy, :].reshape((
                 obj.nx, 1, obj.nz)).astype('float32')), 'd' + quant[-4:]),(obj.nx, obj.nz))

          return output
        else:
          return cstagger.do(var.astype('float32'), 'd' + quant[-4:])
  else:
    return None


def get_center(obj,quant, *args, **kwargs):

  CENTER_QUANT = ['xc', 'yc', 'zc']
  obj.description['CENTER'] = ('Allows to center any vector(Bifrost'
                                ' units). It must end with ' +
                                ', '.join(CENTER_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['CENTER']

  if (quant == ''):
      return None

  if quant[-2:] in CENTER_QUANT:
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
    # 2D
    if getattr(obj, 'n' + axis) < 5 or obj.cstagop == False:
      return var
    else:
      if len(transf) == 2:
        if obj.lowbus:
          output = np.zeros_like(var)
          if transf[0][0] != 'z':
            for iiz in range(obj.nz):
              output[:, :, iiz] = np.reshape(cstagger.do(
                  (var[:, :, iiz].reshape((obj.nx, obj.ny, 1))).astype('float32'),
                  transf[0]), (obj.nx, obj.ny))
          else:
              for iiy in range(obj.ny):
                output[:, iiy, :] = np.reshape(cstagger.do(
                    (var[:, iiy, :].reshape((obj.nx, 1, obj.nz))).astype('float32'),
                    transf[0]), (obj.nx, obj.nz))

          if transf[1][0] != 'z':
              for iiz in range(obj.nz):
                output[:, :, iiz] = np.reshape(cstagger.do(
                    output[:, :, iiz].reshape((obj.nx, obj.ny, 1)),
                    transf[1]), (obj.nx, obj.ny))
          else:
              for iiy in range(obj.ny):
                output[:, iiy, :] = np.reshape(cstagger.do(
                    (output[:, iiy, :].reshape((obj.nx, 1, obj.nz))).astype('float32'),
                    transf[1]), (obj.nx, obj.nz))
          return output
        else:
            tmp = cstagger.do(var.astype('float32'), transf[0])
            return cstagger.do(tmp.astype('float32'), transf[1])
      else:
          if obj.lowbus:
            output = np.zeros_like(var)
            if axis != 'z':
              for iiz in range(obj.nz):
                  output[:, :, iiz] = np.reshape(cstagger.do(
                      (var[:, :, iiz].reshape((obj.nx, obj.ny, 1))).astype('float32'),
                      transf[0]), (obj.nx, obj.ny))
            else:
              for iiy in range(obj.ny):
                  output[:, iiy, :] = np.reshape(cstagger.do(
                      (var[:, iiy, :].reshape((obj.nx, 1, obj.nz))).astype('float32'),
                      transf[0]), (obj.nx, obj.nz))
            return output
          else:
            return cstagger.do(var.astype('float32'), transf[0])
  else:
    return None


def get_module(obj,quant):

  MODULE_QUANT = ['mod', 'h']  # This one must be called the last
  obj.description['MODULE'] = ('Module (starting with mod) or horizontal '
                   '(ending with h) component of vectors (Bifrost units)')
  obj.description['ALL'] += "\n"+ obj.description['MODULE']

  if (quant == ''):
    return None

  if ((quant[:3] in MODULE_QUANT) or (quant[-1] in MODULE_QUANT)):
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
  else:
    return None


def get_horizontal_average(obj,quant):
    HORVAR_QUANT = ['horvar']
    obj.description['HORVAR'] = ('Horizontal average (Bifrost units).'
                                  ' Starting with: ' + ', '.join(HORVAR_QUANT))
    obj.description['ALL'] += "\n"+ obj.description['HORVAR']

    if (quant == ''):
      return None

    if quant[:6] in HORVAR_QUANT:
      # Compares the variable with the horizontal mean
      if quant[:6] == 'horvar':
        result = np.zeros_like(obj.r)
        result += obj.get_var(quant[6:])  # base variable
        horv = np.mean(np.mean(result, 0), 0)
        for iix in range(0, getattr(obj, 'nx')):
          for iiy in range(0, getattr(obj, 'ny')):
            result[iix, iiy, :] = result[iix, iiy, :] / horv[:]
      return result
    else:
        return None


def get_gradients_vect(obj,quant):
  GRADVECT_QUANT = ['div', 'rot', 'she', 'chkdiv', 'chbdiv', 'chhdiv']
  obj.description['GRADVECT'] = ('Vectorial derivative opeartions '
      '(Bifrost units). '
      'The following show divergence, rotational, shear, ratio of the '
      'divergence with the maximum of the abs of each spatial derivative, '
      'with the sum of the absolute of each spatial derivative, with '
      'horizontal averages of the absolute of each spatial derivative '
      'respectively when starting with: ' + ', '.join(GRADVECT_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['GRADVECT']

  if (quant == ''):
      return None

  if quant[:6] in GRADVECT_QUANT or quant[:3] in GRADVECT_QUANT:
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
  else:
    return None


def get_gradients_scalar(obj,quant):
  GRADSCAL_QUANT = ['gra']
  obj.description['GRADSCAL'] = ('Gradient of a scalar (Bifrost units)'
          ' starts with: ' + ', '.join(GRADSCAL_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['GRADSCAL']

  if (quant == ''):
    return None

  if quant[:3] in GRADSCAL_QUANT:
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
  else:
    return None


def get_square(obj,quant):
  SQUARE_QUANT = ['2']  # This one must be called the towards the last
  obj.description['SQUARE'] = ('Square of a variable (Bifrost units)'
          ' ends with: ' + ', '.join(SQUARE_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['SQUARE']

  if (quant == ''):
    return None

  if quant[-1] in SQUARE_QUANT:
    try: 
      result = obj.get_var(quant[:-1] + 'xc') ** 2
      result += obj.get_var(quant[:-1] + 'yc') ** 2
      result += obj.get_var(quant[:-1] + 'zc') ** 2
      return result
    except:
      return None
  else: 
    return None


def get_lg(obj,quant):
  LG_QUANT = ['lg']  
  obj.description['LG'] = ('Logarithmic of a variable'
          ' starts with: ' + ', '.join(LG_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['LG']

  if (quant == ''):
    return None

  if quant[:2] in LG_QUANT:
    try: 
      return np.log10(obj.get_var(quant[2:]))
    except:
      return None
  else: 
    return None


def get_ratios(obj,quant):
  RATIO_QUANT = 'rat'
  obj.description['RATIO'] = ('Ratio of two variables (Bifrost units)'
          'have in between: ' + ', '.join(RATIO_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['RATIO']

  if (quant != ''):
    return None

  if RATIO_QUANT in quant:
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
  else:
    return None


def get_projections(obj,quant):
  PROJ_QUANT = ['par', 'per']
  obj.description['PROJ'] = ('Projected vectors (Bifrost units).'
      ' Parallel and perpendicular have in the middle the following: ' +
      ', '.join(PROJ_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['PROJ']

  if (quant == ''):
    return None

  if quant[1:4] in PROJ_QUANT:
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
  else:
    return None


def get_vector_product(obj,quant):
  VECO_QUANT = ['times']
  obj.description['VECO'] = ('vectorial products (Bifrost units).'
      ' have in the middle the following: ' +
      ', '.join(VECO_QUANT))
  obj.description['ALL'] += "\n"+ obj.description['VECO']

  if (quant == ''):
    return None

  if quant[1:6] in VECO_QUANT:
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
  else:
    return None


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
