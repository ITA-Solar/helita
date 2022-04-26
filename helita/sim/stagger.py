"""
Stagger mesh methods using numba.

set stagger_kind = 'numba' or 'numpy' to use these methods. (not 'cstagger')
stagger_kind = 'numba' is the default for BifrostData and EbysusData.

STAGGER KINDS DEFINED HERE:
    numba          - original 5th order scheme using numba.
                        functions wrapped in njit
    numba_nopython - original 5th order scheme using numba.
                        functions wrapped in jit with nopython=True.
    numpy          - original 5th order scheme using numpy.
    numpy_improved - improved 5th order scheme using numpy.
                        the improvement refers to improved precision for "shift" operations.
                        the improved scheme is also an implemented option in ebysus.
    o1_numpy       - 1st order scheme using numpy.
                        "simplest" method available.
                        good enough, for most uses.
                        ~20% faster than numpy and numpy_improved methods


METHODS DEFINED HERE (which an end-user might want to access):
    do:
        perform the indicated stagger operation.
        interface for the low-level _xshift, _yshift, _zshift functions.

    _xup, _xdn, _yup, _ydn, _zup, _zdn, _ddxup, _ddxdn, _ddyup, _ddydn, _ddzup, _ddzdn:
        peform the corresponding stagger operation on the input array.
        These behave like functions; e.g. stagger._xup(arr) does the 'xup' operation.
        These are interfaces to stagger.do, for all the possible operations which 'do' can handle.

    xup, xdn, yup, ydn, zup, zdn, ddxup, ddxdn, ddyup, ddydn, ddzup, ddzdn:
        Similar to their underscore counterparts, (e.g. xup is like _xup),
        with the additional benefit that they can be chained togther, E.g:
            stagger.xup.ddzdn.yup(arr, diffz=arr_diff) is equivalent to:
            stagger.xup(stagger.ddzdn(stagger.yup(arr), diffz=arr_diff)))

    ^ those methods (xup, ..., ddzdn) in a StaggerInterface:
        Additional benefit that the defaults for pad_mode and diff will be determined based on obj,
        and also if arr is a string, first do arr = obj(arr) (calls 'get_var').
        Example:
            dd = helita.sim.bifrost.BifrostData(...)
            dd.stagger.xdn.yup.ddzdn('r')
            # performs the operations xdn(yup(ddzdn(r)),
            #    using dd.get_param('periodic_x') (or y, z, as appropriate) to choose pad_mode,
            #    and dd.dzidzdn for diff during the ddzdn operation.
        If desired to use non-defaults, the kwargs available are:
            padx, pady, padz kwargs to force a specific pad_mode for a given axis,
            diffx, diffy, diffz kwargs to force a specific diff for a given axis.

TODO:
    - fix ugly printout during verbose==1, for interfaces to 'do', e.g. stagger.xup(arr, verbose=1).
"""

# import built-in modules
import time
import weakref
import collections
import warnings

# import internal modules
from . import tools

# import public external modules
import numpy as np

try:
    from numba import jit, njit, prange
except ImportError:
    numba, prange = tools.ImportFailed('numba', "This module is required to use stagger_kind='numba'.")
    # we still need to set jit and njit, since they are used in top-level of this module as decorators.
    def boring_decorator_factory(*args, **kw):
        def boring_decorator(f):
            return f
        return boring_decorator
    jit = njit = boring_decorator_factory


""" ------------------------ defaults ------------------------ """

PAD_PERIODIC    = 'wrap'     # how to pad periodic dimensions, by default
PAD_NONPERIODIC = 'reflect'  # how to pad nonperiodic dimensions, by default
PAD_DEFAULTS = {'x': PAD_PERIODIC, 'y': PAD_PERIODIC, 'z': PAD_NONPERIODIC}   # default padding for each dimension.
DEFAULT_STAGGER_KIND = 'numba'  # which stagger kind to use by default.
VALID_STAGGER_KINDS  = tuple(('numba', 'numba_nopython',
                              'numpy', 'numpy_improved', 'o1_numpy',
                              'cstagger'))  # list of valid stagger kinds.
PYTHON_STAGGER_KINDS = tuple(('numba', 'numba_nopython',
                              'numpy', 'numpy_improved', 'o1_numpy',
                              ))  # list of valid stagger kinds from stagger.py.
ALIAS_STAGGER_KIND   = {'stagger': 'numba', 'numba': 'numba',   # dict of aliases for stagger kinds.
                       'numba_nopython': 'numba_nopython', 
                       'numpy': 'numpy',
                       'numpy_improved': 'numpy_improved', 'numpy_i': 'numpy_improved',
                       'o1_numpy': 'o1_numpy',
                       'cstagger': 'cstagger'}
DEFAULT_MESH_LOCATION_TRACKING = False   # whether mesh location tracking should be enabled, by default.

def STAGGER_KIND_PROPERTY(internal_name='_stagger_kind', default=DEFAULT_STAGGER_KIND):
    '''creates a property which manages stagger_kind.
    uses the internal name provided, and returns the default if property value has not been set.

    only allows setting of stagger_kind to valid names (as determined by VALID_STAGGER_KINDS).
    '''
    def get_stagger_kind(self):
        return getattr(self, internal_name, default)

    def set_stagger_kind(self, value):
        '''sets stagger_kind to VALID_STAGGER_KINDS[value]'''
        try:
            kind = ALIAS_STAGGER_KIND[value]
        except KeyError:
            class KeyErrorMessage(str):  # KeyError(msg) uses repr(msg), so newlines don't show up.
                def __repr__(self): return str(self)    # this is a workaround. Makes the message prettier.
            errmsg = (f"stagger_kind = {repr(value)} was invalid!" + "\n" + 
                      f"Expected value from: {VALID_STAGGER_KINDS}." + "\n" + 
                      f"Advanced: to add a valid value, edit helita.sim.stagger.ALIAS_STAGGER_KINDS")
            raise KeyError(KeyErrorMessage(errmsg)) from None
        setattr(self, internal_name, kind)

    doc = f"Tells which method to use for stagger operations. Options are: {VALID_STAGGER_KINDS}"

    return property(fset=set_stagger_kind, fget=get_stagger_kind, doc=doc)

""" ------------------------ stagger constants ------------------------ """

StaggerConstants = collections.namedtuple('StaggerConstants', ('a', 'b', 'c'))

## FIFTH ORDER SCHEME ##
# derivatives
c = (-1 + (3**5 - 3) / (3**3 - 3)) / (5**5 - 5 - 5 * (3**5 - 3))
b = (-1 - 120*c) / 24
a = (1 - 3*b - 5*c)
CONSTANTS_DERIV = StaggerConstants(a, b, c)

# shifts (i.e. not a derivative)
c = 3.0 / 256.0
b = -25.0 / 256.0
a = 0.5 - b - c
CONSTANTS_SHIFT = StaggerConstants(a, b, c)


## FIRST ORDER SCHEME ##
CONSTANTS_DERIV_o1 = StaggerConstants(1.0, 0, 0)
CONSTANTS_SHIFT_o1 = StaggerConstants(0.5, 0, 0)


## GENERIC ##
CONSTANTS_DERIV_ODICT = {5: CONSTANTS_DERIV, 1: CONSTANTS_DERIV_o1}
CONSTANTS_SHIFT_ODICT = {5: CONSTANTS_SHIFT, 1: CONSTANTS_SHIFT_o1}
def GET_CONSTANTS_DERIV(order):
    return CONSTANTS_DERIV_ODICT[order]
def GET_CONSTANTS_SHIFT(order):
    return CONSTANTS_SHIFT_ODICT[order]

# remove temporary variables from the module namespace
del c, b, a


""" ------------------------ 'do' - stagger interface ------------------------ """

def do(var, operation='xup', diff=None, pad_mode=None, stagger_kind=DEFAULT_STAGGER_KIND):
    """
    Do a stagger operation on `var` by doing a 6th order polynomial interpolation of 
    the variable from cell centres to cell faces (down operations), or cell faces
    to cell centres (up operations).
    
    Parameters
    ----------
    var : 3D array
        Variable to work on.
        if not 3D, makes a warning but still tries to return a reasonable result:
            for non-derivatives, return var (unchanged).
            for derivatives, return 0 (as an array with same shape as var).
    operation: str
        Type of operation. Currently supported values are
        * 'xup', 'xdn', 'yup', 'ydn', 'zup', 'zdn'
        * 'ddxup', 'ddxdn', 'ddyup', 'ddydn', 'ddzup', 'ddzdn' (derivatives)
    diff: None or 1D array
        If operation is one of the derivatives, you must supply `diff`,
        an array with the distances between each cell in the direction of the
        operation must be same length as array along that direction. 
        For non-derivative operations, `diff` must be None.
    pad_mode : None or str
        Mode for padding array `var` to have enough points for a 6th order
        polynomial interpolation. Same as supported by np.pad.
        if None, use default: `wrap` (periodic) for x and y; `reflect` for z.
    stagger_kind: 'numba', 'numpy', or 'numpy_improved'
        Mode for stagger operations.
        numba --> numba methods ('_xshift', '_yshift', '_zshift')
        numpy --> numpy methods ('_np_stagger')
        numpy_improved --> numpy implmentation of improved method. ('_np_stagger_improved')
        For historical reasons, stagger_kind='stagger' --> stagger_kind='numba'.

    Returns
    -------
    3D array
        Array of same type and dimensions to var, after performing the 
        stagger operation.
    """
    # initial bookkeeping
    AXES = ('x', 'y', 'z')
    operation = operation_orig = operation.lower()
    stagger_kind = ALIAS_STAGGER_KIND[stagger_kind]
    # order
    if stagger_kind == 'o1_numpy':
        order = 1
        stagger_kind = 'numpy'
    else:
        order = 5
    # derivative, diff
    if operation[:2] == 'dd':  # For derivative operations
        derivative = True
        operation = operation[2:]
        if diff is None:
            raise ValueError(f"diff not provided for derivative operation: {operation_orig}")
    else:
        derivative = False
        if diff is not None:
            raise ValueError(f"diff must not be provided for non-derivative operation: {operation}")
    # make sure var is 3D. make warning then handle appropriately if not.
    if var.ndim != 3:
        warnmsg = f'can only stagger 3D array but got {var.ndim}D.'
        if derivative:
            warnings.warn(warnmsg + f' returning 0 for operation {operation_orig}')
            return np.zeros_like(var)
        else:
            warnings.warn(warnmsg + f' returning original array for operation {operation_orig}')
            return var
    # up/dn
    up_str = operation[-2:]  # 'up' or 'dn'
    if up_str == 'up':
        up = True
    elif up_str == 'dn':
        up = False
    else: 
        raise ValueError(f"Invalid operation; must end in 'up' or 'dn': {operation}")
    # x, dim_index (0 for x, 1 for y, 2 for z)
    x = operation[:-2]
    if x not in AXES:
        raise ValueError(f"Invalid operation; axis must be 'x', 'y', or 'z': {operation}")
    if pad_mode is None:
        pad_mode = PAD_DEFAULTS[x]
    dim_index = AXES.index(x)
    # padding
    extra_dims = (2, 3) if up else (3, 2)
    if not derivative:
        diff = np.ones(var.shape[dim_index], dtype=var.dtype)
    padding = [(0, 0)] * 3
    padding[dim_index] = extra_dims
    # interpolating
    if var.shape[dim_index] <= 5:   # don't interpolate along axis with size 5 or less...
        if derivative:
            result = np.zeros_like(var)   # E.g. ( dvardzup, where var has shape (Nx, Ny, 1) ) --> 0
        else:
            result = var
    else:
        out = np.pad(var, padding, mode=pad_mode)
        out_diff = np.pad(diff, extra_dims, mode=pad_mode)
        if stagger_kind=='numba':
            func = {'x':_xshift, 'y':_yshift, 'z':_zshift}[x]
            result = func(out, out_diff, up=up, derivative=derivative)
        elif stagger_kind=='numba_nopython':
            result = _numba_stagger(out, out_diff, up, derivative, dim_index)
        elif stagger_kind=='numpy':   # 'numpy' or 'o1_numpy', originally.
            result = _np_stagger(out, out_diff, up, derivative, dim_index, order=order)
        elif stagger_kind=='numpy_improved':
            result = _np_stagger_improved(out, out_diff, up, derivative, dim_index)
        else:
            raise ValueError(f"invalid stagger_kind: '{stagger_kind}'. Options are: {PYTHON_STAGGER_KINDS}")
    # tracking mesh location.
    meshloc = getattr(var, 'meshloc', None)
    if meshloc is not None:  # (input array had a meshloc attribute)
        result = ArrayOnMesh(result, meshloc=meshloc)
        result._shift_location(f'{x}{up_str}')
    # output.
    return result


""" ------------------------ numba stagger ------------------------ """

## STAGGER_KIND = NUMBA ##
@njit(parallel=True)
def _xshift(var, diff, up=True, derivative=False):
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    if derivative:
        pm, (a, b, c) = -1, CONSTANTS_DERIV
    else:
        pm, (a, b, c) =  1, CONSTANTS_SHIFT
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(nz): 
        for j in prange(ny):
            for i in prange(start, nx + end):
                out[i, j, k] = diff[i] * (a * (var[i+ grdshf, j, k] + pm * var[i - 1 + grdshf, j, k]) +
                                b * (var[i + 1 + grdshf, j, k] + pm * var[i - 2 + grdshf, j, k]) +
                                c * (var[i + 2 + grdshf, j, k] + pm * var[i - 3 + grdshf, j, k]))

    return out[start:end, :, :]

@njit(parallel=True)
def _yshift(var, diff, up=True, derivative=False):
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    if derivative:
        pm, (a, b, c) = -1, CONSTANTS_DERIV
    else:
        pm, (a, b, c) =  1, CONSTANTS_SHIFT
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(nz): 
        for j in prange(start, ny + end):
            for i in prange(nx):
                out[i, j, k] = diff[j] * (a * (var[i, j + grdshf, k] + pm * var[i, j - 1 + grdshf, k]) +
                                b * (var[i, j + 1 + grdshf, k] + pm * var[i, j - 2 + grdshf, k]) +
                                c * (var[i, j + 2 + grdshf, k] + pm * var[i, j - 3 + grdshf, k]))
    return out[:, start:end, :]

@njit(parallel=True)
def _zshift(var, diff, up=True, derivative=False):
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    if derivative:
        pm, (a, b, c) = -1, CONSTANTS_DERIV
    else:
        pm, (a, b, c) =  1, CONSTANTS_SHIFT
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(start, nz + end): 
        for j in prange(ny):
            for i in prange(nx):
                out[i, j, k] = diff[k] * (a * (var[i, j, k + grdshf] + pm * var[i, j, k - 1 + grdshf]) +
                                b * (var[i, j, k + 1 + grdshf] + pm * var[i, j, k - 2 + grdshf]) +
                                c * (var[i, j, k + 2 + grdshf] + pm * var[i, j, k - 3 + grdshf]))
    return out[:, :, start:end]

## STAGGER_KIND = NUMBA_NOPYTHON ##
def _numba_stagger(var, diff, up, derivative, x):
    '''stagger along x axis. x should be 0, 1, or 2. Corresponds to stagger_kind='numba_compiled'.

    The idea is to put the numba parts of _xshift, _yshift, _zshift
        in their own functions, so that we can compile them with nopython=True.

    brief testing (by SE on 2/18/22) showed no speed improvement compared to stagger_kind='numba' method.
    '''
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    if derivative:
        pm, (a, b, c) = -1, CONSTANTS_DERIV
    else:
        pm, (a, b, c) =  1, CONSTANTS_SHIFT
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    _nopython_shift = {0:_nopython_xshift, 1:_nopython_yshift, 2:_nopython_zshift}[x]
    return _nopython_shift(var, diff, out, nx, ny, nz, start, end, grdshf, a, b, c, pm)

@jit(parallel=True, nopython=True)
def _nopython_xshift(var, diff, out, nx, ny, nz, start, end, grdshf, a, b, c, pm):
    for k in prange(nz): 
        for j in prange(ny):
            for i in prange(start, nx + end):
                out[i, j, k] = diff[i] * (a * (var[i+ grdshf, j, k] + pm * var[i - 1 + grdshf, j, k]) +
                                b * (var[i + 1 + grdshf, j, k] + pm * var[i - 2 + grdshf, j, k]) +
                                c * (var[i + 2 + grdshf, j, k] + pm * var[i - 3 + grdshf, j, k]))
    return out[start:end, :, :]

@jit(parallel=True, nopython=True)
def _nopython_yshift(var, diff, out, nx, ny, nz, start, end, grdshf, a, b, c, pm):
    for k in prange(nz): 
        for j in prange(start, ny + end):
            for i in prange(nx):
                out[i, j, k] = diff[j] * (a * (var[i, j + grdshf, k] + pm * var[i, j - 1 + grdshf, k]) +
                                b * (var[i, j + 1 + grdshf, k] + pm * var[i, j - 2 + grdshf, k]) +
                                c * (var[i, j + 2 + grdshf, k] + pm * var[i, j - 3 + grdshf, k]))
    return out[:, start:end, :]

@jit(parallel=True, nopython=True)
def _nopython_zshift(var, diff, out, nx, ny, nz, start, end, grdshf, a, b, c, pm):
    for k in prange(start, nz + end): 
        for j in prange(ny):
            for i in prange(nx):
                out[i, j, k] = diff[k] * (a * (var[i, j, k + grdshf] + pm * var[i, j, k - 1 + grdshf]) +
                                b * (var[i, j, k + 1 + grdshf] + pm * var[i, j, k - 2 + grdshf]) +
                                c * (var[i, j, k + 2 + grdshf] + pm * var[i, j, k - 3 + grdshf]))
    return out[:, :, start:end]


""" ------------------------ numpy stagger ------------------------ """

def slicer_at_ax(slicer, ax):
    '''return tuple of slices which, when applied to an array, takes slice along axis number <ax>.
    slicer: a slice object, or integer, or tuple of integers.
        slice or integer  -> use slicer directly.
        tuple of integers -> use slice(*slicer).
    ax: a number (negative ax not supported here).
    '''
    try:
        slicer[0]
    except TypeError: #slicer is a slice or an integer.
        pass  
    else: #assume slicer is a tuple of integers.
        slicer = slice(*slicer)
    return (slice(None),)*ax + (slicer,)

## STAGGER_KIND = NUMPY ##
def _np_stagger(var, diff, up, derivative, x, order=5):
    """stagger along x axis. x should be 0, 1, or 2."""
    # -- same constants and setup as numba method -- #
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    if derivative:
        pm, (a, b, c) = -1, GET_CONSTANTS_DERIV(order)
    else:
        pm, (a, b, c) =  1, GET_CONSTANTS_SHIFT(order)
    # -- begin numpy syntax -- #
    nx = var.shape[x]
    def slx(shift):
        '''return slicer at x axis from (start + shift) to (nx + end + shift).'''
        return slicer_at_ax((start+shift, nx+end+shift), x)
    def sgx(shift):
        '''return slicer at x axis from (start + shift + grdshf) to (nx + end + shift + grdshf)'''
        return slx(shift + grdshf)
    diff = np.expand_dims(diff,  axis=tuple( set((0,1,2)) - set((x,)) )  )   # make diff 3D (with size 1 for axes other than x)

    if order == 5:
        out = diff[slx(0)] * (a * (var[sgx(0)] + pm * var[sgx(-1)]) + 
                              b * (var[sgx(1)] + pm * var[sgx(-2)]) +
                              c * (var[sgx(2)] + pm * var[sgx(-3)]))
    elif order == 1:
        out = diff[slx(0)] * (a * (var[sgx(0)] + pm * var[sgx(-1)]))  # b=c=0 for order 1.

    return out

## STAGGER_KIND = NUMPY_IMPROVED ##
def _np_stagger_improved(var, diff, up, derivative, x):
    """stagger along x axis. x should be 0, 1, or 2.
    uses the "improved" stagger method, as implemented in stagger_mesh_improved_mpi.f90.
        It subtracts f_0 from each term before multiplying, then adds f0 again at the end.
        since a + b + c = 0.5 by definition,
            a X + b Y + c Z == a (X - 2 f_0) + b (Y - 2 f_0) + c (Z - 2 f_0) + f_0
    """
    grdshf = 1 if up else 0
    start  =   int(3. - grdshf)  
    end    = - int(2. + grdshf)
    # -- begin numpy syntax -- #
    nx = var.shape[x]
    def slx(shift):
        '''return slicer at x axis from (start + shift) to (nx + end + shift).'''
        return slicer_at_ax((start+shift, nx+end+shift), x)
    def sgx(shift):
        '''return slicer at x axis from (start + shift + grdshf) to (nx + end + shift + grdshf)'''
        return slx(shift + grdshf)
    diff = np.expand_dims(diff,  axis=tuple( set((0,1,2)) - set((x,)) )  )   # make diff 3D (with size 1 for axes other than x)

    if derivative:
        # formula is exactly the same as regular numpy method. (though we use '-' instead of 'pm' with pm=-1)
        a, b, c = CONSTANTS_DERIV
        out = diff[slx(0)] * (a * (var[sgx(0)] - var[sgx(-1)]) + 
                              b * (var[sgx(1)] - var[sgx(-2)]) +
                              c * (var[sgx(2)] - var[sgx(-3)]))
    else:
        # here is where we see the 'improved' stagger method.
        a, b, c = CONSTANTS_SHIFT
        f0 = var[sgx(0)]
        out = diff[slx(0)] * (a * (                   var[sgx(-1)] - f0) +    # note: the f0 - f0 term went away.
                              b * (var[sgx(1)] - f0 + var[sgx(-2)] - f0) +
                              c * (var[sgx(2)] - f0 + var[sgx(-3)] - f0)
                              + f0)

    return out


""" ------------------------ MeshLocation, ArrayOnMesh ------------------------ """
# The idea is to associate arrays with a location on the mesh,
#   update that mesh location info whenever a stagger operation is performed,
#   and enforce arrays have the same location when doing arithmetic.

class MeshLocation():
    '''class defining a location on a mesh.
    Also provides shifting operations.

    Examples:
        m = MeshLocation([0, 0.5, 0])
        m.xup
        >>> MeshLocation([0.5, 0.5, 0])
        m.xup.ydn.zdn
        >>> MeshLocation([0.5, 0, -0.5])
    '''
    def __init__(self, loc=[0,0,0]):
        self.loc = list(loc)

    def __repr__(self):
        return f'{type(self).__name__}({self.loc})'  # TODO even spacing (length == len('-0.5'))

    def _new(self, *args, **kw):
        return type(self)(*args, **kw)

    ## LIST-LIKE BEHAVIOR ##

    def __iter__(self):
        return iter(self.loc)

    def __len__(self):
        return len(self.loc)

    def __getitem__(self, i):
        return self.loc[i]

    def __setitem__(self, i, value):
        self.loc[i] = value

    def __eq__(self, other):
        if len(other) != len(self):
            return False
        return all(s == o for s, o in zip(self, other))

    def copy(self):
        return MeshLocation(self)

    ## MESH LOCATION ARITHMETIC ##
    def __add__(self, other):
        '''element-wise addition of self + other, returned as a MeshLocation.'''
        return self._new([s + o for s, o in zip(self, other)])

    def __sub__(self, other):
        '''element-wise subtraction of self - other, returned as a MeshLocation.'''
        return self._new([s - o for s, o in zip(self, other)])

    def __radd__(self, other):
        '''element-wise addition of other + self, returned as a MeshLocation.'''
        return self._new([o + s for s, o in zip(self, other)])

    def __rsub__(self, other):
        '''element-wise subtraction of other - self, returned as a MeshLocation.'''
        return self._new([o - s for s, o in zip(self, other)])

    ## MESH LOCATION AS OPERATION LIST ##
    def as_operations(self):
        '''returns self, viewed as a list of operations. (returns a list of strings.)
        equivalently, returns "steps needed to get from (0,0,0) to self". 
        
        Examples:
            MeshLocation([0.5, 0, 0]).as_operations()
            >>> ['xup']
            MeshLocation([0, -0.5, -0.5]).as_operations()
            >>> ['ydn', 'zdn']
            MeshLocation([1.0, -0.5, -1.5]).as_operations()
            >>> ['xup', 'xup', 'ydn', 'zdn', 'zdn', 'zdn']
        '''
        AXES = ('x', 'y', 'z')
        result = []
        for x, val in zip(AXES, self):
            if val == 0:
                continue
            n = val / 0.5   # here we expect n to be an integer-valued float. (e.g. 1.0)
            assert getattr(n, 'is_integer', lambda: True)(), f"Expected n/0.5 to be an integer. n={n}, self={self}"
            up = 'up' if val > 0 else 'dn'
            n = abs(int(n)) # convert n to positive integer (required for list multiplication)
            result += ([f'{x}{up}'] * n)  # list addition; list multiplication.
        return result

    as_ops = property(lambda self: self.as_operations, doc='alias for as_operations')

    def steps_from(self, other):
        '''return the steps needed to get FROM other TO self. (returns a list of strings.)

        Examples:
            MeshLocation([0.5, 0, 0]).steps_from([0,0,0])
            >>> ['xup']
            MeshLocation([-0.5, 0, 0]).steps_from(MeshLocation([0.5, -0.5, -0.5]))
            >>> ['xdn', 'xdn', 'yup', 'zup']
        '''
        return (self - other).as_operations()

    def steps_to(self, other):
        '''return the steps needed to get TO other FROM self. (returns a list of strings.)

        Examples:
            MeshLocation([0.5, 0, 0]).steps_to([0,0,0])
            >>> ['xdn']
            MeshLocation([-0.5, 0, 0]).steps_to(MeshLocation([0.5, -0.5, -0.5]))
            >>> ['xup', 'xup', 'ydn', 'zdn']
        '''
        return (other - self).as_operations()

    ## MESH LOCATION DESCRIPTION ##
    def describe(self):
        '''returns a description of self.
        The possible descriptions are:
            ('center', None),
            ('face', 'x'), ('face', 'y'), ('face', 'z'),
            ('edge', 'x'), ('edge', 'y'), ('edge', 'z'),
            ('unknown', None)
        They mean:
            'center' --> location == (0,0,0)
            'face_x' --> location == (-0.5, 0, 0)    # corresponds to x-component of a face-centered vector like magnetic field.
            'edge_x' --> location == (0, -0.5, -0.5) # correspodns to x-component of an edge-centered vector like electric field.
            'unknown' --> location is not center, face, or edge.
        face_y, face_z, edge_y, edge_z take similar meanings as face_x, edge_x, but for the y, z directions instead.

        returns one of the tuples above.
        '''
        lookup = {-0.5: True, 0: False}
        xdn = lookup.get(self[0], None)
        ydn = lookup.get(self[1], None)
        zdn = lookup.get(self[2], None)
        pos = (xdn, ydn, zdn)
        if all(p is True for p in pos):
            return ('center', None)
        if any(p is None for p in pos) or all(p is True for p in pos):
            return ('unknown', None)
        if xdn:
            if   ydn: return ('edge', 'z')
            elif zdn: return ('edge', 'y')
            else:     return ('face', 'x')
        elif ydn:
            if   zdn: return ('edge', 'x')
            else:     return ('face', 'y')
        elif zdn:
            return ('face', 'z')
        # could just return ('unknown', None) if we reach this line.
        # But we expect the code to have handled all cases by this line.
        # So if this error is ever raised, we made a mistake in the code of this function.
        assert False, f"Expected all meshlocs should have been accounted for, but this one was not: {self}"

    ## MESH LOCATION SHIFTING ##
    def shifted(self, xup):
        '''return a copy of self shifted by xup.
        xup: 'xup', 'xdn', 'yup', 'ydn', 'zup', or 'zdn'.
        '''
        return getattr(self, xup)

    # the properties: xup, xdn, yup, ydn, zup, zdn
    #   are added to the class after its initial definition.

## MESH LOCATION SHIFTING ##
def _mesh_shifter(x, up):
    '''returns a function which returns a copy of MeshLocation but shifted by x and up.
    x should be 'x', 'y', or 'z'.
    up should be 'up' or 'dn'.
    '''
    ix = {'x':0, 'y':1, 'z':2}[x]
    up_value = {'up':0.5, 'dn':-0.5}[up]
    def mesh_shifted(self):
        '''returns a copy of self shifted by {x}{up}'''
        copy = self.copy()
        copy[ix] += up_value
        return copy
    mesh_shifted.__doc__  = mesh_shifted.__doc__.format(x=x, up=up)
    mesh_shifted.__name__ = f'{x}{up}'
    return mesh_shifted

def _mesh_shifter_property(x, up):
    '''returns a property which calls a function that returns a copy of MeshLocation shifted by x and up.'''
    shifter = _mesh_shifter(x, up)
    return property(fget=shifter, doc=shifter.__doc__)

# actually set the functions xup, ..., zdn, as methods of MeshLocation.
for x in ('x', 'y', 'z'):
    for up in ('up', 'dn'):
        setattr(MeshLocation, f'{x}{up}', _mesh_shifter_property(x, up))


class ArrayOnMesh(np.ndarray):
    '''numpy array associated with a location on a mesh grid.
    
    Examples:
        ArrayOnMesh(x, meshloc=[0,0,0])
        ArrayOnMesh(y, meshloc=[0,0,0.5])
        with x, y numpy arrays (or subclasses).
    
    The idea is to enforce that arrays are at the same mesh location before doing any math.
    When arrays are at different locations, raise an AssertionError instead.

    The operations xup, ..., zdn are intentionally not provided here.
        This is to avoid potential confusion of thinking stagger is being performed when it is not.
        ArrayOnMesh does not know how to actually do any of the stagger operations.
        Rather, the stagger operations are responsible for properly tracking mesh location;
            they can use the provided _relocate or _shift_location methods to do so.
    
    meshloc: list, tuple, MeshLocation object, or None
        None --> default. If input has meshloc, use meshloc of input; else use [0,0,0]
        else --> use this value as the mesh location.
    '''
    def __new__(cls, input_array, meshloc=None):
        obj = np.asanyarray(input_array).view(cls)   # view input_array as an ArrayOnMesh.
        if meshloc is None:
            obj.meshloc = getattr(obj, 'meshloc', [0,0,0])
        else:
            obj.meshloc = meshloc
        return obj

    def __array_finalize__(self, obj):
        '''handle other ways of creating this array, e.g. copying an existing ArrayOnMesh.'''
        if obj is None: return
        self.meshloc = getattr(obj, 'meshloc', [0,0,0])

    def describe_mesh_location(self):
        '''returns a description of the mesh location of self.
        The possible descriptions are:
            ('center', None),
            ('face', 'x'), ('face', 'y'), ('face', 'z'),
            ('edge', 'x'), ('edge', 'y'), ('edge', 'z'),
            ('unknown', None)
        They mean:
            'center' --> location == (0,0,0)
            'face_x' --> location == (-0.5, 0, 0)    # corresponds to x-component of a face-centered vector like magnetic field.
            'edge_x' --> location == (0, -0.5, -0.5) # correspodns to x-component of an edge-centered vector like electric field.
            'unknown' --> location is not center, face, or edge.
        face_y, face_z, edge_y, edge_z take similar meanings as face_x, edge_x, but for the y, z directions instead.

        returns one of the tuples above.
        '''
        return self.meshloc.describe()

    @property
    def meshloc(self):
        return self._meshloc
    @meshloc.setter
    def meshloc(self, newloc):
        if not isinstance(newloc, MeshLocation):
            newloc = MeshLocation(newloc)
        self._meshloc = newloc

    def _relocate(self, new_meshloc):
        '''changes the location associated with self to new_meshloc.
        DOES NOT PERFORM ANY STAGGER OPERATIONS -
            the array contents will be unchanged; only the mesh location label will be affected.
        '''
        self.meshloc = new_meshloc

    def _shift_location(self, xup):
        '''shifts the location associated with self by xup.
        DOES NOT PERFORM ANY STAGGER OPERATIONS -
            the array contents will be unchanged; only the mesh location label will be affected.
        '''
        self._relocate(self.meshloc.shifted(xup))

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        '''does the ufunc but first ensures all arrays are at the same meshloc.
        
        The code here follows the format of the example from the numpy subclassing docs.
        '''
        args = []
        meshloc = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, type(self)):
                if meshloc is None:
                    meshloc = input_.meshloc
                else:
                    assert meshloc == input_.meshloc, f"Inputs' mesh locations differ: {meshloc}, {input_.meshloc}"
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)
                
        assert meshloc is not None  # meshloc should have been set to some value by this point.

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, type(self)):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], type(self)):
                inputs[0].meshloc = meshloc
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(type(self))
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], type(self)):
            results[0].meshloc = meshloc

        return results[0] if len(results) == 1 else results
        
    def __repr__(self):
        result = super().__repr__()
        return f'{result} at {self.meshloc}'


# predefined mesh locations
def mesh_location_center():
    '''returns MeshLocation at center of box. (0,0,0)'''
    return MeshLocation([0,0,0])

def mesh_location_face(x):
    '''returns MeshLocation centered at face x.
    x: 'x', 'y', or 'z'.
        'x' --> [-0.5,  0  ,  0  ]
        'y' --> [ 0  , -0.5,  0  ]
        'z' --> [ 0  ,  0  , -0.5]
    '''
    loc = {'x' : [-0.5,  0  ,  0  ],
           'y' : [ 0  , -0.5,  0  ],
           'z' : [ 0  ,  0  , -0.5]}
    return MeshLocation(loc[x])

def mesh_location_edge(x):
    '''returns MeshLocation centered at edge x.
    x: 'x', 'y', or 'z'.
        'x' --> [ 0  , -0.5, -0.5]
        'y' --> [-0.5,  0  , -0.5]
        'z' --> [-0.5, -0.5,  0  ]
    '''
    loc = {'x' : [ 0  , -0.5, -0.5],
           'y' : [-0.5,  0  , -0.5],
           'z' : [-0.5, -0.5,  0  ]}
    return MeshLocation(loc[x])

# describing mesh locations (for a "generic object")
def get_mesh_location(obj, *default):
    if len(default) > 0:
        return getattr(obj, 'meshloc', default[0])
    else:
        return getattr(obj, 'meshloc')

def has_mesh_location(obj):
    return hasattr(obj, 'meshloc')

def describe_mesh_location(obj):
    '''returns a description of the mesh location of obj
    The possible descriptions are:
        ('center', None),
        ('face', 'x'), ('face', 'y'), ('face', 'z'),
        ('edge', 'x'), ('edge', 'y'), ('edge', 'z'),
        ('unknown', None)
        ('none', None)
    They mean:
        'center' --> location == (0,0,0)
        'face_x' --> location == (-0.5, 0, 0)    # corresponds to x-component of a face-centered vector like magnetic field.
        'edge_x' --> location == (0, -0.5, -0.5) # correspodns to x-component of an edge-centered vector like electric field.
        'unknown' --> location is not center, face, or edge.
        'none'   --> obj is not a MeshLocation and does not have attribute meshloc.
    face_y, face_z, edge_y, edge_z take similar meanings as face_x, edge_x, but for the y, z directions instead.

    returns one of the tuples above.
    '''
    if isinstance(obj, MeshLocation):
        return obj.describe()
    elif hasattr(obj, 'meshloc'):
        return obj.meshloc.describe()
    else:
        return ('unknown', None)

# mesh location tracking property
def MESH_LOCATION_TRACKING_PROPERTY(internal_name='_mesh_location_tracking', default=DEFAULT_MESH_LOCATION_TRACKING):
    '''creates a property which manages mesh_location_tracking.
    uses the internal name provided, and returns the default if property value has not been set.

    checks self.do_stagger and self.stagger_kind for compatibility (see doc of the produced property for details).
    '''
    doc = f'''whether mesh location tracking is enabled. (default is {default})
        True --> arrays from get_var will be returned as stagger.ArrayOnMesh objects,
                 which track the location on mesh but also require locations of
                 arrays (if they are ArrayOnMesh) to match before doing arithmetic.
        False --> stagger.ArrayOnMesh conversion will be disabled.

        Tied directly to self.do_stagger and self.stagger_kind.
            when self.do_stagger or self.stagger_kind are INCOMPATIBLE with mesh_location_tracking,
                mesh_location_tracking will be disabled, until compatibility requirements are met.
                trying to set mesh_location_tracking = True will make a ValueError.
            INCOMPATIBLE when one or more of the following are True: 
                1) bool(self.do_stagger) != True
                2) self.stagger_kind not in stagger.PYTHON_STAGGER_KINDS
                    (compatible stagger_kinds are {PYTHON_STAGGER_KINDS})
        '''
    def _mesh_location_tracking_incompatible(obj):
        '''returns attributes of obj with present values incompatible with mesh_location_tracking.
        e.g. ['do_stagger', 'stagger_kind'], or ['stagger_kind'], or ['do_stagger'] or [].

        non-existing attributes do not break compatibility.
        E.g. if do_stagger and stagger_kind are unset, result will be [] (i.e. "fully compatible").
        '''
        result = []
        if not getattr(obj, 'do_stagger', True):
            result.append('do_stagger')
        if not getattr(obj, 'stagger_kind', PYTHON_STAGGER_KINDS[0]) in PYTHON_STAGGER_KINDS:
            result.append('stagger_kind')
        return result

    def get_mesh_location_tracking(self):
        result = getattr(self, '_mesh_location_tracking', default)
        if result:
            # before returning True, check compatibility.
            incompatible = _mesh_location_tracking_incompatible(self)
            if incompatible:
                return False
        return result

    def set_mesh_location_tracking(self, value):
        if value:
            # before setting to True, check compatibility.
            incompatible = _mesh_location_tracking_incompatible(self)
            if incompatible:
                # make a ValueError with helpful instructions.
                errmsg = f"present values of attributes {incompatible} are incompatible" +\
                    "with mesh_location_tracking. To enable mesh_location_tracking, first you must"
                if 'do_stagger' in incompatible:
                    errmsg += " enable do_stagger"
                if len(incompatible) > 1:
                    errmsg += " and"
                if 'stagger_kind' in incompatible:
                    errmsg += f" set stagger_kind to one of the python stagger kinds: {PYTHON_STAGGER_KINDS}"
                errmsg += "."
                raise ValueError(errmsg)
        self._mesh_location_tracking = value

    return property(fget=get_mesh_location_tracking, fset=set_mesh_location_tracking, doc=doc)


""" ------------------------ Aliases ------------------------ """
# Here we define the 12 supported operations, using the 'do' function defined above. The ops are:
#    xdn, xup, ydn, yup, zdn, zup, ddxdn, ddxup, ddydn, ddyup, ddzdn, ddzup
# This is for convenience; the 'work' of the stagger methods is handled by the functions above.

# The definitions here all put a leading underscore '_'. E.g.: _xup, _ddydn.
# This is because users should prefer the non-underscored versions defined in the next section,
# since those can be chained together, e.g. ddxdn.xup.ydn(arr) equals to _ddxdn(_xup(_ydn(arr))).


class _stagger_factory():
    def __init__(self, x, up, opstr_fmt):
        self.x = x
        self.up = up
        self.opstr = opstr_fmt.format(x=x, up=up)
        self.__doc__ = self.__doc__.format(up=up, x=x, pad_default=PAD_DEFAULTS[x])
        self.__name__ = f'_{self.opstr}'

    def __call__(self, arr, pad_mode=None, verbose=False,
                 padx=None, pady=None, padz=None,
                 **kw__do):
        if pad_mode is None:
            pad_mode = {'x':padx, 'y': pady, 'z':padz}[self.x]
        if verbose:
            end = '\n' if verbose>1 else '\r\r'
            msg = f'interpolating: {self.opstr:>5s}.'
            print(msg, end=' ', flush=True)
            now = time.time()
        result = do(arr, self.opstr, pad_mode=pad_mode, **kw__do)
        if verbose:
            print(f'Completed in {time.time()-now:.4f} seconds.', end=end, flush=True)
        return result


class _stagger_spatial(_stagger_factory):
    '''interpolate data one half cell {up} in {x}.
    arr: 3D array
        the data to be interpolated.
    pad_mode: None or str.
        pad mode for the interpolation; same options as those supported by np.pad.
        if None, the default for this operation will be used: '{pad_default}'
    verbose: 0, 1, 2
        0 --> no verbosity
        1 --> print, end with '\r'.
        2 --> print, end with '\n'.
    padx, pady, padz: None or string
        pad_mode, but only applies for operation in the corresponding axis.
        (For convenience. E.g. if all pad_modes are known, can enter padx, pady, padz,
        without needing to worry about which type of operation is being performed.)

    **kw__None:
        additional kwargs are ignored.

    TODO: fix ugly printout during verbose==1.
    '''
    def __init__(self, x, up):
        super().__init__(x, up, opstr_fmt='{x}{up}')

    def __call__(self, arr, pad_mode=None, verbose=False,
                 padx=None, pady=None, padz=None, stagger_kind=DEFAULT_STAGGER_KIND, **kw__None):
        return super().__call__(arr, pad_mode=pad_mode, verbose=verbose,
                                padx=padx, pady=pady, padz=padz, stagger_kind=stagger_kind)


class _stagger_derivate(_stagger_factory):
    '''take derivative of data, interpolating one half cell {up} in {x}.
    arr: 3D array
        the data to be interpolated.
    diff: 1D array
        array of distances between each cell along the {x} axis;
        length of array must equal the number of points in {x}.
    pad_mode: None or str.
        pad mode for the interpolation; same options as those supported by np.pad.
        if None, the default for this operation will be used: '{pad_default}'
    verbose: 0, 1, 2
        0 --> no verbosity
        1 --> print, end with r'\r'.
        2 --> print, end with r'\n'.
    padx, pady, padz: None or string
        pad_mode, but only applies for operation in the corresponding axis.
        (For convenience. E.g. if all pad_modes are known, can enter padx, pady, padz,
        without needing to worry about which type of operation is being performed.)
    diffx, diffy, diffz: None or array
        diff, but only applies for operation in the corresponding axis.
        (For convenience. E.g. if all diffs are known, can enter diffx, diffy, and diffz,
        without needing to worry about which type of operation is being performed.)

    TODO: fix ugly printout during verbose==1.
    '''
    def __init__(self, x, up):
        super().__init__(x, up, opstr_fmt='dd{x}{up}')

    def __call__(self, arr, diff=None, pad_mode=None, verbose=False,
                 padx=None, pady=None, padz=None,
                 diffx=None, diffy=None, diffz=None,
                 stagger_kind=DEFAULT_STAGGER_KIND, **kw__None):
        if diff is None:
            diff = {'x':diffx, 'y':diffy, 'z':diffz}[self.x]
        return super().__call__(arr, diff=diff, pad_mode=pad_mode, verbose=verbose,
                                padx=padx, pady=pady, padz=padz, stagger_kind=stagger_kind)

_STAGGER_ALIASES = {}
for x in ('x', 'y', 'z'):
    _pad_default = PAD_DEFAULTS[x]
    for up in ('up', 'dn'):
        # define _xup (or _xdn, _yup, _ydn, _zup, _zdn).
        _STAGGER_ALIASES[f'_{x}{up}']   = _stagger_spatial(x, up)
        # define _ddxup (or _ddxdn, _ddyup, _ddydn, _ddzup, _ddzdn).
        _STAGGER_ALIASES[f'_dd{x}{up}'] = _stagger_derivate(x, up)

## << HERE IS WHERE WE ACTUALLY PUT THE FUNCTIONS INTO THE MODULE NAMESPACE >> ##
for _opstr, _op in _STAGGER_ALIASES.items():
    locals()[_opstr] = _op

del x, _pad_default, up, _opstr, _op   # << remove "temporary variables" from module namespace

# << At this point, the following functions have all been defined in the module namespace:
#    _xdn, _xup, _ydn, _yup, _zdn, _zup, _ddxdn, _ddxup, _ddydn, _ddyup, _ddzdn, _ddzup
# Any of them may be referenced. E.g. import helita.sim.stagger; stagger._ddydn  # < this has been defined.


""" ------------------------ Chainable Interpolation Objects ------------------------ """
# Here is where we define:
#    xdn, xup, ydn, yup, zdn, zup, ddxdn, ddxup, ddydn, ddyup, ddzdn, ddzup
# They can be called as you would expect, e.g. xdn(arr),
# or chained together, e.g. xdn.ydn.zdn.ddzup(arr)  would do xdn(ydn(xdn(ddzup(arr)))).

def _trim_leading_underscore(name):
    return name[1:] if name[0]=='_' else name

class BaseChain():  # base class. Inherit from this class before creating the chain. See e.g. _make_chain().
    """
    object which behaves like an interpolation function (e.g. xup, ddydn),
    but can be chained to other interpolations, e.g. xup.ydn.zup(arr)

    This object in particular behaves like: {undetermined}.

    Helpful tricks:
        to pass diff to derivatives, use kwargs diffx, diffy, diffz.
        to apply in reverse order, use kwarg reverse=True.
            default order is A.B.C(val) --> A(B(C(val))).
    """
    ## ESSENTIAL BEHAVIORS ##
    def __init__(self, f_self, *funcs):
        self.funcs = [f_self, *funcs]
        # bookkeeping (non-essential, but makes help() more helpful and repr() prettier)
        self.__name__ = _trim_leading_underscore(f_self.__name__)
        self.__doc__  = self.__doc__.format(undetermined=self.__name__)
    
    def __call__(self, x, reverse=False, **kw):
        '''apply the operations. If reverse, go in reverse order.'''
        itfuncs = self.funcs[::-1] if reverse else self.funcs
        for func in itfuncs:
            x = func(x, **kw)
        return x

    ## CONVNIENT BEHAVIORS ##
    def op(self, opstr):
        '''get link opstr from self. (For using dynamically-named links)

        Equivalent to getattr(self, opstr).
        Example:
            self.op('xup').op('ddydn')   is equivalent to   self.xup.ddydn
        '''
        return getattr(self, opstr)

    def __getitem__(self, i):
        return self.funcs[i]

    def __iter__(self):
        return iter(self.funcs)

    def __repr__(self):
        funcnames = ' '.join([_trim_leading_underscore(f.__name__) for f in self])
        return f'{self.__class__.__name__} at <{hex(id(self))}> with operations: {funcnames}'

class ChainCreator():
    """for creating and manipulating a chain."""
    def __init__(self, name='Chain', base=BaseChain):
        self.Chain = type(name, (base,), {'__doc__': BaseChain.__doc__})
        self.links = []

    def _makeprop(self, link):
        Chain = self.Chain
        return property(lambda self: Chain(link, *self.funcs))

    def _makelink(self, func):
        return self.Chain(func)
        
    def link(self, prop, func):
        '''adds the (prop, func) link to chain.'''
        link = self._makelink(func)
        setattr(self.Chain, prop, self._makeprop(link))
        self.links.append(link)

def _make_chain(*prop_func_pairs, name='Chain', base=BaseChain,
                creator=ChainCreator, **kw__creator):
    """create new chain with (propertyname, func) pairs as indicated, named Chain.
    (propertyname, func): str, function
        name of attribute to associate with performing func when called.
        
    returns Chain, (list of instances of Chain associated with each func)
    """
    Chain = creator(name, base=base, **kw__creator)
    for prop, func in prop_func_pairs:
        Chain.link(prop, func)
    
    return tuple((Chain.Chain, Chain.links))

props, funcs = [], []
for dd in ('', 'dd'):
    for x in ('x', 'y', 'z'):
        for up in ('up', 'dn'):
            opstr = f'{dd}{x}{up}'
            props.append(opstr)                  # e.g. 'xup'
            funcs.append(locals()[f'_{opstr}'])  # e.g. _xup

## << HERE IS WHERE WE ACTUALLY PUT THE FUNCTIONS INTO THE MODULE NAMESPACE >> ##
InterpolationChain, links = _make_chain(*zip(props, funcs), name='InterpolationChain')
for prop, link in zip(props, links):
    locals()[prop] = link    # set function in the module namespace (e.g. xup, ddzdn)

del props, funcs, dd, x, up, opstr, links, prop, link   # << remove "temporary variables" from module namespace


# << At this point, the following functions have all been defined in the module namespace:
#    xdn, xup, ydn, yup, zdn, zup, ddxdn, ddxup, ddydn, ddyup, ddzdn, ddzup


""" ------------------------ StaggerData (wrap methods in a class) ------------------------ """

class StaggerInterface():
    """
    Interface to stagger methods, with defaults implied by an object.
    Examples:
        self.stagger = StaggerInterface(self)

        # interpolate arr by ddxdn:
        self.stagger.ddxdn(arr)
        # Note, this uses the defaults:
            # pad_mode     = '{PAD_PERIODIC}' if self.get_param('periodic_x') else '{PAD_NONPERIODIC}'
            # diff         = self.dxidxdn
            # stagger_kind = self.stagger_kind

        # interpolate arr via xup( ydn(ddzdn(arr)) ), using defaults as appropriate:
        self.stagger.xup.ydn.ddzdn(arr)  

    Available operations:
          xdn,   xup,   ydn,   yup,   zdn,   zup,
        ddxdn, ddxup, ddydn, ddyup, ddzdn, ddzup
    Available convenience method:
        do(opstr, arr, ...)  # << does the operation implied by opstr; equivalent to getattr(self, opstr)(arr, ...)
        Example:
            self.stagger.do('zup', arr)  # equivalent to self.stagger.zup(arr)

    Each method will call the appropriate method from stagger.py.
    Additionally, for convenience:
        named operations can be chained together.
            For example:
                self.stagger.xup.ydn.ddzdn(arr)
            This does not apply when using the 'do' function.
            For dynamically-named chaining, see self.op.
        default values are supplied for the extra paramaters:
            pad_mode:
                periodic = self.get_param('periodic_x') (or y, z)
                periodic True --> pad_mode = stagger.PAD_PERIODIC (=='{PAD_PERIODIC}')
                periodic False -> pad_mode = stagger.PAD_NONPERIODIC (=='{PAD_NONPERIODIC}')
            diff:
                self.dxidxup   with x --> x, y, or z; up --> up or dn.
            stagger_kind:
                self.stagger_kind
        if the operation is called on a string instead of an array,
            first pass the string to a call of self.
            E.g. self.xup('r') will do stagger.xup(self('r'))
    """
    _PAD_PERIODIC = PAD_PERIODIC
    _PAD_NONPERIODIC = PAD_NONPERIODIC

    def __init__(self, obj):
        self._obj_ref = weakref.ref(obj)  # weakref to avoid circular reference.
        prop_func_pairs = [(_trim_leading_underscore(prop), func) for prop, func in _STAGGER_ALIASES.items()]
        self._make_bound_chain(*prop_func_pairs, name='BoundInterpolationChain')

    obj = property(lambda self: self._obj_ref())

    def do(self, arr, opstr, *args, **kw):
        '''does the operation implied by opstr (e.g. 'xup', ..., 'ddzdn').
        Equivalent to getattr(self, opstr)(arr, *args, **kw)
        '''
        return getattr(self, opstr)(arr, *args, **kw)

    def op(self, opstr):
        '''gets the operation which opstr would apply.
        For convenience. Equivalent to getattr(self, opstr).

        Can be chained. For example:
        self.op('xup').op('ddydn')   is equivalent to   self.xup.ddydn.
        '''
        return getattr(self, opstr)

    def _make_bound_chain(self, *prop_func_pairs, name='BoundChain'):
        """create new bound chain, linking all props to same-named attributes of self."""
        Chain, links = _make_chain(*prop_func_pairs, name=name,
                           base=BoundBaseChain, creator=BoundChainCreator, obj=self)
        props, funcs = zip(*prop_func_pairs)
        for prop, link in zip(props, links):
            setattr(self, prop, link)

    ## __INTERPOLATION_CALL__ ##
    # this function will be called whenever an interpolation method is used.
    # To edit the behavior of calling an interpolation method, edit this function.
    # E.g. here is where to connect properties of obj to defaults for interpolation.
    def _pad_modes(self):
        '''return dict of padx, pady, padz, with values the appropriate strings for padding.'''
        def _booly_to_mode(booly):
            return {None: None, True: self._PAD_PERIODIC, False: self._PAD_NONPERIODIC}[booly]
        return {f'pad{x}': _booly_to_mode(self.obj.get_param(f'periodic_{x}')) for x in ('x', 'y', 'z')}

    def _diffs(self):
        '''return dict of diffx, diffy, diffz, with values the appropriate arrays.
        CAUTION: assumes dxidxup == dxidxdn == diffx, and similar for y and z.
        '''
        return {f'diff{x}': getattr(self.obj, f'd{x}id{x}up') for x in ('x', 'y', 'z')}

    def _stagger_kind(self):
        return {'stagger_kind': self.obj.stagger_kind}

    def __interpolation_call__(self, func, arr, *args__get_var, **kw):
        '''call interpolation function func on array arr with the provided kw.

        use defaults implied by self (e.g. padx implied by periodic_x), for any kw not entered.
        if arr is a string, first call self(arr, *args__get_var, **kw).
        '''
        __tracebackhide__ = True
        kw_to_use = {**self._pad_modes(), **self._diffs(), **self._stagger_kind()}  # defaults based on obj.
        kw_to_use.update(kw)   # exisitng kwargs override defaults.
        if isinstance(arr, str):
            arr = self.obj(arr, *args__get_var, **kw)
        return func(arr, **kw_to_use)

StaggerInterface.__doc__ = StaggerInterface.__doc__.format(PAD_PERIODIC=PAD_PERIODIC, PAD_NONPERIODIC=PAD_NONPERIODIC)


class BoundBaseChain(BaseChain):
    """BaseChain structure but bound to a class."""
    def __init__(self, obj, f_self, *funcs):
        self.obj = obj
        super().__init__(f_self, *funcs)

    def __call__(self, x, reverse=False, **kw):
        '''apply the operations. If reverse, go in reverse order'''
        itfuncs = self.funcs[::-1] if reverse else self.funcs
        for func in itfuncs:
            x = self.obj.__interpolation_call__(func, x, **kw)
        return x

    def __repr__(self):
        funcnames = ' '.join([_trim_leading_underscore(f.__name__) for f in self])
        return f'<{self.__class__.__name__} at <{hex(id(self))}> with operations: {funcnames}> bound to {self.obj}'

class BoundChainCreator(ChainCreator):
    """for creating and manipulating a bound chain"""
    def __init__(self, *args, obj=None, **kw):
        if obj is None: raise TypeError('obj must be provided')
        self.obj = obj
        super().__init__(*args, **kw)

    def _makeprop(self, link):
        Chain = self.Chain
        return property(lambda self: Chain(self.obj, link, *self.funcs))

    def _makelink(self, func):
        return self.Chain(self.obj, func)