"""
Stagger mesh methods using numba.

set stagger_kind = 'stagger' to use these methods. (not 'cstagger')
stagger_kind = 'stagger' is the default for BifrostData and EbysusData.


Methods defined here (which an end-user might want to access):
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
"""

# import built-in modules
import time

# import public external modules
import numpy as np
from numba import jit, njit, prange


""" ------------------------ defaults ------------------------ """

PAD_PERIODIC    = 'wrap'     # how to pad periodic dimensions, by default
PAD_NONPERIODIC = 'reflect'  # how to pad nonperiodic dimensions, by default
PAD_DEFAULTS = {'x': PAD_PERIODIC, 'y': PAD_PERIODIC, 'z': PAD_NONPERIODIC}   # default padding for each dimension.


""" ------------------------ 'do' - stagger interface ------------------------ """

def do(var, operation='xup', diff=None, pad_mode=None, stagger_kind='numba'):
    """
    Do a stagger operation on `var` by doing a 6th order polynomial interpolation of 
    the variable from cell centres to cell faces (down operations), or cell faces
    to cell centres (up operations).
    
    Parameters
    ----------
    var : 3D array
        Variable to work on.
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
    stagger_kind: 'numba' or 'numpy'
        Mode for stagger operations.
        numba --> numba methods ('_xshift', '_yshift', '_zshift')
        numpy --> numpy methods ('_np_stagger')

    Returns
    -------
    3D array
        Array of same type and dimensions to var, after performing the 
        stagger operation.
    """
    AXES = {
        'x': _xshift,
        'y': _yshift,
        'z': _zshift,
    }
 
    operation = operation.lower()
    # up/dn
    if operation[-2:] == 'up':
        up = True
    elif operation[-2:] == 'dn':
        up = False
    else: 
        raise ValueError(f"Invalid operation; must end in 'up' or 'dn': {operation}")
    # derivative, diff
    if operation[:2] == 'dd':  # For derivative operations
        derivative = True
        operation = operation[2:]
        if diff is None:
            raise ValueError(f"diff not provided for derivative operation: {operation}")
    else:
        derivative = False
        if diff is not None:
            raise ValueError(f"diff must not be provided for non-derivative operation: {operation}")
    # op (aka 'axis'), dim_index (0 for x, 1 for y, 2 for z)
    op = operation[:-2]
    if op not in AXES:
        raise ValueError(f"Invalid operation; axis must be 'x', 'y', or 'z': {operation}")
    if pad_mode is None:
        pad_mode = PAD_DEFAULTS[op]
    dim_index = 'xyz'.find(op[-1])
    # padding
    extra_dims = (2, 3) if up else (3, 2)
    if not derivative:
        diff = np.ones(var.shape[dim_index], dtype=var.dtype)
    padding = [(0, 0)] * 3
    padding[dim_index] = extra_dims
    # interpolating
    if var.shape[dim_index] == 1:   # don't interpolate along axis with size 1...
        return var
    else:
        out = np.pad(var, padding, mode=pad_mode)
        out_diff = np.pad(diff, extra_dims, mode=pad_mode)
        if stagger_kind=='numba':
            func = AXES[op]
            return func(out, out_diff, up=up, derivative=derivative)
        elif stagger_kind=='numpy':
            return _np_stagger(out, out_diff, up, derivative, dim_index)
        else:
            raise ValueError(f"invalid {stagger_kind = }; expected 'numba' or 'numpy'")


""" ------------------------ numba stagger ------------------------ """

@njit(parallel=True)
def _xshift(var, diff, up=True, derivative=False):
    if up:
        grdshf = 1
    else:
        grdshf = 0
    if derivative:
        pm = -1
        c = (-1 + (3**5 - 3) / (3**3 - 3)) / (5**5 - 5 - 5 * (3**5 - 3))
        b = (-1 - 120*c) / 24
        a = (1 - 3*b - 5*c)
    else:
        pm = 1
        c = 3.0 / 256.0
        b = -25.0 / 256.0
        a = 0.5 - b - c
    start = int(3. - grdshf)  
    end = - int(2. + grdshf)
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(nz): 
        for j in prange(ny):
            for i in prange(start, nx + end):
                out[i, j, k] = diff[i] * (a * (var[i+ grdshf, j, k] + pm * var[i - 1 + grdshf, j, k]) +
                                b * (var[i + 1 + grdshf, j, k] + pm * var[i - 2 + grdshf, j, k]) +
                                c * (var[i + 2 + grdshf, j, k] + pm * var[i - 3 + grdshf, j, k]))

    return out[start:end]


@njit(parallel=True)
def _yshift(var, diff, up=True, derivative=False):
    if up:
        grdshf = 1
    else:
        grdshf = 0
    if derivative:
        pm = -1
        c = (-1 + (3**5 - 3) / (3**3 - 3)) / (5**5 - 5 - 5 * (3**5 - 3))
        b = (-1 - 120*c) / 24
        a = (1 - 3*b - 5*c)
    else:
        pm = 1
        c = 3.0 / 256.0
        b = -25.0 / 256.0
        a = 0.5 - b - c
    start = int(3. - grdshf)  
    end = - int(2. + grdshf)
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(nz): 
        for j in prange(start, ny + end):
            for i in prange(nx):
                out[i, j, k] = diff[j] * (a * (var[i, j + grdshf, k] + pm * var[i, j - 1 + grdshf, k]) +
                                b * (var[i, j + 1 + grdshf, k] + pm * var[i, j - 2 + grdshf, k]) +
                                c * (var[i, j + 2 + grdshf, k] + pm * var[i, j - 3 + grdshf, k]))
    return out[:, start:end]


@njit(parallel=True)
def _zshift(var, diff, up=True, derivative=False):
    if up:
        grdshf = 1
    else:
        grdshf = 0
    if derivative:
        pm = -1
        c = (-1 + (3**5 - 3) / (3**3 - 3)) / (5**5 - 5 - 5 * (3**5 - 3))
        b = (-1 - 120*c) / 24
        a = (1 - 3*b - 5*c)
    else:
        pm = 1
        c = 3.0 / 256.0
        b = -25.0 / 256.0
        a = 0.5 - b - c
    start = int(3. - grdshf)  
    end = - int(2. + grdshf)
    nx, ny, nz = var.shape
    out=np.zeros((nx,ny,nz))
    for k in prange(start, nz + end): 
        for j in prange(ny):
            for i in prange(nx):
                out[i, j, k] = diff[k] * (a * (var[i, j, k + grdshf] + pm * var[i, j, k - 1 + grdshf]) +
                                b * (var[i, j, k + 1 + grdshf] + pm * var[i, j, k - 2 + grdshf]) +
                                c * (var[i, j, k + 2 + grdshf] + pm * var[i, j, k - 3 + grdshf]))
    return out[..., start:end]


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

def _np_stagger(var, diff, up, derivative, x):
    """stagger along x axis. x should be 0, 1, or 2."""
    # -- same constants and setup as numba method -- #
    if up:
        grdshf = 1
    else:
        grdshf = 0
    if derivative:
        pm = -1
        c = (-1 + (3**5 - 3) / (3**3 - 3)) / (5**5 - 5 - 5 * (3**5 - 3))
        b = (-1 - 120*c) / 24
        a = (1 - 3*b - 5*c)
    else:
        pm = 1
        c = 3.0 / 256.0
        b = -25.0 / 256.0
        a = 0.5 - b - c
    start = int(3. - grdshf)  
    end = - int(2. + grdshf)
    nx = var.shape[x]
    out=np.zeros(var.shape)
    # -- begin numpy syntax -- #
    def slx(shift):
        '''return slicer at x axis from (start + shift) to (nx + end + shift).'''
        return slicer_at_ax((start+shift, nx+end+shift), x)
    def sgx(shift):
        '''return slicer at x axis from (start + shift + grdshf) to (nx + end + shift + grdshf)'''
        return slx(shift + grdshf)
    diff = np.expand_dims(diff,  axis=tuple( set((0,1,2)) - set((x,)) )  )   # make diff 3D (with size 1 for axes other than x)

    out = diff[slx(0)] * (a * (var[sgx(0)] + pm * var[sgx(-1)]) + 
                          b * (var[sgx(1)] + pm * var[sgx(-2)]) +
                          c * (var[sgx(2)] + pm * var[sgx(-3)]))

    return out


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
                 padx=None, pady=None, padz=None, stagger_kind='numba', **kw__None):
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
                 stagger_kind='numba', **kw__None):
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

class StaggerData():
    """
    Class with stagger methods.

    Available operations:
          xdn,   xup,   ydn,   yup,   zdn,   zup,
        ddxdn, ddxup, ddydn, ddyup, ddzdn, ddzup

    Each method will call the appropriate method from stagger.py.
    Additionally, for convenience:
        default values are supplied for the extra paramaters:
            pad_mode:
                periodic = self.get_param('periodic_x') (or y, z)
                periodic True --> pad_mode = stagger.PAD_PERIODIC (=='{PAD_PERIODIC}')
                periodic False -> pad_mode = stagger.PAD_NONPERIODIC (=='{PAD_NONPERIODIC}')
            diff:
                self.d{x}id{x}{up}   with {x} --> x, y, or z; {up} --> up or dn.
        if the operation is called on a string instead of an array,
            first pass the string to a call of self.
            E.g. self.xup('r') will do stagger.xup(self('r'))
    """
    _PAD_PERIODIC = PAD_PERIODIC
    _PAD_NONPERIODIC = PAD_NONPERIODIC

    def __init__(self, *args__None, **kw__None):
        _make_bound_chain(self, *_STAGGER_ALIASES.items(), name='BoundInterpolationChain')

    def _pad_modes(self):
        '''return dict of padx, pady, padz, with values the appropriate strings for padding.'''
        def _booly_to_mode(booly):
            return {None: None, True: self._PAD_PERIODIC, False: self._PAD_NONPERIODIC}[booly]
        return {f'pad{x}': _booly_to_mode(self.get_param(f'periodic_{x}')) for x in ('x', 'y', 'z')}

    def _diffs(self):
        '''return dict of diffx, diffy, diffz, with values the appropriate arrays.
        CAUTION: assumes dxidxup == dxidxdn == diffx, and similar for y and z.
        '''
        return {f'diff{x}': getattr(self, f'd{x}id{x}up') for x in ('x', 'y', 'z')}

    def __interpolation_call__(self, func, arr, *args__get_var, **kw):
        '''call interpolation function func on array arr with the provided kw.

        use defaults implied by self (e.g. padx implied by periodic_x), for any kw not entered.
        if arr is a string, first call self(arr, *args__get_var, **kw).
        '''
        __tracebackhide__ = True
        kw_to_use = {**self._pad_modes(), **self._diffs()}  # defaults based on obj.
        kw_to_use.update(kw)   # exisitng kwargs override defaults.
        if isinstance(arr, str):
            arr = self(arr, *args__get_var, **kw)
        return func(arr, **kw_to_use)


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

def _make_bound_chain(obj, *prop_func_pairs, name='BoundChain'):
    """create new bound chain, linking all props to same-named attributes of obj."""
    Chain, links = _make_chain(*prop_func_pairs, name=name,
                       base=BoundBaseChain, creator=BoundChainCreator, obj=obj)
    props, funcs = zip(*prop_func_pairs)
    for prop, link in zip(props, links):
        setattr(obj, prop, link)

StaggerData.__doc__ = StaggerData.__doc__.format(PAD_PERIODIC=PAD_PERIODIC, x='x', up='up', PAD_NONPERIODIC=PAD_NONPERIODIC)