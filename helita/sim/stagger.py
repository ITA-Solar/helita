import numpy as np
from numba import jit, njit, prange


def do(var, operation='xup', diff=None, pad_mode=None):
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
    pad_mode : str
        Mode for padding array `var` to have enough points for a 6th order
        polynomial interpolation. Same as supported by np.pad. Default is
        `wrap` (periodic horizontal) for x and y, and `reflect` for z operations.

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
    DEFAULT_PAD = {'x': 'wrap', 'y': 'wrap', 'z': 'reflect'}
    if operation[-2:].lower() == 'up':
        up = True
    elif operation[-2:].lower() == 'dn':
        up = False
    else: 
        raise ValueError(f"Invalid operation {operation}")
    if operation[:2].lower() == 'dd':  # For derivative operations
        derivative = True
        operation = operation[2:]
        if diff is None:
            raise ValueError("diff not provided for derivative operation")
    else:
        derivative = False
        if diff is not None:
            raise ValueError("diff must not be provided for non-derivative operation")
    op = operation[:-2]
    if op not in AXES:
        raise ValueError(f"Invalid operation {operation}")
    func = AXES[op]
    if pad_mode is None:
        pad_mode = DEFAULT_PAD[op]
    dim_index = 'xyz'.find(op[-1])
    extra_dims = [(3, 2), (2, 3)][up]
    if not derivative:
        diff = np.ones(var.shape[dim_index], dtype=var.dtype)
    padding = [(0, 0)] * 3
    padding[dim_index] = extra_dims
    s = var.shape
    if s[dim_index] == 1:
        return var
    else:
        out = np.pad(var, padding, mode=pad_mode)
        out_diff = np.pad(diff, extra_dims, mode=pad_mode)
        return func(out, out_diff, up=up, derivative=derivative)


@njit(parallel=True)
def _xshift(var, diff, up=True, derivative=False):
    if up:
        sign = 1
    else:
        sign = -1
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
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(nz): 
        for j in prange(ny):
            for i in prange(start, nx + end):
                var[i, j, k] = diff[i] * (a * (var[i, j, k] + pm * var[i + sign, j, k]) +
                                b * (var[i - sign*1, j, k] + pm * var[i + sign*2, j, k]) +
                                c * (var[i - sign*2, j, k] + pm * var[i + sign*3, j, k]))
    return var[start:end]


@njit(parallel=True)
def _yshift(var, diff, up=True, derivative=False):
    if up:
        sign = 1
    else:
        sign = -1
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
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(nz): 
        for j in prange(start, ny + end):
            for i in prange(nx):
                var[i, j, k] = diff[j] * (a * (var[i, j, k] + pm * var[i, j + sign, k]) +
                                b * (var[i, j - sign*1, k] + pm * var[i, j + sign*2, k]) +
                                c * (var[i, j - sign*2, k] + pm * var[i, j + sign*3, k]))
    return var[:, start:end]


@njit(parallel=True)
def _zshift(var, diff, up=True, derivative=False):
    if up:
        sign = 1
    else:
        sign = -1
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
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(start, nz + end): 
        for j in prange(ny):
            for i in prange(nx):
                var[i, j, k] = diff[k] * (a * (var[i, j, k] + pm * var[i, j, k + sign]) +
                                b * (var[i, j, k - sign*1] + pm * var[i, j, k + sign*2]) +
                                c * (var[i, j, k - sign*2] + pm * var[i, j, k + sign*3]))
    return var[..., start:end]