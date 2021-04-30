import numpy as np
from numba import jit, njit, prange


def do(var, operation='xup', pad_mode='wrap'):
    """
    Do a stagger operation on `var`. These are generally divided into
    'up' or 'down' operations that interpolate values from cell faces
    to cell centres (up), or from cell centres to cell faces (down).

    Supported operations are currently:
    * 'xup', 'xdn', 'yup', 'ydn', 'zup', 'zdn'.
    """
    op_func = {
        'x': _xshift,
        'y': _yshift,
        'z': _zshift,
    }
    if operation[-2:].lower() == 'up':
        up = True
    elif operation[-2:].lower() == 'dn':
        up = False
    else: 
        raise ValueError(f"Invalid operation {operation}")
    op = operation[:-2]
    if op not in op_func:
        raise ValueError(f"Invalid operation {operation}")
    func = op_func[op]
    dim_index = 'xyz'.find(op[-1])
    extra_dims = [(3, 2), (2, 3)][up]
    padding = [(0, 0)] * 3
    padding[dim_index] = extra_dims
    s = var.shape
    if s[dim_index] == 1:
        return var
    else:
        out = np.pad(var, padding, mode=pad_mode)
        return func(out, up=up)


@njit(parallel=True)
def _xshift(var, up=True):
    c = 3.0 / 256.0
    b = -25.0 / 256.0
    a = 0.5 - b - c
    if up:
        sign = 1
    else:
        sign = -1
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(nz): 
        for j in prange(ny):
            for i in prange(start, nx + end):
                var[i, j, k] = (a * (var[i, j, k] + var[i + sign, j, k]) +
                                b * (var[i - sign*1, j, k] + var[i + sign*2, j, k]) +
                                c * (var[i - sign*2, j, k] + var[i + sign*3, j, k]))
    return var[start:end]


@njit(parallel=True)
def _yshift(var, up=True):
    c = 3.0 / 256.0
    b = -25.0 / 256.0
    a = 0.5 - b - c
    if up:
        sign = 1
    else:
        sign = -1
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(nz): 
        for j in prange(start, ny + end):
            for i in prange(nx):
                var[i, j, k] = (a * (var[i, j, k] + var[i, j + sign, k]) +
                                b * (var[i, j - sign*1, k] + var[i, j + sign*2, k]) +
                                c * (var[i, j - sign*2, k] + var[i, j + sign*3, k]))
    return var[:, start:end]


@njit(parallel=True)
def _zshift(var, up=True):
    c = 3.0 / 256.0
    b = -25.0 / 256.0
    a = 0.5 - b - c
    if up:
        sign = 1
    else:
        sign = -1
    start = int(2.5 - sign*0.5)  
    end = - int(2.5 + sign*0.5)
    nx, ny, nz = var.shape
    for k in prange(start, nz + end): 
        for j in prange(ny):
            for i in prange(nx):
                var[i, j, k] = (a * (var[i, j, k] + var[i, j, k + sign]) +
                                b * (var[i, j, k - sign*1] + var[i, j, k + sign*2]) +
                                c * (var[i, j, k - sign*2] + var[i, j, k + sign*3]))
    return var[..., start:end]