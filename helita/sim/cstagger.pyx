cimport cython
cimport numpy as np
import numpy as np

ctypedef fused FLOAT_t:
    np.float32_t
    np.float64_t

nz = 0   # initialise nz
dxc = 0  # initialise dxc
dyc = 0  # initialise dyc

###
###  C functions
###
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef xup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    xup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell faces to cell centre, x direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for j in range(my):
            l = j*mx + k*mx*my
            o[l + (m-3)] = (
              a*(f[l + (m-2)]+f[l + (m-3)]) +
              b*(f[l + (m-1)]+f[l + (m-4)]) +
              c*(f[l + (0)]+f[l + (m-5)]))
            o[l + (m-2)] = (
              a*(f[l + (m-1)]+f[l + (m-2)]) +
              b*(f[l + (0)]+f[l + (m-3)]) +
              c*(f[l + (1)]+f[l + (m-4)]))
            o[l + (m-1)] = (
              a*(f[l + (0)]+f[l + (m-1)]) +
              b*(f[l + (1)]+f[l + (m-2)]) +
              c*(f[l + (2)]+f[l + (m-3)]))
            o[l + (0)] = (
              a*(f[l + (1)]+f[l + (0)]) +
              b*(f[l + (2)]+f[l + (m-1)]) +
              c*(f[l + (3)]+f[l + (m-2)]))
            o[l + (1)] = (
              a*(f[l + (2)]+f[l + (1)]) +
              b*(f[l + (3)]+f[l + (0)]) +
              c*(f[l + (4)]+f[l + (m-1)]))
        for j in range(my):
            l = j*mx + k*mx*my
            for i in range(2,mx-3):
                o[l + (i)] = (
                    a*(f[l + (i+1)] + f[l + i]) +
                    b*(f[l + (i+2)] + f[l + (i-1)]) +
                    c*(f[l + (i+3)] + f[l + (i-2)]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef yup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    yup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell faces to cell centre, y direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for i in range(mx):
            l = i + k*mx*my
            o[l + (m-3)*mx] = (
                a*(f[l + (m-2)*mx]+f[l + (m-3)*mx]) +
                b*(f[l + (m-1)*mx]+f[l + (m-4)*mx]) +
                c*(f[l + (0)*mx]+f[l + (m-5)*mx]))
            o[l + (m-2)*mx] = (
                a*(f[l + (m-1)*mx]+f[l + (m-2)*mx]) +
                b*(f[l + (0)*mx]+f[l + (m-3)*mx]) +
                c*(f[l + (1)*mx]+f[l + (m-4)*mx]))
            o[l + (m-1)*mx] = (
                a*(f[l + (0)*mx]+f[l + (m-1)*mx]) +
                b*(f[l + (1)*mx]+f[l + (m-2)*mx]) +
                c*(f[l + (2)*mx]+f[l + (m-3)*mx]))
            o[l + (0)*mx] = (
                a*(f[l + (1)*mx]+f[l + (0)*mx]) +
                b*(f[l + (2)*mx]+f[l + (m-1)*mx]) +
                c*(f[l + (3)*mx]+f[l + (m-2)*mx]))
            o[l + (1)*mx] = (
                a*(f[l + (2)*mx]+f[l + (1)*mx]) +
                b*(f[l + (3)*mx]+f[l + (0)*mx]) +
                c*(f[l + (4)*mx]+f[l + (m-1)*mx]))
        for j in range(2, my - 3):
            for i in range(mx):
                l = i + k*mx*my
                o[l + (j)*mx] = (
                    a*(f[l + (j+1)*mx] + f[l + j*mx]) +
                    b*(f[l + (j+2)*mx] + f[l + (j-1)*mx]) +
                    c*(f[l + (j+3)*mx] + f[l + (j-2)*mx]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef zup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    zup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell faces to cell centre, z direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef long i, j, k, l, m=mx
    cdef FLOAT_t d
    cdef np.ndarray[FLOAT_t, ndim=2] zz = zupc
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    tmp = <FLOAT_t *> zz.data
    # Pure C part
    for k in range(mz):
        m = k - 2
        if (k < 3):
            m = 0
        if (k > mz - 4):
            m = mz - 6
        for j in range(my):
            for i in range(mx):
                d = 0
                for l in range(6):
                    d += tmp[k*6 + l] * f[((m + l)*my + j)*mx + i]
                o[(k * my + j) * mx + i] = d
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef xdn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    xdn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell centre to cell faces, x direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for j in range(my):
            l = j*mx + k*mx*my
            o[l + (m-2)] = (
              a*(f[l + (m-2)]+f[l + (m-3)]) +
              b*(f[l + (m-1)]+f[l + (m-4)]) +
              c*(f[l + (0)]+f[l + (m-5)]))
            o[l + (m-1)] = (
              a*(f[l + (m-1)]+f[l + (m-2)]) +
              b*(f[l + (0)]+f[l + (m-3)]) +
              c*(f[l + (1)]+f[l + (m-4)]))
            o[l + (0)] = (
              a*(f[l + (0)]+f[l + (m-1)]) +
              b*(f[l + (1)]+f[l + (m-2)]) +
              c*(f[l + (2)]+f[l + (m-3)]))
            o[l + (1)] = (
              a*(f[l + (1)]+f[l + (0)]) +
              b*(f[l + (2)]+f[l + (m-1)]) +
              c*(f[l + (3)]+f[l + (m-2)]))
            o[l + (2)] = (
              a*(f[l + (2)]+f[l + (1)]) +
              b*(f[l + (3)]+f[l + (0)]) +
              c*(f[l + (4)]+f[l + (m-1)]))
        for j in range(my):
            l = j*mx + k*mx*my
            for i in range(2, mx - 3):
                o[l + (i+1)] = (
                  a*(f[l + (i+1)] + f[l + i]) +
                  b*(f[l + (i+2)] + f[l + (i-1)]) +
                  c*(f[l + (i+3)] + f[l + (i-2)]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ydn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ydn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell centre to cell faces, y direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for i in range(mx):
            l = i + k*mx*my
            o[l + (m-2)*mx] = (
                a*(f[l + (m-2)*mx]+f[l + (m-3)*mx]) +
                b*(f[l + (m-1)*mx]+f[l + (m-4)*mx]) +
                c*(f[l + (0)*mx]+f[l + (m-5)*mx]))
            o[l + (m-1)*mx] = (
                a*(f[l + (m-1)*mx]+f[l + (m-2)*mx]) +
                b*(f[l + (0)*mx]+f[l + (m-3)*mx]) +
                c*(f[l + (1)*mx]+f[l + (m-4)*mx]))
            o[l + (0)*mx] = (
                a*(f[l + (0)*mx]+f[l + (m-1)*mx]) +
                b*(f[l + (1)*mx]+f[l + (m-2)*mx]) +
                c*(f[l + (2)*mx]+f[l + (m-3)*mx]))
            o[l + (1)*mx] = (
                a*(f[l + (1)*mx]+f[l + (0)*mx]) +
                b*(f[l + (2)*mx]+f[l + (m-1)*mx]) +
                c*(f[l + (3)*mx]+f[l + (m-2)*mx]))
            o[l + (2)*mx] = (
                a*(f[l + (2)*mx]+f[l + (1)*mx]) +
                b*(f[l + (3)*mx]+f[l + (0)*mx]) +
                c*(f[l + (4)*mx]+f[l + (m-1)*mx]))
        for j in range(2, my - 3):
            for i in range(mx):
                l = i + k*mx*my
                o[l + (j+1)*mx] = (
                  a*(f[l + (j+1)*mx] + f[l + j*mx]) +
                  b*(f[l + (j+2)*mx] + f[l + (j-1)*mx]) +
                  c*(f[l + (j+3)*mx] + f[l + (j-2)*mx]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef zdn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    zdn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings quantity from cell centre to cell faces, z direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef long i, j, k, l, m=mx
    cdef FLOAT_t d
    cdef np.ndarray[FLOAT_t, ndim=2] zz = zdnc
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    tmp = <FLOAT_t *> zz.data
    # Pure C part
    for k in range(mz):
        m = k-3
        if (k < 3):
            m = 0
        if (k > mz-4):
            m = mz-6
        for j in range(my):
            for i in range(mx):
                d = 0
                for l in range(6):
                    d += tmp[k*6 + l] * f[((m + l)*my + j)*mx + i];
                o[(k * my + j) * mx + i] = d
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddxup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddxup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell faces to cell centre, x direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t a = 300 / 281. / dxc, b = -50 / 843. / dxc, c = 6 / 1405. / dxc
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
      for j in range(my):
        l = j*mx + k*mx*my
        o[l + (m-3)] = (
          a*(f[l + (m-2)]-f[l + (m-3)]) +
          b*(f[l + (m-1)]-f[l + (m-4)]) +
          c*(f[l + (0)]-f[l + (m-5)]))
        o[l + (m-2)] = (
          a*(f[l + (m-1)]-f[l + (m-2)]) +
          b*(f[l + (0)]-f[l + (m-3)]) +
          c*(f[l + (1)]-f[l + (m-4)]))
        o[l + (m-1)] = (
          a*(f[l + (0)]-f[l + (m-1)]) +
          b*(f[l + (1)]-f[l + (m-2)]) +
          c*(f[l + (2)]-f[l + (m-3)]))
        o[l + (0)] = (
          a*(f[l + (1)]-f[l + (0)]) +
          b*(f[l + (2)]-f[l + (m-1)]) +
          c*(f[l + (3)]-f[l + (m-2)]));
        o[l + (1)] = (
          a*(f[l + (2)]-f[l + (1)]) +
          b*(f[l + (3)]-f[l + (0)]) +
          c*(f[l + (4)]-f[l + (m-1)]));
      for j in range(my):
        l = j*mx + k*mx*my;
        for i in range(2,mx-3):
          o[l + (i)] = (
            a*(f[l + (i+1)] - f[l + i]) +
            b*(f[l + (i+2)] - f[l + (i-1)]) +
            c*(f[l + (i+3)] - f[l + (i-2)]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddyup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddyup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell faces to cell centre, y direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t a = 300 / 281. / dxc, b = -50 / 843. / dxc, c = 6 / 1405. / dxc
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
      for i in range(mx):
        l = i + k*mx*my
        o[l + (m-3)*mx] = (
          a*(f[l + (m-2)*mx]-f[l + (m-3)*mx]) +
          b*(f[l + (m-1)*mx]-f[l + (m-4)*mx]) +
          c*(f[l + (0)*mx]-f[l + (m-5)*mx]))
        o[l + (m-2)*mx] = (
          a*(f[l + (m-1)*mx]-f[l + (m-2)*mx]) +
          b*(f[l + (0)*mx]-f[l + (m-3)*mx]) +
          c*(f[l + (1)*mx]-f[l + (m-4)*mx]))
        o[l + (m-1)*mx] = (
          a*(f[l + (0)*mx]-f[l + (m-1)*mx]) +
          b*(f[l + (1)*mx]-f[l + (m-2)*mx]) +
          c*(f[l + (2)*mx]-f[l + (m-3)*mx]))
        o[l + (0)*mx] = (
          a*(f[l + (1)*mx]-f[l + (0)*mx]) +
          b*(f[l + (2)*mx]-f[l + (m-1)*mx]) +
          c*(f[l + (3)*mx]-f[l + (m-2)*mx]))
        o[l + (1)*mx] = (
          a*(f[l + (2)*mx]-f[l + (1)*mx]) +
          b*(f[l + (3)*mx]-f[l + (0)*mx]) +
          c*(f[l + (4)*mx]-f[l + (m-1)*mx]))
      for j in range(2,my-3):
        for i in range(mx):
          l = i + k*mx*my
          o[l + (j)*mx] = (
            a*(f[l + (j+1)*mx] - f[l + j*mx]) +
            b*(f[l + (j+2)*mx] - f[l + (j-1)*mx]) +
            c*(f[l + (j+3)*mx] - f[l + (j-2)*mx]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddzup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddzup(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell faces to cell centre, z direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef long i, j, k, l, m=mx
    cdef FLOAT_t d
    cdef np.ndarray[FLOAT_t, ndim=2] zz = dzupc
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    tmp = <FLOAT_t *> zz.data
    # Pure C part
    for k in range(mz):
        m = k-2
        if (k < 3):
            m = 0
        if (k > mz - 4):
            m = mz - 6
        for j in range(my):
            for i in range(mx):
                d = 0
                for l in range(6):
                    d += tmp[k*6+l]*f[((m+l)*my + j)*mx + i]
                o[(k * my + j) * mx + i] = d
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddxdn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddxdn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell centre to cell faces, x direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t a = 300 / 281. / dxc, b = -50 / 843. / dxc, c = 6 / 1405. / dxc
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for j in range(my):
            l = j*mx + k*mx*my
            o[l + (m-2)] = (
                a*(f[l + (m-2)]-f[l + (m-3)]) +
                b*(f[l + (m-1)]-f[l + (m-4)]) +
                c*(f[l + (0)]-f[l + (m-5)]))
            o[l + (m-1)] = (
                a*(f[l + (m-1)]-f[l + (m-2)]) +
                b*(f[l + (0)]-f[l + (m-3)]) +
                c*(f[l + (1)]-f[l + (m-4)]))
            o[l + (0)] = (
                a*(f[l + (0)]-f[l + (m-1)]) +
                b*(f[l + (1)]-f[l + (m-2)]) +
                c*(f[l + (2)]-f[l + (m-3)]))
            o[l + (1)] = (
                a*(f[l + (1)]-f[l + (0)]) +
                b*(f[l + (2)]-f[l + (m-1)]) +
                c*(f[l + (3)]-f[l + (m-2)]))
            o[l + (2)] = (
                a*(f[l + (2)]-f[l + (1)]) +
                b*(f[l + (3)]-f[l + (0)]) +
                c*(f[l + (4)]-f[l + (m-1)]))
    for j in range(my):
        l = j*mx + k*mx*my;
        for i in range(2,mx-3):
            o[l + (i+1)] = (
              a*(f[l + (i+1)] - f[l + i]) +
              b*(f[l + (i+2)] - f[l + (i-1)]) +
              c*(f[l + (i+3)] - f[l + (i-2)]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddydn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddydn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell centre to cell faces, y direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef FLOAT_t a = 300 / 281. / dxc, b = -50 / 843. / dxc, c = 6 / 1405. / dxc
    cdef long i, j, k, l, m=mx
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    # Pure C part
    for k in range(mz):
        for i in range(mx):
          l = i + k*mx*my
          o[l + (m-2)*mx] = (
            a*(f[l + (m-2)*mx]-f[l + (m-3)*mx]) +
            b*(f[l + (m-1)*mx]-f[l + (m-4)*mx]) +
            c*(f[l + (0)*mx]-f[l + (m-5)*mx]))
          o[l + (m-1)*mx] = (
            a*(f[l + (m-1)*mx]-f[l + (m-2)*mx]) +
            b*(f[l + (0)*mx]-f[l + (m-3)*mx]) +
            c*(f[l + (1)*mx]-f[l + (m-4)*mx]))
          o[l + (0)*mx] = (
            a*(f[l + (0)*mx]-f[l + (m-1)*mx]) +
            b*(f[l + (1)*mx]-f[l + (m-2)*mx]) +
            c*(f[l + (2)*mx]-f[l + (m-3)*mx]))
          o[l + (1)*mx] = (
            a*(f[l + (1)*mx]-f[l + (0)*mx]) +
            b*(f[l + (2)*mx]-f[l + (m-1)*mx]) +
            c*(f[l + (3)*mx]-f[l + (m-2)*mx]))
          o[l + (2)*mx] = (
            a*(f[l + (2)*mx]-f[l + (1)*mx]) +
            b*(f[l + (3)*mx]-f[l + (0)*mx]) +
            c*(f[l + (4)*mx]-f[l + (m-1)*mx]))
    for j in range(2,my-3):
          for i in range(0,mx):
            l = i + k*mx*my
            o[l + (j+1)*mx] = (
              a*(f[l + (j+1)*mx] - f[l + j*mx]) +
              b*(f[l + (j+2)*mx] - f[l + (j-1)*mx]) +
              c*(f[l + (j+3)*mx] - f[l + (j-2)*mx]))
    return outarr


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ddzdn(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    ddzdn(np.ndarray[FLOAT_t, ndim=3] inarr)

    Brings derivative of quantity from cell faces to cell centre, z direction.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).

    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    cdef int mx = inarr.shape[0], my = inarr.shape[1], mz = inarr.shape[2]
    cdef long i, j, k, l, m=mx
    cdef FLOAT_t d
    cdef np.ndarray[FLOAT_t, ndim=2] zz = dzdnc
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros_like(inarr)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    f = <FLOAT_t *> inarr.data
    o = <FLOAT_t *> outarr.data
    tmp = <FLOAT_t *> zz.data
    # Pure C part
    for k in range(mz):
        m = k-3
        if (k < 3):
            m = 0
        if (k > mz-4):
            m = mz - 6
        for j in range(my):
          for i in range(mx):
            d = 0
            for l in range(6):
                d += tmp[k*6 + l] * f[((m + l) * my + j) * mx + i]
            o[(k * my + j) * mx + i] = d
    return outarr


###
### init functions
###
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_stagger_inv(np.ndarray[FLOAT_t, ndim=1] x, int n, int y, int o,
                     np.ndarray[FLOAT_t, ndim=1] r):
    ''' Auxiliary function for init_stagger. '''

    cdef FLOAT_t c[6]
    cdef FLOAT_t b[6]
    cdef FLOAT_t t
    cdef long i, j

    for i in range(n+1): c[i] = 0.
    c[n] = -x[0]
    for i in range(1,n + 1):
        for j in range(n-i, n):
            c[j] -= x[i] * c[j + 1]
        c[n] -= x[i]
    for i in range(6):
        r[i] = 0.
    for i in range(n + 1):
        t = 1.
        b[n] = 1.
        for j in range(n, 0, -1):
            b[j - 1] = c[j] + x[i] * b[j]
            t = x[i] * t + b[j-1]
        r[i + o] = b[y] / t
    return


@cython.boundscheck(False)
@cython.wraparound(False)
def init_stagger(int mz, FLOAT_t dx, FLOAT_t dy, np.ndarray[FLOAT_t, ndim=1] z,
                 np.ndarray[FLOAT_t, ndim=1] zdn,
                 np.ndarray[FLOAT_t, ndim=1] dzup,
                 np.ndarray[FLOAT_t, ndim=1] dzdn):
    '''
    init_stagger(int mz, dx, dy, z, zdn, dzup, dzdn)

    Initialises zupc and zdnc structures (for using with zdn and zup).
    From init_stagger.c and init_stagger.pro

    Parameters
    ----------
    mz - integer
       Number of z points
    z - 1-D ndarray, float
        z scale
    zdn - 1-D ndarray, float
        z scale derivative
    dzdup - 1-D ndarray, float
        z scale up derivative
    dzdn - 1-D ndarray, float
        z scale down derivative
    Returns
    -------
    None. Results saved in cstagger.zupc, cstagger.zdnc, cstagger.dzupc,
    cstagger.dzdnc.
    '''
    cdef int i, j, k
    global zupc, zdnc, nz, dxc, dyc, dzupc, dzdnc
    zupc = np.zeros((mz, 6), dtype=z.dtype)
    zdnc = np.zeros((mz, 6), dtype=z.dtype)
    dzupc = np.zeros((mz, 6), dtype=z.dtype)
    dzdnc = np.zeros((mz, 6), dtype=z.dtype)
    nz = mz
    dxc = dx
    dyc = dy

    cdef np.ndarray[FLOAT_t, ndim=1] zh = np.sort(np.concatenate([z,zdn]))
    cdef np.ndarray[FLOAT_t, ndim=1] a = np.zeros(6, dtype=z.dtype)

    iordl = np.array([1, 3, 4, 5])
    iordu = np.array([1, 3, 4, 5])
    dordl = np.array([2, 3, 4, 5])
    dordu = np.array([2, 3, 4, 5])

    for k in range(1, 4):
        for j in range(6):
            a[j] = zh[2 * j] - zh[k*2 - 1]
        calc_stagger_inv(a, iordl[k], 0, 0, zupc[k-1])
        calc_stagger_inv(a, dordl[k], 1, 0, dzupc[k-1])
        for j in range(6):
            a[j] = zh[2*j + 1] - zh[k*2 - 2]
        calc_stagger_inv(a, iordl[k-1], 0, 0, zdnc[k-1])
        calc_stagger_inv(a, dordl[k-1], 1, 0, dzdnc[k-1])

    for k in range(4, mz - 2):
        for j in range(6):
            a[j] = zh[2 * (k-2+j) - 2] - zh[k*2 - 1]
        calc_stagger_inv(a, 5, 0, 0, zupc[k-1])
        calc_stagger_inv(a, 5, 1, 0, dzupc[k-1])
        for j in range(6):
            a[j] = zh[2 * (k-3+j) - 1] - zh[k*2 - 2];
        calc_stagger_inv(a, 5, 0, 0, zdnc[k-1])
        calc_stagger_inv(a, 5, 1, 0, dzdnc[k-1])

    for k in range(mz-2, mz+1):
        for j in range(iordu[mz - k] + 1):
            a[j] = zh[2 * (mz + j - iordu[mz-k]) - 2] - zh[k*2 - 1]
        for j in range(iordu[mz - k] + 1, 6):
            a[j] = 0
        calc_stagger_inv(a, iordu[mz - k], 0, 5 - iordu[mz - k], zupc[k - 1])
        for j in range(dordu[mz - k] + 1):
            a[j] = zh[2 * (mz + j - dordu[mz - k]) - 2] - zh[k*2 - 1]
        for j in range(dordu[mz-k]+1, 6):
            a[j] = 0
        calc_stagger_inv(a, dordu[mz - k], 1, 5 - dordu[mz - k], dzupc[k - 1])
        for j in range(iordu[mz - k + 1] + 1):
            a[j] = zh[2 * (mz + j - iordu[mz - k + 1]) - 1] - zh[k*2 - 2]
        for j in range(iordu[mz - k + 1] + 1, 6):
            a[j] = 0
        calc_stagger_inv(a, iordu[mz - k + 1], 0, 5 - iordu[mz - k + 1],
                         zdnc[k - 1])
        for j in range(dordu[mz - k + 1] + 1):
            a[j] = zh[2 * (mz + j - dordu[mz - k + 1]) - 1] - zh[k*2 - 2]
        for j in range(dordu[mz - k + 1] + 1, 6):
            a[j] = 0
        calc_stagger_inv(a, dordu[mz - k + 1], 1, 5 - iordu[mz - k + 1],
                         dzdnc[k - 1])
    return


# Wrapper function for all the C stuff:
def do(np.ndarray[FLOAT_t, ndim=3] inarr, operation='xup'):
    """
    do(np.ndarray[FLOAT_t, ndim=3] inarr, operation='xup')

    Wrapper for C stagger operations.

    Parameters
    ----------
    inarr - 3-D array, floating type
        Input array. If not F contiguous, will make a copy (slower).
    operation - string
        What operation to perform. Possible values are:
        'xup', 'yup', 'zup': brings quantity from cell faces to cell centre
                             along x/y/z axes
        'xdn', 'ydn', 'zdn': brings quantity from cell centre to cell faces
                             along x/y/z axes
        'ddxdn', 'ddydn', 'ddzdn': brings derivative from cell centre to
                                   cell faces along x/y/z axes.
        'ddxup', 'ddyup', 'ddzup': brings derivative from cell faces to
                                   cell centre along x/y/z axes.
    Returns
    -------
    result - 3-D array, floating type
        Interpolated quantity. Same dtype as inarr.
    """
    OPERATIONS = {'xup': xup, 'yup': yup, 'zup': zup,
                  'xdn': xdn, 'ydn': ydn, 'zdn': zdn,
                  'ddxup': ddxup, 'ddyup': ddyup, 'ddzup': ddzup,
                  'ddxdn': ddxdn, 'ddydn': ddydn, 'ddzdn': ddzdn}
    func = OPERATIONS[operation]
    return func(inarr)
