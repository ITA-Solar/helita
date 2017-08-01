cimport cython
cimport numpy as np
import numpy as np

ctypedef fused FLOAT_t:
    np.float32_t
    np.float64_t

nz = 0  # initialise nz

###
###  C functions
###
cdef inline void xup_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o):
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef int i, j, k, l, m=mx

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
    return


cdef inline void yup_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o):
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef int i, j, k, l, m=my

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
    return


cdef inline void zup_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o,
                       FLOAT_t *zzupc):
    cdef FLOAT_t d
    cdef int i, j, k, l, m=mz

    for k in range(mz):
        m = k-2
        if (k < 3): m = 0
        if (k > mz-4): m = mz-6
        for j in range(my):
            for i in range(mx):
                d = 0
                for l in range(6):
                    d += zzupc[k*6 + l]*f[((m+l)*my + j)*mx + i];
                o[(k * my + j) * mx + i] = d
    return


cdef inline void xdn_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o):
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef int i, j, k, l, m=mx

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
    return


cdef inline void ydn_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o):
    cdef FLOAT_t c = 3./256., b = -25./256., a = 150./256.
    cdef int i, j, k, l, m=my

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
    return


cdef inline void zdn_c(int mx, int my, int mz, FLOAT_t *f, FLOAT_t *o,
                       FLOAT_t *zzdnc):
    cdef FLOAT_t d
    cdef int i, j, k, l, m=mz

    for k in range(mz):
        m = k-3
        if (k < 3): m = 0
        if (k > mz-4): m = mz-6
        for j in range(my):
            for i in range(mx):
                d = 0
                for l in range(6):
                    d += zzdnc[k*6 + l] * f[((m + l)*my + j)*mx + i];
                o[(k * my + j) * mx + i] = d
    return


###
### Python wrappers
###
### init functions
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_stagger_inv(np.ndarray[FLOAT_t, ndim=1] x, int n, int y, int o,
                     np.ndarray[FLOAT_t, ndim=1] r):
    ''' Auxiliary function for init_stagger. '''

    cdef FLOAT_t c[6]
    cdef FLOAT_t b[6]
    cdef FLOAT_t t
    cdef int i, j

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
def init_stagger(int mz, np.ndarray[FLOAT_t, ndim=1] z,
                 np.ndarray[FLOAT_t, ndim=1] zdn):
    '''
    init_stagger(int mz, np.ndarray[FLOAT_t] z, np.ndarray[FLOAT_t] zdn)

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

    Returns
    -------
    None. Results saved in cstagger.zupc, cstagger.zdnc.
    '''
    cdef int i, j, k
    global zupc, zdnc, nz
    zupc = np.zeros((mz,6),dtype=z.dtype)
    zdnc = np.zeros((mz,6),dtype=z.dtype)
    nz = mz

    cdef np.ndarray[FLOAT_t, ndim=1] zh = np.sort(np.concatenate([z,zdn]))
    cdef np.ndarray[FLOAT_t, ndim=1] a = np.zeros(6, dtype=z.dtype)

    iordl = np.array([1,3,4,5])
    iordu = np.array([1,3,4,5])
    dordl = np.array([2,3,4,5])
    dordu = np.array([2,3,4,5])

    for k in range(1, 4):
        for j in range(6):
            a[j] = zh[2 * j] - zh[k*2 - 1]
        calc_stagger_inv(a, iordl[k], 0, 0, zupc[k-1])
        for j in range(6):
            a[j] = zh[2*j + 1] - zh[k*2 - 2]
        calc_stagger_inv(a, iordl[k-1], 0, 0, zdnc[k-1])
    for k in range(4, mz - 2):
        for j in range(6):
            a[j] = zh[2 * (k-2+j) - 2] - zh[k*2 - 1]
        calc_stagger_inv(a, 5, 0, 0, zupc[k-1])
        for j in range(6):
            a[j] = zh[2 * (k-3+j) - 1] - zh[k*2 - 2];
        calc_stagger_inv(a, 5, 0, 0, zdnc[k-1])
    for k in range(mz-2, mz+1):
        for j in range(iordu[mz - k] + 1):
            a[j] = zh[2 * (mz + j - iordu[mz-k]) - 2] - zh[k*2 - 1]
        for j in range(iordu[mz - k] + 1, 6):
            a[j] = 0
        calc_stagger_inv(a, iordu[mz - k], 0, 5 - iordu[mz - k], zupc[k - 1])
        for j in range(iordu[mz - k + 1] + 1):
            a[j] = zh[2 * (mz + j - iordu[mz - k + 1]) - 1] - zh[k*2 - 2]
        for j in range(iordu[mz - k + 1] + 1, 6):
            a[j] = 0
        calc_stagger_inv(a, iordu[mz - k + 1], 0, 5 - iordu[mz - k + 1],
                         zdnc[k - 1])
    return


#------------------------------------------------------------------------------
# up and down functions
#------------------------------------------------------------------------------
def xup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    xup(np.ndarray[FLOAT_t, ndim=3] inarr)

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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    xup_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))


def yup(np.ndarray[FLOAT_t, ndim=3] inarr):
    """
    yup(np.ndarray[FLOAT_t, ndim=3] inarr)

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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    yup_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))


def zup(np.ndarray[FLOAT_t, ndim=3] inarr):
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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    if (mz != nz):
        raise ValueError('zup: nz mismatch, must run init_stagger first!')
    cdef np.ndarray[FLOAT_t, ndim=2] zz = zupc
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    zup_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data,
          <FLOAT_t *> zz.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))


def xdn(np.ndarray[FLOAT_t, ndim=3] inarr):
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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    xdn_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))


def ydn(np.ndarray[FLOAT_t, ndim=3] inarr):
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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    ydn_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))


def zdn(np.ndarray[FLOAT_t, ndim=3] inarr):
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
    cdef np.ndarray[FLOAT_t, ndim=3] outarr = np.zeros((mx, my, mz),
                                                       dtype=inarr.dtype)
    inarr = np.reshape(np.transpose(inarr), (mx, my, mz))
    if (mz != nz):
        raise ValueError('zdn: nz mismatch, must run init_stagger first!')
    cdef np.ndarray[FLOAT_t, ndim=2] zz = zdnc
    if not inarr.flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    zdn_c(mx, my, mz, <FLOAT_t *> inarr.data, <FLOAT_t *> outarr.data,
          <FLOAT_t *> zz.data)
    return np.transpose(np.reshape(outarr, (mz, my, mx)))
