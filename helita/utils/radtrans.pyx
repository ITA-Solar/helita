cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport exp

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def piecewise_1D(np.ndarray[DTYPE_t, ndim=1] height,
                 np.ndarray[DTYPE_t, ndim=1] chi,
                 np.ndarray[DTYPE_t, ndim=1] S):
    """
    Performs the piecewise quadratic integration of radiative transfer
    equation in 1D. Always assumes radiation is travelling in direction
    of increasing depth index. Boundary conditions are zero radiation
    everywhere (no thermalised boundary cond.).

    Calling sequence
    ----------------
    I = piecewise_1D(height, chi, S)

    Parameters
    ----------
    height, chi, S:  1D arrays, float32
        height scale, absorption coefficient, source function. Height 
        and chi must have consistent units (typically m and m^-1, respectively).

    Returns
    -------
    I_upw : float
        Outgoing intensity.
    """
    cdef DTYPE_t dtau, dS, I_upw
    cdef int k, ndep
    ndep = len(chi)
    cdef np.ndarray[DTYPE_t, ndim=1] I = np.zeros(ndep, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.empty(3, dtype=DTYPE)

    I_upw = 0.
    dtau = 0.5 * (chi[1] + chi[0]) * abs(height[0] - height[1])
    if dtau != 0.0:
        dS = (S[0] - S[1]) / dtau
    else:
        dS = 0.0
    for k in range(ndep):
        w3(dtau, <DTYPE_t *> w.data)
        I[k] = (1.0 - w[0]) * I_upw + w[0] * S[k] + w[1] * dS
        I_upw = I[k]
        if k != ndep - 1:
            dtau = 0.5 * (chi[k] + chi[k + 1]) * abs(height[k] - height[k + 1])
            if dtau != 0.0:
                dS = (S[k] - S[k + 1]) / dtau
            else:
                dS = 0.0

    return I_upw

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)
def piecewise_3D(np.ndarray[DTYPE_t, ndim=1] height,
                 np.ndarray[DTYPE_t, ndim=3] chi,
                 np.ndarray[DTYPE_t, ndim=3] S):
    """
    Performs the piecewise quadratic integration of radiative transfer
    equation in 1D, for every column [:, i, j] in a 3D array.
    Always assumes radiation is travelling in direction of increasing depth,
    which is the first index. Boundary conditions are zero radiation
    everywhere (no thermalised boundary cond.).

    Calling sequence
    ----------------
    I = piecewise_1D(height, chi, S)

    Parameters
    ----------
    height : 1-D array, float32
         Height scale
    chi, S: 3-D arrays, float32
         Absorption coefficient, source function. Height and chi must have
         consistent units (typically m and m^-1, respectively).

    Returns
    -------
    I : 2-D array, float32
         Outgoing radiation intensity.
    """

    cdef DTYPE_t dtau, dS, I_upw
    cdef int k
    cdef int ndep = chi.shape[0]
    cdef int nz = chi.shape[1]
    cdef int nw = chi.shape[2]
    cdef np.ndarray[DTYPE_t, ndim=1] I = np.zeros(ndep, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] w = np.empty(3, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] res = np.empty((nz, nw), dtype=DTYPE)

    for iz in range(nz):
        for iw in range(nw):
            I_upw = 0.
            dtau = 0.5 * (chi[1, iz, iw] + chi[0, iz, iw]) * abs(height[0] -
                                                                 height[1])
            if dtau != 0.0:
                dS = (S[0, iz, iw] - S[1, iz, iw]) / dtau
            else:
                dS = 0.0
            for k in range(ndep):
                w3(dtau, <DTYPE_t *> w.data)
                I[k] = (1.0 - w[0]) * I_upw + w[0] * S[k, iz, iw] + w[1] * dS
                I_upw = I[k]
                if k != ndep - 1:
                    dtau = 0.5 * (chi[k, iz, iw] + chi[k + 1, iz, iw]) * \
                       abs(height[k] - height[k + 1])
                    if dtau != 0.0:
                        dS = (S[k, iz, iw] - S[k + 1, iz, iw]) / dtau
                    else:
                        dS = 0.0
            res[iz, iw] = I_upw
    return res


cdef inline void w3(DTYPE_t dtau, DTYPE_t *w):
    cdef DTYPE_t delta, expdt

    if dtau < 5.e-4:
        w[0]   = dtau * (1.0 - 0.5 * dtau)
        delta  = dtau * dtau
        w[1]   = delta * (0.5 - 0.33333333 * dtau)
        delta *= dtau
        w[2]   = delta * (0.33333333 - 0.25 * dtau)
    elif (dtau > 50.0):
        w[1] = w[0] = 1.0;
        w[2] = 2.0;
    else:
        expdt = exp(-dtau);
        w[0]  = 1.0 - expdt;
        w[1]  = w[0] - dtau * expdt;
        w[2]  = 2.0 * w[1] - dtau * dtau * expdt;
    return
