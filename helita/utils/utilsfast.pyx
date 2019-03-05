#
# Collection of cython-ized fast math or image processing libraries.
#
# Currently, replace_nans and sincinterp come from:
# https://github.com/gasagna/openpiv-python/blob/master/openpiv/src/lib.pyx

import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t

DTYPEl = np.int64
ctypedef np.int64_t DTYPEl_t

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_nans(np.ndarray[DTYPEf_t, ndim=2] array, int max_iter, float tol,
                 int kernel_size=1, str method='localmean'):
    """
    Replace NaN elements in an array using an iterative image
    inpainting algorithm.

    The algorithm is the following:

    1) For each element in the input array, replace it by a weighted average
       of the neighbouring elements which are not NaN themselves. The weights
       depends on the method type. If ``method=localmean`` weight are equal
       to 1/( (2*kernel_size+1)**2 -1 )

    2) Several iterations are needed if there are adjacent NaN elements.
       If this is the case, information is "spread" from the edges of the
       missing  regions iteratively, until the variation is below a certain
      threshold.

    Parameters
    ----------
    array : 2d np.ndarray
        an array containing NaN elements that have to be replaced
    max_iter : int
        the number of iterations
   tol : float
        tolerance
    kernel_size : int
        the size of the kernel, default is 1
    method : str
        the method used to replace invalid values. Valid options are
        `localmean`.

    Returns
    -------
    filled : 2d np.ndarray
        a copy of the input array, where NaN elements have been replaced.
    """

    cdef int i, j, I, J, it, n, k, l
    cdef int n_invalids

    cdef np.ndarray[DTYPEf_t, ndim=2] filled = np.empty([array.shape[0],
                                                         array.shape[1]],
                                                        dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] kernel = np.empty((2 * kernel_size + 1,
                                                         2 * kernel_size + 1),
                                                        dtype=DTYPEf )

    cdef np.ndarray[np.int_t, ndim=1] inans
    cdef np.ndarray[np.int_t, ndim=1] jnans

    # indices where array is NaN
    inans, jnans = np.nonzero( np.isnan(array) )

    # number of NaN elements
    n_nans = len(inans)

    # arrays which contain replaced values to check for convergence
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_new = np.zeros(n_nans,
                                                              dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=1] replaced_old = np.zeros(n_nans,
                                                              dtype=DTYPEf)

    # depending on kernel type, fill kernel array
    if method == 'localmean':
        for i in range(2 * kernel_size + 1):
            for j in range(2 * kernel_size + 1):
                kernel[i,j] = 1.0
    else:
        raise ValueError( 'method not valid. Should be one of `localmean`.')

    # fill new array with input elements
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filled[i,j] = array[i,j]

    # make several passes
    # until we reach convergence
    for it in range(max_iter):
        # for each NaN element
        for k in range(n_nans):
            i = inans[k]
            j = jnans[k]
            # initialize to zero
            filled[i,j] = 0.0
            n = 0
            # loop over the kernel
            for I in range(2*kernel_size+1):
                for J in range(2*kernel_size+1):
                    # if we are not out of the boundaries
                    if i+I-kernel_size < array.shape[0] and i+I-kernel_size >= 0:
                        if j+J-kernel_size < array.shape[1] and j+J-kernel_size >= 0:
                            # if the neighbour element is not NaN itself.
                            if filled[i+I-kernel_size, j+J-kernel_size] == \
                                filled[i+I-kernel_size, j+J-kernel_size] :
                                # do not sum itself
                                if I-kernel_size != 0 and J-kernel_size != 0:
                                    # convolve kernel with original array
                                    filled[i,j] = filled[i,j] + \
                                        filled[i+I-kernel_size, j+J-kernel_size] * kernel[I, J]
                                    n = n + 1
            # divide value by effective number of added elements
            if n != 0:
                filled[i,j] = filled[i,j] / n
                replaced_new[k] = filled[i,j]
            else:
                filled[i,j] = np.nan
        # check if mean square difference between values of replaced
        # elements is below a certain tolerance
        if np.mean( (replaced_new-replaced_old)**2 ) < tol:
            break
        else:
            for l in range(n_nans):
                replaced_old[l] = replaced_new[l]
    return filled


def sincinterp(np.ndarray[DTYPEi_t, ndim=2] image,
               np.ndarray[DTYPEf_t, ndim=2] x, np.ndarray[DTYPEf_t, ndim=2] y,
               int kernel_size=3 ):
    """
    Re-sample an image at intermediate positions between pixels.

    This function uses a cardinal interpolation formula which limits
    the loss of information in the resampling process. It uses a limited
    number of neighbouring pixels.


    The new image :math:`im^+` at fractional locations :math:`x` and :math:`y` is computed as:

    .. math::

       im^+(x,y) = \sum_{i=-\mathtt{kernel\_size}}^{i=\mathtt{kernel\_size}} \sum_{j=-\mathtt{kernel\_size}}^{j=\mathtt{kernel\_size}} \mathtt{image}(i,j)  sin[\pi(i-\mathtt{x})]  sin[\pi(j-\mathtt{y})]  / \pi(i-\mathtt{x}) / \pi(j-\mathtt{y})


    Parameters
    ----------
    image : np.ndarray, dtype np.int32
        the image array.

    x : two dimensions np.ndarray of floats
        an array containing fractional pixel row
        positions at which to interpolate the image

    y : two dimensions np.ndarray of floats
        an array containing fractional pixel column
        positions at which to interpolate the image

    kernel_size : int
        interpolation is performed over a ``(2*kernel_size+1)*(2*kernel_size+1)``
        submatrix  in the neighbourhood of each interpolation point.

    Returns
    -------

    im : np.ndarray, dtype np.float64
        the interpolated value of ``image`` at the points specified
        by ``x`` and ``y``

    """
    # indices
    cdef int i, j, I, J
    # the output array
    cdef np.ndarray[DTYPEf_t, ndim=2] r = np.zeros([x.shape[0], x.shape[1]],
                                                   dtype=DTYPEf)
    # fast pi
    cdef float pi = 3.1419
    # for each point of the output array
    for I in range(x.shape[0]):
        for J in range(x.shape[1]):
            #loop over all neighbouring grid points
            for i in range( int(x[I,J])-kernel_size, int(x[I,J])+kernel_size+1 ):
                for j in range( int(y[I,J])-kernel_size, int(y[I,J])+kernel_size+1 ):
                    # check that we are in the boundaries
                    if i >= 0 and i <= image.shape[0] and j >= 0 and j <= image.shape[1]:
                        if (i-x[I,J]) == 0.0 and (j-y[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j]
                        elif (i-x[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j] * sin( pi*(j-y[I,J]) )/( pi*(j-y[I,J]) )
                        elif (j-y[I,J]) == 0.0:
                            r[I,J] = r[I,J] + image[i,j] * sin( pi*(i-x[I,J]) )/( pi*(i-x[I,J]) )
                        else:
                            r[I,J] = r[I,J] + image[i,j] * sin( pi*(i-x[I,J]) )*sin(pi*(j-y[I,J])) \
                                /( pi*pi*(i-x[I,J])*(j-y[I,J]))
    return r


cdef extern from "math.h":
    double sin(double)


@cython.boundscheck(False)
@cython.wraparound(False)
def peakdetect2(np.ndarray[DTYPEf_t, ndim=1] y_axis,
                np.ndarray[DTYPEf_t, ndim=1] x_axis=None,
                float delta_max=0., float delta_min=0.,
                int lookahead_max=300, int lookahead_min=300):
    """
    peakdetect2(y_axis, x_axis=None, delta_max=0., delta_min=0.,
                lookahead_max=300, lookahead_min=300)

    Function for detecting local maximas and minimas in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    Parameters
    ----------
    y_axis : 1-D ndarray (double type)
        Array containg the signal over which to find peaks
    x_axis : 1-D ndarray (double type), optional
        Array whose values correspond to the y_axis array and is used
        in the return to specify the postion of the peaks. If omitted an
        index of the y_axis is used. (default: None)
    lookahead_max, lookahead_min : int, optional
        Distance to look ahead from a peak candidate to determine if it
        is the actual peak. '(sample / period) / f' where '4 >= f >= 1.25'
        might be a good value. lookahead_max is for maximum peaks,
        lookahead_min for minimum peaks.
    delta_max, delta_min : float, optional
        Specifies a minimum difference between a peak and the following
        points, before a peak may be considered a peak. Useful to hinder
        the function from picking up false peaks towards to end of the
        signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
        delta function causes a 20% decrease in speed, when omitted.
        Correctly used it can double the speed of the function
        delta_max is for maximum peaks, delta_min for minimum peaks.

    Returns
    -------
    max_peaks, min_peaks : 2-D ndarrays
        Two arrays containing the maxima and minima location. Each array
        has a shape of (2, len(y_axis)). First dimension is for peak position
        (first index) or peak value (second index). Second dimension is
        for the number of peaks (maxima or minima).

    Notes
    -----
    Downloaded and adapted from https://gist.github.com/1178136

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html
    """
    cdef int length = len(y_axis)
    cdef np.ndarray[DTYPEf_t, ndim=2] max_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] min_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef int nmax = 0
    cdef int nmin = 0
    cdef int dump = 0
    cdef float mxpos, mnpos
    cdef float mn = np.inf
    cdef float mx = -np.inf
    cdef int i, lookahead

    lookahead = max(lookahead_max, lookahead_min)
    assert lookahead > 0

    for i in range(length - lookahead):
        if y_axis[i] > mx:
            mx = y_axis[i]
            mxpos = x_axis[i]
        if y_axis[i] < mn:
            mn = y_axis[i]
            mnpos = x_axis[i]

        #### look for max ####
        if (y_axis[i] < mx - delta_max) and (mx != np.Inf):
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead_max].max() < mx:
                max_peaks[0, nmax] = mxpos
                max_peaks[1, nmax] = mx
                nmax += 1
                if mx == y_axis[0]:
                    dump = 1
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if i + lookahead_max >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue

        #### look for min ####
        if (y_axis[i] > mn + delta_min) and (mn != -np.Inf):
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead_min].min() > mn:
                min_peaks[0, nmin] = mnpos
                min_peaks[1, nmin] = mn
                nmin += 1
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if i + lookahead_min >= length:
                    # end is within lookahead no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    if dump == 1:
        max_peaks = max_peaks[:, 1:]
        nmax -= 1
    else:
        min_peaks = min_peaks[:, 1:]
        nmin -= 1

    return max_peaks[:, :nmax], min_peaks[:, :nmin]


@cython.boundscheck(False)
@cython.wraparound(False)
def peakdetect(np.ndarray[DTYPEf_t, ndim=1] y_axis,
               np.ndarray[DTYPEf_t, ndim=1] x_axis=None,
               float delta=0., int lookahead=300):
    """
    peakdetect(y_axis, x_axis=None, delta=0., lookahead=300)

    Function for detecting local maximas and minimas in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    Parameters
    ----------
    y_axis : 1-D ndarray (double type)
        Array containg the signal over which to find peaks
    x_axis : 1-D ndarray (double type), optional
        Array whose values correspond to the y_axis array and is used
        in the return to specify the postion of the peaks. If omitted an
        index of the y_axis is used. (default: None)
    lookahead : int, optional
        Distance to look ahead from a peak candidate to determine if it
        is the actual peak. '(sample / period) / f' where '4 >= f >= 1.25'
        might be a good value.
    delta : float, optional
        Specifies a minimum difference between a peak and the following
        points, before a peak may be considered a peak. Useful to hinder
        the function from picking up false peaks towards to end of the
        signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
        delta function causes a 20% decrease in speed, when omitted.
        Correctly used it can double the speed of the function


    Returns
    -------
    max_peaks, min_peaks : 2-D ndarrays
        Two arrays containing the maxima and minima location. Each array
        has a shape of (2, len(y_axis)). First dimension is for peak position
        (first index) or peak value (second index). Second dimension is
        for the number of peaks (maxima or minima).

    Notes
    -----
    Downloaded and adapted from https://gist.github.com/1178136

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html
    """
    assert lookahead > 0
    cdef int length = len(y_axis)
    cdef np.ndarray[DTYPEf_t, ndim=2] max_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=2] min_peaks = np.zeros((2, length),
                                                         dtype=DTYPEf)
    cdef int nmax = 0
    cdef int nmin = 0
    cdef int dump = 0
    cdef float mxpos, mnpos
    cdef float mn = np.inf
    cdef float mx = -np.inf
    cdef int i

    for i in range(length - lookahead):
        if y_axis[i] > mx:
            mx = y_axis[i]
            mxpos = x_axis[i]
        if y_axis[i] < mn:
            mn = y_axis[i]
            mnpos = x_axis[i]

        #### look for max ####
        if (y_axis[i] < mx - delta) and (mx != np.Inf):
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead].max() < mx:
                max_peaks[0, nmax] = mxpos
                max_peaks[1, nmax] = mx
                nmax += 1
                if mx == y_axis[0]:
                    dump = 1
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if i + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue

        #### look for min ####
        if (y_axis[i] > mn + delta) and (mn != -np.Inf):
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead].min() > mn:
                min_peaks[0, nmin] = mnpos
                min_peaks[1, nmin] = mn
                nmin += 1
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if i + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break

    # Remove the false hit on the first value of the y_axis
    if dump == 1:
        max_peaks = max_peaks[:, 1:]
        nmax -= 1
    else:
        min_peaks = min_peaks[:, 1:]
        nmin -= 1

    return max_peaks[:, :nmax], min_peaks[:, :nmin]


@cython.boundscheck(False)
@cython.wraparound(False)
def peakdetect_3d(np.ndarray[DTYPEf_t, ndim=3] y_axis,
                  np.ndarray[DTYPEf_t, ndim=1] x_axis = None,
                  int lookahead = 300, float delta=0., mask_value = np.nan):
    """
    Adaptation of peakdetect for a 3D y_axis input. Not the most efficient
    method at the moment.
    """
    assert lookahead > 0
    cdef int length = y_axis.shape[2]
    cdef int nx = y_axis.shape[0]
    cdef int ny = y_axis.shape[1]
    cdef np.ndarray[DTYPEf_t, ndim=4] max_peaks = np.zeros((nx, ny, 2, length),
                                                         dtype=DTYPEf)
    cdef np.ndarray[DTYPEf_t, ndim=4] min_peaks = np.zeros((nx, ny, 2, length),
                                                         dtype=DTYPEf)
    cdef int nmax, nmin, dump
    cdef float mn = np.inf, mx = -np.inf, mxpos, mnpos
    cdef int i, j, k

    for k in range(nx):
        for j in range(ny):
            nmax = 0
            nmin = 0
            dump = 0
            mn = np.inf
            mx = -np.inf
            max_peaks[k, j] = mask_value
            min_peaks[k, j] = mask_value
            for i in range(length - lookahead):
                if y_axis[k, j, i] > mx:
                    mx = y_axis[k, j, i]
                    mxpos = x_axis[i]
                if y_axis[k, j, i] < mn:
                    mn = y_axis[k, j, i]
                    mnpos = x_axis[i]

                #### look for max ####
                if (y_axis[k, j, i] < mx - delta) and (mx != np.Inf):
                    # Maxima peak candidate found
                    # look ahead in signal to ensure that this is a peak and not jitter
                    if y_axis[k, j, i:i + lookahead].max() < mx:
                        max_peaks[k, j, 0, nmax] = mxpos
                        max_peaks[k, j, 1, nmax] = mx
                        nmax += 1
                        if mx == y_axis[k, j, 0]:
                            dump = 1
                        # set algorithm to only find minima now
                        mx = np.Inf
                        mn = np.Inf
                        if i + lookahead >= length:
                            # end is within lookahead no more peaks can be found
                            break
                        continue

                #### look for min ####
                if (y_axis[k, j, i] > mn + delta) and (mn != -np.Inf):
                    # Minima peak candidate found
                    # look ahead in signal to ensure that this is a peak and not jitter
                    if y_axis[k, j, i:i + lookahead].min() > mn:
                        min_peaks[k, j, 0, nmin] = mnpos
                        min_peaks[k, j, 1, nmin] = mn
                        nmin += 1
                        # set algorithm to only find maxima now
                        mn = -np.Inf
                        mx = -np.Inf
                        if i + lookahead >= length:
                            # end is within lookahead no more peaks can be found
                            break

            # Remove the false hit on the first value of the y_axis
            if dump == 1:
                max_peaks[k, j, 0, 0] = mask_value
            else:
                min_peaks[k, j, 0, 0] = mask_value


    return max_peaks, min_peaks

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef interp3d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=3] y,
             np.ndarray[DTYPEf_t, ndim=2] new_x):
    """
    interp3d(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 2D array new_x. Thus, interpolate
    y[i, j, :] for new_x[i, j].

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 3-D ndarray (double type)
        Array containing the y values to interpolate.
    x_new: 2-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 3-D ndarray
        Interpolated values.
    """
    cdef int nx = y.shape[0]
    cdef int ny = y.shape[1]
    cdef int nz = y.shape[2]
    cdef int i, j, k
    cdef np.ndarray[DTYPEf_t, ndim=2] new_y = np.zeros((nx, ny), dtype=DTYPEf)

    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz):
                 if x[k] > new_x[i, j]:
                     new_y[i, j] = (y[i, j, k] - y[i, j, k - 1]) * \
                    (new_x[i, j] - x[k-1]) / (x[k] - x[k - 1]) + y[i, j, k - 1]
                     break
    return new_y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef interp3d_vec(np.ndarray[DTYPEf_t, ndim=1] x,
                   np.ndarray[np.float32_t, ndim=3] y,
                   np.ndarray[DTYPEf_t, ndim=3] new_x):
    """
    interp3d_vec(x, y, new_x)

    Performs linear interpolation over the last dimension of a 3D array,
    according to new values from a 3D array new_x. Similar to interp3d(),
    but the new_x array has an extra dimension, meaning that for each
    1D slice y[i, j, :], the values are interpolated to a 1D array
    new_x[i, j, :] and not just a single point like interp3d().

    Thus, interpolates
    (x[:], y[i, j, :]) for new_x[i, j, :].

    For new_x[i, j, k] outside the range of x, it will extrapolate.

    Parameters
    ----------
    x : 1-D array (double type)
       Independent axis, must be monotonically increasing.
    y : 3-D array (float type)
       Array containing the values to interpolate. Last axis
       is the interpolation axis.
    new_x : 3-D array (double type)
       New points to interpolate. Last axis is the interpolation axis,
       doesn't need to be the same number of points as y.shape[-1].

    Returns
    -------
    result : 3-D array (float type)
        Interpolated values.
    """
    cdef int nx = y.shape[0]
    cdef int ny = y.shape[1]
    cdef int nw = y.shape[2]
    cdef int npts = new_x.shape[2]
    cdef int i, j, k
    cdef np.ndarray[np.float32_t, ndim=3] result = np.empty((nx, ny, npts),
                                                            dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            k = 0   # index of x just above new_x[i, j, k]
            for w in range(npts):
                while (x[k] < new_x[i, j, w]) and (k < nw -1):
                    k += 1
                if k == 0:  # extrapolation at start of domain
                    result[i, j, w] = ((y[i, j, 1] - y[i, j, 0]) *
                                       (new_x[i, j, w] - x[1]) /
                                       (x[1] - x[0])) + y[i, j, 0]
                elif k < nw - 1:  # normal interpolation
                    result[i, j, w] = ((y[i, j, k] - y[i, j, k - 1]) *
                                       (new_x[i, j, w] - x[k - 1]) /
                                       (x[k] - x[k - 1])) + y[i, j, k - 1]
                else:  # extrapolation at end of domain
                    result[i, j, w] = ((y[i, j, nw - 1] - y[i, j, nw - 2]) *
                                       (new_x[i, j, w] - x[nw - 2]) /
                                      (x[nw - 1] - x[nw - 2])) + y[i, j, nw - 2]
    return result


cdef inline float f_max(float a, float b):
    return a if a >= b else b


cdef inline float f_min(float a, float b):
    return a if a <= b else b


cdef inline int i_max(int a, int b):
    return a if a >= b else b


cdef inline int i_min(int a, int b):
    return a if a <= b else b

cdef extern from "math.h":
    double sqrt(double)


cpdef stat2d_idx(np.ndarray[DTYPEf_t, ndim=3] x,
                 np.ndarray[DTYPEl_t, ndim=2] idx_low,
                 np.ndarray[DTYPEl_t, ndim=2] idx_high):
    """
    vz2d(x, idx_low, idx_high)

    Performs statistics on the last dimension of a three-dimensional array,
    and for a range of indices along the last dimension.

    Parameters
    ----------
    x : 3-D ndarray (double type)
        Array containg the data
    idx_low : 2-D ndarray (integer type)
        Array containing the minimum indices of the last dimension to which
        the statistics will be calculated, for each (x, y) point.
    idx_high : 2-D ndarray (integer type)
        Array containing the maximum indices of the last dimension to which
        the statistics will be calculated, for each (x, y) point.

    Returns
    -------
    res : 3-D ndarray
        Array containing the results. The shape of this array is (4, nx, ny).
        The indices of the first dimension correspond to:
        res[0, :, :] : minimum
        res[1, :, :] : maximum
        res[2, :, :] : mean
        res[3, :, :] : sqrt(rms)
    """
    cdef int nx = x.shape[0]
    cdef int ny = x.shape[1]
    cdef int nz = x.shape[2]
    cdef int i, j, k
    cdef double amin, amax, avg, rms, count
    cdef np.ndarray[DTYPEf_t, ndim=3] res = np.zeros((4, nx, ny), dtype=DTYPEf)

    for i in range(nx):
        for j in range(ny):
            if (idx_low[i, j] < idx_high[i, j]) and \
               (idx_low[i, j] >= 0) and (idx_high[i, j] < nz):
                amin = np.inf
                amax = -np.inf
                count = 0.
                avg = 0.
                rms = 0.
                for k in range(idx_low[i, j], idx_high[i, j]):
                    amin = f_min(amin, x[i, j, k])
                    amax = f_max(amax, x[i, j, k])
                    avg += x[i, j, k]
                    rms += x[i, j, k] * x[i, j, k]
                    count += 1.
                if count > 0:
                    avg /= count
                    rms /= count
                res[0, i, j] = amin
                res[1, i, j] = amax
                res[2, i, j] = avg
                res[3, i, j] = sqrt(rms)
            else:
                res[:, i, j] = np.nan
    return res
