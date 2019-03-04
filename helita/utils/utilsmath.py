import numpy as np
from numba import vectorize, float32, float64
from math import exp


def hist2d(x, y, nbins=30, norm=False, rx=0.08):
    '''
    Computes the 2D histogram of the data and the x,y coordinates
    of the middle of the bins. OPTIONS; nbins (number of bins), norm,
    rx (range in x as % of the min/max, previous default 0.08).
    '''
    # Increments in x and y
    xinc = (np.max(x) - np.min(x)) / nbins
    yinc = (np.max(y) - np.min(y)) / nbins
    # Define bin edges: either from rx or 1.5 increments,
    # whichever is wider
    xlow = min(np.min(x) * (1 - rx), np.min(x) - 1.5 * xinc)
    xhigh = max(np.max(x) * (1 + rx), np.max(x) + 1.5 * xinc)
    ylow = min(np.min(y) * (1 - rx), np.min(y) - 1.5 * yinc)
    yhigh = max(np.max(y) * (1 + rx), np.max(y) + 1.5 * yinc)
    r = [[xlow, xhigh], [ylow, yhigh]]
    hist, xi, yi = np.histogram2d(x, y, bins=nbins, range=r, normed=norm)
    # Take the middle point of the bins
    xbin = (xi[1:] + xi[:-1]) / 2.
    ybin = (yi[1:] + yi[:-1]) / 2.
    return np.transpose(hist), xbin, ybin


def stat2d(x, y, x_range=None, nbins=10, percentiles=[25, 50, 75]):
    """
    Computes the median, and two quartils for 2D data by binning in the
    x axis. Range of binning set by x_range, number of bins by nbins.

    Parameters
    ----------
    x:   1-D array
         Abcissa for the data.
    y:   1-D array
         Values for the data.
    x_range:  2-element list/array
         Min and max for the binning. If not set, will use x.max() and x.min()
    nbins: integer
         Number of bins to use.
    percentiles: list with 3 elements
         Percentile values for q1, q2, q3

    Returns
    -------
    xbins:  1-D array
         Contains the abcissa for the bins
    q1:     1-D array
         First quartile
    q2:     1-D array
         Second quartile (median)
    q3:     1-D array
         Third quartile.
    """
    if x_range is None:
        x_range = [np.min(x), np.max(x)]
    bins = np.linspace(x_range[0], x_range[1], nbins + 1)
    xbins = 0.5 * (bins[1:] + bins[:-1])
    q1 = np.zeros(nbins)
    q2 = np.zeros(nbins)
    q3 = np.zeros(nbins)
    for i in range(nbins):
        idx = (x >= bins[i]) & (x < bins[i + 1])
        sdata = y[idx]
        q1[i] = np.percentile(sdata, percentiles[0])
        q2[i] = np.percentile(sdata, percentiles[1])
        q3[i] = np.percentile(sdata, percentiles[2])
    return xbins, q1, q2, q3


def planck(wavelength, temp, dist='wavelength'):
    """
    Calculates the Planck function, either per wavelength or per
    frequency.

    Parameters
    ----------
    wavelength : `Quantity` object (number or sequence, length units)
        Wavelength(s) to calculate.
    temp : `Quantity` object (number or sequence, temperature units)
        Temperature(s) to calculate
    dist: str, optional
        How to calculate the distribution. Options are 'wavelength' (default),
        or 'frequency'.

    Returns
    -------
    planck : `Quantity` object (number or sequence)
        Planck distribution. Units are energy per time per area
        per frequency (or wavelength) per solid angle.

    Notes
    -----
    For solid angle integrated one must multiply it by pi.

    """
    from astropy.constants import c, h, k_B
    import astropy.units as u

    wave = wavelength.to('nm')
    if temp.shape and wave.shape:
        temp = temp[:, np.newaxis]  # array broadcast when T and wave are arrays
    if dist.lower() == 'wavelength':
        iplanck = 2 * h * c**2 / wave**5 / (np.exp(h * c / (wave * k_B * temp)) - 1)
        return (iplanck / u.sr).to('erg / (s cm2 Angstrom sr)')
    elif dist.lower() == 'frequency':
        iplanck = 2 * h * c / wave**3 / (np.exp(h * c / (wave * k_B * temp)) - 1)
        return (iplanck / u.sr).to('W / (m2 Hz sr)')
    else:
        raise ValueError('invalid distribution ' % dist)


def int_to_bt(inu, wave):
    """
    Converts radiation intensity to brightness temperature.

    Parameters
    ----------
    inu : `Quantity` object (number or sequence)
        Radiation intensity in units of energy per second per area
        per frequency per solid angle.
    wave: `Quantity` object (number or sequence)
        Wavelength in length units

    Returns
    -------
    brightness_temp : `Quantity` object (number or sequence)
        Brightness temperature in SI units of temperature.
    """
    from astropy.constants import c, h, k_B
    import astropy.units as u

    bt = h * c / (wave * k_B * np.log(2 * h * c / (wave**3 * inu * u.rad**2) + 1))
    return bt.si


def trapz2d(z, x=None, y=None, dx=1., dy=1.):
    ''' Integrates a regularly spaced 2D grid using the composite
        trapezium rule.
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval

    --Tiago, 20090501
    '''
    if x != None:
        dx = (x[-1] - x[0]) / (np.shape(x)[0] - 1)
    if y != None:
        dy = (y[-1] - y[0]) / (np.shape(y)[0] - 1)
    s1 = z[0, 0] + z[-1, 0] + z[0, -1] + z[-1, -1]
    s2 = np.sum(z[1:-1, 0]) + np.sum(z[1:-1, -1]) + \
        np.sum(z[0, 1:-1]) + np.sum(z[-1, 1:-1])
    s3 = np.sum(z[1:-1, 1:-1])
    return 0.25 * dx * dy * (s1 + 2 * s2 + 4 * s3)


def translate(data, z, mu, phi, dx=1, dy=1):
    """
    Horizontally rotates a 3D array with periodic horizontal boundaries
    by a polar and azimuthal angle. Uses cubic splines, modifies data in-place
    (therefore the rotation leads to an array with the same dimensions).

    Parameters
    ----------
    data : 3D array, 32-bit float, F contiguous
        Array with values. Last index should be height, the
        non-periodic dimension. The rotation keeps the top and
        bottom layers
    z : 1D array, 32-bit float
        Array with heights.
    mu : float
        Cosine of polar angle.
    phi : float
        Azimuthal angle in radians.
    dx : float, optional
        Grid separation in x dimension (same units as height). Default is 1.
    dy : float, optional
        Grid separation in y dimension (same units as height). Default is 1.

    Returns
    -------
    None, data are modified in-place.
    """
    from math import acos, sin, cos
    try:
        from .trnslt import trnslt
    except ModuleNotFoundError:
        raise ModuleNotFoundError('trnslt not found, helita probably installed'
                                  ' without a fortran compiler!')
    assert data.shape[-1] == z.shape[0]
    assert data.flags['F_CONTIGUOUS']
    assert data.dtype == np.dtype("float32")
    theta = acos(mu)
    sinth = sin(theta)
    tanth = sinth / mu
    cosphi = cos(phi)
    sinphi = sin(phi)
    dxdz = tanth * cosphi
    dydz = tanth * sinphi
    trnslt(dx, dy, z, data, dxdz, dydz)


@vectorize([float32(float32, float32), float64(float64, float64)])
def voigt(a, v):
    """
    Returns the Voigt function:

    H(a,v) = a/pi * \int_{-Inf}^{+Inf} exp(-y**2)/[(v-y)**2 + a**2] dy

    Based on approximation from old Fortran routine voigtv from Aake Nordlund.
    Makes use of numba vectorize, can be used as numpy ufunc.

    Parameters
    ----------
    a : scalar or n-D array (float)
        Parameter 'a' in Voigt function, typically a scalar. If n-D, must
        have same shape of v.
    v : scalar or n-D array (float)
        Velocity or Doppler value for Voigt function, typically a 1D array.

    Returns
    -------
    h : scalar or n-D array (float)
        Voigt function. Same shape and type as inputs.
    """
    a0 = 122.607931777104326
    a1 = 214.382388694706425
    a2 = 181.928533092181549
    a3 = 93.155580458138441
    a4 = 30.180142196210589
    a5 = 5.912626209773153
    a6 = 0.564189583562615
    b0 = 122.607931773875350
    b1 = 352.730625110963558
    b2 = 457.334478783897737
    b3 = 348.703917719495792
    b4 = 170.354001821091472
    b5 = 53.992906912940207
    b6 = 10.479857114260399
    if a == 0:
        return exp(-v ** 2)
    z = v * 1j + a
    h = (((((((a6 * z + a5) * z + a4) * z + a3) * z + a2) * z + a1) * z + a0) /
     (((((((z + b6) * z + b5) * z + b4) * z + b3) * z + b2) * z + b1) * z + b0))
    return h.real


def voigt_sigma(sigma, gamma, r):
    ''' Returns the Voigt function, defined in terms of sigma (Gaussian sdev)
    and gamma (Lorentzian FWHM). '''
    tt = np.sqrt(2) * sigma
    v = r / tt
    a = gamma / (2 * tt)
    return voigt(a, v) / (tt * np.sqrt(np.pi))


def stat(a):
    ''' Returns some statistics on a given array '''
    mm = np.nanmean(a)
    ss = float(np.nanstd(a))  # float for the memmap bug
    mi = np.nanmin(a)
    ma = np.nanmax(a)
    print(('aver =    %.3e' % mm))
    print(('rms  =    %.3e    rms/aver =    %.3e' % (ss, ss / mm)))
    print(('min  =    %.3e    min/aver =    %.3e' % (mi, mi / mm)))
    print(('max  =    %.3e    max/aver =    %.3e' % (ma, ma / mm)))


def bin_quantities(x, y, bins, func, *args, **kwargs):
    """
    Perform a certain function on x-bins of a x/y relation.

    Parameters
    ----------

    x - n-D array
       Array with the abcissa. If more than 1D, it will be flattened.
    y - n-D array
       Array with the coordinates. If more than 1D, it will be flattened.
    bins - array-like (1D)
       Values for the abcissa bins.
    func - [numpy] function
       Function to operate. Must work on arrays.
    *args, **kwargs: arguments and keyword arguments for func.

    Returns
    -------
    result - 1D array
       Array with same shape as bins, containing the results of running
       func in the different regions.
    """
    xx = x.ravel()
    yy = y.ravel()
    idx = np.digitize(xx, bins)
    result = np.zeros(len(bins))
    for i in range(len(bins)):
        if np.sum(idx == i) > 0:
            result[i] = func(yy[idx == i], *args, **kwargs)
    return result


def peakdetect(y_axis, x_axis=None, lookahead=300, delta=0):
    """
    peakdetect(y_axis, x_axis=None, delta=0., lookahead=300)

    Function for detecting local maximas and minimas in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively.

    This is the pure python version. See utilsfast.peakdetect2 for a
    much faster version in Cython.

    Parameters
    ----------
    y_axis : 1-D ndarray or list
        Array containg the signal over which to find peaks
    x_axis : 1-D ndarray or list
        Array whose values correspond to the y_axis array and is used
        in the return to specify the postion of the peaks. If omitted an
        index of the y_axis is used. (default: None)
    lookahead : int, optional
        Distance to look ahead from a peak candidate to determine if it
        is the actual peak. '(sample / period) / f' where '4 >= f >= 1.25'
        might be a good value.
    delta : number, optional
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
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false
    # Store data length for later use
    length = len(y_axis)
    # Perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    # maxima and minima candidates are temporarily stored in
    # mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    # Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        # look for max
        if y < mx - delta and mx != np.Inf:
            # Maxima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
                continue
        # look for min
        if y > mn + delta and mn != -np.Inf:
            # Minima peak candidate found
            # look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                # set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index + lookahead >= length:
                    # end is within lookahead no more peaks can be found
                    break
    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        # no peaks were found
        pass
    return [np.array(max_peaks).T, np.array(min_peaks).T]


def lclxtrem(vec, in_width, maxima=False):
    """
    Finds peaks in data. Converted from lclxtrem.
    """
    width = abs(in_width)
    # First derivative
    vecp = np.diff(vec)
    # Collapse the derivative to just +1, 0, or -1
    vecps = np.zeros(vecp.shape, dtype='i')
    vecps[vecp > 0.] = 1
    vecps[vecp < 0.] = -1
    # Derivative of the sign vectors
    vecpps = np.diff(vecps)
    # Keep the appropriate extremum
    if maxima:
        z = np.where(vecpps < 0)[0]
    else:
        z = np.where(vecpps > 0)[0]
    nidx = len(z)
    flags = np.ones(nidx, dtype=np.bool)
    # Create an index vector with just the good points.
    if nidx == 0:
        if maxima:
            idx = (vec == np.max(vec))
        else:
            idx = (vec == np.min(vec))
    else:
        idx = z + 1
    # Sort the extrema (actually, the absolute value)
    sidx = idx[np.argsort(np.abs(vec[idx]))[::-1]]
    # Scan down the list of extrema, start with the brightest and take out
    #   all extrema within width of the position.  Any that are too close should
    #   be removed from further consideration.
    if width > 1:
        i = 0
        for i in range(nidx - 1):
            if flags[i]:
                flags[i + 1:][np.abs(sidx[i + 1:] - sidx[i]) <= width] = False
    #  The ones that survive are returned.
    return np.sort(sidx[flags])


def peakdetect_lcl(y_axis, x_axis=None, lookahead=300, delta=0):
    """
    Wrapper to lclxtrem to mimic the behaviour of peakdetect.
    """
    maxima = lclxtrem(y_axis, lookahead, maxima=True)
    minima = lclxtrem(y_axis, lookahead, maxima=False)
    return np.array([x_axis[maxima], y_axis[maxima]]), \
        np.array([x_axis[minima], y_axis[minima]])


def pinterp3d(x, y, new_x):
    """
    pinterp3d(x, y, new_x)

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
    nx = y.shape[0]
    ny = y.shape[1]
    nz = y.shape[2]
    new_y = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            for k in range(1, nz):
                if x[k] > new_x[i, j]:
                    new_y[i, j] = (y[i, j, k] - y[i, j, k - 1]) * \
                        (new_x[i, j] - x[k - 1]) / \
                        (x[k] - x[k - 1]) + y[i, j, k - 1]
                    break
    return new_y


def pystat2d_idx(x, idx_low, idx_high):
    """
    vz2d(x, idx_low, idx_high)

    """
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]
    res = np.empty((4, nx, ny), dtype='d')
    for i in range(nx):
        for j in range(ny):
            if (idx_low[i, j] < idx_high[i, j]) and \
               (idx_low[i, j] >= 0) and (idx_high[i, j] < nz):
                arr = x[i, j, idx_low[i, j]:idx_high[i, j]]
                res[0, i, j] = np.max(arr)
                res[1, i, j] = np.min(arr)
                res[2, i, j] = np.mean(arr)
                res[3, i, j] = np.std(arr)
    return res


def madmax(image):
    """
    Computes a multidirectional maximum of (weighted second order difference)
    using 8 directions step=2 in horizontal and vertical directions
    weight=distance between extreme pixels.

    Uses algorithm from ; Koutchmy,O. and Koutchmy, S. (1988),
    Proc. of the 10th NSO/SPO, 1989, 217, O. von der Luhe Ed.

    Adapted from madmax.pro.

    Parameters
    ----------
    image : 2D array
        Image data to calculate madmax filter.

    Returns
    -------
    mad : 2D array
        Filtered image.
    """
    from scipy import ndimage

    nx, ny = image.shape
    # Determine some constants and arrays.
    h1 = 0.5
    h2 = 0.2 * np.sqrt(5.)
    h3 = 0.25 * np.sqrt(2.)
    d = np.empty((nx, ny, 8))
    mat = image.copy()
    shifts = [[(0, -2), (0,  2)], [(-1, -2), (1,  2)], [(-2, -2), (2,  2)],
              [(-2, -1), (2,  1)], [(-2,  0), (2,  0)], [(-2,  1), (2, -1)],
              [(-2,  2), (2, -2)], [(-1,  2), (1, -2)]]
    hh = [h1, h2, h3, h2, h1, h2, h3, h2]
    for i, h, shft in zip(list(range(8)), hh, shifts):
        s1, s2 = shft
        d[..., i] = h * (mat - 0.5 *
                         (np.roll(np.roll(mat, s1[0], axis=0), s1[1], axis=1)
                          + np.roll(np.roll(mat, s2[0], axis=0), s2[1], axis=1)))
    mat = d.max(-1)
    del d
    # border
    mat = ndimage.map_coordinates(mat[4:-4, 4:-4],
                                  np.mgrid[0:nx - 9:nx * 1j, 0:ny - 9:ny * 1j],
                                  order=3)
    return mat - mat.min()   # make matrix always positive


def make_composite_array(f1, f2, f3, l1, l2, l3):
    """
    Makes a composite RGB image based on the arrays f1, f2, f3 at saturations
    l1, l2, l3.
    """
    a1 = f1.copy()
    a1[a1 > l1] = l1
    a1[a1 < 0] = 0.
    a1 /= a1.max()
    a2 = f2.copy()
    a2[a2 > l2] = l2
    a2[a2 < 0] = 0.
    a2 /= a2.max()
    a3 = f3.copy()
    a3[a3 > l3] = l3
    a3[a3 < 0] = 0.
    a3 /= a3.max()
    return np.transpose(np.array([a1, a2, a3]), axes=[1, 2, 0])


def make_composite_array2(f1, f2, l1, l2, color=None, negative=False):
    """
    Makes a composite RGB image based on two arrays f1, f2 at saturations
    l1, l2. Colour can be set as RGB tuple (each value from 0 to 255),
    otherwise a default will be used. "color" sets the colour of the first
    array f1, the second array f2 will have its complementary.
    """
    a1 = f1.copy()
    a1[a1 > l1] = l1
    a1[a1 < 0] = 0.
    a1 /= a1.max()
    a2 = f2.copy()
    a2[a2 > l2] = l2
    a2[a2 < 0] = 0.
    a2 /= a2.max()
    if color is None:
        if negative:
            result = np.transpose(np.array([a1, a2, a2]), axes=[1, 2, 0])
        else:
            result = np.transpose(np.array([a2, a1, a1]), axes=[1, 2, 0])
    else:
        result = np.zeros(f1.shape + (3,))
        if negative:
            a1 = 1 - a1
            a2 = 1 - a2
        result[..., 0] = a1 * color[0] + a2 * (255 - color[0])
        result[..., 1] = a1 * color[1] + a2 * (255 - color[1])
        result[..., 2] = a1 * color[2] + a2 * (255 - color[2])
        result /= 255.
    return result


def make_composite_array3(f1, f2, f3, l1, l2, l3, color1=None, color2=None,
                          negative=False):
    """
    Makes a composite RGB image based on three arrays f1, f2, f3, at
    saturations l1, l2, l3. Colours for first two can be set as RGB tuple
    (each value from 0 to 255). The third array f3 will have the complementary
    colour.
    """
    a1 = f1.copy()
    a1[a1 > l1] = l1
    a1[a1 < 0] = 0.
    a1 /= a1.max()
    a2 = f2.copy()
    a2[a2 > l2] = l2
    a2[a2 < 0] = 0.
    a2 /= a2.max()
    a3 = f3.copy()
    a3[a3 > l3] = l3
    a3[a3 < 0] = 0.
    a3 /= a3.max()
    # color to complement
    color3 = 255 - (np.array(color1) + np.array(color2))
    color3[color3 > 255] = 255.
    color3[color3 < 0] = 0.
    print(color3)
    result = np.zeros(f1.shape + (3,))
    if negative:
        a1 = 1 - a1
        a2 = 1 - a2
        a3 = 1 - a3
    result[..., 0] = a1 * color1[0] + a2 * color2[0] + a3 * color3[0]
    result[..., 1] = a1 * color1[1] + a2 * color2[1] + a3 * color3[1]
    result[..., 2] = a1 * color1[2] + a2 * color2[2] + a3 * color3[2]
    result[result > 255] = 255.
    result /= 255.
    return result


def get_equidistant_points(x, y, scale=1., npts=100, order=3):
    """
    Returns a (x, y) set of points equidistant points from a smoothed cubic
    spline to the original. Equidistance is approximate (within 1% of scale)
    due to the numerical scheme used.

    Parameters
    ----------
    x : 1D array
       Input x axis.
    y : 1D array
       Input y axis
    scale : float, optional
       Distance between points (in pixels). Default is 1.
    npts : int, optional
       Number of points to use. Default is 100. If npts implies total length
       larger than the distance given by (x, y), then it is truncated.

    Returns
    -------
    result : 2D array
       Array of points. First index is coordinate (x, y), second index is
       point number.
    """
    from scipy import interpolate as interp
    incr = 0.01
    newy = np.arange(y.min(), y.max(), incr / scale)
    newx = interp.splev(newy, interp.splrep(y, x, k=order, s=3))
    res = []
    st = 0
    for i in range(npts):
        d = np.sqrt((newx - newx[st]) ** 2 + (newy - newy[st]) ** 2)
        idx = np.argmin(np.abs(d[st + 1:] - scale))
        st += idx
        res.append([newx[st], newy[st]])
        if (newx.shape[0] - st < scale / incr):   # limit of points reached
            break
    return np.array(res).T


def doppler_shift(wave, data, vel, order="linear"):
    """
    Doppler shifts a quantity that is a function of wavelength.

    Parameters
    ----------
    wave : 1-D array
        Wavelength values in nm.
    data : ndarray (1-D or 2-D)
        Data to shift. The last dimension should correspond to wavelength.
    vel : number or 1-D array
        Velocities in km/s.
    order : string, optional
        Interpolation order. Could be 'linear' (default), 'nearest', 'linear'
        'quadratic', 'cubic'. The last three refer to spline interpolation of
        first, second, and third order.

    Returns
    -------
    data_shift : ndarray
        Shifted values, same shape as data.
    """
    from scipy.constants import c
    from scipy.interpolate import interp1d
    wave_shift = wave * (1. + 1.e3 / c * vel)
    fill = {"linear": "extrapolate", "nearest": 0., "slinear": 0.,
            "quadratic": 0., "cubic": 0}
    f = interp1d(wave, data, kind=order, bounds_error=False,
                 fill_value=fill[order])
    return f(wave_shift)
