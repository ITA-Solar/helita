import numpy as np


def hist2d(x, y, nbins=30, norm=False, rx=0.08):
    ''' Computes the 2D histogram of the data and the x,y coordinates
    of the middle of the bins. OPTIONS; nbins (number of bins), norm,
    rx (range in x as % of the min/max, previous default 0.08).'''
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


def planck(w, T, units='cgs_AA'):
    ''' Returns the Planck function for wavelength in nm and T in Kelvin.
    Units depend on input:

    cgs_AA: erg s^-1 cm^-2 AA^-1 sr^-1
    cgs_nm: erg s^-1 cm^-2 nm^-1 sr^-1
    Hz    : J   s^-1 m^-2  Hz^-1 sr^-1

    If using the brightness temperature units, then w must be a single value
    (T can be an array).

    For solid angle integrated one must multiply it by pi.'''
    from scipy.constants import c, h, k

    JOULE_TO_ERG = 1.e7
    CM_TO_M = 1.e-2
    NM_TO_M = 1.e-9
    AA_TO_M = 1.e-10
    if units in ['cgs_AA', 'cgs_nm']:
        wave = w * 10.  # to AA
        c /= AA_TO_M
        h *= JOULE_TO_ERG
        k *= JOULE_TO_ERG
        iplanck = 2 * h * c**2 / wave**5 / (np.exp(h * c / (wave * k * T)) - 1)
        # convert from AA-2 to cm-2
        iplanck *= (1e8)**2
        if units == 'cgs_nm':
            iplanck *= 10.
    elif units == 'Hz':
        wave = w * NM_TO_M  # wave in m
        iplanck = 2 * h * c / wave**3 / (np.exp(h * c / (wave * k * T)) - 1)
    else:
        raise ValueError('planck: invalid units (%s)' % units)
    return iplanck


def int2bt(inu, w):
    ''' Converts from radiation intensity (in J s^-1 m^-2  Hz^-1 sr^-1 units)
        to brightness temperature units (in K), at a given wavelength wave
        (in nm). '''
    from scipy.constants import c, h, k
    return h * c / (w * 1e-9 * k * np.log(2 * h * c / (wave**3 * inu) + 1))


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


def voigt(a, v):
    ''' Returns the Voigt function:

    H(a,v) = a/pi * \int_{-Inf}^{+Inf} exp(-y**2)/[(v-y)**2 + a**2] dy

    Uses voigtv.f.

    IN:
    a -- scalar
    v -- scalar or 1D array

    OUT:
    h -- same dimensions as v.

    --Tiago, 20090728
    '''
    from voigtv import voigtv

    if hasattr(a, '__len__'):
        raise TypeError('voigt: a must be a scalar!')
    if hasattr(v, '__len__'):
        n = len(v)
    else:
        n = 1
    if a != 0:
        return voigtv(np.repeat(a, n), v)
    else:  # Gaussian limit
        return np.exp(-v**2)


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
    Tiago: downloaded from https://gist.github.com/1178136

    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks

    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)

    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value

    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
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
