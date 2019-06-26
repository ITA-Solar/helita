from scipy.odr import odrpack as odr
from scipy.odr import models
import numpy as np


def gaussian(B, x):
    ''' Returns the gaussian function for B=m,stdev,max,offset '''
    return B[3] + B[2] / (B[1] * np.sqrt(2 * np.pi)) * \
        np.exp(- ((x - B[0]) ** 2 / (2 * B[1] ** 2)))


def double_gaussian(B, x):
    """
    Returns a sum of two gaussian functions for
     B = mean1, stdev1, max1, mean2, stdev2, max2, offset
    """
    return B[2] / (B[1] * np.sqrt(2 * np.pi)) * np.exp(-((x - B[0])**2 / (2 * B[1]**2))) + \
           B[5] / (B[4] * np.sqrt(2 * np.pi)) * \
           np.exp(-((x - B[3])**2 / (2 * B[4]**2))) + B[6]


def sine(B, x):
    """
    Returns a sine function. Note that frequency is linear, not angular.
    """
    return B[0] * np.sin(2 * np.pi * B[1] * x + B[2]) + B[3]


def gauss_lsq(x, y, weight_x=1., weight_y=1., verbose=False, itmax=200,
              iparams=[]):
    """
    Performs a Gaussian least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.

    Parameters
    ----------
    x : 1D array-like
        Observed data, independent variable
    y : 1D array-like
        Observed data, dependent variable
    weight_x: array-like, optional
        Weights for independent variable. This is typically based on the errors,
        if any. With errors, normal weights should be 1/err**2. If weight is a
        scalar, the same weight will be used for all points and therefore its
        value is irrelevant.
    weight_y: array-like, optional.
        Weights for independent variable. This is typically based on the errors,
        if any. With errors, normal weights should be 1/err**2. For Poisson
        weighing, should be 1/y.
    verbose: boolean or int
        If True, will print out more detailed information about the result.
        If 2, will print out additional information.
    itmax: integer, Optional
        Maximum number of iterations, default is 200.
    iparams: list, optional
        Starting guess of Gaussian parameters. Optional but highly recommended
        to use realistic values!

    Returns
    -------
    output: tuple
        Tuple with containing (coeff, err, itlim), where coeff are the fit
        resulting coefficients (same order as Gaussian function above), err are
        the errors on each coefficient, and itlim is the number of iterations.

    Notes
    -----
    See documentation of scipy.odr.ordpack for more information.
    """

    def _gauss_fjd(B, x):
        # Analytical derivative of gaussian with respect to x
        return (B[0] - x) / B[1]**2 * gaussian(np.concatenate((B[:3], [0.])), x)

    def _gauss_fjb(B, x):
        gauss1 = gaussian(np.concatenate((B[:3], [0.])), x)
        # Analytical derivatives of gaussian with respect to parameters
        _ret = np.concatenate(((x - B[0]) / B[1]**2 * gauss1,
                               ((B[0] - x)**2 - B[1]**2) / B[1]**3 * gauss1,
                               gauss1 / B[2],
                               np.ones(x.shape, float)))
        _ret.shape = (4,) + x.shape
        return _ret

    # Centre data in mean(x) (makes better conditioned matrix)
    mx = np.mean(x)
    x2 = x - mx
    if not any(iparams):
        iparams = np.array([x2[np.argmax(y)], np.std(y),
                            np.sqrt(2 * np.pi) * np.std(y) * max(y), 1.])
    gauss = odr.Model(gaussian, fjacd=_gauss_fjd, fjacb=_gauss_fjb)
    mydata = odr.Data(x2, y, wd=weight_x, we=weight_y)
    myodr = odr.ODR(mydata, gauss, beta0=iparams, maxit=itmax)
    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)
    fit = myodr.run()
    # Display results:
    if verbose:
        fit.pprint()
        print('Re-centered Beta: [%f  %f  %f %f]' %
              (fit.beta[0] + mx, fit.beta[1], fit.beta[2], fit.beta[3]))
    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        print('(WWW) gauss_lsq: Iteration limit reached, result not reliable!')
    # Results and errors
    coeff = fit.beta
    coeff[0] += mx  # Recentre in original axis
    err = fit.sd_beta
    return coeff, err, itlim


def double_gauss_lsq(x, y, verbose=False, itmax=200, iparams=[]):
    """
    Performs a double Gaussian least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.

    Parameters
    ----------
    x, y : 1-D arrays
        Data to fit.
    verbose: boolean or int
        If True, will print out more detailed information about the result.
        If 2, will print out additional information.
    itmax : int, optional
        Maximum number of iterations.

    Returns
    -------
    coeff :  1-D array
        Polynomial coefficients, lowest order first.
    err :  1-D array
        Standard error (1-sigma) on the coefficients.
    """

    def _dgauss_fjd(B, x):
        # Analytical derivative of gaussian with respect to x
        return (B[0] - x) / B[1]**2 * gaussian(np.concatenate((B[:3], [0.])), x) + \
               (B[3] - x) / B[4]**2 * gaussian(np.concatenate((B[3:6], [0.])), x)

    def _dgauss_fjb(B, x):
        # Analytical derivatives of gaussian with respect to parameters
        gauss1 = gaussian(np.concatenate((B[:3], [0.])), x)
        gauss2 = gaussian(np.concatenate((B[3:6], [0.])), x)
        _ret = np.concatenate(((x - B[0]) / B[1]**2 * gauss1,
                               ((B[0] - x)**2 - B[1]**2) / B[1]**3 * gauss1,
                               gauss1 / B[2],
                               (x - B[3]) / B[4]**2 * gauss2,
                               ((B[3] - x)**2 - B[4]**2) / B[4]**3 * gauss2,
                               gauss2 / B[5],
                               np.ones(x.shape, float)))
        _ret.shape = (7,) + x.shape
        return _ret

    # Centre data in mean(x) (makes better conditioned matrix)
    mx = np.mean(x)
    x2 = x - mx
    if not any(iparams):
        iparams = np.array([x2[np.argmax(y)], np.std(y),
                            np.sqrt(2 * np.pi) * np.std(y) * max(y),
                            x2[np.argmax(y)], np.std(y),
                            np.sqrt(2 * np.pi) * np.std(y) * max(y), 1.])
    dgauss = odr.Model(double_gaussian, fjacd=_dgauss_fjd, fjacb=_dgauss_fjb)
    mydata = odr.Data(x2, y)
    myodr = odr.ODR(mydata, dgauss, beta0=iparams, maxit=itmax)
    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)
    fit = myodr.run()
    # Display results:
    if verbose:
        fit.pprint()
        print('Re-centered Beta: [%f  %f  %f  %f  %f  %f  %f]' %
              (fit.beta[0] + mx, fit.beta[1], fit.beta[2],
               fit.beta[3] + mx, fit.beta[4], fit.beta[5], fit.beta[6]))
    itlim = False
    if fit.stopreason[0] == 'Iteration limit reached':
        itlim = True
        print('(WWW) gauss_lsq: Iteration limit reached, result not reliable!')
    # Results and errors
    coeff = fit.beta
    coeff[[0, 3]] += mx  # Recentre in original axis
    err = fit.sd_beta
    return coeff, err, itlim


def poly_lsq(x, y, n, verbose=False, itmax=200):
    """
    Performs a polynomial least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.

    Parameters
    ----------
    x, y : 1-D arrays
        Data to fit.
    n : int
        Polynomial order
    verbose : bool or int, optional
        Can be 0,1,2 for different levels of output (False or True
        are the same as 0 or 1)
    itmax : int, optional
        Maximum number of iterations.

    Returns
    -------
    coeff :  1-D array
        Polynomial coefficients, lowest order first.
    err :  1-D array
        Standard error (1-sigma) on the coefficients.
    """
    func = models.polynomial(n)
    mydata = odr.Data(x, y)
    myodr = odr.ODR(mydata, func, maxit=itmax)
    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)
    fit = myodr.run()
    # Display results:
    if verbose:
        fit.pprint()
    if fit.stopreason[0] == 'Iteration limit reached':
        print('(WWW) poly_lsq: Iteration limit reached, result not reliable!')
    # Results and errors
    coeff = fit.beta
    err = fit.sd_beta
    return coeff, err


def quad_lsq(x, y, verbose=False, itmax=200, iparams=[]):
    """
    Fits a parabola to the data, more handy as it fits for
    parabola parameters in the form y = B_0 * (x - B_1)**2 + B_2.
    This is computationally slower than poly_lsq, so beware of its usage
    for time consuming operations. Uses scipy odrpack, but for least squares.

    Parameters
    ----------
    x, y : 1-D arrays
        Data to fit.
    verbose : bool or int, optional
        Can be 0,1,2 for different levels of output (False or True
        are the same as 0 or 1)
    itmax : int, optional
        Maximum number of iterations.
    iparams : 1D array, optional
        Initial parameters B_0, B_1, B_2.

    Returns
    -------
    coeff :  1-D array
        Parabola coefficients
    err :  1-D array
        Standard error (1-sigma) on the coefficients.
    """
    # Internal definition of quadratic
    def _quadratic(B, x):
        return B[0] * (x - B[1]) * (x - B[1]) + B[2]

    def _quad_fjd(B, x):
        return 2 * B[0] * (x - B[1])

    def _quad_fjb(B, x):
        _ret = np.concatenate((np.ones(x.shape, float),
                               2 * B[0] * (B[1] - x),
                               x * x - 2 * B[1] * x + B[1] * B[1],))
        _ret.shape = (3,) + x.shape
        return _ret

    if any(iparams):
        def _quad_est(data):
            return tuple(iparams)
    else:
        def _quad_est(data):
            return (1., 1., 1.)
    quadratic = odr.Model(_quadratic, fjacd=_quad_fjd, fjacb=_quad_fjb,
                          estimate=_quad_est)
    mydata = odr.Data(x, y)
    myodr = odr.ODR(mydata, quadratic, maxit=itmax)
    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)
    fit = myodr.run()
    # Display results:
    if verbose:
        fit.pprint()
    if fit.stopreason[0] == 'Iteration limit reached':
        print('(WWW) quad_lsq: iteration limit reached, result not reliable!')
    # Results and errors
    coeff = fit.beta
    err = fit.sd_beta
    return coeff, err


def circle_lsq(x, y, up=True, verbose=False, itmax=200, iparams=[]):
    """
    Method to compute a (half) circle fit, It fits for circle
    parameters in the form y = B_2 +/- sqrt(B_0^2 - (x - B_1)^2), the sign
    is negative if up is False.

    Parameters
    ----------
    x, y : 1-D arrays
        Data to fit.
    up : bool, optional
        Whether the half circle is up or down.
    verbose : bool or int, optional
        Can be 0,1,2 for different levels of output (False or True
        are the same as 0 or 1)
    itmax : int, optional
        Maximum number of iterations.
    iparams : 1D array, optional
        Initial parameters B_0, B_1, B_2.

    Returns
    -------
    coeff :  1-D array
        Parabola coefficients
    err :  1-D array
        Standard error (1-sigma) on the coefficients.
    """
    # circle functions for B=r,x0,y0
    def circle_up(B, x):
        return B[2] + sqrt(B[0]**2 - (x - B[1])**2)

    def circle_dn(B, x):
        return B[2] - sqrt(B[0]**2 - (x - B[1])**2)

    # Derivative of function in respect to x
    def circle_fjd_up(B, x):
        return -(x - B[1]) / (sqrt(B[0]**2 - (x - B[1])**2))

    def circle_fjd_dn(B, x):
        return (x - B[1]) / (sqrt(B[0]**2 - (x - B[1])**2))

    # Derivative of function in respect to B[i]
    def circle_fjb_up(B, x):
        _ret = np.concatenate((B[0] / (sqrt(B[0]**2 - (x - B[1])**2)),
                               - circle_fjd_up(B, x),
                               np.ones(x.shape, float),))
        _ret.shape = (3,) + x.shape
        return _ret

    def circle_fjb_dn(B, x):
        _ret = np.concatenate((B[0] / (sqrt(B[0]**2 - (x - B[1])**2)),
                               - circle_fjd_dn(B, x),
                               np.ones(x.shape, float),))
        _ret.shape = (3,) + x.shape
        return _ret

    if any(iparams):
        def circle_est(data):
            return tuple(iparams)
    else:
        def circle_est(data):
            return (1., 1., 1.)
    if up:
        circle_fit = odr.Model(circle_up, fjacd=circle_fjd_up, fjacb=circle_fjb_up,
                               estimate=circle_est)
    else:
        circle_fit = odr.Model(circle_dn, fjacd=circle_fjd_dn, fjacb=circle_fjb_dn,
                               estimate=circle_est)
    mydata = odr.Data(x, y)
    myodr = odr.ODR(mydata, circle_fit, maxit=itmax)
    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2:
        myodr.set_iprint(final=2)
    fit = myodr.run()
    # Display results:
    if verbose:
        fit.pprint()
    if fit.stopreason[0] != 'Sum of squares convergence':
        if verbose:
            print('(WWW): circle_lsq: fit result not reliable')
        success = 0
    else:
        success = 1
    # Results and errors
    coeff = fit.beta
    err = fit.sd_beta
    return coeff, success, err


def sine_lsq(x, y, iparams=None):
    """
    Fits a sine to the data.
    """
    from scipy.optimize import curve_fit

    def _sine(x, amp, freq, phase, offset):
        return amp * np.sin(2 * np.pi * freq * x + phase) + offset

    coeff, cov = curve_fit(_sine, x, y, p0=iparams)
    return coeff, np.sqrt(np.diag(cov))
