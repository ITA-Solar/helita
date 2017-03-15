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


def gauss_lsq(x, y, verbose=False, itmax=200, iparams=[]):
    ''' Performs a Gaussian least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.'''

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
    mx = mean(x)
    x2 = x - mx
    if not any(iparams):
        iparams = array([x2[argmax(y)], np.std(y),
                         np.sqrt(2 * np.pi) * np.std(y) * max(y), 1.])
    gauss = odr.Model(gaussian, fjacd=_gauss_fjd, fjacb=_gauss_fjb)
    mydata = odr.Data(x2, y)
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
    ''' Performs a double gaussian least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.'''

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
    mx = mean(x)
    x2 = x - mx
    if not any(iparams):
        iparams = array([x2[argmax(y)], np.std(y),
                         np.sqrt(2 * np.pi) * np.std(y) * max(y),
                         x2[argmax(y)], np.std(y),
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
    ''' Performs a polynomial least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.

    IN:
       x,y (arrays) - data to fit
       n (int)      - polinomial order
       verbose      - can be 0,1,2 for different levels of output
                      (False or True are the same as 0 or 1)
       itmax (int)  - optional maximum number of iterations

    OUT:
       coeff -  polynomial coefficients, lowest order first
       err   - standard error (1-sigma) on the coefficients

    --Tiago, 20071114
    '''
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
    ''' Method to compute a parabola fit, more handy as it fits for
    parabola parameters in the form y = B_0 * (x - B_1)**2 + B_2.
    This is computationally slower than poly_lsq, so beware of its usage
    for time consuming operations.

    IN:
       x,y (arr)     - data to fit
       n (int)       - polinomial order
       verbose       - can be 0,1,2 for different levels of output
                         (False or True are the same as 0 or 1)
       itmax (int)   - optional maximum number of iterations.
       iparams (arr) - optional initial parameters b0,b1,b2

    OUT:
       coeff -  polynomial coefficients, lowest order first
       err   - standard error (1-sigma) on the coefficients


    --Tiago, 20071115
    '''

    # Tiago's internal new definition of quadratic
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
    ''' Method to compute a circle fit, It fits for circle
    parameters in the form y = B_2 +/- sqrt(B_0^2-(x-B_1)^2), the sign
    is negative if up is False.

    IN:
       x,y (arr)     - data to fit
       n (int)       - polinomial order
       verbose       - can be 0,1,2 for different levels of output
                         (False or True are the same as 0 or 1)
       itmax (int)   - optional maximum number of iterations.
       iparams (arr) - optional initial parameters b0,b1,b2

    OUT:
       coeff -  polynomial coefficients, lowest order first
       err   - standard error (1-sigma) on the coefficients

    --Tiago, 20080120
    '''
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
