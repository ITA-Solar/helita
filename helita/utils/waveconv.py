import numpy as np


def waveconv(wave, mode='vac2air', gravred=False, verbose=False):
    ''' Converts given wavelength from air to vacuum (use mode=\'air2vac\')
    or vacuum to air (use mode=\'vac2air\'). Can also account for the
    Solar gravitational redshift (use gravred=True).
    Uses the formula from P. Ciddor, Applied Optics vol 35, no 9, 1566 (1996)
    20070306: Coded --Tiago
    '''
    # All these constants are in um^-2
    k = np.array([238.0185, 5792105., 57.362, 167917.], dtype='d')
    wn = np.array(1 / (1.e-3 * wave), dtype='d')  # to wave number in um^-1
    # Index of refraction of dry air (15deg C, 101325 Pa, 450 ppm CO2)
    n = 1. + 1.e-8 * (k[1] / (k[0] - wn**2) + k[3] / (k[2] - wn**2))
    if verbose:
        print('*** Index of refraction is %s' % n)
    if mode == 'air2vac':
        result = wave * n
    elif mode == 'vac2air':
        result = wave / n
    else:
        print('(EEE) waveconv: Mode not valid, exiting...')
        return 0
    # Account for solar gravitational redshift?
    if gravred:
        # Multiplying by the 636m/s velocity shift / vac. speed of light
        result += result * 636. / 299792458.
    return result


def waveconv_regner(wave, mode='vac2air', gravred=False, verbose=False):
    ''' Converts given wavelength from air to vacuum (use mode=\'air2vac\')
        or vacuum to air (use mode=\'vac2air\'). Can also account for the
        Solar gravitational redshift (use gravred=True).
        Based on Regner\'s IDL routines, just porting to a real programming
        language. Input wavelength MUST be in Angstroms!
        20070301: Coded --Tiago
        '''
    # Magic numbers that probably only Regner knows about...
    aa = np.array([2.71709889e-3, 2.72550686e-4, 1.21847797e2,
                  7.81341589e+2])
    if mode == 'vac2air':
        a = aa[1] + 1.
        b = aa[0] - wave - aa[3] * (1. + aa[1])
        c = aa[2] - aa[3] * (aa[0] - wave)
        d = b * b - 4. * a * c
        result = .5 * (np.sqrt(d) - b) / a
    elif mode == 'air2vac':
        result = wave + aa[0] + aa[1] * wave + aa[2] / (wave - aa[3])
    else:
        print('(EEE): waveconv_regner: Mode not valid, exiting...')
        return 0
    if verbose:
        print('*** Index of refraction is ' + str(wave / result))
    # Account for solar gravitational redshift?
    if gravred:
        # Multiplying by the 636m/s velocity shift / vac. speed of light
        result += result * 636. / 299792458.
    return result
