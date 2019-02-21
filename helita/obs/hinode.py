"""
set of tools to deal with Hinode observations
"""
import os
import numpy as np
from pkg_resources import resource_filename


def bfi_filter(wave, band='CAH', norm=True):
    """
    Returns the BFI filter transmission profile for a given wavelength grid,
    from data extracted from solarsoft.

    Parameters:
    -----------
    wave - 1D array
         Wavelength values (for interpolation)
    band - string, optional
         Band to use, one of 'CN', 'CAH', 'GBAND', 'BLUE', 'GREEN', 'RED'
    norm - bool
         Defines weather resulting filter is normalised (ie, has unit area,
         NOT sum)

    Returns:
    --------
    wfilt - 1D array
         Array with wavelength filter.
    """
    from scipy import interpolate as interp
    band = band.upper()
    filt_names = {'CN': '3883', 'CAH': '3968', 'GBAND': '4305',
                  'BLUE': '4504', 'GREEN': '5550', 'RED': '6684'}
    if band not in list(filt_names.keys()):
        msg = "Band name must be one of %s" % ', '.join(filt_names.keys())
        raise(ValueError, "Invalid band. " + msg + ".")
    cfile = resource_filename('helita',
                              'data/BFI_filter_%s.txt' % filt_names[band])
    wave_filt, filt = np.loadtxt(cfile, unpack=True)
    f = interp.interp1d(wave_filt, filt, bounds_error=False, fill_value=0)
    wfilt = f(wave)
    if norm:
        widx = wfilt != 0
        wfilt /= np.trapz(wfilt[widx], x=wave[widx])
    return wfilt
