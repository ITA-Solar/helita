"""
Set of programs to degrade/convolve synthetic images/spectra to observational
conditions
"""
import math
import os
import scipy.interpolate as interp
import numpy as np
from scipy import ndimage, signal


def spec_conv(spec, wave, conv_type='IRIS', ww=None, wpts=200, winterp='linear',
              xMm=16.5491, graph=False, lscale=1.):
    ''' Convolves a 2D spectrogram to observational conditions (both on the
        spatial and spectral axes)

    IN:
      spec:       2D array, wavelength on last axis
      wave:       wavelength array, could be irregular
      conv_type:  string, type of convolution to make: 'IRIS', 'SUMI', etc.
      ww:         2-element list, tuple containing the first and last
                  wavelengths to be used in the wavelength interpolation.
                  If unset, will use the min and max of wave.
      wpts:       integer, number of interpolated wavelength points.
      winterp:    string, type of wavelength interpolation for interp1d
                  ('linear', 'cubic', etc.)
      xMm:        physical size (in Mm) of spatial dimension
      graph:      if true, will show images
      lscale:     luminance scale for images

    OUT:
      nspec:      2D array of new spectrogram.

    --Tiago, 20110819
    '''
    if graph:
        import matplotlib.pyplot as p
        vmin = np.min(spec)
        vmax = np.max(spec) * lscale
    # Convolution parameters. This is a 4-element list with the following:
    #  [spatial res. FWHM (arcsec), spatial pixel size (arcsec),
    #   spectral res FWHM (nm), spectral pixel size (nm)]
    cp = {'IRIS': [0.4, 0.167, 0.008, 0.0025],
          'SUMI': [1.,   1.,  0.0043, 0.0022],
          'SUMI2': [2.,   1.,  0.0043, 0.0022],
          'SUMI3': [3.,   1.,  0.0043, 0.0022]}
    if conv_type not in list(cp.keys()):
        raise ValueError('Invalid convolution type'
                         ' %s. Supported values are %s' %
                         (conv_type, str([a for a in list(cp.keys())])))
    pix2asec = xMm / (spec.shape[0] * 696. / 959.9)  # from pixels to arcsec
    # wavelength interpolation
    if ww is None:
        ww = (np.min(wave), np.max(wave))
    nwave = np.arange(ww[0], ww[1], cp[conv_type][3] / 3.)
    f = interp.interp1d(wave, spec, kind=winterp)
    nspec = f(nwave)
    # convolution
    wstep = nwave[1] - nwave[0]
    wsigma = cp[conv_type][2] / \
        (wstep * 2 * math.sqrt(2 * math.log(2)))  # from fwhm to sigma
    dstep = pix2asec
    dsigma = cp[conv_type][0] / (dstep * 2 * math.sqrt(2 * math.log(2)))
    nspec = ndimage.gaussian_filter(nspec, [dsigma, wsigma])
    if graph:
        p.subplot(121)
        p.pcolormesh(wave, np.arange(spec.shape[0]) * pix2asec, spec,
                     shading='gouraud', cmap=p.cm.gist_gray, vmin=vmin,
                     vmax=vmax)
        p.xlim(ww[0], ww[1])
        p.ylim(0, spec.shape[0] * pix2asec)
    # pixelisation
    coords = np.mgrid[0.: spec.shape[0]:cp[conv_type][1] / pix2asec,
                      0.:nwave.shape[0]:cp[conv_type][3] / wstep]
    nspec = ndimage.map_coordinates(nspec, coords, order=1, mode='nearest')
    nwave = interp.interp1d(np.arange(nwave.shape[0]), nwave)(
        coords[1][0].astype('i'))
    if graph:
        p.subplot(122)
        p.imshow(nspec,
                 extent=(nwave[0], nwave[-1], 0, spec.shape[0] * pix2asec),
                 cmap=p.cm.gist_gray, aspect='auto', interpolation='nearest',
                 vmin=vmin, vmax=vmax)
        p.xlim(ww[0], ww[1])
        p.ylim(0, spec.shape[0] * pix2asec)
    return nspec, nwave


def spec3d_conv(spec, wave, conv_type='IRIS', ww=None, wpts=200,
                winterp='linear', xMm=16.5491):
    ''' Convolves a 3D spectrogram to observational conditions (both on the
        spatial andspectral axes)

    IN:
      spec:       3D array, wavelength on last axis
      wave:       wavelength array, could be irregular
      conv_type:  string, type of convolution to make: 'IRIS', 'SUMI', etc.
      ww:         2-element list, tuple containing the first and last
                  wavelengths to be used in the wavelength interpolation.
                  If unset, will use the min and max of wave.
      wpts:       integer, number of interpolated wavelength points.
      winterp:    string, type of wavelength interpolation for interp1d
                  ('linear', 'cubic', etc.)
      xMm:        physical size (in Mm) of spatial dimension
      graph:      if true, will show images
      lscale:     luminance scale for images

    OUT:
      nspec:      3D array of new spectrogram.

    --Tiago, 20110819
    '''
    # Convolution parameters. This is a 4-element list with the following:
    #  [spatial res. FWHM (arcsec), spatial pixel size (arcsec),
    #   spectral res FWHM (nm), spectral pixel size (nm)]
    cp = {'IRIS': [0.4, 0.167, 0.008, 0.0025],
          'SUMI': [1.,   1.,  0.0043, 0.0022],
          'SUMI2': [2.,   1.,  0.0043, 0.0022],
          'SUMI3': [3.,   1.,  0.0043, 0.0022]}

    if conv_type not in list(cp.keys()):
        raise ValueError('Invalid convolution type %s. Supported values are %s' %
                         (conv_type, str([a for a in list(cp.keys())])))
    pix2asec = xMm / (spec.shape[0] * 696. / 959.9)  # from pixels to arcsec
    nwave = spec.shape[-1]
    # Spatial convolution
    dstep = pix2asec
    dsigma = cp[conv_type][0] / (dstep * 2 * math.sqrt(2 * math.log(2)))
    #nspec = ndimage.gaussian_filter(nspec,[dsigma,dsigma,wsigma])
    for w in range(nwave):
        spec[:, :, w] = ndimage.gaussian_filter(spec[:, :, w], dsigma)
    # Spatial pixelisation
    coords = np.mgrid[0.: spec.shape[0]:cp[conv_type][1] / pix2asec,
                      0.: spec.shape[1]:cp[conv_type][1] / pix2asec]
    nspec = np.empty(coords.shape[1:] + (nwave,), dtype='Float32')
    for w in range(nwave):
        nspec[:, :, w] = ndimage.map_coordinates(
            spec[:, :, w], coords, order=1, mode='nearest')
    # Spectral convolution
    # wavelength interpolation to fixed scale
    if ww is None:
        ww = (np.min(wave), np.max(wave))
    nwave = np.arange(ww[0], ww[1], cp[conv_type][3] / 3.)
    f = interp.interp1d(wave, nspec, kind=winterp)
    nspec = f(nwave)
    # convolve with Gaussian
    wstep = nwave[1] - nwave[0]
    wsigma = cp[conv_type][2] / \
        (wstep * 2 * math.sqrt(2 * math.log(2)))  # from fwhm to sigma
    nspec = ndimage.gaussian_filter1d(nspec, wsigma, axis=-1, mode='nearest')
    # Spectral pixelisation
    nspec = nspec[:, :, ::3]
    return nspec, nwave[::3]


def img_conv(spec, wave, psf, psfx, conv_type='IRIS_MgII_core', xMm=16.5491,
             wfilt=None, graph=False, lscale=1., pixelise=True):
    ''' Convolves a 3D spectrogram to observational slit-jaw conditions
        (does spatial convolution and pixelisation, and spectral filtering)

    IN:
      spec:       3D array, wavelength on last axis
      wave:       wavelength array, could be irregular
      conv_type:  string, type of convolution to make: 'IRIS',
      xMm:        physical size (in Mm) of first spatial dimension
      graph:      if true, will show images
      lscale:     luminance scale for images

    OUT:
      nspec:      2D array of image.

    --Tiago, 20110820
    '''
    from ..utils.fitting import gaussian

    if graph:
        import matplotlib.pyplot as p
    # some definitions
    asec2Mm = 696. / 959.5         # conversion between arcsec and Mm
    pix2Mm = xMm / spec.shape[0]   # size of simulation's pixels in Mm
    pix2asec = pix2Mm / asec2Mm    # from pixels to arcsec
    # Convolution parameters. This is a 4-element list with the following:
    #  [spatial res. FWHM (arcsec), spatial pixel size (arcsec),
    #   spectral central wavelength (nm), spectral FWHM (nm)]
    cp = {'IRIS_MGII_CORE': [0.4, 0.166, 279.518, 0.4],  # 279.6 nm in vac
          'IRIS_MGII_WING': [0.4, 0.166, 283.017, 0.4],  # 283.1 nm in vac
          'IRIS_CII':       [0.4, 0.166, 133.279, 4.0],  # 133.5 nm in vac
          'IRIS_SIV':       [0.4, 0.166, 139.912, 4.0],  # 140.0 nm in vac
          'IRIS_TEST':      [0.4, 0.166, 279.518, 1.5],
          # scaled IRIS resolution to Hinode Ca H
          'IRIS_HINODE_CAH': [0.4 * 397 / 280., 0.166, 396.85, 0.3],
          # scaled IRIS resolution to Hinode red BFI
          'IRIS_HINODE_450': [0.4 * 450 / 280., 0.166, 450.45, 0.4],
          # scaled IRIS resolution to Hinode green BFI
          'IRIS_HINODE_555': [0.4 * 555 / 280., 0.166, 555.05, 0.4],
          # scaled IRIS resolution to Hinode blue BFI
          'IRIS_HINODE_668': [0.4 * 668 / 280., 0.166, 668.40, 0.4]}
    conv_type = conv_type.upper()
    if wfilt is None:
        wcent = cp[conv_type][2]
        wfwhm = cp[conv_type][3]
        # selecting wavelengths within 4 FWHM
        widx = (wave[:] > wcent - 2. * wfwhm) & (wave[:] < wcent + 2. * wfwhm)
        # filtering function, here set to Gaussian
        wfilt = gaussian([wcent, wfwhm / (2 * math.sqrt(2 * math.log(2))),
                                                        1., 0.], wave[widx])
        wfilt /= np.trapz(wfilt, x=wave[widx])
    else:
        widx = wfilt != 0
        wfilt = wfilt[widx]
    # multiply by filter and integrate
    nspec = np.trapz(spec[:, :, widx] * wfilt, x=wave[widx], axis=-1)
    if graph:
        vmin = np.min(nspec)
        vmax = np.max(nspec) * lscale
        p.subplot(211)
        aa = nspec
        if hasattr(aa, 'mask'):
            aa[aa.mask] = np.mean(nspec)
        p.imshow(np.transpose(aa), extent=(0, spec.shape[0] * pix2asec, 0,
                                           spec.shape[1] * pix2asec),
                 vmin=vmin, vmax=vmax, cmap=p.cm.gist_gray)
        p.title('Filter only')
        p.xlabel('arcsec')
        p.ylabel('arcsec')
    # spatial convolution
    psf_x = psfx * asec2Mm
    sep = np.mean(psf_x[1:] - psf_x[:-1])
    coords = np.mgrid[0: psf_x.shape[0]: pix2Mm / sep,
                      0: psf_x.shape[0]: pix2Mm / sep]
    npsf = ndimage.map_coordinates(psf, coords, order=1, mode='nearest')
    npsf /= np.sum(npsf)
    im = np.concatenate([nspec[-50:], nspec, nspec[:50]])
    im = np.concatenate([im[:, -50:], im, im[:, :50]], axis=1)
    nspec = signal.fftconvolve(im, npsf, mode='same')[50:-50, 50:-50]
    # pixelisation
    if pixelise:
        coords = np.mgrid[0.:spec.shape[0]:cp[conv_type][1] / pix2asec,
                          0.:spec.shape[1]:cp[conv_type][1] / pix2asec]
        nspec = ndimage.map_coordinates(nspec, coords, order=1, mode='nearest')
    if graph:
        p.subplot(212)
        p.imshow(np.transpose(nspec),
                              extent=(0, spec.shape[0] * pix2asec, 0,
                                      spec.shape[1] * pix2asec),
                              vmin=vmin, vmax=vmax,
                 interpolation='nearest', cmap=p.cm.gist_gray)
        p.title('Filter + convolved %s' % (conv_type))
        p.xlabel('arcsec')
        p.ylabel('arcsec')
    return nspec


def get_hinode_psf(wave, psfdir='.'):
    """
    Gets the Hinode PSF (from Sven Wedemeyer's work) for a given
    wavelength in nm. Assumes Hinode's ideal PSF is on psfdir.
    Returns x scale (in arcsec), and psf (2D array, normalised).
    """
    from astropy.io import fits as pyfits
    from ..utils import utilsmath
    # Get ideal PSF
    ipsf = pyfits.getdata(os.path.join(psfdir, 'hinode_ideal_psf_555nm.fits'))
    ix = pyfits.getdata(os.path.join(psfdir, 'hinode_ideal_psf_scale_555nm.fits'))
    # Scale ideal PSF to our wavelength and simulation pixels
    cwave = np.mean(wave)  # our wavelength
    ix *= cwave / 555.
    sep = np.mean(ix[1:] - ix[:-1])
    # Get non-ideal PSF for our wavelength
    # these are the tabulated values in Wedemeyer-Boem , A&A 487, 399 (2008),
    # for voigt function and Mercury transit
    gamma_data = [4., 5., 6.]
    sigma = 8.
    wave_data = [450.45, 555., 668.4]
    # interpolate gamma for our wavelength
    gamma = interp.interp1d(wave_data, gamma_data, kind='linear')(cwave)
    gamma *= 1e-3
    sigma *= 1e-3
    uu = np.mgrid[-sep * 600: sep * 599: sep, -sep * 600: sep * 599: sep]
    xm = np.arange(-sep * 600, sep * 599, sep)
    r = np.sqrt(uu[0]**2 + uu[1]**2)
    rf = np.ravel(r)
    npsf = np.reshape(utilsmath.voigt(a, rf / b) /
                      (b * np.sqrt(np.pi)), (xm.shape[0], xm.shape[0]))
    # Convolve ideal PSF with non-ideal PSF
    psf = signal.fftconvolve(ipsf, npsf, mode='same')
    # Recentre from convolution, remove pixels outside non-ideal PSF kernel
    ix = ix[:-1][2000:-2000]
    psf = psf[1:, 1:][2000:-2000, 2000:-2000]
    return ix, psf


def spectral_convolve(w):
    '''
    Spectral convolution function for imgspec_conv. Interpolates to new
    wavelength and does a gaussian convolution in the last index of the
    spectrum array. Input is a single tuple to make it easier to
    parallelise with multiprocessing. Requires the existence of a global
    array called result.
    '''
    i, wsigma, wave, nwave, spec = w   # get arguments from tuple
    f = interp.interp1d(wave, spec, kind='linear')
    spec = f(nwave)
    result[i] = ndimage.gaussian_filter1d(spec, wsigma, axis=-1,
                                          mode='nearest')


def spatial_convolve(w):
    '''
    Spatial convolution function for imgspec_conv. Does a convolution with
    given psf in Fourier space. Input is a single tuple to make it easier to
    parallelise with multiprocessing. Requires the existence of a global
    array called result.
    '''
    i, im, psf = w
    result[:, :, i] = signal.fftconvolve(im, psf, mode='same')[50:-50, 50:-50]


def var_conv(var, xMm, psf, psfx, obs='iris_nuv', parallel=False,
             pixelise=False, mean2=False):
    """
    Spatially convolves a single atmos variable.
    """
    import multiprocessing
    import ctypes
    global result

    # some definitions
    asec2Mm = 696. / 959.5         # conversion between arcsec and Mm
    pix2Mm = xMm / var.shape[0]    # size of simulation's pixels in Mm
    if obs.lower() == 'hinode_sp':
        obs_pix2Mm = 0.16 * asec2Mm    # size of instrument spatial pixels in Mm
    elif obs.lower() == 'iris_nuv':
        obs_pix2Mm = 0.166 * asec2Mm   # size of instrument spatial pixels in Mm
    nwave = var.shape[-1]   # This is really depth, not wavelength...
    # convert PSF kernel to the spectrogram's pixel scale
    psf_x = psfx * asec2Mm
    sep = np.mean(psf_x[1:] - psf_x[:-1])
    coords = np.mgrid[0: psf_x.shape[0]: pix2Mm / sep,
                      0: psf_x.shape[0]: pix2Mm / sep]
    npsf = ndimage.map_coordinates(psf, coords, order=1, mode='nearest')
    npsf /= np.sum(npsf)
    im = np.concatenate([var[-50:], var, var[:50]])
    im = np.concatenate([im[:, -50:], im, im[:, :50]], axis=1)
    itr = ((i, im[:, :, i], npsf) for i in range(nwave))

    # Spatial convolution
    if parallel:
        # multiprocessing shared object to collect output
        result_base = multiprocessing.Array(ctypes.c_float, np.prod(var.shape))
        result = np.ctypeslib.as_array(result_base.get_obj())
        result = result.reshape(var.shape)
        pool = multiprocessing.Pool()       # by default use all CPUs
        pool.map(spatial_convolve, itr)
        pool.close()
        pool.join()
    else:
        result = np.empty_like(var)
        for w in itr:
            spatial_convolve(w)

    # Spatial pixelisation
    if pixelise:
        coords = np.mgrid[0.: var.shape[0]: obs_pix2Mm / pix2Mm,
                          0.: var.shape[1]: obs_pix2Mm / pix2Mm]
        nvar = np.empty(coords.shape[1:] + (nwave,), dtype='Float32')
        for w in range(nwave):
            nvar[:, :, w] = ndimage.map_coordinates(result[:, :, w], coords,
                                                    order=1, mode='nearest')
        if mean2:
            # average 2 pixels along second dimension
            si = nvar.shape
            nvar = np.reshape(nvar, (si[0], si[1] / 2, 2, si[2])).mean(2)
    else:
        nvar = result[:]
    return nvar


def imgspec_conv(spec, wave, xMm, psf, psfx, obs='hinode_sp', verbose=False,
                 pixelise=True, parallel=False, mean2=False):
    '''
    Convolves a 3D spectrogram to observational conditions (does spatial
    convolution, spectral convolution and pixelisation, in that order)

    IN:
      spec:       3D array, wavelength on last axis
      wave:       wavelength array, could be irregular
      xMm:        physical size (in Mm) of first spatial dimension
      psf:        2D array with PSF
      psf_x:      1D array with PSF radial coordinates in arcsec
      obs:        type of observations. Options: 'hinode_sp', 'iris_nuv'.
      parallel:   if True, will run in parallel using all available CPUs
      pixelise:   if True, will pixelise into the observational conditions
      mean2:      if True and pixelise is True, will average every 2 pixels on
                  second dimension (to mimick the size of IRIS's slit width)

    OUT:
      nspec:      3D array of spectrogram.
      nwave:      1D array of resulting wavelength

    --Tiago, 20120105
    '''
    import multiprocessing
    import ctypes
    global result

    # some definitions
    asec2Mm = 696. / 959.5         # conversion between arcsec and Mm
    pix2Mm = xMm / spec.shape[0]   # size of simulation's pixels in Mm
    if obs.lower() == 'hinode_sp':
        obs_pix2Mm = 0.16 * asec2Mm    # size of instrument spatial pixels in Mm
        obs_pix2nm = 0.00215         # size of instrument spectral pixels in nm
        # instrument spectral resolution (Gaussian FWHM in nm)
        obs_spect_res = 0.0025
    elif obs.lower() == 'iris_nuv':
        # Tiago: updated to Iris Technical Note 1
        obs_pix2Mm = 0.166 * asec2Mm   # size of instrument spatial pixels in Mm
        obs_pix2nm = 0.002546        # size of instrument spectral pixels in nm
        # instrument spectral resolution (Gaussian FWHM in nm)
        obs_spect_res = 0.0060
        wavei = 278.1779
        wavef = 283.3067 + obs_pix2nm
    elif obs.lower() == 'iris_fuvcont':
        obs_pix2Mm = 0.166 * asec2Mm   # size of instrument spatial pixels in Mm
        obs_pix2nm = 0.002546 / 2.   # size of instrument spectral pixels in nm
        # instrument spectral resolution (Gaussian FWHM in nm)
        obs_spect_res = 0.0026
        wavei = 132.9673
        wavef = 141.6682 + obs_pix2nm
    else:
        raise ValueError('imgspec_conv: unsupported instrument %s' % obs)
    nwave = spec.shape[-1]
    # Spatial convolution
    if verbose:
        print('Spatial convolution...')
    # convert PSF kernel to the spectrogram's pixel scale
    psf_x = psfx * asec2Mm
    sep = np.mean(psf_x[1:] - psf_x[:-1])
    coords = np.mgrid[0: psf_x.shape[0]: pix2Mm / sep,
                      0: psf_x.shape[0]: pix2Mm / sep]
    npsf = ndimage.map_coordinates(psf, coords, order=1, mode='nearest')
    npsf /= np.sum(npsf)
    im = np.concatenate([spec[-50:], spec, spec[:50]])
    im = np.concatenate([im[:, -50:], im, im[:, :50]], axis=1)
    itr = ((i, im[:, :, i], npsf) for i in range(nwave))

    if parallel:
        # multiprocessing shared object to collect output
        result_base = multiprocessing.Array(
            ctypes.c_float, np.prod(spec.shape))
        result = np.ctypeslib.as_array(result_base.get_obj())
        result = result.reshape(spec.shape)
        pool = multiprocessing.Pool(parallel)       # by default use all CPUs
        pool.map(spatial_convolve, itr)
        pool.close()
        pool.join()
    else:
        result = np.empty_like(spec)
        for w in itr:
            spatial_convolve(w)

    # Spatial pixelisation
    if pixelise:
        if verbose:
            print('Spatial pixelisation...')
        coords = np.mgrid[0.: spec.shape[0]: obs_pix2Mm / pix2Mm,
                          0.: spec.shape[1]: obs_pix2Mm / pix2Mm]
        nspec = np.empty(coords.shape[1:] + (nwave,), dtype='Float32')
        for w in range(nwave):
            nspec[:, :, w] = ndimage.map_coordinates(result[:, :, w], coords,
                                                     order=1, mode='nearest')
        if mean2:
            # average 2 pixels along second dimension
            si = nspec.shape
            nspec = np.reshape(nspec, (si[0], si[1] / 2, 2, si[2])).mean(2)
    else:
        nspec = result[:]
    # Spectral convolution
    if verbose:
        print('Spectral convolution...')
    if obs.lower() in ['iris_nuv', 'iris_fuvcont']:
        nwave = np.arange(wavei, wavef, obs_pix2nm / 3.)
    else:
        nwave = np.arange(wave[0], wave[-1], obs_pix2nm / 3.)
    wstep = nwave[1] - nwave[0]
    wsigma = obs_spect_res / \
        (wstep * 2 * math.sqrt(2 * math.log(2)))  # fwhm to sigma
    itr = ((i, wsigma, wave, nwave, nspec[i]) for i in range(nspec.shape[0]))

    if parallel:
        result_base = multiprocessing.Array(ctypes.c_float,
                                            np.prod(nspec.shape[:-1]) *
                                            len(nwave))
        result = np.ctypeslib.as_array(result_base.get_obj())
        result = result.reshape(nspec.shape[:-1] + nwave.shape)
        pool = multiprocessing.Pool()        # by default use all CPUs
        pool.map(spectral_convolve, itr)
        pool.close()
        pool.join()
    else:
        result = np.empty(nspec.shape[:-1] + nwave.shape, dtype='f')
        for w in itr:
            spectral_convolve(w)

    # Spectral pixelisation
    nspec = result[:, :, ::3]
    return nspec
