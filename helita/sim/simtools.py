"""
Tools to use with the simulation's syntetic spectra
"""
import numpy as np
from scipy import ndimage
import scipy.interpolate as interp


def psf_diffr(ang, wave=777, D=1., pix=True):
    ''' Returns the diffraction PSF (Airy disc) for a circular
    aperture telescope. See R. Wilson, Reflecting Telescope Optics I,
    pp. 287-289.

    IN:

    ang  - angular coordinate in arcsecs, unless pix is True (then in
           simulation pixels)
    wave - wavelength (in nanometers)
    D    - telescope diameter (in meters)
    pix  - if true, then angular coordinate is read in simulation pixels
           (assuming 50x50x82: 50 pixels are 6 Mm)
    '''
    from scipy import special as sp

    a = ang * 1.
    if pix:
        solr = 696.  # solar radius in Mm
        sold = 959.5  # solar angular radius in arcsec
        a *= 6 / 50. * sold / solr
    # Put x in rad, then in normalized angle
    a *= np.pi / 648000
    a *= np.pi * D / (wave * 1e-9)
    # Remove zeros, both array and non array cases
    try:
        if a == 0:
            a = 1e-20
    except ValueError:
        a[np.where(a == 0)] = 1e-20
    # Airy disk function
    return (2 * sp.j1(a) / a)**2


def psf_atm(x, a=1, b=1.):
    '''Atmospheric PSF, similar to the one used in Shelyag et al. (2003),
    with influences from Nordlund (1984), Collados & Vazquez (1987).

    IN:
    x - angular coordinate
    a - parameter defining the width of the distribution
    b - height of the distribution
    '''
    # return b * a**3/(np.sqrt(x**2+a**2))**3 # initial function
    return a / (x**2 + a**2) + b / (x**2 + b**2)  # pure lorentzian


def psf_kernel(a, b=0.1, n=100, mu=1., phi=0., norm=True, threshold=1e-2,
               minpts=11):
    ''' Returns a lorentzian shaped 2D centered matrix kernel
    to use in convolutions.

    IN:

    a  - lorentzian fwhm
    n  - size of the square matrix (should be even, so that the matrix
        has n-1 even number, better for symmetry)
    b  - if different than zero, kernel won\'t conserve intensity
    mu - to be used in simulations with different mu values
    threshold - value from which smaller values are ignored in the psf

    --Tiago, 20080130
    '''
    a = float(a)
    b = float(b)
    kernel = np.zeros((n - 1, n - 1))
    kernel_atm = np.zeros((n - 1, n - 1))
    kernel_dif = np.zeros((n - 1, n - 1))
    for i in range(n - 1):  # This is the axis where mu dist will occur
        for j in range(n - 1):
            # mu acts on the x axis, convention of phi=0 is along x axis
            r = np.sqrt(mu * (i - n / 2 + 1)**2 + (j - n / 2 + 1)**2)
            #kernel[i,j] = psf_diffr(r) + psf_atm(r,a,b)
            # new way, separate both components and then convolve them
            kernel_atm[i, j] = psf_atm(r, a, b)
            kernel_dif[i, j] = psf_diffr(r)
    #kernel = ndimage.convolve(kernel_atm,kernel_dif)
    if norm:
        kernel /= np.sum(kernel)
        kernel_atm /= np.sum(kernel_atm)
        kernel_dif /= np.sum(kernel_dif)
    # if phi is nonzero, rotate the matrix:
    if phi != 0 and mu != 1:
        # phi in degrees
        kernel = ndimage.rotate(kernel, phi, reshape=False)
        kernel_atm = ndimage.rotate(kernel_atm, phi, reshape=False)
        kernel_dif = ndimage.rotate(kernel_dif, phi, reshape=False)
    # Select elements contributing less than 0.1% to the middle row integral
    # first for kernel_atm
    kernel_atm = psf_trim(kernel_atm, threshold, minpts=minpts)
    # second for kernel_dif
    # kernel_dif = psf_trim(kernel_dif,threshold,5) # not essential
    # convolve with diffraction PSF
    kernel = ndimage.convolve(kernel_atm, kernel_dif)
    if norm:
        kernel /= np.sum(kernel)
    return kernel


def psf_trim(psf, threshold, minpts=11):
    ''' Trims PSF by removing elements contributing less than the threshold
        fraction to the middle row integral. The minimum number of points for
        the resulting psf is minpts.'''
    n = psf.shape[0]
    (stix, stiy) = (0, 0)
    # find x cutoff value
    uu = np.cumsum(psf[:, n // 2]) / np.sum(psf[:, n // 2])
    if np.any(np.where(uu < threshold)):
        stix = np.max(np.where(uu < threshold))
        # leave at least minpts elements
        if stix > n // 2 - minpts // 2:
            stix = n // 2 - minpts // 2
    # find y cutoff value
    uu = np.cumsum(psf[n // 2, :]) / np.sum(psf[n // 2, :])
    if np.any(np.where(uu < threshold)):
        stiy = np.max(np.where(uu < threshold))
        # leave at least minpts elements
        if stiy > n // 2 - minpts // 2:
            stiy = n // 2 - minpts // 2
    return psf[stix:n - stix, stiy:n - stiy]


def gaussconv(spec, wave, resolution, fixed=False):
    ''' Convolves spectra with a gaussian, given a resolution
    and wavelength array.

    IN:
    spec - spectrum array (can be 1D, 2D or nD as long as last dimension is wave)
    wave - wavelength array
    resolution - resolving power in dw/w
    fixed - if true, will treat resolution as fixed FWHM (in wave units)

    OUT:
    convolved spectrum

    --Tiago, 20080201
    '''
    ishp = spec.shape
    if len(ishp) > 1:
        # Make the spectrum a 2D array [spatial point,wave point]
        a = ishp[0]
        for i in range(1, len(ishp) - 1):
            a *= ishp[i]
        nspec = np.reshape(spec, (a, ishp[-1]))
    else:
        nspec = np.array([spec])
    out = np.zeros(nspec.shape)
    # mean wavelengh step
    step = abs(np.mean(wave[1:] - wave[:-1]))
    # note: this fwhm and sigma are in 'pixels' (or array units),
    # hence the need to divide by step
    if not fixed:
        fwhm = np.mean(wave) / (resolution * step)
    else:  # use a fixed fwhm, given by resolution in wavelength units
        fwhm = resolution / step
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    for n in range(nspec.shape[0]):
        out[n] = ndimage.gaussian_filter1d(nspec[n], sigma)
    if len(ishp) > 1:
        out.shape = ishp
    else:
        out = out[0]
    return out


def sincconv(spec, wavein, resolution, fine=True, lobes=100):
    ''' Convolves spectra with a sinc function, given a resolution
    and wavelength array.

    IN:
    spec - spectrum array (can be 1D, 2D or 3D)
    wave - wavelength array
    resolution - resolving power in dw/w
    fine - if True, will interpolate the spectrum to a finer wavelength grid,
           in the end interpolating back to the original grid.
    lobes - number of zeros of sinc. Must be even number!
            At last 80 recommended.

    OUT:
    convolved spectrum

    --Tiago, 20090127
    '''
    ishp = spec.shape
    if len(ishp) > 1:
        # Make the spectrum a 2D array [spatial point,wave point]
        a = ishp[0]
        for i in range(1, len(ishp) - 1):
            a *= ishp[i]
        nspec = np.reshape(spec, (a, ishp[-1]))
    else:
        nspec = np.array([spec])
    out = np.zeros(nspec.shape)
    if fine:
        # interpolate to wavelength grid 5x higher than required resolution
        res = np.mean(wavein) / resolution
        wave = np.arange(wavein[0], wavein[-1], res / 5.)
        nspec2 = np.zeros((nspec.shape[0], len(wave)))
        for i in range(nspec.shape[0]):
            nspec2[i] = interp.splev(wave, interp.splrep(
                wavein, nspec[i], k=3, s=0), der=0)
        nspec = nspec2
    else:
        wave = wavein.copy()
    # mean wavelengh step
    step = abs(np.mean(wave[1:] - wave[:-1]))
    # note: this fwhm is in wavelength units! (later divided by step)
    fwhm = np.mean(wave) / (resolution)
    # Make sinc function out to 20th zero-crossing on either side. Error due to
    # ignoring additional lobes is less than 0.2% of continuum. Reducing extent
    # to 10th zero-crossing doubles maximum error.
    hwhm = fwhm / 2.                            # half width at half maximum
    # lobes = nr of zeros of sinc (radians)
    xxrange = lobes * np.pi
    nhalf = int(xxrange / np.pi * fwhm / step +
                0.999)  # nr. points in half sinc
    nsinc = 2 * nhalf + 1	                      # nr. points in sinc (odd!)
    wsinc = (np.arange(nsinc) - nhalf) * step     # abcissa (wavelength)
    xsinc = wsinc / (hwhm) * np.pi  # abcissa (radians)
    xsinc[nhalf] = 1.0              # avoid divide by zero
    sinc = np.sin(xsinc) / xsi      # calculate sinc
    sinc[nhalf] = 1.0               # insert midpoint
    sinc /= np.sum(sinc)            # normalize sinc
    # convolve
    for n in range(nspec.shape[0]):
        result = ndimage.convolve1d(nspec[n], sinc)

        if fine:  # interpolate back to original wave grid
            out[n] = interp.splev(wavein, interp.splrep(
                wave, result, k=3, s=0), der=0)
        else:
            out[n] = result
    if len(ishp) > 1:
        out.shape = ishp
    else:
        out = out[0]
    return out


######################################
### INPUT FILE GENERATING PROGRAMS ###
######################################
def buildltein(line, mu, phi, sim='fsun201', nts=20, nta=3, multphi_nt=False):
    # Get initial data
    f = open('lte.in.' + line + '.source', 'r')
    ll_ini = f.readlines()
    f.close()
    phi = np.array([phi]).ravel()
    for p in range(len(phi)):
        ll = ll_ini
        ll[0] = "'scr0/%s_nopack.int','linetab/oxyobs/%s.%s.tab','lineprof/oxyobs/%s.%s_phi%s_mu%s.I'\n" % \
            (sim, line, sim, line, sim, str(p), str(mu))
        if multiphi_nt:
            # Advance nta snapshots in each different phi
            if len(phi) * nta > nts:
                print('(EEE) buildltein: nts not big enough to cover '
                      'separation from all phi angles.')
                return

            ll[2] = '  %i, 98,  %i,  1,     .ns1,ns2,ns3,ns4 (snapshots)\n' % (
                (i + 1) * nta, nts)
        else:
            ll[2] = '  1, 98,  %i,  1,    .ns1,ns2,ns3,ns4 (snapshots)\n' % nts
        ll[9] = ' 1, 1, 1,%.3f,%.2f,  .nfl,nmy,nphi,xmu1,phi1\n' % (mu, phi[p])

        outfile = 'lte.in.%s_phi%s_mu%s' % (line, phi[p], mu)
        out = open(outfile, 'w')
        out.writelines(ll)
        out.close()
        print(('*** Wrote ' + outfile))
    return
