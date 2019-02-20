"""
set of tools to deal with IRIS observations
"""
import numpy as np
from pkg_resources import resource_filename


def si2dn(wave, band='NUV'):
    """
    Computes the coefficient to convert from flux SI units (W m^2 Hz^-1 sr^-1)
    to IRIS DN (counts). Most of this code came from Viggo's routines.

    Parameters
    ----------
    wave : 1-D array
           Wavelengths of the observations in nm
    band : string, optional
           Band of the observations. At the moment only 'NUV' is supported.

    Returns
    -------
    result : float
             Conversion factor from W / (m2 Hz sr) to IRIS DN.
    """
    from scipy import constants as const
    from scipy import interpolate as interp
    from scipy.io.idl import readsav
    band = band.upper()
    # File with IRIS effective area
    CFILE = resource_filename('helita', 'data/iris_sra_20130211.geny')
    arcsec2sr = (2. * const.pi / 360. / 60.**2)**2
    assert band == 'NUV', 'Only NUV band supported'
    dellambda = {'NUV': 25.46e-3}   # spectral pixel size
    eperph = {'NUV': 1.}          # electrons per photon (in CCD)
    resx = {'NUV': 1 / 3.}          # x resolution in arcsec
    resy = 1 / 6.                   # y resolution (along slit) in arcsec
    enph = 1.e7 * const.h * const.c / (wave * 1.e-9)  # energy per photon (erg)
    # convert from W m-2 Hz-1 sr-1 to erg s-1 cm-2 AA-1 sr-1
    iconv = 1e3 * const.c * 1e9 / wave**2 / 10  # from I_nu to I_lambda
    if band == 'NUV':
        ea = readsav(CFILE).p0
        fint = interp.interp1d(ea['lambda'][0], ea['area_sg'][0][1])
    else:
        raise ValueError
    effective_area = fint(wave)
    sr = arcsec2sr * resx[band] * resy  # from arcsec^2 to sr
    return iconv * effective_area * sr * dellambda[band] * eperph[band] / enph


def add_iris_noise(spec, exptime=1.):
    """
    Adds realistic IRIS noise to Mg II spectra from RH, using an approximate
    recipe assuming that the counts/sec at 282.0 nm are as given in the
    instrument description (~3500), and adding shot noise from Poisson and
    from a realistic effective gain and read noise.

    Parameters
    ----------
    spec : 3-D array
           Spectrum from RH (ideally spectrally and spatially convolved)
    exptime: float, optional
           Exposure time in seconds

    Returns
    -------
    result : 3-D array
            Spectrum with added noise
    """
    rh2cnts = 3500 / 2.6979319e-09  # ad hoc conversion from SI units to counts
    gain1 = 14.5                  # photon counts/DN, pristine gain
    gain2 = 16.0                  # gain measured with charge spreading
    read_noise = 1.2              # DN
    spec_cts = spec.copy() * rh2cnts * (gain1 / gain2) * exptime
    # Add poisson noise. Parallelise this part?
    spec_cts = np.random.poisson(spec_cts)
    spec_cts /= gain1 / gain2 * exptime
    rn = np.empty(spec_cts.shape, dtype='f')
    rn[:] = read_noise
    rn = np.random.poisson(rn)
    return (spec_cts + rn) / rh2cnts


def sj_filter(wave, band='IRIS_MGII_CORE', norm=True):
    """
    Returns the Solc filter for a given wavelength grid, for one of the
    NUV slit-jaw bands. Reads effective area.

    Parameters:
    -----------
    wave - 1D array
         Wavelength values (for interpolation)
    band - string, optional
         Band to use, either 'IRIS_MGII_CORE' or 'IRIS_MGII_WING'
    norm - bool
         Defines weather resulting filter is normalised (ie, has unit area,
         NOT sum)

    Returns:
    --------
    wfilt - 1D array
         Array with wavelength filter.
    """
    from scipy import interpolate as interp
    from scipy.io.idl import readsav
    # File with IRIS effective area
    CFILE = resource_filename('helita', 'data/iris_sra_20130211.geny')
    ea = readsav(CFILE).p0
    wave_filt = ea['lambda'][0]
    if band.upper() == 'IRIS_MGII_CORE':
        filt = ea['area_sji'][0][2]
    elif band.upper() == 'IRIS_MGII_WING':
        filt = ea['area_sji'][0][3]
    else:
        raise ValueError
    wfilt = interp.splev(wave, interp.splrep(wave_filt, filt, k=3, s=0))
    if band.upper() == 'IRIS_MGII_CORE':
        widx = (wave > 277.8) & (wave < 283.5)
    elif band.upper() == 'IRIS_MGII_WING':
        widx = (wave > 281) & (wave < 285)
    if norm:
        wfilt /= np.trapz(wfilt[widx], x=wave[widx])
    wfilt[~widx] = 0.
    return wfilt


def make_fits_level3_skel(filename, dtype, naxis, times, waves, wsizes,
                          desc="File from make_fits_level3_skel", descw=None,
                          cwaves=None, header_extra={}):
    """
    Creates a FITS file compliant with IRIS level 3 for use in CRISPEX. The
    file structure will be created and dummy data will be written, so that
    it can be later populated.

    Parameters
    ----------
    filename : string
        Name of file to write.
    dtype : string or numpy.dtype object
        Numpy datatype. Must be one of 'uint8', 'int16', 'int32', 'float32',
        or 'float64'.
    naxis : list of two integers
        Spatial dimensions of the file (nx, ny).
    times : array_like
        Array with times for each raster and slit position. Can either be
        1 dimension (same time for all slit positions), or 2 dimensions as
        (ntime, nsteps).
        The values are the time in seconds since DATE_OBS.
    waves : array_like
        Array with wavelength values (in Angstrom) for all the windows.
    wsizes : list of ints
        Sizes of the wavelength window(s). Can have one or more items, but
        the sum of all sizes must be less than len(waves). Each size has to
        be more than 2.
    desc : string, optional
        String describing observations (to go into OBS_DESC).
    descw : list of strings, optional
        Description of each wavelength window, to write in the WDESCx cards.
        Must have same size as wsizes.
    cwaves : list of floats, optional
        Central wavelengths of each wavelength window. IMPORTANT: must be
        increasing (smallest wavelength comes first). If not given, will use
        average from each window.
    header_extra : dictionary or pyfits.hdu.header object, optional
        Extra header information. This should be used to write important
        information such as CDELTx, CRVALx, CPIXx, XCEN, YCEN, DATE_OBS.
    """
    from astropy.io import fits as pyfits
    from datetime import datetime
    VERSION = '001'
    FITSBLOCK = 2880  # FITS blocksize in bytes
    # Consistency checks
    VALID_DTYPES = ['uint8', 'int16', 'int32', 'float32', 'float64']
    if type(dtype) != np.dtype:
        dtype = dtype.lower()
        if dtype not in VALID_DTYPES:
            raise TypeError("dtype %s not one of %s" % (dtype,
                                                        repr(VALID_DTYPES)))
    stime = times.shape
    if len(stime) == 2:
        if stime[1] != naxis[0]:
            raise ValueError("Second dimension of times must be same as nx")
    if np.any(wsizes < 3):
        raise ValueError("All wavelength windows must be bigger than 2.")
    if np.sum(wsizes) != len(waves):
        raise ValueError("wsizes do not add up to len(waves)")
    if np.any(np.diff(waves) < 0):
        raise ValueError("Wavelengths must be increasing!")
    if cwaves is None:
        cwaves = []
        s = 0
        for w in wsizes:
            cwaves.append(waves[s:s + w].mean())
            s += w
    if descw is None:
        descw = []
        for i in wsizes:
            descw.append("Some wavelength")
    # Create header
    tmp = np.zeros((1, 1, 1, 1), dtype=dtype)
    hdu = pyfits.PrimaryHDU(data=tmp)
    hd = hdu.header
    hd['NAXIS1'] = naxis[0]
    hd['NAXIS2'] = naxis[1]
    hd['NAXIS3'] = len(waves)
    hd['NAXIS4'] = len(times)
    hd['EXTEND'] = (True, 'FITS data may contain extensions')
    hd['INSTRUME'] = ('IRIS', 'Data generated in IRIS format')
    hd['DATA_LEV'] = (3., 'Data level')
    hd['LVL_NUM'] = (3., 'Data level')
    hd['VER_RF3'] = (VERSION, 'Version number of make_fits_level3_skel')
    hd['OBJECT'] = ('Sun', 'Type of solar area')
    hd['OBSID'] = (0000000000, 'obsid')
    hd['OBS_DESC'] = (desc, '')
    hd['DATE_OBS'] = (str(np.datetime64(datetime.today()))[:-5], '')
    hd['STARTOBS'] = (str(np.datetime64(datetime.today()))[:-5], '')
    hd['BTYPE'] = ('Intensity', '')
    hd['BUNIT'] = ('Corrected DN', '')
    hd['CDELT1'] = (0.34920, '[arcsec] x-coordinate increment')
    hd['CDELT2'] = (0.16635, '[arcsec] y-coordinate increment')
    hd['CDELT3'] = (0.02596, '[AA] wavelength increment')
    hd['CDELT4'] = (51.8794, '[s] t-coordinate axis increment')
    hd['CRPIX1'] = (1., 'reference pixel x-coordinate')
    hd['CRPIX2'] = (1., 'reference pixel y-coordinate')
    hd['CRPIX3'] = (1., 'reference pixel lambda-coordinate')
    hd['CRPIX4'] = (1., 'reference pixel t-coordinate')
    hd['CRVAL1'] = (0., '[arcsec] Position refpixel x-coordinate')
    hd['CRVAL2'] = (0., '[arcsec] Position refpixel y-coordinate')
    hd['CRVAL3'] = (1332.7, '[Angstrom] wavelength refpixel lambda-coordina')
    hd['CRVAL4'] = (0., '[s] Time mid-pixel t-coordinate')
    hd['CTYPE1'] = ('x', '[arcsec]')
    hd['CTYPE2'] = ('y', '[arcsec]')
    hd['CTYPE3'] = ('wave', '[Angstrom]')
    hd['CTYPE4'] = ('time', '[s]')
    hd['CUNIT1'] = ('arcsec', '')
    hd['CUNIT2'] = ('arcsec', '')
    hd['CUNIT3'] = ('Angstrom', '')
    hd['CUNIT4'] = ('s', '')
    hd['XCEN'] = (0., '[arcsec] x-coordinate center of FOV 1 raster')
    hd['YCEN'] = (0., '[arcsec] y-coordinate center of FOV 1 raster')
    # HERE PUT WCS STUFF
    nwin = len(wsizes)
    hd['NWIN'] = (nwin, 'Number of windows concatenated')
    istart = 0
    for i in range(nwin):
        iss = str(i + 1)
        hd['WSTART' + iss] = (istart, 'Start pixel for subwindow')
        hd['WWIDTH' + iss] = (wsizes[i], 'Width of subwindow')
        hd['WDESC' + iss] = (descw[i], 'Name of subwindow')
        hd['TWAVE' + iss] = (cwaves[i], 'Line center wavelength in subwindow')
        istart += wsizes[i]
    hd['COMMENT'] = 'Index order is (x,y,lambda,t)'
    for key, value in list(header_extra.items()):
        hd[key] = value
    # add some empty cards for contingency
    for i in range(10):
        hd.append()
    hd['DATE'] = (str(np.datetime64(datetime.today()))[:10],
                  'Creation UTC (CCCC-MM-DD) date of FITS header')
    hd.tofile(filename)
    # fill up empty file with correct size
    with open(filename, 'rb+') as fobj:
        fsize = len(hd.tostring()) + (naxis[0] * naxis[1] * len(waves) *
                                      len(times) * np.dtype(dtype).itemsize)
        fobj.seek(int(np.ceil(fsize / float(FITSBLOCK)) * FITSBLOCK) - 1)
        fobj.write(b'\0')
    # put wavelengths and times as extensions
    pyfits.append(filename, waves)
    pyfits.append(filename, times)
    f = pyfits.open(filename, mode='update', memmap=True)
    f[1].header['EXTNAME'] = 'lambda-coordinate'
    f[1].header['BTYPE'] = 'lambda axis'
    f[1].header['BUNIT'] = '[AA]'
    f[2].header['EXTNAME'] = 'time-coordinates'
    f[2].header['BTYPE'] = 't axes'
    f[2].header['BUNIT'] = '[s]'
    f.close()
    return


def transpose_fits_level3(filename, outfile=None):
    """
    Transposes an 'im' level 3 FITS file into 'sp' file
    (ie, transposed). INCOMPLETE, ONLY HEADER SO FAR.
    """
    from astropy.io import fits as pyfits
    hdr_in = pyfits.getheader(filename)
    hdr_out = hdr_in.copy()
    (nx, ny, nz, nt) = [hdr_in['NAXIS*'][i] for i in range(1, 5)]
    TRANSP_KEYS = ['NAXIS', 'CDELT', 'CRPIX', 'CRVAL', 'CTYPE', 'CUNIT']
    ORDER = [2, 3, 0, 1]
    for item in TRANSP_KEYS:
        for i in range(4):
            hdr_out[item + str(ORDER[i] + 1)] = hdr_in[item + str(i + 1)]
    return hdr_out


def rh_to_fits_level3(filelist, outfile, windows, window_desc, times=None,
                      xsize=24., clean=False, time_collapse_2d=False,
                      cwaves=None, make_sp=False, desc=None, wave2vac=None,
                      wave_select=np.array([False])):
    """
    Converts a sequence of RH netCDF/HDF5 ray files to a FITS file
    compliant with IRIS level 3 for use in CRISPEX.

    Parameters
    ----------
    filelist : list
        Name(s) of RH ray files.
    outfile : string
        Name of output file.
    windows : list of lists / tuples
        List with start/end wavelengths of different windows to write.
        E.g. windows=[(279.0, 281.0), (656.1, 656.4)]. Wavelengths given
        in air and nm. WINDOWS CANNOT BE OVERLAPPING!
    window_desc : list of strings
        List with same number of elements as windows, with description
        strings for the different windows
    times : array_like, optional
        Times for different snapshots in list. If none given, assumed to
        be 10 seconds between snapshots.
    xsize : float, optional.
        Size of x dimension in Mm. Default is 24 Mm.
    clean : bool, optional
        If True, will clean up any masked values using an inpainting
        algorithm. Default is False.
    time_collapse_2d : bool, optional
        If True, will collapse the y dimension into a time dimension.
        Use for 2D models with time as the y dimension. Default is False.
    cwaves : list of floats, optional
        Central wavelengths of each wavelength window, in AA. IMPORTANT: must
        be increasing (smallest wavelength comes first). If not given, will
        use average from each window.
    make_sp : bool, optional
        If True, will also produce an sp cube. Default is False.
    desc : string, optional
        Description string.
    wave2vac : list, optional
        Defines whether to convert the wavelengths from vacuum to air.If not
        None, should be a boolean list with the same number of elements
        as there are windows. Windows that are True will be converted. Default
        is set to None, so no conversion takes place.
    wave_select : array, optional
        If present, will only use wavelengths that are contained in this
        array. Must be exact match. Useful to combine output files that
        have common wavelengths.
    """
    from ..sim import rh15d
    from specutils.utils.wcs_utils import air_to_vac
    from astropy.io import fits as pyfits
    from astropy import units as u
    nt = len(filelist)
    robj = rh15d.Rh15dout()
    robj.read_ray(filelist[0])
    hd = robj.ray.params.copy()
    # Consistency checks to make sure all files are compatible
    if nt > 1:
        for f in filelist[1:]:
            robj.read_ray(f)
            hd_tmp = robj.ray.params
            if wave_select is None:
                assert hd_tmp['nwave'] == hd['nwave']
            assert hd_tmp['nx'] == hd['nx']
            assert hd_tmp['ny'] == hd['ny']
    nx, ny = hd['nx'], hd['ny']
    if time_collapse_2d:
        nt = ny
        ny = 1
    if wave2vac is None:
        wave2vac = [False] * len(windows)
    wave_full = robj.ray.wavelength[:]
    if wave_select.size == robj.ray.wavelength.size:
        wave_full = wave_full[wave_select]
    nwave = len(wave_full)
    waves = np.array([])
    indices = np.zeros(nwave, dtype='bool')
    nwaves = np.array([])
    for (wi, wf), air_conv in zip(windows, wave2vac):
        idx = (wave_full > wi) & (wave_full < wf)
        tmp = wave_full[idx]
        if air_conv:
            # RH converts to air using Edlen (1966) method
            tmp = air_to_vac(tmp * u.nm, method='edlen1966', scheme='iteration').value
        waves = np.append(waves, tmp)
        nwaves = np.append(nwaves, len(tmp))
        indices += idx
    waves *= 10.   # in Angstrom
    if times is None:
        times = np.arange(0., nt) * 10.
    asec2Mm = 696. / 959.5
    xres = xsize / nx / asec2Mm  # in arcsec
    tres = 0
    if nt > 1:
        tres = np.median(np.diff(times))
    header_extra = {"XCEN": 0.0, "YCEN": 0.0,
                    "CRPIX1": ny // 2, "CRPIX2": nx // 2, "CRPIX3": 1,
                    "CRPIX4": 1, "CRVAL1": 0.0, "CRVAL2": 0.0,
                    "CRVAL3": waves[0], "CRVAL4": times[0], "CDELT1": xres,
                    "CDELT2": xres, "CDELT3": np.median(np.diff(waves)),
                    "CDELT4": tres}
    desc = "Calculated from %s" % (robj.ray.params['atmosID'])
    make_fits_level3_skel(outfile, robj.ray.intensity.dtype,
                          (ny, nx), times, waves, nwaves, descw=window_desc,
                          cwaves=cwaves, header_extra=header_extra)
    fobj = pyfits.open(outfile, mode="update", memmap=True)
    if time_collapse_2d:
        robj.read_ray(filelist[0])
        tmp = robj.ray.intensity[:]
        if wave_select.size == robj.ray.wavelength.size:
            tmp = tmp[..., wave_select]
        tmp = tmp[:, :, indices]
        if clean:
            tmp = rh15d.clean_var(tmp, only_positive=True)
        else:   # always clean up for NaNs, Infs, masked, and negative
            idx = (~np.isfinite(tmp)) | (tmp < 0) | (tmp > 9e36)
            tmp[idx] = 0.0
        fobj[0].data[:] = tmp[:, :, np.newaxis].transpose((1, 3, 0, 2))
    else:
        for i, f in enumerate(filelist):
            robj.read_ray(f)
            tmp = robj.ray.intensity[:]
            if wave_select.size == robj.ray.wavelength.size:
                # common wavelengths, if applicable
                tmp = tmp[..., wave_select]
            tmp = tmp[:, ::-1, indices]  # right-handed system
            if clean:
                tmp = rh15d.clean_var(tmp, only_positive=True)
            else:   # always clean up for NaNs, Infs, masked, and negative
                idx = (~np.isfinite(tmp)) | (tmp < 0) | (tmp > 9e36)
                tmp[idx] = 0.0
            fobj[0].data[i] = tmp.T
    fobj.close()
    return
