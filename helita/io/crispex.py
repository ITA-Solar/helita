"""
set of tools to deal with crispex data
"""
import numpy as np
import scipy.interpolate as interp


def write_buf(intensity, outfile, wave=None, stokes=False):
    ''' Writes crispex image and spectral cubes, for when the data is already
        resident in memory. To be used when there is ample memory for all
        the cubes.

        IN:
           intensity: array with intensities (possibly IQUV). Its shape depends
                      on the value of stokes. If stokes=False, then its shape is
                      [nt, nx, ny, nwave]. If stokes=True, then its shape is
                      [4, nt, nx, ny, nwave], where the first index corresponds
                      to I, Q, U, V.
           outfile:   name of files to be writtenp. Will be prefixed by im_ and
                      sp_.
           stokes:    If True, will write full stokes.

        '''
    from . import lp

    if not stokes:
        nt, nx, ny, nw = intensity.shape
        ax = [(1, 2, 0, 3), (3, 0, 2, 1)]
        rs = [(nx, ny, nt * nw), (nw, nt, ny * nx)]
        extrahd = ''
    else:
        ns, nt, nx, ny, nw = intensity.shape
        ax = [(2, 3, 1, 0, 4), (4, 1, 3, 2, 0)]
        rs = [(nx, ny, nt * ns * nw), (nw, nt, ny * nx * ns)]
        extrahd = ', stokes=[I,Q,U,V], ns=4'
    # this is the image cube:
    im = np.transpose(intensity, axes=ax[0])
    im = im.reshape(rs[0])
    # this is the spectral cube
    sp = np.transpose(intensity, axes=ax[1])
    sp = sp.reshape(rs[1])
    # write lp.put, etc.
    # , extraheader_sep=False)
    lp.writeto('im_' + outfile, im, extraheader=extrahd)
    # , extraheader_sep=False)
    lp.writeto('sp_' + outfile, sp, extraheader=extrahd)
    return


def write_from_rh(files, outfile, stokes=False, waveidx=None, waveinterp=None,
                  verbose=False):
    ''' Writes crispex image cube from RH 1.5D netcdf output.'''
    from . import ncdf, lp
    from ..utils.shell import progressbar

    # open first file to get some data
    ii = ncdf.getvar(files[0], 'intensity', memmap=True)
    nx, ny, nw = ii.shape
    nt = len(files)
    dtype = ii.dtype
    del ii
    wave = ncdf.getvar(files[0], 'wavelength', memmap=False)
    if waveidx is not None:
        wave = wave[waveidx]
    if waveinterp is None:
        nw = len(wave)
    else:
        nw = len(waveinterp)
    if stokes:
        try:
            ii = ncdf.getvar(files[0], 'stokes_V', memmap=True)
            del ii
        except KeyError:
            print('(WWW) write_from_rh: stokes selected but no data in file.')
            stokes = False
    if stokes:
        vars = ['intensity', 'stokes_Q', 'stokes_U', 'stokes_V']
        extrahd = ', stokes=[I,Q,U,V], ns=4'
    else:
        vars = ['intensity']
        extrahd = ''
    ns = len(vars)
    # write image cube
    print('writing image cube, %i files' % nt)
    for i, f in enumerate(files):
        for v in vars:
            ii = ncdf.getvar(f, v, memmap=True)
            ii = np.array(ii)  # Tiago new
            if waveidx is not None:
                ii = ii[:, :, waveidx]
            if waveinterp is not None:
                fint = interp.interp1d(wave, ii, kind='linear')
                ii = fint(waveinterp).astype(dtype)
            lp.writeto('im_' + outfile, ii, append=True,
                       extraheader=extrahd, extraheader_sep=False)
            del ii
        if verbose:
            progressbar(i + 1, nt)
    print()
    return
    # old stuff, NOT IN USE
    # write spectral cube
    print('\nwriting spectral cube, %i rows' % ny)
    isave = np.empty((nw, nt, nx * ns), dtype=dtype)
    for y in range(ny):
        for i, f in enumerate(files):
            for j, v in enumerate(vars):
                ii = ncdf.getvar(f, v, memmap=True)[:, y]
                if waveidx is not None:
                    ii = ii[:, waveidx]
                if waveinterp is not None:
                    fint = interp.interp1d(wave, ii, kind='linear')
                    ii = fint(waveinterp).astype(dtype)
                isave[:, i, j::ns] = np.transpose(ii)
        lp.writeto('sp_' + outfile, isave, append=True,
                   extraheader=extrahd, extraheader_sep=False)
        if verbose:
            progressbar(y + 1, ny)
    print()
    return


def write_from_rh_sp(files, outfile, stokes=False, waveidx=None,
                     waveinterp=None, verbose=False):
    ''' Writes crispex spectral cubes only, from RH 1.5D netcdf output.'''
    from . import ncdf, lp
    from ..utils.shell import progressbar

    # open first file to get some data
    ii = ncdf.getvar(files[0], 'intensity', memmap=True)
    nx, ny, nw = ii.shape
    nt = len(files)
    dtype = ii.dtype
    del ii
    wave = ncdf.getvar(files[0], 'wavelength', memmap=False)
    if waveidx is not None:
        wave = wave[waveidx]
    if waveinterp is None:
        nw = len(wave)
    else:
        nw = len(waveinterp)
    if stokes:
        try:
            ii = ncdf.getvar(files[0], 'stokes_V', memmap=True)
            del ii
        except KeyError:
            print('(WWW) write_from_rh: stokes selected but no data in file.')
            stokes = False
    if stokes:
        vars = ['intensity', 'stokes_Q', 'stokes_U', 'stokes_V']
        extrahd = ', stokes=[I,Q,U,V], ns=4'
    else:
        vars = ['intensity']
        extrahd = ''
    ns = len(vars)
    # write spectral cube
    print('\nwriting spectral cube, %i rows' % ny)
    isave = np.empty((nw, nt, nx * ns), dtype=dtype)
    for y in range(ny):
        for i, f in enumerate(files):
            for j, v in enumerate(vars):
                ii = ncdf.getvar(f, v, memmap=True)[:, y]
                if waveidx is not None:
                    ii = ii[:, waveidx]
                if waveinterp is not None:
                    fint = interp.interp1d(wave, ii, kind='linear')
                    ii = fint(waveinterp).astype(dtype)
                isave[:, i, j::ns] = np.transpose(ii)
        lp.writeto('sp_' + outfile, isave, append=True,
                   extraheader=extrahd, extraheader_sep=False)
        if verbose:
            progressbar(y + 1, ny)
    print()
    return


def sp_from_im(infile, outfile, nwave, maxmem=4, verbose=True):
    ''' Creates a CRISPEX spectral cube from a quasi-transposition of an
        image cube.

        IN:
          infile  - lp image cube file to read.
          outfile - lp spectral cube file to write. Overwritten if exists.
          nwave   - number of spectral points.
          maxmem  - maximum memory (in GB) to use when creating temporary arrays
    '''

    from . import lp
    from ..utils.shell import progressbar

    GB = 2**30
    nx, ny, ntl = lp.getheader(infile)[0]
    ns = 1  # for now
    nt = ntl / nwave
    if (ntl % nwave != 0):
        raise ValueError('sp_from_im: image cube nlt axis not multiple of' +
                         ' given nwave (%i).' % (nwave) + ' Check values!')
    ninc = maxmem * GB / (ntl * nx * ns * 4)
    if ninc < 1:
        raise MemoryError('sp_from_im: memory supplied for temporary arrays' +
                          ' (%i GB) not enough.' % (maxmem) +
                          ' Need at least %f.2 GB.' % (ntl * nx * ns * 4. / GB))
    for i in range(ny / ninc + 1):
        imc = lp.getdata(infile)
        isave = imc[:, i * ninc:(i + 1) * ninc]
        sy = isave.shape[1]
        isave = np.transpose(
            np.transpose(isave).reshape(nt, nwave, nx * sy), axes=(1, 0, 2))
        lp.writeto(outfile, isave, append=i != 0, extraheader='',
                   extraheader_sep=False)
        imc.close()
        if verbose:
            progressbar(i + 1, ny / ninc + 1)
    print()
    return
