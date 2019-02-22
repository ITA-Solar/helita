"""
set of tools to deal with crispex data
"""
import xarray
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


def write_from_rh(files, outfile, mode='im', stokes=False, waveidx=None,
                  waveinterp=None, verbose=True):
    '''
    Writes crispex image cube from RH 1.5D netcdf output.

    Mode can be 'im' or 'sp'

    '''
    from . import lp

    # open first file to get some data
    dataset = xarray.open_dataset(files[0])
    nx, ny, nwave = dataset.nx, dataset.ny, dataset.nwave
    nt = len(files)
    dtype = dataset.intensity.dtype
    wave = dataset.wavelength
    if waveidx is not None:
        wave = wave[waveidx]
    if waveinterp is None:
        nwave = len(wave)
    else:
        nwave = len(waveinterp)
    if stokes:
        if not hasattr(dataset, 'stokes_V'):
            print('(WWW) write_from_rh: stokes selected but no data in file.')
            stokes = False
    if stokes:
        variables = ['intensity', 'stokes_Q', 'stokes_U', 'stokes_V']
        extrahd = ', stokes=[I,Q,U,V], ns=4'
    else:
        variables = ['intensity']
        extrahd = ''
    ns = len(variables)
    dataset.close()
    if mode.lower() == 'im':
        # write image cube
        print('writing image cube, %i files' % nt)
        iterator = enumerate(files)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(files), total=len(files))
            except ModuleNotFoundError:
                pass
        for i, f in iterator:
            dataset = xarray.open_dataset(f)
            for v in variables:
                data = dataset[v].data
                if waveidx is not None:
                    data = data[:, :, waveidx]
                if waveinterp is not None:
                    fint = interp.interp1d(wave, data, kind='linear')
                    data = fint(waveinterp).astype(dtype)
                lp.writeto('im_' + outfile, data, append=True,
                           extraheader=extrahd)
            dataset.close()
        print()
    elif mode.lower() == 'sp':
        # write spectral cube
        print('writing spectral cube, %i rows' % ny)
        isave = np.empty((nwave, nt, nx * ns), dtype=dtype)
        iterator = range(ny)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(ny))
            except ModuleNotFoundError:
                pass
        for y in iterator:
            for i, f in enumerate(files):
                dataset = xarray.open_dataset(f)
                for j, v in enumerate(variables):
                    data = dataset[v].data[:, y]
                    if waveidx is not None:
                        data = data[:, waveidx]
                    if waveinterp is not None:
                        fint = interp.interp1d(wave, data, kind='linear')
                        data = fint(waveinterp).astype(dtype)
                    isave[:, i, j::ns] = np.transpose(data)
                dataset.close()
            lp.writeto('sp_' + outfile, isave, append=True,
                       extraheader=extrahd)


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
    iterator = range(ny / ninc + 1)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(ny / ninc + 1))
        except ModuleNotFoundError:
            pass
    for i in iterator:
        imc = lp.getdata(infile)
        isave = imc[:, i * ninc:(i + 1) * ninc]
        sy = isave.shape[1]
        isave = np.transpose(
            np.transpose(isave).reshape(nt, nwave, nx * sy), axes=(1, 0, 2))
        lp.writeto(outfile, isave, append=i != 0, extraheader='')
        imc.close()
