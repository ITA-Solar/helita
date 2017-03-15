"""
Set of tools to read and write 'La Palma' cubes
"""
import numpy as np
import os


def make_header(image):
    ''' Creates header for La Palma images. '''
    from struct import pack

    ss = image.shape
    # only 2D or 3D arrays
    if len(ss) not in [2, 3]:
        raise IndexError('make_header: input array must be 2D or 3D, got %iD'
                         % len(ss))
    dtypes = {'int8': ['(byte)', 1], 'int16': ['(integer)', 2],
              'int32': ['(long)', 3], 'float32': ['(float)', 4]}
    if str(image.dtype) not in dtypes:
        raise ValueError('make_header: array type' +
                         ' %s not supported, must be one of %s' %
                         (image.dtype, list(dtypes.keys())))
    sdt = dtypes[str(image.dtype)]
    header = ' datatype=%s %s, dims=%i, nx=%i, ny=%i' % \
             (sdt[1], sdt[0], len(ss), ss[0], ss[1])
    if len(ss) == 3:
        header += ', nt=%i' % (ss[2])
    # endianess
    if pack('@h', 1) == pack('<h', 1):
        header += ', endian=l'
    else:
        header += ', endian=b'
    return header


def writeto(filename, image, extraheader='', dtype=None, verbose=False,
            append=False):
    '''Writes image into cube, La Palma format. Analogous to IDL's lp_write.'''
    # Tiago notes: seems to have problems with 2D images, but not sure if that
    # even works in IDL's routines...
    if not os.path.isfile(filename):
        append = False
    # use dtype from array, if none is specified
    if dtype is None:
        dtype = image.dtype
    image = image.astype(dtype)
    if append:
        # check if image sizes/types are consistent with file
        sin, t, h = getheader(filename)
        if sin[:2] != image.shape[:2]:
            raise IOError('writeto: trying to write' +
                          ' %s images, but %s has %s images!' %
                          (repr(image.shape[:2]), filename, repr(sin[:2])))
        if np.dtype(t) != image.dtype:
            raise IOError('writeto: trying to write' +
                          ' %s type images, but %s nas %s images' %
                          (image.dtype, filename, np.dtype(t)))
        # add the nt of current image to the header
        hloc = h.lower().find('nt=')
        new_nt = str(sin[-1] + image.shape[-1])
        header = h[:hloc + 3] + new_nt + h[hloc + 3 + len(str(sin[-1])):]
    else:
        header = make_header(image)
    if extraheader:
        header += ' : ' + extraheader
    # convert string to [unsigned] byte array
    hh = np.zeros(512, dtype='uint8')
    for i, ss in enumerate(header):
        hh[i] = ord(ss)
    # write header to file
    file_arr = np.memmap(filename, dtype='uint8', mode=append and 'r+' or 'w+',
                         shape=(512,))
    file_arr[:512] = hh[:]
    del file_arr
    # offset if appending
    apoff = append and np.prod(sin) * image.dtype.itemsize or 0
    # write array to file
    file_arr = np.memmap(filename, dtype=dtype, mode='r+', order='F',
                         offset=512 + apoff, shape=image.shape)
    file_arr[:] = image[:]
    del file_arr
    if verbose:
        if append:
            print(('Appended %s %s array into %s.' % (image.shape, dtype,
                                                      filename)))
        else:
            print(('Wrote %s, %s array of shape %s' % (filename, dtype,
                                                       image.shape)))
    return


def writeheader(filename, header):
    """
    Writes header (proper format, from make_header) into existing filename.
    """
    # convert string to [unsigned] byte array
    hh = np.zeros(512, dtype='uint8')
    for i, ss in enumerate(header):
        hh[i] = ord(ss)
    # write header to file
    file_arr = np.memmap(filename, dtype='uint8', mode='r+', shape=(512,))
    file_arr[:512] = hh[:]
    del file_arr
    return


def getheader(filename):
    """
    Reads header from La Palma format cube.

    Returns a list with the following:
    shape tuple (nx, ny [, nt], datatype (with endianness), header string.
    """
    # read header and convert to string
    h = np.fromfile(filename, dtype='uint8', count=512)
    header = ''
    for s in h[h > 0]:
        header += chr(s)
    # start reading at 'datatype'
    hd = header[header.lower().find('datatype'):]
    hd = hd.split(':')[0].replace(',', ' ').split()
    # Types:   uint8  int16 int32 float32
    typelist = ['u1', 'i2', 'i4', 'f4']
    # extract datatype
    try:
        dtype = typelist[int(hd[0].split('=')[1]) - 1]
    except:
        print(header)
        raise IOError('getheader: datatype invalid or missing')
    # extract endianness
    try:
        if hd[-1].split('=')[0].lower() != 'endian':
            raise IndexError()
        endian = hd[-1].split('=')[1]
    except IndexError:
        print(header)
        raise IOError('getheader: endianess missing.')
    if endian.lower() == 'l':
        dtype = '<' + dtype
    else:
        dtype = '>' + dtype
    # extract dims
    try:
        if hd[2].split('=')[0].lower() != 'dims':
            raise IndexError()
        dims = int(hd[2].split('=')[1])
        if dims not in [2, 3]:
            raise ValueError('Invalid dims=%i (must be 2 or 3)' % dims)
    except IndexError:
        print(header)
        raise IOError('getheader: dims invalid or missing.')
    try:
        if hd[3].split('=')[0].lower() != 'nx':
            raise IndexError()
        nx = int(hd[3].split('=')[1])
    except:
        print(header)
        raise IOError('getheader: nx invalid or missing.')
    try:
        if hd[4].split('=')[0].lower() != 'ny':
            raise IndexError()
        ny = int(hd[4].split('=')[1])
    except:
        print(header)
        raise IOError('getheader: ny invalid or missing.')
    if dims == 3:
        try:
            if hd[5].split('=')[0].lower() != 'nt':
                raise IndexError()
            nt = int(hd[5].split('=')[1])
        except:
            print(header)
            raise IOError('getheader: nt invalid or missing.')
        shape = (nx, ny, nt)
    else:
        shape = (nx, ny)
    return [shape, dtype, header]


def getdata(filename, rw=False, verbose=False):
    ''' Reads La Palma format cube (into a memmap object). If rw is True, then
        any change on the data will be written to the file.'''
    sh, dt, header = getheader(filename)
    if verbose:
        print(('Reading %s...\n%s' % (filename, header)))
    mode = ['c', 'r+']
    return np.memmap(filename, mode=mode[rw], shape=sh, dtype=dt, order='F',
                     offset=512)
