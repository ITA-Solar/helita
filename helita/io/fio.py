"""
Fortran unformatted I/O helper functions.
"""
import numpy as np


def fra(filename, dim=[], dtype='d', it=1, big_endian=0):
    """ Reads fortran unformatted binary data in arbitrary format.

    dim   : list/tuple/array with dimensions of array to be read
    dtype : data type of array to be read (d=double, f=float, i=int, etc.)
    it    : number of times an array of dim is read from filename. Useful
            when a given quantity is written sequentially into file. Setting
            it > 1 will give an extra dimension to the output (first dimension)
    big_endian : if != 0 will swap endianness from the system's default.

    --20100302, Tiago
    """

    file = open(filename, 'r')
    if not dim:
        raise ValueError
    if it < 1:
        raise ValueError
    if it == 1:
        dim = np.array(dim)[::-1]
        xtra = np.reshape(fort_read(file, np.prod(dim), dtype,
                          big_endian=big_endian), dim)
        # Invert axes
        xtra = np.transpose(xtra, axes=list(range(xtra.ndim))[::-1])
    else:
        xtra = np.empty((it,) + dim, dtype=dtype)
        dim = np.array(dim)[::-1]
        for i in range(it):
            tmp = np.reshape(fort_read(file, np.prod(dim), dtype,
                             big_endian=big_endian), dim)
            xtra[i] = np.transpose(tmp, axes=list(range(tmp.ndim))[::-1])
    file.close()
    return xtra


def fort_read(fid, num, read_type, mem_type=None, big_endian=0, length=4):
    """read fortran unformatted binary data

    the meaning of argument is the same as scipy.io.fread except for `length',
    which is is the size (in bytes) of header/footer.
    """
    # Swapped use of scipy.io.fread (removed from scipy >= 0.8) for np.fromfile
    # Now big_endian flag is changed to big_endian = 0,1
    # --Tiago, 20101004
    if big_endian:
        read_type = '>' + read_type
    else:
        read_type = '<' + read_type
    if mem_type is None:
        mem_type = read_type
    if length == 0:  # without header/footer
        return np.fromfile(fid, dtype=read_type, count=num)
    # with header/footer
    fid.read(length)  # header
    result = np.fromfile(fid, dtype=read_type, count=num)
    fid.read(length)  # footer
    return result


def fort_write(fid, num, myarray, write_type=None, big_endian=0, length=4):
    """write fortran unformatted binary data

    the meaning of argumetn is the same as scipy.io.fread except for `length',
    which is is the size (in bytes) of header/footer.
    """
    import struct

    # Swapped use of scipy.io.fread (removed from scipy >= 0.8) for np.fromfile
    # Now big_endian flag is changed to big_endian = 0,1.
    # --Tiago, 20101004
    if write_type is None:
        write_type = myarray.dtype.char
    if big_endian:
        write_type = '>' + write_type
        marker = '>'
    else:
        write_type = '<' + write_type
        marker = '<'
    if length == 0:
        return myarray.astype(write_type).tofile(fid, sep='')
    elif length == 4:
        marker += 'i'
    elif length == 8:
        marker += 'l'
    else:
        raise ValueError("length argument should be either 0, 4, or 8")
    # write
    nbyte = num * myarray.itemsize
    fid.write(struct.pack(marker, nbyte))  # header
    result = myarray.astype(write_type).tofile(fid, sep='')
    fid.write(struct.pack(marker, nbyte))  # footer
    return
