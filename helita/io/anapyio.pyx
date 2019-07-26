import os
import numpy as np
cimport numpy as np
from stdlib cimport free, malloc
cdef extern from "stdlib.h":
     void *memcpy(void *dst, void *src, long n)

ctypedef unsigned char uint8_t
ctypedef signed char int8_t
ctypedef short int16_t
ctypedef int int32_t
ctypedef long long int64_t
DTYPES = [np.int8, np.int16, np.int32, np.float32, np.float64, np.int64]

cdef extern uint8_t *ana_fzread(char *file_name,int **ds,int *nd, char **header,int *type,int *osz)
cdef extern void ana_fzwrite(uint8_t *data,char *file_name,int *ds, int nd,char *header,int type)
cdef extern void ana_fcwrite(uint8_t *data,char *file_name,int *ds, int nd,char *header,int type,int slice)

# array copy function
cdef inline np.ndarray c2npy(uint8_t *data, int type, int size,int nd, int *ds):
    shapes = tuple([ds[j] for j in range(1, nd + 1)])
    cdef np.ndarray result = np.zeros(shapes[::-1], dtype=DTYPES[type])
    if type == 0:
        memcpy(result.data, <int8_t *>data, size)
    elif type == 1:
        memcpy(result.data, <int16_t *>data, size)
    elif type == 2:
        memcpy(result.data, <int32_t *>data, size)
    elif type == 3:
        memcpy(result.data, <float *>data, size)
    elif type == 4:
        memcpy(result.data, <double *>data, size)
    elif type == 5:
        memcpy(result.data, <int64_t *>data, size)
    free(data)
    return result


def fzread(filename, verbose=False):
    """fzread(filename,verbose=False)

    Reads ANA formatted files (compressed and uncompressed). Now can read
    arbitrary sizes and all supported datatypes.

    Parameters
    ----------
    filename : string
        Name of file to open
    verbose : bool, optional
        If True displays some extra information

    Returns
    -------
    (result,header) : (ndarray with data, string with header)
    """

    cdef char *hdr
    cdef int nd, type,*ds, size
    cdef uint8_t *data

    data = ana_fzread(str.encode(filename), &ds, &nd, &hdr, &type, &size)
    header = hdr.decode()
    if not data:
        raise IOError('fzread: could not get data.')
    if verbose:
        print('*** File %s successfully read.' % filename)
        print('*** Array shape: ', tuple([ds[j] for j in range(1,nd+1)])[::-1])
        print('*** Type: %i (%s)' % (type, str(DTYPES[type])))
        print('*** Header: \n', header)
    if type in range(6):
        # convert from c array to numpy
        result = c2npy(data, type, size, nd, ds)
    else:
        raise ValueError('fzread: type must be between 0-5, got %i' % type)
    return result, header


def fzwrite(filename, np.ndarray inarr, header='', comp=True, slice=5):
    ''' fzwrite(filename, inarray, header='', comp=True, slice=5)

    Writes numpy array to file, using the ANA format. Can write uncompressed
    or compressed (only available for int types). Can write in any dimension.
    Supported numpy datatypes: int8, int16, int32, int64, float32, float64.

    Parameters
    ----------
    filename: string with name of file
    inarray : numpy array to write
    header  : (optional) string with header
    comp    : if True, will write compressed files (only for int types)
    slice   : parameter of the compression. Don't change unless you know what
              you're doing.

    Returns
    -------
    None
    '''
    cdef int nd = inarr.ndim
    cdef int *ds,type
    cdef char *hd = header
    cdef int sl = slice

    if inarr.dtype == np.int8:
        type = 0
    elif inarr.dtype == np.int16:
        type = 1
    elif inarr.dtype == np.int32:
        type = 2
    elif inarr.dtype == np.float32:
        type = 3
    elif inarr.dtype == np.float64:
        type = 4
    elif inarr.dtype == np.int64:
        type = 5
    else:
        free(ds)
        raise TypeError('fzwrite: invalid type: %s ' % str(inarr.dtype))

    # Get shape
    s = (<object>inarr).shape[::-1]
    ds = <int *>malloc(ds[1] * ds[2] * sizeof(int))
    for i in range(nd):
        ds[i + 1] = s[i]
    # Compression ?
    if comp and type in [3,4]:
        print('(WWW) fzwrite: type is float, not compressing!')
        comp = False
    # Make sure array is contiguous
    if not (<object>inarr).flags["C_CONTIGUOUS"]:
        inarr = inarr.copy('C')
    # Check if we can write
    f = open(filename, 'w')
    f.close()
    # Write
    if comp:
        ana_fcwrite(<uint8_t *>inarr.data, str.encode(filename), ds, nd, hd,type, sl)
    else:
        ana_fzwrite(<uint8_t *>inarr.data, str.encode(filename), ds, nd, hd, type)
    free(ds)
    return
