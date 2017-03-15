"""
Set of tools to read SDF format.

First coded: 20111227 by Tiago Pereira (tiago.pereira@nasa.gov)
"""
import numpy as np


class SDFHeader:
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        self.query(filename)


    def query(self, filename, verbose=False):
        ''' Queries the file, returning datasets and shapes.'''
        f = open(filename, 'r')
        h = f.read(11)
        hdstr = str(h[:-1])
        if hdstr != 'SDF format':
            raise IOError('SDF header not found in' +
                          ' %s, probably wrong or corrupt file.' % filename)
        self.hdrpos = np.fromfile(f, dtype='>l', count=1)[0]
        self.datapos = np.fromfile(f, dtype='>l', count=1)[0]
        self.norder = np.fromfile(f, dtype='>i', count=1)[0]
        self.hdrsize = np.fromfile(f, dtype='>l', count=1)[0]
        header = f.read(self.hdrpos - f.tell())
        self.header = header
        if self.verbose:
            print(header)
        f.close()
        self.header_data(header)
        return


    def header_data(self, header):
        ''' Breaks header string into variable informationp. '''
        self.variables = {}
        offset = 19 + self.hdrsize
        for line in header.split('\n')[:-1]:
            l = line.split()
            label = l.pop(1)
            order = int(l[0])
            dtype = '>' + l[1] + l[2]  # force big endian
            nbpw = int(l[2])
            ndims = int(l[3])
            shape = ()
            for i in range(ndims):
                shape += (int(l[4 + i]),)
            nbytes = nbpw * np.prod(shape)
            if dtype[1] == 'c':
                nbytes *= 2
            if dtype[1:] == 'c4':  # these are the same internally to numpy
                dtype = '>c8'
            self.variables[label] = [order, dtype, nbpw, offset, shape]
            offset += nbytes
        return


def getvar(filename, variable, memmap=False):
    ''' Reads variable from SDF file.

        IN:
            filename - string with filename
            variable - string with variable name
            memmap   - [OPTIONAL] booleanp. If true, will return a memmap object
                       (ie, data is only loaded into memory when needed)
        OUT:
            data - array with data
    '''
    ff = SDFHeader(filename, verbose=False)
    if variable not in ff.variables:
        raise KeyError(
            '(EEE) getvar: variable %s not found in %s' %
            (variable, filename))
    order, dtype, nbpw, offset, shape = ff.variables[variable]
    if memmap:
        data = np.memmap(filename, dtype=dtype, mode='r', shape=shape,
                         offset=offset, order='F')
    else:
        f = open(filename, 'r')
        f.seek(offset)
        data = np.fromfile(f, dtype=dtype,
                           count=np.prod(shape)).reshape(shape[::-1]).T
        f.close()
    return data


def getall(filename, memmap=False):
    ''' Reads all the variables of an SDF file. Loads into a dictionary indexed
        by variable name. '''
    ff = SDFHeader(filename, verbose=False)
    result = {}
    for v in ff.variables:
        result[v] = getvar(filename, v, memmap)
    return result
