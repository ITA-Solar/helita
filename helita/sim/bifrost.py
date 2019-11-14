"""
Set of programs to read and interact with output from Bifrost
"""

import os
import warnings
from glob import glob
import numpy as np
from . import cstagger
from scipy import interpolate
from scipy.ndimage import map_coordinates
from multiprocessing.dummy import Pool as ThreadPool


class BifrostData(object):
    """
    Reads data from Bifrost simulations in native format.

    Parameters
    ----------
    file_root - string
        Basename for all file names (without underscore!). Snapshot number
        will be added afterwards, and directory will be added before.
    snap - integer, optional
        Snapshot number. If None, will read first snapshot in sequence.
    meshfile - string, optional
        File name (including full path) for file with mesh. If set
        to None (default), a uniform mesh will be created.
    fdir - string, optional
        Directory where simulation files are. Must be a real path.
    verbose - bool, optional
        If True, will print out more diagnostic messages
    dtype - string, optional
        Data type for reading variables. Default is 32 bit float.
    big_endian - string, optional
        If True, will read variables in big endian. Default is False
        (reading in little endian).
    ghost_analyse - bool, optional
        If True, will read data from ghost zones when this is saved
        to files. Default is never to read ghost zones.
    cstagop - bool, optional
        Use true only if data is too big to load. Danger:
        it will do quantity operations without correcting the stagger mesh.
    lowbus  - bool, optional
        Use True only if data is too big to load. It will do cstagger
        operations layer by layer using threads (slower).
    numThreads - integer, optional
        number of threads for certain operations that use parallelism.

    Examples
    --------
    This reads snapshot 383 from simulation "cb24bih", whose file
    root is "cb24bih", and is found at directory /data/cb24bih:

    >>> a = BifrostData("cb24bih", snap=383, fdir="/data/cb24bih")

    Scalar variables do not need de-staggering and are available as
    memory map (only loaded to memory when needed), e.g.:

    >>> a.r.shape
    (504, 504, 496)

    Composite variables need to be obtained by get_var():

    >>> vx = a.get_var("ux")
    """
    snap = None

    def __init__(self, file_root, snap=None, meshfile=None, fdir='.',
                 verbose=True, dtype='f4', big_endian=False, cstagop=True,
                 ghost_analyse=False, lowbus=False, numThreads=1):
        """
        Loads metadata and initialises variables.
        """
        self.fdir = fdir
        self.verbose = verbose
        self.cstagop = cstagop
        self.lowbus = lowbus
        self.numThreads = numThreads
        self.file_root = os.path.join(self.fdir, file_root)
        self.root_name = file_root
        self.meshfile = meshfile
        self.ghost_analyse = ghost_analyse
        self.cstagop = cstagop
        self.lowbus = lowbus
        self.numThreads = numThreads
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        self.hion = False
        self.heion = False
        self.set_snap(snap)
        try:
            tmp = find_first_match("%s*idl" % file_root, fdir)
        except IndexError:
            try:
                tmp = find_first_match("%s*idl.scr" % file_root, fdir)
            except IndexError:
                try:
                    tmp = find_first_match("mhd.in", fdir)
                except IndexError:
                    raise ValueError(("(EEE) init: no .idl or mhd.in files "
                                    "found"))
        self.uni = Bifrost_units(filename=tmp,fdir=fdir)

    def _set_snapvars(self):
        """
            Sets list of avaible variables
        """
        self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        self.auxvars = self.params['aux'][self.snapInd].split()
        if self.do_mhd:
            self.snapvars += ['bx', 'by', 'bz']
        self.hionvars = []
        self.heliumvars = []
        if 'do_hion' in self.params:
            if self.params['do_hion'][self.snapInd] > 0:
                self.hionvars = ['hionne', 'hiontg', 'n1',
                                 'n2', 'n3', 'n4', 'n5', 'n6', 'nh2']
                self.hion = True
        if 'do_helium' in self.params:
            if self.params['do_helium'][self.snapInd] > 0:
                self.heliumvars = ['nhe1', 'nhe2', 'nhe3']
                self.heion = True
        self.compvars = ['ux', 'uy', 'uz', 's', 'ee']
        self.simple_vars = self.snapvars + self.auxvars + self.hionvars + \
            self.heliumvars
        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.auxxyvars.append('ixy1')
        self.vars2d = []
        # special case for 2D variables, stored in a separate file
        for var in self.auxvars:
            if any(i in var for i in ('xy', 'yz', 'xz')):
                self.auxvars.remove(var)
                self.vars2d.append(var)

    def set_snap(self, snap):
        """
        Reads metadata and sets variable memmap links for a given snapshot
        number.
        Parameters
        ----------
        snap - integer or array
            Number of simulation snapshot to load.
        """
        if snap is None:
            try:
                tmp = sorted(glob("%s*idl" % self.file_root))[0]
                snap_string = tmp.split(self.file_root + '_')[-1].split(".idl")[0]
                if snap_string.isdigit():
                    snap = int(snap_string)
                else:
                    tmp = glob("%s.idl" % self.file_root)
                    snap = 0
            except:
                try:
                    tmp = sorted(glob("%s*idl.scr" % self.file_root))[0]
                    snap = -1
                except IndexError:
                    try:
                        tmp = glob("%s.idl" % self.file_root)
                        snap = 0
                    except IndexError:
                        raise ValueError(("(EEE) set_snap: snapshot not defined "
                                      "and no .idl files found"))
        self.snap = snap
        if np.size(snap) > 1:
            self.snap_str = []
            for num in snap:
                self.snap_str.append('_%03i' % int(num))
        else:
            if snap == 0:
                self.snap_str = ''
            else:
                self.snap_str = '_%03i' % snap
        self.snapInd = 0

        self._read_params()
        # Read mesh for all snaps because meshfiles could differ
        self.__read_mesh(self.meshfile)
        # variables: lists and initialisation
        self._set_snapvars()
        self._init_vars()

    def _read_params(self):
        """
        Reads parameter file (.idl)
        """
        if np.shape(self.snap) is ():
            snap = [self.snap]
            snap_str = [self.snap_str]
        else:
            snap = self.snap
            snap_str = self.snap_str

        filename = []
        self.paramList = []

        for i, num in enumerate(snap):
            if num < 0:
                filename.append(self.file_root + '.idl.scr')
            elif num == 0:
                filename.append(self.file_root + '.idl')
            else:
                filename.append(self.file_root + snap_str[i] + '.idl')

        for file in filename:
            self.paramList.append(read_idl_ascii(file))

        # assign some parameters as attributes
        for params in self.paramList:
            for p in ['x', 'y', 'z', 'b']:
                try:
                    setattr(self, 'n' + p, params['m' + p])
                except KeyError:
                    raise KeyError(('read_params: could not find '
                                    'm%s in idl file!' % p))
            for p in ['dx', 'dy', 'dz', 'do_mhd']:
                try:
                    setattr(self, p, params[p])
                except KeyError:
                    raise KeyError(('read_params: could not find '
                                    '%s in idl file!' % p))
            try:
                if params['boundarychk'] == 1:
                    self.nzb = self.nz + 2 * self.nb
                else:
                    self.nzb = self.nz
            except KeyError:
                self.nzb = self.nz
            # check if units are there, if not use defaults and print warning
            unit_def = {'u_l': 1.e8, 'u_t': 1.e2, 'u_r': 1.e-7,
                        'u_b': 1.121e3, 'u_ee': 1.e12}
            for unit in unit_def:
                if unit not in params:
                    print(("(WWW) read_params:"" %s not found, using "
                           "default of %.3e" % (unit, unit_def[unit])))
                    params[unit] = unit_def[unit]

        self.params = {}
        for key in self.paramList[0]:
            self.params[key] = np.array(
                [self.paramList[i][key] for i in range(
                    0, len(self.paramList))])

    def __read_mesh(self, meshfile):
        """
        Reads mesh file
        """
        if meshfile is None:
            meshfile = os.path.join(
                self.fdir, self.params['meshfile'][self.snapInd].strip())
        if os.path.isfile(meshfile):
            f = open(meshfile, 'r')
            for p in ['x', 'y', 'z']:
                dim = int(f.readline().strip('\n').strip())
                assert dim == getattr(self, 'n' + p)
                # quantity
                setattr(self, p, np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # quantity "down"
                setattr(self, p + 'dn', np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # up derivative of quantity
                setattr(self, 'd%sid%sup' % (p, p), np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # down derivative of quantity
                setattr(self, 'd%sid%sdn' % (p, p), np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
            f.close()
            if self.ghost_analyse:
                # extend mesh to cover ghost zones
                self.z = np.concatenate((self.z[0] - np.linspace(
                    self.dz * self.nb, self.dz, self.nb),
                    self.z, self.z[-1] + np.linspace(
                    self.dz, self.dz * self.nb, self.nb)))
                self.zdn = np.concatenate((self.zdn[0] - np.linspace(
                    self.dz * self.nb, self.dz, self.nb),
                    self.zdn, (self.zdn[-1] + np.linspace(
                        self.dz, self.dz * self.nb, self.nb))))
                self.dzidzup = np.concatenate((
                    np.repeat(self.dzidzup[0], self.nb),
                    self.dzidzup,
                    np.repeat(self.dzidzup[-1], self.nb)))
                self.dzidzdn = np.concatenate((
                    np.repeat(self.dzidzdn[0], self.nb),
                    self.dzidzdn,
                    np.repeat(self.dzidzdn[-1], self.nb)))
                self.nz = self.nzb
        else:  # no mesh file
            print('(WWW) Mesh file %s does not exist.' % meshfile)
            if self.dx == 0.0:
                self.dx = 1.0
            if self.dy == 0.0:
                self.dy = 1.0
            if self.dz == 0.0:
                self.dz = 1.0
            print(('(WWW) Creating uniform grid with [dx,dy,dz] = '
                   '[%f,%f,%f]') % (self.dx, self.dy, self.dz))
            # x
            self.x = np.arange(self.nx) * self.dx
            self.xdn = self.x - 0.5 * self.dx
            self.dxidxup = np.zeros(self.nx) + 1. / self.dx
            self.dxidxdn = np.zeros(self.nx) + 1. / self.dx
            # y
            self.y = np.arange(self.ny) * self.dy
            self.ydn = self.y - 0.5 * self.dy
            self.dyidyup = np.zeros(self.ny) + 1. / self.dy
            self.dyidydn = np.zeros(self.ny) + 1. / self.dy
            # z
            if self.ghost_analyse:
                self.nz = self.nzb
            self.z = np.arange(self.nz) * self.dz
            self.zdn = self.z - 0.5 * self.dz
            self.dzidzup = np.zeros(self.nz) + 1. / self.dz
            self.dzidzdn = np.zeros(self.nz) + 1. / self.dz

        if self.nz > 1:
            self.dz1d = np.gradient(self.z)
        else:
            self.dz1d = np.zeros(self.nz)

    def _init_vars(self, *args, **kwargs):
        """
        Memmaps "simple" variables, and maps them to methods.
        Also, sets file name[s] from which to read a data
        """
        self.variables = {}
        for var in self.simple_vars:
            try:
                self.variables[var] = self._get_simple_var(
                    var, *args, **kwargs)
                setattr(self, var, self.variables[var])
            except Exception:
                if self.verbose:
                    print(('(WWW) init_vars: could not read '
                           'variable %s' % var))
        for var in self.auxxyvars:
            try:
                self.variables[var] = self._get_simple_var_xy(var, *args,
                                                              **kwargs)
                setattr(self, var, self.variables[var])
            except Exception:
                if self.verbose:
                    print(('(WWW) init_vars: could not read '
                           'variable %s' % var))
        rdt = self.r.dtype
        cstagger.init_stagger(self.nz, self.dx, self.dy, self.z.astype(rdt),
                              self.zdn.astype(rdt), self.dzidzup.astype(rdt),
                              self.dzidzdn.astype(rdt))

    def get_varTime(self, var, snap=None, iix=None, iiy=None, iiz=None,
                    *args, **kwargs):
        """
        Reads a given variable as a function of time.

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be a valid Bifrost variable name,
            see Bifrost.get_var().
        snap - array of integers
            Snapshot numbers to read.
        iix -- integer or array of integers, optional
            reads yz slices.
        iiy -- integer or array of integers, optional
            reads xz slices.
        iiz -- integer or array of integers, optional
            reads xy slices.
        """
        self.iix = iix
        self.iiy = iiy
        self.iiz = iiz

        try:
            if snap is not None:
                if np.size(snap) == np.size(self.snap):
                    if any(snap != self.snap):
                        self.set_snap(snap)
                else:
                    self.set_snap(snap)
        except ValueError:
            print('WWW: snap has to be a numpy.arrange parameter')

        # lengths for dimensions of return array
        self.xLength = 0
        self.yLength = 0
        self.zLength = 0

        for dim in ('iix', 'iiy', 'iiz'):
            if getattr(self, dim) is None:
                if dim[2] == 'z':
                    setattr(self, dim[2] + 'Length',
                            getattr(self, 'n' + dim[2] + 'b'))
                else:
                    setattr(self, dim[2] + 'Length',
                            getattr(self, 'n' + dim[2]))
                setattr(self, dim, slice(None))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)

        snapLen = np.size(self.snap)
        value = np.empty([self.xLength, self.yLength, self.zLength, snapLen])

        for i in range(0, snapLen):
            self.snapInd = i
            self._set_snapvars()
            self._init_vars()

            value[..., i] = self.get_var(var, self.snap[i], iix=self.iix,
                                         iiy=self.iiy, iiz=self.iiz)
        return value

    def set_domain_iiaxis(self, iinum=slice(None), iiaxis='x'):
        """
        Sets length of each dimension for get_var based on iix/iiy/iiz

        Parameters
        ----------
        iinum - int, list, or array
            Slice to be taken from get_var quantity in that axis (iiaxis)
        iiaxis - string
            Axis from which the slice will be taken ('x', 'y', or 'z')
        """
        if iinum is None:
            iinum = slice(None)

        dim = 'ii' + iiaxis
        setattr(self, dim, iinum)
        setattr(self, iiaxis + 'Length', np.size(iinum))

        if np.size(getattr(self, dim)) == 1:
            if getattr(self, dim) == slice(None):
                if dim[2] == 'z':
                    setattr(self, dim[2] + 'Length',
                            getattr(self, 'n' + dim[2] + 'b'))
                else:
                    setattr(self, dim[2] + 'Length',
                            getattr(self, 'n' + dim[2]))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)
                if indSize == 1:
                    temp = np.asarray(getattr(self, dim))
                    setattr(self, dim, temp.item())
        else:
            indSize = np.size(getattr(self, dim))
            setattr(self, dim[2] + 'Length', indSize)
            if indSize == 1:
                temp = np.asarray(getattr(self, dim))
                setattr(self, dim, temp.item())

    def get_var(self, var, snap=None, *args, iix=slice(None), iiy=slice(None),
                iiz=slice(None), **kwargs):
        """
        Reads a variable from the relevant files.

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        snap - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot
            by running self.set_snap(snap).
        """
        if self.verbose:
            print('(get_var): reading ', var)

        if not hasattr(self, 'iix'):
            self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            self.set_domain_iiaxis(iinum=iiz, iiaxis='z')
        else:
            if (iix != slice(None)) and np.any(iix != self.iix):
                if self.verbose:
                    print('(get_var): iix ', iix, self.iix)
                self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            if (iiy != slice(None)) and np.any(iiy != self.iiy):
                if self.verbose:
                    print('(get_var): iiy ', iiy, self.iiy)
                self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            if (iiz != slice(None)) and np.any(iiz != self.iiz):
                if self.verbose:
                    print('(get_var): iiz ', iiz, self.iiz)
                self.set_domain_iiaxis(iinum=iiz, iiaxis='z')

        if self.cstagop and ((self.iix != slice(None)) or
                             (self.iiy != slice(None)) or
                             (self.iiz != slice(None))):
            self.cstagop = False
            print('WARNING: cstagger use has been turned off,',
                  'turn it back on with "dd.cstagop = True"')

        if var in ['x', 'y', 'z']:
            return getattr(self, var)

        if (snap is not None) and np.any(snap != self.snap):
            if self.verbose:
                print('(get_var): setsnap ', snap, self.snap)
            self.set_snap(snap)

        if var in self.simple_vars:  # is variable already loaded?
            val = self._get_simple_var(var, *args, **kwargs)
            if self.verbose:
                print('(get_var): reading simple ', np.shape(val))
        elif var in self.auxxyvars:
            val = self._get_simple_var_xy(var, *args, **kwargs)
        elif var in self.compvars:  # add to variable list
            self.variables[var] = self._get_composite_var(var, *args, **kwargs)
            setattr(self, var, self.variables[var])
            val = self.variables[var]
        else:
            val = self.get_quantity(var, *args, **kwargs)

        if np.shape(val) != (self.xLength, self.yLength, self.zLength):
            # at least one slice has more than one value
            if np.size(self.iix) + np.size(self.iiy) + np.size(self.iiz) > 3:
                # x axis may be squeezed out, axes for take()
                axes = [0, -2, -1]

                for counter, dim in enumerate(['iix', 'iiy', 'iiz']):
                    if (np.size(getattr(self, dim)) > 1 or
                            getattr(self, dim) != slice(None)):
                        # slicing each dimension in turn
                        val = val.take(getattr(self, dim), axis=axes[counter])
            else:
                # all of the slices are only one int or slice(None)
                val = val[self.iix, self.iiy, self.iiz]

            # ensuring that dimensions of size 1 are retained
            val = np.reshape(val, (self.xLength, self.yLength, self.zLength))

        return val

    def _get_simple_var(self, var, order='F', mode='r', *args, **kwargs):
        """
        Gets "simple" variable (ie, only memmap, not load into memory).
        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        order - string, optional
            Must be either 'C' (C order) or 'F' (Fortran order, default).
        mode - string, optional
            numpy.memmap read mode. By default is read only ('r'), but
            you can use 'r+' to read and write. DO NOT USE 'w+'.
        Returns
        -------
        result - numpy.memmap array
            Requested variable.
        """
        if np.size(self.snap) > 1:
            currSnap = self.snap[self.snapInd]
            currStr = self.snap_str[self.snapInd]
        else:
            currSnap = self.snap
            currStr = self.snap_str
        if currSnap < 0:
            filename = self.file_root
            fsuffix_b = '.scr'
        elif currSnap == 0:
            filename = self.file_root
            fsuffix_b = ''
        else:
            filename = self.file_root + currStr
            fsuffix_b = ''

        if var in self.snapvars:
            fsuffix_a = '.snap'
            idx = (self.snapvars).index(var)
            filename += fsuffix_a + fsuffix_b
        elif var in self.auxvars:
            fsuffix_a = '.aux'
            idx = self.auxvars.index(var)
            filename += fsuffix_a + fsuffix_b
        elif var in self.hionvars:
            idx = self.hionvars.index(var)
            isnap = self.params['isnap'][self.snapInd]
            if isnap <= -1:
                filename = filename + '.hion.snap.scr'
            elif isnap == 0:
                filename = filename + '.hion.snap'
            elif isnap > 0:
                filename = '%s.hion_%03d.snap' % (self.file_root, isnap)
                if not os.path.isfile(filename):
                    filename = '%s_.hion%s.snap' % (self.file_root, isnap)
        elif var in self.heliumvars:
            idx = self.heliumvars.index(var)
            isnap = self.params['isnap'][self.snapInd]
            if isnap <= -1:
                filename = filename + '.helium.snap.scr'
            elif isnap == 0:
                filename = filename + '.helium.snap'
            elif isnap > 0:
                filename = '%s.helium_%s.snap' % (self.file_root, isnap)
        else:
            raise ValueError(('_get_simple_var: could not find variable '
                              '%s. Available variables:' % (var) +
                              '\n' + repr(self.simple_vars)))
        dsize = np.dtype(self.dtype).itemsize
        if self.ghost_analyse:
            offset = self.nx * self.ny * self.nzb * idx * dsize
            ss = (self.nx, self.ny, self.nzb)
        else:
            offset = (self.nx * self.ny *
                      (self.nzb + (self.nzb - self.nz) // 2) * idx * dsize)
            ss = (self.nx, self.ny, self.nz)

        if var in self.heliumvars:
            return np.exp(np.memmap(filename, dtype=self.dtype, order=order,
                                    mode=mode, offset=offset, shape=ss))
        else:
            return np.memmap(filename, dtype=self.dtype, order=order,
                             mode=mode, offset=offset, shape=ss)

    def _get_simple_var_xy(self, var, order='F', mode='r'):
        """
        Reads a given 2D variable from the _XY.aux file
        """
        if var in self.auxxyvars:
            fsuffix = '_XY.aux'
            idx = self.auxxyvars.index(var)
            filename = self.file_root + fsuffix
        else:
            raise ValueError(('_get_simple_var_xy: variable'
                              ' %s not available. Available vars:'
                              % (var) + '\n' + repr(self.auxxyvars)))
        # Now memmap the variable
        if not os.path.isfile(filename):
            raise IOError(('_get_simple_var_xy: variable'
                           ' %s should be in %s file, not found!' %
                           (var, filename)))
        # size of the data type
        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * idx * dsize
        return np.memmap(filename, dtype=self.dtype, order=order, mode=mode,
                         offset=offset, shape=(self.nx, self.ny))

    def _get_composite_var(self, var, *args, **kwargs):
        """
        Gets composite variables (will load into memory).
        """
        if var in ['ux', 'uy', 'uz']:  # velocities
            p = self.get_var('p' + var[1], order='F')
            if getattr(self, 'n' + var[1]) < 5 or not self.cstagop:
                return p / self.get_var('r')  # do not recentre for 2D cases
            else:  # will call xdn, ydn, or zdn to get r at cell faces
                return p / cstagger.do(self.get_var('r'), var[1] + 'dn')
        elif var == 'ee':   # internal energy
            return self.get_var('e') / self.get_var('r')
        elif var == 's':   # entropy?
            return np.log(self.get_var('p', *args, **kwargs)) - \
                self.params['gamma'][self.snapInd] * np.log(
                    self.get_var('r', *args, **kwargs))

    def get_quantity(self, quant, *args, **kwargs):
        """
        Calculates a quantity from the simulation quantiables.

        Parameters
        ----------
        quant - string
            Name of the quantity to calculate (see below for some categories).

        Returns
        -------
        array - ndarray
            Array with the dimensions of the simulation.

        Notes
        -----
        All possible rules for quantity names are described in the dictionary
        "self.description", e.g.:

        >>> dd.get_quantity('')
        >>> dd.description.keys()
        >>> dd.description['DERIV']
        """
        quant = quant.lower()
        self.description = {}
        DERIV_QUANT = ['dxup', 'dyup', 'dzup', 'dxdn', 'dydn', 'dzdn']
        self.description['DERIV'] = ('Spatial derivative (Bifrost units). '
                                     'It must start with d and end with: '
                                     ', '.join(DERIV_QUANT))

        CENTRE_QUANT = ['xc', 'yc', 'zc']
        self.description['CENTRE'] = ('Allows to center any vector(Bifrost'
                                      ' units). It must end with ' +
                                      ', '.join(CENTRE_QUANT))

        MODULE_QUANT = ['mod', 'h']  # This one must be called the last
        self.description['MODULE'] = ('Module (starting with mod) or horizontal '
                         '(ending with h) component of vectors (Bifrost units)')

        HORVAR_QUANT = ['horvar']
        self.description['HORVAR'] = ('Horizontal average (Bifrost units).'
                                      ' Starting with: ' + ', '.join(HORVAR_QUANT))

        GRADVECT_QUANT = ['div', 'rot', 'she', 'chkdiv', 'chbdiv', 'chhdiv']
        self.description['GRADVECT'] = ('vectorial derivative opeartions '
            '(Bifrost units). '
            'The following show divergence, rotational, shear, ratio of the '
            'divergence with the maximum of the abs of each spatial derivative, '
            'with the sum of the absolute of each spatial derivative, with '
            'horizontal averages of the absolute of each spatial derivative '
            'respectively when starting with: ' + ', '.join(GRADVECT_QUANT))

        GRADSCAL_QUANT = ['gra']
        self.description['GRADSCAL'] = ('Gradient of a scalar (Bifrost units)'
                ' starts with: ' + ', '.join(GRADSCAL_QUANT))

        SQUARE_QUANT = ['2']  # This one must be called the towards the last
        self.description['SQUARE'] = ('Square of a variable (Bifrost units)'
                ' ends with: ' + ', '.join(SQUARE_QUANT))

        RATIO_QUANT = 'rat'
        self.description['RATIO'] = ('Ratio of two variables (Bifrost units)'
                'have in between: ' + ', '.join(RATIO_QUANT))

        EOSTAB_QUANT = ['ne', 'tg', 'pg', 'kr', 'eps', 'opa', 'temt', 'ent']
        self.description['EOSTAB'] = ('Variables from EOS table. All of them '
            'are in cgs except ne which is in SI. The electron density '
            '[m^-3], temperature [K], pressure [dyn/cm^2], Rosseland opacity '
            '[cm^2/g], scattering probability, opacity, thermal emission and '
            'entropy are as follows: ' + ', '.join(EOSTAB_QUANT))

        TAU_QUANT = 'tau'
        self.description['TAU'] = ('tau at 500 is: ' + ', '.join(TAU_QUANT))

        PROJ_QUANT = ['par', 'per']
        self.description['PROJ'] = ('Projected vectors (Bifrost units).'
            ' Parallel and perpendicular have in the middle the following: ' +
            ', '.join(PROJ_QUANT))

        CURRENT_QUANT = ['ix', 'iy', 'iz', 'wx', 'wy', 'wz']
        self.description['CURRENT'] = ('Calculates currents (bifrost units) or'
            'rotational components of the velocity as follows ' +
            ', '.join(CURRENT_QUANT))

        FLUX_QUANT = ['pfx', 'pfy', 'pfz', 'pfex', 'pfey', 'pfez', 'pfwx',
                      'pfwy', 'pfwz']
        self.description['FLUX'] = ('Poynting flux, Flux emergence, and'
            'Poynting flux from "horizontal" motions: ' +
            ', '.join(FLUX_QUANT))

        PLASMA_QUANT = ['beta', 'va', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                        'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky',
                        'kz']
        self.description['PLASMA'] = ('Plasma beta, alfven velocity (and its'
            'components), sound speed, entropy, kinetic energy flux'
            '(and its components), magnetic and sonic Mach number'
            'pressure scale height, and each component of the total energy'
            'flux (if applicable, Bifrost units): ' +
            ', '.join(PLASMA_QUANT))

        WAVE_QUANT = ['alf', 'fast', 'long']
        self.description['WAVE'] = ('Alfven, fast and longitudinal wave'
            'components (Bifrost units): ' + ', '.join(WAVE_QUANT))

        CYCL_RES = ['n6nhe2', 'n6nhe3', 'nhe2nhe3']
        self.description['CYCL_RES'] = ('Resonant cyclotron frequencies'
            '(only for do_helium) are (SI units): ' + ', '.join(CYCL_RES))

        elemlist = ['h', 'he', 'c', 'o', 'ne', 'na', 'mg', 'al', 'si', 's',
                    'k', 'ca', 'cr', 'fe', 'ni']
        GYROF_QUANT = ['gfe'] + ['gf' + clist for clist in elemlist]
        self.description['GYROF'] = ('gyro freqency are (Hz): ' +
            ', '.join(GYROF_QUANT))

        DEBYE_LN_QUANT = ['debye_ln']
        self.description['DEBYE'] = ('Debye length in ... units:',
            ', '.join(DEBYE_LN_QUANT))

        COULOMB_COL_QUANT = ['coucol' + clist for clist in elemlist]
        self.description['COULOMB_COL'] = ('Coulomb collision frequency in Hz'
            'units: ' + ', '.join(COULOMB_COL_QUANT))

        CROSTAB_QUANT = ['h_' + clist for clist in elemlist]
        for iel in elemlist:
            CROSTAB_QUANT = CROSTAB_QUANT + [
                iel + '_' + clist for clist in elemlist]
        self.description['CROSTAB'] = ('Cross section between species'
            '(in cgs): ' + ', '.join(CROSTAB_QUANT))

        COLFRE_QUANT = ['nu' + clist for clist in CROSTAB_QUANT]
        self.description['COLFRE'] = ('Collision frequency (elastic and charge'
            'exchange) between different species in (cgs): ' +
            ', '.join(COLFRE_QUANT))

        COLFRI_QUANT = ['nu_ni', 'nu_en', 'nu_ei']
        COLFRI_QUANT = COLFRI_QUANT + \
            ['nu' + clist + '_i' for clist in elemlist]
        COLFRI_QUANT = COLFRI_QUANT + \
            ['nu' + clist + '_n' for clist in elemlist]
        self.description['COLFRI'] = ('Collision frequency (elastic and charge'
            'exchange) between fluids in (cgs): ' + ', '.join(COLFRI_QUANT))

        IONP_QUANT = ['n' + clist + '-' for clist in elemlist]
        IONP_QUANT = IONP_QUANT + ['r' + clist + '-' for clist in elemlist]
        self.description['IONP'] = ('densities for specific ionized species as'
            'follow (in SI): ' + ', '.join(IONP_QUANT))

        if quant == '':
            help(self.get_quantity)
            return -1

        if np.size(self.snap) > 1:
            currSnap = self.snap[self.snapInd]
        else:
            currSnap = self.snap

        if RATIO_QUANT in quant:
            # Calculate module of vector quantity
            q = quant[:quant.find(RATIO_QUANT)]
            if q[0] == 'b':
                if not self.do_mhd:
                    raise ValueError("No magnetic field available.")
            result = self.get_var(q)
            q = quant[quant.find(RATIO_QUANT) + 3:]
            if q[0] == 'b':
                if not self.do_mhd:
                    raise ValueError("No magnetic field available.")
            return result / (self.get_var(q) + 1e-19)

        elif quant[0] == 'd' and quant[-4:] in DERIV_QUANT:
            # Calculate derivative of quantity
            axis = quant[-3]
            q = quant[1:-4]  # base variable
            var = self.get_var(q)

            def deriv_loop(var, quant):
                return cstagger.do(var, 'd' + quant[0])

            if getattr(self, 'n' + axis) < 5:  # 2D or close
                print('(WWW) get_quantity: DERIV_QUANT: '
                      'n%s < 5, derivative set to 0.0' % axis)
                return np.zeros_like(var)
            else:
                if self.numThreads > 1:
                    if self.verbose:
                        print('Threading')
                    quantlist = [quant[-4:] for numb in range(self.numThreads)]
                    if axis != 'z':
                        return threadQuantity_z(
                            deriv_loop, self.numThreads, var, quantlist)
                    else:
                        return threadQuantity_y(
                            deriv_loop, self.numThreads, var, quantlist)
                else:
                    if self.lowbus:
                        output = np.zeros_like(var)
                        if axis != 'z':
                            for iiz in range(self.nz):
                                output[:, :, iiz] = np.reshape(
                                   cstagger.do(var[:, :, iiz].reshape((self.nx, self.ny, 1)),
                                               'd' + quant[-4:]),
                                   (self.nx, self.ny))
                        else:
                            for iiy in range(self.ny):
                                output[:, iiy, :] = np.reshape(
                                   cstagger.do(var[:, iiy, :].reshape((self.nx, 1, self.nz)),
                                               'd' + quant[-4:]),
                                   (self.nx, self.nz))

                        return output
                    else:
                        return cstagger.do(var, 'd' + quant[-4:])

        elif quant[-2:] in CENTRE_QUANT:
            # This brings a given vector quantity to cell centres
            axis = quant[-2]
            q = quant[:-1]  # base variable
            if q[:-1] == 'i' or q == 'e':
                AXIS_TRANSFORM = {'x': ['yup', 'zup'],
                                  'y': ['xup', 'zup'],
                                  'z': ['xup', 'yup']}
            else:
                AXIS_TRANSFORM = {'x': ['xup'],
                                  'y': ['yup'],
                                  'z': ['zup']}
            transf = AXIS_TRANSFORM[axis]

            var = self.get_var(q, **kwargs)

            # 2D
            if getattr(self, 'n' + axis) < 5 or self.cstagop is False:
                return var
            else:
                if len(transf) == 2:
                    if self.lowbus:
                        output = np.zeros_like(var)
                        if transf[0][0] != 'z':
                            for iiz in range(self.nz):
                                output[:, :, iiz] = np.reshape(cstagger.do(
                                    var[:, :, iiz].reshape(
                                        (self.nx, self.ny, 1)),
                                    transf[0]), (self.nx, self.ny))
                        else:
                            for iiy in range(self.ny):
                                output[:, iiy, :] = np.reshape(cstagger.do(
                                    var[:, iiy, :].reshape(
                                        (self.nx, 1, self.nz)),
                                    transf[0]), (self.nx, self.nz))

                        if transf[1][0] != 'z':
                            for iiz in range(self.nz):
                                output[:, :, iiz] = np.reshape(cstagger.do(
                                    output[:, :, iiz].reshape((self.nx, self.ny, 1)),
                                    transf[1]), (self.nx, self.ny))
                        else:
                            for iiy in range(self.ny):
                                output[:, iiy, :] = np.reshape(cstagger.do(
                                    output[:, iiy, :].reshape(
                                        (self.nx, 1, self.nz)),
                                    transf[1]), (self.nx, self.nz))
                        return output
                    else:
                        tmp = cstagger.do(var, transf[0])
                        return cstagger.do(tmp, transf[1])
                else:
                    if self.lowbus:
                        output = np.zeros_like(var)
                        if axis != 'z':
                            for iiz in range(self.nz):
                                output[:, :, iiz] = np.reshape(cstagger.do(
                                    var[:, :, iiz].reshape(
                                        (self.nx, self.ny, 1)),
                                    transf[0]), (self.nx, self.ny))
                        else:
                            for iiy in range(self.ny):
                                output[:, iiy, :] = np.reshape(cstagger.do(
                                    var[:, iiy, :].reshape(
                                        (self.nx, 1, self.nz)),
                                    transf[0]), (self.nx, self.nz))
                        return output
                    else:
                        return cstagger.do(var, transf[0])

        elif quant[:6] in GRADVECT_QUANT or quant[:3] in GRADVECT_QUANT:
            if quant[:3] == 'chk':
                q = quant[6:]  # base variable
                if getattr(self, 'nx') < 5:  # 2D or close
                    varx = np.zeros_like(self.r)
                else:
                    varx = self.get_var('d' + q + 'xdxup')

                if getattr(self, 'ny') > 5:
                    vary = self.get_var('d' + q + 'ydyup')
                else:
                    vary = np.zeros_like(varx)

                if getattr(self, 'nz') > 5:
                    varz = self.get_var('d' + q + 'zdzup')
                else:
                    varz = np.zeros_like(varx)
                return np.abs(varx + vary + varx) / (np.maximum(
                    np.abs(varx), np.abs(vary), np.abs(varz)) + 1.0e-20)

            elif quant[:3] == 'chb':
                q = quant[6:]  # base variable
                varx = self.get_var(q + 'x')
                vary = self.get_var(q + 'y')
                varz = self.get_var(q + 'z')
                if getattr(self, 'nx') < 5:  # 2D or close
                    result = np.zeros_like(varx)
                else:
                    result = self.get_var('d' + q + 'xdxup')

                if getattr(self, 'ny') > 5:
                    result += self.get_var('d' + q + 'ydyup')

                if getattr(self, 'nz') > 5:
                    result += self.get_var('d' + q + 'zdzup')

                return np.abs(result / (np.sqrt(
                    varx * varx + vary * vary + varz * varz) + 1.0e-20))

            elif quant[:3] == 'chh':
                q = quant[6:]  # base variable
                varx = self.get_var(q + 'x')
                vary = self.get_var(q + 'y')
                varz = self.get_var(q + 'z')
                if getattr(self, 'nx') < 5:  # 2D or close
                    result = np.zeros_like(varx)
                else:
                    result = self.get_var('d' + q + 'xdxup')

                if getattr(self, 'ny') > 5:
                    result += self.get_var('d' + q + 'ydyup')

                if getattr(self, 'nz') > 5:
                    result += self.get_var('d' + q + 'zdzup')

                for iiz in range(0, self.nz):
                    result[:, :, iiz] = np.abs(result[:, :, iiz]) / np.mean((
                        np.sqrt(varx[:, :, iiz]**2 + vary[:, :, iiz]**2 +
                                varz[:, :, iiz]**2)))
                return result

            elif quant[:3] == 'div':  # divergence of vector quantity
                q = quant[3:]  # base variable
                if getattr(self, 'nx') < 5:  # 2D or close
                    result = np.zeros_like(self.r)
                else:
                    result = self.get_var('d' + q + 'xdxup')
                if getattr(self, 'ny') > 5:
                    result += self.get_var('d' + q + 'ydyup')
                if getattr(self, 'nz') > 5:
                    result += self.get_var('d' + q + 'zdzup')

            elif quant[:3] == 'rot' or quant[:3] == 'she':
                q = quant[3:-1]  # base variable
                qaxis = quant[-1]
                if qaxis == 'x':
                    if getattr(self, 'ny') < 5:  # 2D or close
                        result = np.zeros_like(self.r)
                    else:
                        result = self.get_var('d' + q + 'zdyup')
                    if getattr(self, 'nz') > 5:
                        if quant[:3] == 'rot':
                            result -= self.get_var('d' + q + 'ydzup')
                        else:  # shear
                            result += self.get_var('d' + q + 'ydzup')
                elif qaxis == 'y':
                    if getattr(self, 'nz') < 5:  # 2D or close
                        result = np.zeros_like(self.r)
                    else:
                        result = self.get_var('d' + q + 'xdzup')
                    if getattr(self, 'nx') > 5:
                        if quant[:3] == 'rot':
                            result -= self.get_var('d' + q + 'zdxup')
                        else:  # shear
                            result += self.get_var('d' + q + 'zdxup')
                elif qaxis == 'z':
                    if getattr(self, 'nx') < 5:  # 2D or close
                        result = np.zeros_like(self.r)
                    else:
                        result = self.get_var('d' + q + 'ydxup')
                    if getattr(self, 'ny') > 5:
                        if quant[:3] == 'rot':
                            result -= self.get_var('d' + q + 'xdyup')
                        else:  # shear
                            result += self.get_var('d' + q + 'xdyup')
            return result

        elif quant[:3] in GRADSCAL_QUANT:
            if quant[:3] == 'gra':
                q = quant[3:]  # base variable
                if getattr(self, 'nx') < 5:  # 2D or close
                    result = np.zeros_like(self.r)
                else:
                    result = self.get_var('d' + q + 'dxup')
                if getattr(self, 'ny') > 5:
                    result += self.get_var('d' + q + 'dyup')
                if getattr(self, 'nz') > 5:
                    result += self.get_var('d' + q + 'dzup')
            return result

        elif quant[:6] in HORVAR_QUANT:
            # Compares the variable with the horizontal mean
            if quant[:6] == 'horvar':
                result = np.zeros_like(self.r)
                result += self.get_var(quant[6:])  # base variable
                horv = np.mean(np.mean(result, 0), 0)
                for iix in range(0, getattr(self, 'nx')):
                    for iiy in range(0, getattr(self, 'ny')):
                        result[iix, iiy, :] = result[iix, iiy, :] / horv[:]
            return result

        elif quant in EOSTAB_QUANT:
            # unit conversion to SI
            # to g/cm^3
            ur = self.params['u_r'][self.snapInd]
            ue = self.params['u_ee'][self.snapInd]        # to erg/g
            if self.hion and quant == 'ne':
                return self.get_var('hionne')
            rho = self.get_var('r')
            rho = rho * ur
            ee = self.get_var('ee')
            ee = ee * ue
            if self.verbose:
                print(quant + ' interpolation...')

            fac = 1.0
            # JMS Why SI?? SI seems to work with bifrost_uvotrt.
            if quant == 'ne':
                fac = 1.e6  # cm^-3 to m^-3
            if quant in ['eps', 'opa', 'temt']:
                radtab = True
            else:
                radtab = False
            eostab = Rhoeetab(fdir=self.fdir, radtab=radtab)
            return eostab.tab_interp(
                rho, ee, order=1, out=quant) * fac

        elif quant[1:4] in PROJ_QUANT:
            # projects v1 onto v2
            v1 = quant[0]
            v2 = quant[4]
            x_a = self.get_var(v1 + 'xc', self.snap)
            y_a = self.get_var(v1 + 'yc', self.snap)
            z_a = self.get_var(v1 + 'zc', self.snap)
            x_b = self.get_var(v2 + 'xc', self.snap)
            y_b = self.get_var(v2 + 'yc', self.snap)
            z_b = self.get_var(v2 + 'zc', self.snap)
            # can be used for threadQuantity() or as is
            def proj_task(x1, y1, z1, x2, y2, z2):
                v2Mag = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
                v2x, v2y, v2z = x2 / v2Mag, y2 / v2Mag, z2 / v2Mag
                parScal = x1 * v2x + y1 * v2y + z1 * v2z
                parX, parY, parZ = parScal * v2x, parScal * v2y, parScal * v2z
                result = np.abs(parScal)
                if quant[1:4] == 'per':
                    perX = x1 - parX
                    perY = y1 - parY
                    perZ = z1 - parZ
                    v1Mag = np.sqrt(perX**2 + perY**2 + perZ**2)
                    result = v1Mag
                return result

            if self.numThreads > 1:
                if self.verbose:
                    print('Threading')

                return threadQuantity(proj_task, self.numThreads,
                                      x_a, y_a, z_a, x_b, y_b, z_b)
            else:
                return proj_task(x_a, y_a, z_a, x_b, y_b, z_b)

        elif quant in CURRENT_QUANT:
            # Calculate derivative of quantity
            axis = quant[-1]
            if quant[0] == 'i':
                q = 'b'
            else:
                q = 'u'
            try:
                var = getattr(self, quant)
            except AttributeError:
                if axis == 'x':
                    varsn = ['z', 'y']
                    derv = ['dydn', 'dzdn']
                elif axis == 'y':
                    varsn = ['x', 'z']
                    derv = ['dzdn', 'dxdn']
                elif axis == 'z':
                    varsn = ['y', 'x']
                    derv = ['dxdn', 'dydn']

                # 2D or close
                if (getattr(self, 'n' + varsn[0]) < 5) or (getattr(self, 'n' + varsn[1]) < 5):
                    return np.zeros_like(self.r)
                else:
                    return (self.get_var('d' + q + varsn[0] + derv[0]) -
                            self.get_var('d' + q + varsn[1] + derv[1]))

        elif quant in FLUX_QUANT:
            axis = quant[-1]
            if axis == 'x':
                varsn = ['z', 'y']
            elif axis == 'y':
                varsn = ['x', 'z']
            elif axis == 'z':
                varsn = ['y', 'x']
            if 'pfw' in quant or len(quant) == 3:
                var = - self.get_var('b' + axis + 'c') * (
                    self.get_var('u' + varsn[0] + 'c') *
                    self.get_var('b' + varsn[0] + 'c') +
                    self.get_var('u' + varsn[1] + 'c') *
                    self.get_var('b' + varsn[1] + 'c'))
            else:
                var = np.zeros_like(self.r)
            if 'pfe' in quant or len(quant) == 3:
                var += self.get_var('u' + axis + 'c') * (
                    self.get_var('b' + varsn[0] + 'c')**2 +
                    self.get_var('b' + varsn[1] + 'c')**2)
            return var

        elif quant in PLASMA_QUANT:
            if quant in ['hp', 's', 'cs', 'beta']:
                var = self.get_var('p')
                if quant == 'hp':
                    if getattr(self, 'nx') < 5:
                        return np.zeros_like(var)
                    else:
                        return 1. / (cstagger.do(var, 'ddzup') + 1e-12)
                elif quant == 'cs':
                    return np.sqrt(self.params['gamma'][self.snapInd] *
                                   var / self.get_var('r'))
                elif quant == 's':
                    return (np.log(var) - self.params['gamma'][self.snapInd] *
                            np.log(self.get_var('r')))
                elif quant == 'beta':
                    return 2 * var / self.get_var('b2')

            if quant in ['mn', 'man']:
                var = self.get_var('modu')
                if quant == 'mn':
                    return var / (self.get_var('cs') + 1e-12)
                else:
                    return var / (self.get_var('va') + 1e-12)

            if quant in ['va', 'vax', 'vay', 'vaz']:
                var = self.get_var('r')
                if len(quant) == 2:
                    return self.get_var('modb') / np.sqrt(var)
                else:
                    axis = quant[-1]
                    return np.sqrt(self.get_var('b' + axis + 'c') ** 2 / var)

            if quant in ['hx', 'hy', 'hz', 'kx', 'ky', 'kz']:
                axis = quant[-1]
                var = self.get_var('p' + axis + 'c')
                if quant[0] == 'h':
                    return ((self.get_var('e') + self.get_var('p')) /
                             self.get_var('r') * var)
                else:
                    return self.get_var('u2') * var * 0.5

            if quant in ['ke']:
                var = self.get_var('r')
                return self.get_var('u2') * var * 0.5

        elif quant == 'tau':

            return self.calc_tau()

        elif quant in WAVE_QUANT:
            bx = self.get_var('bxc')
            by = self.get_var('byc')
            bz = self.get_var('bzc')
            bMag = np.sqrt(bx**2 + by**2 + bz**2)
            bx, by, bz = bx / bMag, by / bMag, bz / bMag  # b is already centered
            # unit vector of b
            unitB = np.stack((bx, by, bz))

            if quant == 'alf':
                uperb = self.get_var('uperb')
                uperbVect = uperb * unitB
                # cross product (uses cstagger bc no variable gets uperbVect)
                curlX = (cstagger.do(cstagger.do(uperbVect[2], 'ddydn'), 'yup') -
                         cstagger.do(cstagger.do(uperbVect[1], 'ddzdn'), 'zup'))
                curlY = (-cstagger.do(cstagger.do(uperbVect[2], 'ddxdn'), 'xup')
                         +cstagger.do(cstagger.do(uperbVect[0], 'ddzdn'), 'zup'))
                curlZ = (cstagger.do(cstagger.do(uperbVect[1], 'ddxdn'), 'xup') -
                         cstagger.do(cstagger.do(uperbVect[0], 'ddydn'), 'yup'))
                curl = np.stack((curlX, curlY, curlZ))
                # dot product
                result = np.abs((unitB * curl).sum(0))
            elif quant == 'fast':
                uperb = self.get_var('uperb')
                uperbVect = uperb * unitB

                result = np.abs(cstagger.do(cstagger.do(
                    uperbVect[0], 'ddxdn'), 'xup') + cstagger.do(cstagger.do(
                        uperbVect[1], 'ddydn'), 'yup') + cstagger.do(
                            cstagger.do(uperbVect[2], 'ddzdn'), 'zup'))
            else:
                dot1 = self.get_var('uparb')
                grad = np.stack((cstagger.do(cstagger.do(dot1, 'ddxdn'),
                                             'xup'), cstagger.do(cstagger.do(
                                                 dot1, 'ddydn'), 'yup'),
                                 cstagger.do(cstagger.do(dot1, 'ddzdn'),
                                             'zup')))
                result = np.abs((unitB * grad).sum(0))
            return result

        elif quant in CYCL_RES:
            if self.hion and self.heion:
                posn = ([pos for pos, char in enumerate(quant) if char == 'n'])
                q2 = quant[posn[-1]:]
                q1 = quant[:posn[-1]]
                if self.hion:
                    nel = self.get_var('hionne')
                else:
                    nel = self.get_var('nel')
                var2 = self.get_var(q2)
                var1 = self.get_var(q1)
                z1= 1.0
                z2= float(quant[-1])
                if q1[:3] == 'n6':
                    omega1 = self.get_var('gfh2')
                else:
                    omega1 = self.get_var('gf'+q1[1:])
                omega2 = self.get_var('gf'+q2[1:])
                return (z1 * var1 * omega2 + z2 * var2 * omega1) / nel
            else:
                raise ValueError(('get_quantity: This variable is only '
                                  'avaiable if do_hion and do_helium is true'))

        elif quant in DEBYE_LN_QUANT:
            tg = self.get_var('tg')
            part = np.copy(self.get_var('ne'))
            # We are assuming a single charge state:
            for iele in elemlist:
                part += self.get_var('n' + iele + '-2')
            if self.heion:
                part += 4.0 * self.get_var('nhe3')
            # check units of n
            return np.sqrt(self.uni.permsi / self.uni.qsi_electron**2 /
                           (self.uni.ksi_b * tg.astype('Float64') *
                            part.astype('Float64') + 1.0e-20))

        elif ''.join([i for i in quant if not i.isdigit()]) in GYROF_QUANT:
            if quant == 'gfe':
                return self.get_var('modb') * self.uni.usi_b * \
                    self.uni.qsi_electron / (self.uni.msi_e)
            else:
                ion = float(''.join([i for i in quant if i.isdigit()]))
                return self.get_var('modb') * self.uni.usi_b * \
                    self.uni.qsi_electron * \
                    (ion - 1.0) / \
                    (self.uni.weightdic[quant[2:-1]] * self.uni.amusi)

        elif quant in COULOMB_COL_QUANT:
            iele = np.where(COULOMB_COL_QUANT == quant)
            tg = self.get_var('tg')
            nel = np.copy(self.get_var('ne'))
            elem = quant.replace('coucol', '')

            const = (self.uni.pi * self.uni.qsi_electron ** 4 /
                     ((4.0 * self.uni.pi * self.uni.permsi)**2 *
                      np.sqrt(self.uni.weightdic[elem] * self.uni.amusi *
                             (2.0 * self.uni.ksi_b) ** 3) + 1.0e-20))


            return (const * nel.astype('Float64') *
                    np.log(12.0 * self.uni.pi * nel.astype('Float64') *
                    self.get_var('debye_ln').astype('Float64') + 1e-50) /
                    (np.sqrt(tg.astype('Float64')**3) + 1.0e-20))

        elif quant in CROSTAB_QUANT:
            tg = self.get_var('tg')
            elem = quant.split('_')
            spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
            spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
            cross_tab = ''
            crossunits = 2.8e-17
            if spic1 == 'h':
                if spic2 == 'h':
                    cross_tab = 'p-h-elast.txt'
                elif spic2 == 'he':
                    cross_tab = 'p-he.txt'
                elif spic2 == 'e':
                    cross_tab = 'e-h.txt'
                    crossunits = 1e-16
                else:
                    cross = self.uni.weightdic[spic2] / self.uni.weightdic['h'] * \
                        self.uni.cross_p * np.ones(np.shape(tg))
            elif spic1 == 'he':
                if spic2 == 'h':
                    cross_tab = 'p-h-elast.txt'
                elif spic2 == 'he':
                    cross_tab = 'he-he.txt'
                    crossunits = 1e-16
                elif spic2 == 'e':
                    cross_tab = 'e-he.txt'
                else:
                    cross = self.uni.weightdic[spic2] / self.uni.weightdic['he'] * \
                        self.uni.cross_he * np.ones(np.shape(tg))
            elif spic1 == 'e':
                if spic2 == 'h':
                    cross_tab = 'e-h.txt'
                elif spic2 == 'he':
                    cross_tab = 'e-he.txt'
            if cross_tab != '':
                crossobj = Cross_sect(cross_tab=[cross_tab])
                cross = crossunits * crossobj.tab_interp(tg)
            else:
                cross = self.uni.weightdic[spic2] / self.uni.weightdic['h'] * \
                    self.uni.cross_p * np.ones(np.shape(tg))
            try:
                return cross
            except Exception:
                print('(WWW) cross-section: wrong combination of species')

        elif ''.join([i for i in quant if not i.isdigit()]) in COLFRE_QUANT:

            elem = quant.split('_')
            spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
            ion1 = ''.join([i for i in elem[0] if i.isdigit()])
            spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
            ion2 = ''.join([i for i in elem[1] if i.isdigit()])

            spic1 = spic1[2:]
            crossarr = self.get_var('%s_%s' % (spic1, spic2))
            nspic2 = self.get_var('n%s-%s' % (spic2, ion2))

            tg = self.get_var('tg')
            awg1 = self.uni.weightdic[spic1] * self.uni.amu
            awg2 = self.uni.weightdic[spic2] * self.uni.amu
            scr1 = np.sqrt(8.0 * self.uni.kboltzmann * tg / self.uni.pi)


            return crossarr * np.sqrt((awg1 + awg2) / (awg1 * awg2)) *\
                scr1 * nspic2 * (awg1 / (awg1 + awg1))

        elif ''.join([i for i in quant if not i.isdigit()]) in COLFRI_QUANT:
            if quant == 'nu_ni':
                result = self.uni.m_h * self.get_var('nh-1') * \
                    self.get_var('nuh1_i') + \
                    self.uni.m_he * self.get_var('nhe-1') * self.get_var('nuhe1_i')
            elif quant == 'nu_ei':
                if self.hion:
                    nel = self.get_var('hionne')
                else:
                    nel = self.get_var('nel')
                culblog = 23. + 1.5 * np.log(self.get_var('tg') / 1.e6) - \
                    0.5 * np.log(nel / 1e6)

                result = 3.759 * nel / (self.get_var('tg')**(1.5)) * culblog

            elif quant == 'nu_en':
                if self.hion:
                    nel = self.get_var('hionne')
                else:
                    nel = self.get_var('nel')
                culblog = 23. + 1.5 * np.log(self.get_var('tg') / 1.e6) - \
                    0.5 * np.log(nel / 1e6)
                scr1 = 3.759 * nel / (self.get_var('tg')**(1.5)) * culblog
                scr2 = 0.0 * nel
                for ielem in elemlist:
                    scr2 += self.get_var('n%s-%s' % (ielem, 2))
                if self.heion and quant[-2:] == '_i':
                    scr2 += self.get_var('%s_%s' % (elem[0], 'he3'))
                result = 5.2e-11 * scr2 / nel * self.get_var('tg')**2 / \
                    culblog * scr1

            else:
                if quant[-2:] == '_i':
                    lvl = '2'
                else:
                    lvl = '1'
                elem = quant.split('_')
                result = np.zeros(np.shape(self.r))
                for ielem in elemlist:
                    if elem[0][2:] != '%s%s' % (ielem, lvl):
                        result += self.get_var('%s_%s%s' %
                                               (elem[0], ielem, lvl))
                if self.heion and quant[-2:] == '_i':
                    result += self.get_var('%s_%s' % (elem[0], 'he3'))
            return result

        elif ''.join([i for i in quant if not i.isdigit()]) in IONP_QUANT:
            elem = quant.split('_')
            spic = ''.join([i for i in elem[0] if not i.isdigit()])
            lvl = ''.join([i for i in elem[0] if i.isdigit()])
            if self.hion and spic[1:-1] == 'h':
                if quant[0] == 'n':
                    mass = 1.0
                else:
                    mass = self.uni.m_h
                if lvl == '1':

                    return mass * (self.get_var('n1') +
                                   self.get_var('n2') + self.get_var('n3') +
                                   self.get_var('n4') + self.get_var('n5'))
                else:
                    return mass * self.get_var('n6')
            elif self.heion and spic[1:-1] == 'he':
                if quant[0] == 'n':
                    mass = 1.0
                else:
                    mass = self.uni.m_he
                if self.verbose:
                    print('get_var: reading nhe%s' % lvl)
                return mass * self.get_var('nhe%s' % lvl)

            else:
                tg = self.get_var('tg')
                r = self.get_var('r')
                nel = self.get_var('ne') / 1e6  # 1e6 conversion from SI to cgs

                if quant[0] == 'n':
                    dens = False
                else:
                    dens = True

                return ionpopulation(r, nel, tg, elem=spic[1:-1], lvl=lvl,
                                     dens=dens)

        elif ((quant[:3] in MODULE_QUANT) or (
                quant[-1] in MODULE_QUANT) or (
                quant[-1] in SQUARE_QUANT and not(quant in CYCL_RES))):
            # Calculate module of vector quantity
            if (quant[:3] in MODULE_QUANT):
                q = quant[3:]
            else:
                q = quant[:-1]
            if q == 'b':
                if not self.do_mhd:
                    raise ValueError("No magnetic field available.")
            result = self.get_var(q + 'xc') ** 2
            result += self.get_var(q + 'yc') ** 2
            if not(quant[-1] in MODULE_QUANT):
                result += self.get_var(q + 'zc') ** 2

            if (quant[:3] in MODULE_QUANT) or (quant[-1] in MODULE_QUANT):
                return np.sqrt(result)
            elif quant[-1] in SQUARE_QUANT:
                return result

        else:
            raise ValueError(('get_quantity: do not know (yet) how to '
                              'calculate quantity %s. Note that simple_var '
                              'available variables are: %s.\nIn addition, '
                              'get_quantity can read others computed variables '
                              'see e.g. help(self.get_quantity) for guidance'
                              '.' % (quant, repr(self.simple_vars))))

    def calc_tau(self):
        """
        Calculates optical depth.

        DEPRECATED, DO NOT USE.
        """
        warnings.warn("Use of calc_tau is discouraged. It is model-dependent, "
                      "inefficient and slow, and will give wrong results in "
                      "many scenarios. DO NOT USE.")

        if not hasattr(self, 'z'):
            print('(WWW) get_tau needs the height (z) in Mm (units code)')

        # grph = 2.38049d-24 uni.GRPH
        # bk = 1.38e-16 uni.KBOLTZMANN
        # EV_TO_ERG=1.60217733E-12 uni.EV_TO_ERG
        if not hasattr(self, 'ne'):
            nel = self.get_var('ne')
        else:
            nel = self.ne

        if not hasattr(self, 'tg'):
            tg = self.get_var('tg')
        else:
            tg = self.tg

        if not hasattr(self, 'r'):
            rho = self.get_var('r') * self.uni.u_r
        else:
            rho = self.r * self.uni.u_r

        tau = np.zeros((self.nx, self.ny, self.nz)) + 1.e-16
        xhmbf = np.zeros((self.nz))
        const = (1.03526e-16 / self.uni.grph) * 2.9256e-17 / 1e6
        for iix in range(self.nx):
            for iiy in range(self.ny):
                for iiz in range(self.nz):
                    xhmbf[iiz] = const * nel[iix, iiy, iiz] / \
                        tg[iix, iiy, iiz]**1.5 * np.exp(0.754e0 *
                        self.uni.ev_to_erg / self.uni.kboltzmann /
                        tg[iix, iiy, iiz]) * rho[iix, iiy, iiz]

                for iiz in range(1, self.nz):
                    tau[iix, iiy, iiz] = tau[iix, iiy, iiz - 1] + 0.5 *\
                        (xhmbf[iiz] + xhmbf[iiz - 1]) *\
                        np.abs(self.dz1d[iiz]) * 1.0e8
        return tau

    def get_electron_density(self, sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Gets electron density

        Parameters
        ----------
        self : BifrostData instance
            A BifrostData object loaded for a given snapshot.
        sx, sy, sz : slice objecs
            Slice objects for x, y, and z dimensions, when not all points
            are needed.
        """
        from astropy.units import Quantity
        if self.hion:
            ne = self.get_var('hionne')[sx, sy, sz]
        else:
            ee = self.get_var('ee')[sx, sy, sz]
            ee = ee * self.uni.u_ee
            eostab = Rhoeetab(fdir=self.fdir)
            rho = self.r[sx, sy, sz] * self.uni.u_r   # to cm^-3
            ne = eostab.tab_interp(rho, ee, order=1)
        return Quantity(ne, unit='1/cm3')

    def get_hydrogen_pops(self, sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Gets hydrogen populations, or total number of hydrogen atoms,
        if hydrogen populations not available.

        Parameters
        ----------
        sx, sy, sz : slice objecs
            Slice objects for x, y, and z dimensions, when not all points
            are needed.
        """
        from astropy.units import Quantity
        if self.hion:
            shape = [6, ]
            # calculate size of slices to determine array shape
            for item, n in zip([sx, sy, sz], [self.nx, self.ny, self.nz]):
                slice_size = np.mgrid[item].size
                if slice_size == 0:
                    slice_size = n
                shape.append(slice_size)
            nh = np.empty(shape, dtype='Float32')
            for k in range(6):
                nv = self.get_var('n%i' % (k + 1))
                nh[k] = nv[sx, sy, sz]
        else:
            rho = self.r[sx, sy, sz] * self.uni.u_r
            subsfile = os.path.join(self.fdir, 'subs.dat')
            tabfile = os.path.join(self.fdir, self.params['tabinputfile'][self.snapInd].strip())
            tabparams = []
            if os.access(tabfile, os.R_OK):
                tabparams = read_idl_ascii(tabfile)
            if 'abund' in tabparams and 'aweight' in tabparams:
                abund = np.array(tabparams['abund'].split()).astype('f')
                aweight = np.array(tabparams['aweight'].split()).astype('f')
                grph = calc_grph(abund, aweight)
            elif os.access(subsfile, os.R_OK):
                grph = subs2grph(subsfile)
            else:
                grph = 2.380491e-24
            nh = rho / grph
            nh = nh[None]  # add extra empty dimension when nhydr = 1
        return Quantity(nh, unit='1/cm3')

    def write_rh15d(self, outfile, desc=None, append=True, sx=slice(None),
                    sy=slice(None), sz=slice(None)):
        """
        Writes snapshot in RH 1.5D format.
        Parameters
        ----------
        outfile - string
            File name to write
        append - bool, optional
            If True (default) will append output as a new snapshot in file.
            Otherwise, creates new file (will fail if file exists).
        desc - string, optional
            Description string
        sx, sy, sz - slice object
            Slice objects for x, y, and z dimensions, when not all points
            are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
            for every second point up to 100.
        Returns
        -------
        None.
        """
        from . import rh15d
        verbose = self.verbose
        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=4)
            except ModuleNotFoundError:
                verbose = False
        # unit conversion to SI
        # Comment to Tiago, params['units'] is hard coded. I
        # strongly recoment to use Bifrost_units.
        ul = self.params['u_l'][self.snapInd] / 1.e2  # to metres
        # to g/cm^3  (for ne_rt_table)
        ur = self.params['u_r'][self.snapInd]
        ut = self.params['u_t'][self.snapInd]         # to seconds
        uv = ul / ut
        ub = self.params['u_b'][self.snapInd] * 1e-4  # to Tesla
        ue = self.params['u_ee'][self.snapInd]        # to erg/g
        if verbose:
            pbar.set_description("Slicing and unit conversion")
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]

        if self.do_mhd:
            Bx = cstagger.xup(self.bx)[sx, sy, sz]
            By = cstagger.yup(self.by)[sx, sy, sz]
            Bz = cstagger.zup(self.bz)[sx, sy, sz]
            # Change sign of Bz (because of height scale) and By
            # (to make right-handed system)
            Bx = Bx * ub
            By = -By * ub
            Bz = -Bz * ub
        else:
            Bx = By = Bz = None

        vz = cstagger.zup(self.pz)[sx, sy, sz] / rho
        vz *= -uv
        x = self.x[sx] * ul
        y = self.y[sy] * (-ul)
        z = self.z[sz] * (-ul)
        if verbose:
            pbar.update()
            pbar.set_description("Getting hydrogen populations")
        nh = self.get_hydrogen_pops(sx, sy, sz).to_value('1/m3')
        if verbose:
            pbar.update()
            pbar.set_description("Getting electron density")
        ne = self.get_electron_density(sx, sy, sz).to_value('1/m3')
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.file_root, repr(sx), repr(sy), repr(sz))
            if self.hion:
                desc = 'hion ' + desc
        # write to file
        if verbose:
            pbar.update()
            pbar.set_description("Writing to file")
        rh15d.make_xarray_atmos(outfile, temp, vz, z, nH=nh, ne=ne, x=x, y=y,
                                append=append, Bx=Bx, By=By, Bz=Bz, desc=desc,
                                snap=self.snap)
        if verbose:
            pbar.update()

    def write_multi3d(self, outfile, mesh='mesh.dat', desc=None,
                      sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Writes snapshot in Multi3D format.
        Parameters
        ----------
        outfile - string
            File name to write
        mesh - string, optional
            File name of the mesh file to write.
        desc - string, optional
            Description string
        sx, sy, sz - slice object
            Slice objects for x, y, and z dimensions, when not all points
            are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
            for every second point up to 100.
        Returns
        -------
        None.
        """
        from .multi3d import Multi3dAtmos
        # unit conversion to cgs and km/s
        ul = self.params['u_l'][self.snapInd]   # to cm
        ur = self.params['u_r'][self.snapInd]   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t'][self.snapInd]   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ue = self.params['u_ee'][self.snapInd]  # to erg/g
        nh = None
        if self.verbose:
            print('Slicing and unit conversion...')
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]
        # Change sign of vz (because of height scale) and vy (to make
        # right-handed system)
        vx = cstagger.xup(self.px)[sx, sy, sz] / rho
        vx *= uv
        vy = cstagger.yup(self.py)[sx, sy, sz] / rho
        vy *= -uv
        vz = cstagger.zup(self.pz)[sx, sy, sz] / rho
        vz *= -uv
        rho = rho * ur  # to cgs
        x = self.x[sx] * ul
        y = self.y[sy] * ul
        z = self.z[sz] * (-ul)
        nh = self.get_hydrogen_pops(sx, sy, sz).to_value('1/cm3')
        ne = self.get_electron_density(sx, sy, sz).to_value('1/cm3')
        # write to file
        print('Write to file...')
        nx, ny, nz = temp.shape
        fout = Multi3dAtmos(outfile, nx, ny, nz, mode="w+")
        fout.ne[:] = ne
        fout.temp[:] = temp
        fout.vx[:] = vx
        fout.vy[:] = vy
        fout.vz[:] = vz
        fout.rho[:] = rho
        # write mesh?
        if mesh:
            fout2 = open(mesh, "w")
            fout2.write("%i\n" % nx)
            x.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % ny)
            y.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % nz)
            z.tofile(fout2, sep="  ", format="%11.5e")
            fout2.close()


class Create_new_br_files:
    def write_mesh(self, x=None, y=None, z=None, nx=None, ny=None, nz=None,
                   dx=None, dy=None, dz=None, meshfile="newmesh.mesh"):
        """
        Writes mesh to ascii file.
        """
        def __xxdn(f):
            '''
            f is centered on (i-.5,j,k)
            '''
            nx = len(f)
            d = -5. / 2048
            c = 49. / 2048
            b = -245. / 2048
            a = .5 - b - c - d
            x = (a * (f + np.roll(f, 1)) +
                 b * (np.roll(f, -1) + np.roll(f, 2)) +
                 c * (np.roll(f, -2) + np.roll(f, 3)) +
                 d * (np.roll(f, -3) + np.roll(f, 4)))
            for i in range(0, 4):
                x[i] = x[4] - (4 - i) * (x[5] - x[4])
            for i in range(1, 4):
                x[nx - i] = x[nx - 4] + i * (x[nx - 4] - x[nx - 5])

            x[nx-3:] = x[nx-3:][::-1] # fixes order in the tail of x
            return x

        def __ddxxup(f, dx=None):
            '''
            X partial up derivative
            '''
            if dx is None:
                dx = 1.
            nx = len(f)
            d = -75. / 107520. / dx
            c = 1029 / 107520. / dx
            b = -8575 / 107520. / dx
            a = 1. / dx - 3 * b - 5 * c - 7 * d
            x = (a * (np.roll(f, -1) - f) +
                 b * (np.roll(f, -2) - np.roll(f, 1)) +
                 c * (np.roll(f, -3) - np.roll(f, 2)) +
                 d * (np.roll(f, -4) - np.roll(f, 3)))
            x[:3] = x[3]
            for i in range(1, 5):
                x[nx - i] = x[nx - 5]
            return x

        def __ddxxdn(f, dx=None):
            '''
            X partial down derivative
            '''
            if dx is None:
                dx = 1.
            nx = len(f)
            d = -75. / 107520. / dx
            c = 1029 / 107520. / dx
            b = -8575 / 107520. / dx
            a = 1. / dx - 3 * b - 5 * c - 7 * d
            x = (a * (f - np.roll(f, 1)) +
                 b * (np.roll(f, -1) - np.roll(f, 2)) +
                 c * (np.roll(f, -2) - np.roll(f, 3)) +
                 d * (np.roll(f, -3) - np.roll(f, 4)))
            x[:4] = x[4]
            for i in range(1, 4):
                x[nx - i] = x[nx - 4]
            return x

        f = open(meshfile, 'w')

        for p in ['x', 'y', 'z']:
            setattr(self, p, locals()[p])
            if (getattr(self, p) is None):
                setattr(self, 'n' + p, locals()['n' + p])
                setattr(self, 'd' + p, locals()['d' + p])
                setattr(self, p, np.linspace(0,
                                             getattr(self, 'n' + p) *
                                             getattr(self, 'd' + p),
                                             getattr(self, 'n' + p)))
            else:
                if (len(locals()[p]) < 1):
                    raise ValueError("(EEE): "+p+" axis has length zero")
                setattr(self, 'n' + p, len(locals()[p]))
            if getattr(self, 'n' + p) > 1:
                xmdn = __xxdn(getattr(self, p))
                dxidxup = __ddxxup(getattr(self, p))
                dxidxdn = __ddxxdn(getattr(self, p))
            else:
                xmdn = getattr(self, p)
                dxidxup = np.array([1.0])
                dxidxdn = np.array([1.0])
            f.write(str(getattr(self, 'n' + p)) + "\n")
            f.write(" ".join(map("{:.5f}".format, getattr(self, p))) + "\n")
            f.write(" ".join(map("{:.5f}".format, xmdn)) + "\n")
            f.write(" ".join(map("{:.5f}".format, dxidxup)) + "\n")
            f.write(" ".join(map("{:.5f}".format, dxidxdn)) + "\n")
        f.close()


def polar2cartesian(r, t, grid, x, y, order=3):
    '''
    Converts polar grid to cartesian grid
    '''


    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X * X + Y * Y)
    new_t = np.arctan2(X, Y)

    ir = interpolate.interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interpolate.interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r) - 1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]),
                           order=order).reshape(new_r.shape)


def cartesian2polar(x, y, grid, r, t, order=3):
    '''
    Converts cartesian grid to polar grid
    '''

    R, T = np.meshgrid(r, t)

    new_x = R * np.cos(T)
    new_y = R * np.sin(T)

    ix = interpolate.interp1d(x, np.arange(len(x)), bounds_error=False)
    iy = interpolate.interp1d(y, np.arange(len(y)), bounds_error=False)

    new_ix = ix(new_x.ravel())
    new_iy = iy(new_y.ravel())

    new_ix[new_x.ravel() > x.max()] = len(x) - 1
    new_ix[new_x.ravel() < x.min()] = 0

    new_iy[new_y.ravel() > y.max()] = len(y) - 1
    new_iy[new_y.ravel() < y.min()] = 0

    return map_coordinates(grid, np.array([new_ix, new_iy]),
                           order=order).reshape(new_x.shape)


class Bifrost_units(object):

    def __init__(self,filename='mhd.in',fdir='./'):
        import scipy.constants as const
        from astropy import constants as aconst
        from astropy import units

        if os.path.isfile(os.path.join(fdir,filename)):
            self.params = read_idl_ascii(os.path.join(fdir,filename))
            try:
                self.u_l = self.params['u_l']
                self.u_t = self.params['u_t']
                self.u_r = self.params['u_r']
                # --- ideal gas
                self.gamma = self.params['gamma']

            except:
                print('(WWW) the filename does not have u_l, u_t and u_r.'
                      ' Default Solar Bifrost units has been selected')
                self.u_l = 1.0e8
                self.u_t = 1.0e2
                self.u_r = 1.0e-7
                # --- ideal gas
                self.gamma = self.params['gamma']
        else:
            print('(WWW) selected filename is not available.'
                  ' Default Solar Bifrost units has been selected')
            self.u_l = 1.0e8
            self.u_t = 1.0e2
            self.u_r = 1.0e-7
            # --- ideal gas
            self.gamma = 1.667

        self.u_u = self.u_l / self.u_t
        self.u_p = self.u_r * (self.u_l / self.u_t)**2    # Pressure [dyne/cm2]
        self.u_kr = 1 / (self.u_r * self.u_l)             # Rosseland opacity [cm2/g]
        self.u_ee = self.u_u**2
        self.u_e = self.u_r * self.u_ee
        self.u_te = self.u_e / self.u_t * self.u_l  # Box therm. em. [erg/(s ster cm2)]
        self.mu = 0.8
        self.u_n = 3.00e+10                      # Density number n_0 * 1/cm^3
        self.k_b = aconst.k_B.to_value('erg/K')  # 1.380658E-16 Boltzman's cst. [erg/K]
        self.m_h = const.m_n / const.gram        # 1.674927471e-24
        self.m_he = 6.65e-24
        self.m_p = self.mu * self.m_h            # Mass per particle
        self.m_e = aconst.m_e.to_value('g')
        self.u_tg = (self.m_h / self.k_b) * self.u_ee
        self.u_tge = (self.m_e / self.k_b) * self.u_ee
        self.pi = const.pi
        self.u_b = self.u_u * np.sqrt(4. * self.pi * self.u_r)


        self.usi_l = self.u_l * const.centi  # 1e6
        self.usi_r = self.u_r * const.gram   # 1e-4
        self.usi_u = self.usi_l / self.u_t
        self.usi_p = self.usi_r * (self.usi_l / self.u_t)**2  # Pressure [N/m2]
        self.usi_kr = 1 / (self.usi_r * self.usi_l)           # Rosseland opacity [m2/kg]
        self.usi_ee = self.usi_u**2
        self.usi_e = self.usi_r * self.usi_ee
        self.usi_te = self.usi_e / self.u_t * self.usi_l      # Box therm. em. [J/(s ster m2)]
        self.ksi_b = aconst.k_B.to_value('J/K')               # Boltzman's cst. [J/K]
        self.msi_h = const.m_n                                # 1.674927471e-27
        self.msi_he = 6.65e-27
        self.msi_p = self.mu * self.msi_h                     # Mass per particle
        self.usi_tg = (self.msi_h / self.ksi_b) * self.usi_ee
        self.msi_e = const.m_e  # 9.1093897e-31
        self.usi_b = self.u_b * 1e-4

        # Solar gravity
        self.gsun = (aconst.GM_sun / aconst.R_sun**2).cgs.value  # solar surface gravity

        # --- physical constants and other useful quantities
        self.clight = aconst.c.to_value('cm/s')   # Speed of light [cm/s]
        self.hplanck = aconst.h.to_value('erg s') # Planck's constant [erg s]
        self.kboltzmann = aconst.k_B.to_value('erg/K')  # Boltzman's cst. [erg/K]
        self.amu = aconst.u.to_value('g')        # Atomic mass unit [g]
        self.amusi = aconst.u.to_value('kg')     # Atomic mass unit [kg]
        self.m_electron = aconst.m_e.to_value('g')  # Electron mass [g]
        self.q_electron = aconst.e.esu.value     # Electron charge [esu]
        self.qsi_electron = aconst.e.value       # Electron charge [C]
        self.rbohr = aconst.a0.to_value('cm')    #  bohr radius [cm]
        self.e_rydberg = aconst.Ryd.to_value('erg', equivalencies=units.spectral())
        self.eh2diss = 4.478007          # H2 dissociation energy [eV]
        self.pie2_mec = (np.pi * aconst.e.esu **2 / (aconst.m_e * aconst.c)).cgs.value
        # 5.670400e-5 Stefan-Boltzmann constant [erg/(cm^2 s K^4)]
        self.stefboltz = aconst.sigma_sb.cgs.value
        self.mion = self.m_h            # Ion mass [g]
        self.r_ei = 1.44E-7        # e^2 / kT = 1.44x10^-7 T^-1 cm

        # --- Unit conversions
        self.ev_to_erg = units.eV.to('erg')
        self.ev_to_j = units.eV.to('J')
        self.nm_to_m = const.nano   # 1.0e-09
        self.cm_to_m = const.centi  # 1.0e-02
        self.km_to_m = const.kilo   # 1.0e+03
        self.erg_to_joule = const.erg  # 1.0e-07
        self.g_to_kg = const.gram   # 1.0e-03
        self.micron_to_nm = units.um.to('nm')
        self.megabarn_to_m2 = units.Mbarn.to('m2')
        self.atm_to_pa = const.atm  # 1.0135e+05 atm to pascal (n/m^2)
        self.dyne_cm2_to_pascal = (units.dyne / units.cm**2).to('Pa')
        self.k_to_ev = units.K.to('eV', equivalencies=units.temperature_energy())
        self.ev_to_k = 1. / self.k_to_ev
        self.ergd2wd = 0.1
        self.grph = 2.27e-24
        self.permsi = aconst.eps0.value  # Permitivitty in vacuum (F/m)
        self.cross_p = 1.59880e-14
        self.cross_he = 9.10010e-17

        # Dissociation energy of H2 [eV] from Barklem & Collet (2016)
        self.di = self.eh2diss

        self.atomdic = {'h': 1, 'he': 2, 'c': 3, 'n': 4, 'o': 5, 'ne': 6, 'na': 7,
                   'mg': 8, 'al': 9, 'si': 10, 's': 11, 'k': 12, 'ca': 13,
                   'cr': 14, 'fe': 15, 'ni': 16}
        self.abnddic = {'h': 12.0, 'he': 11.0, 'c': 8.55, 'n': 7.93, 'o': 8.77,
                   'ne': 8.51, 'na': 6.18, 'mg': 7.48, 'al': 6.4, 'si': 7.55,
                   's': 5.21, 'k': 5.05, 'ca': 6.33, 'cr': 5.47, 'fe': 7.5,
                   'ni': 5.08}
        self.weightdic = {'h': 1.008, 'he': 4.003, 'c': 12.01, 'n': 14.01,
                     'o': 16.00, 'ne': 20.18, 'na': 23.00, 'mg': 24.32,
                     'al': 26.97, 'si': 28.06, 's': 32.06, 'k': 39.10,
                     'ca': 40.08, 'cr': 52.01, 'fe': 55.85, 'ni': 58.69}
        self.xidic = {'h': 13.595, 'he': 24.580, 'c': 11.256, 'n': 14.529,
                 'o': 13.614, 'ne': 21.559, 'na': 5.138, 'mg': 7.644,
                 'al': 5.984, 'si': 8.149, 's': 10.357, 'k': 4.339,
                 'ca': 6.111, 'cr': 6.763, 'fe': 7.896, 'ni': 7.633}
        self.u0dic = {'h': 2., 'he': 1., 'c': 9.3, 'n': 4., 'o': 8.7,
                 'ne': 1., 'na': 2., 'mg': 1., 'al': 5.9, 'si': 9.5, 's': 8.1,
                 'k': 2.1, 'ca': 1.2, 'cr': 10.5, 'fe': 26.9, 'ni': 29.5}
        self.u1dic = {'h': 1., 'he': 2., 'c': 6., 'n': 9.,  'o': 4.,  'ne': 5.,
                 'na': 1., 'mg': 2., 'al': 1., 'si': 5.7, 's': 4.1, 'k': 1.,
                 'ca': 2.2, 'cr': 7.2, 'fe': 42.7, 'ni': 10.5}


class Rhoeetab:
    def __init__(self, tabfile=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True, radtab=False):
        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.eosload = False
        self.radload = False
        self.entload = False
        # read table file and calculate parameters
        if tabfile is None:
            tabfile = '%s/tabparam.in' % (fdir)
        self.param = self.read_tab_file(tabfile)
        try:
            tmp = find_first_match("*idl", fdir)
        except IndexError:
            try:
                tmp = find_first_match("*idl.scr", fdir)
            except IndexError:
                try:
                    tmp = find_first_match("mhd.in", fdir)
                except IndexError:
                    tmp = ''
                    print("(WWW) init: no .idl or mhd.in files found." +
                          "Units set to 'standard' Bifrost units.")
        self.uni = Bifrost_units(filename=tmp,fdir=fdir)
        # load table(s)
        self.load_eos_table()
        if radtab:
            self.load_rad_table()

    def read_tab_file(self, tabfile):
        ''' Reads tabparam.in file, populates parameters. '''
        self.params = read_idl_ascii(tabfile)
        if self.verbose:
            print(('*** Read parameters from ' + tabfile))
        p = self.params
        # construct lnrho array
        self.lnrho = np.linspace(
            np.log(p['rhomin']), np.log(p['rhomax']), p['nrhobin'])
        self.dlnrho = self.lnrho[1] - self.lnrho[0]
        # construct ei array
        self.lnei = np.linspace(
            np.log(p['eimin']), np.log(p['eimax']), p['neibin'])
        self.dlnei = self.lnei[1] - self.lnei[0]

    def load_eos_table(self, eostabfile=None):
        ''' Loads EOS table. '''
        if eostabfile is None:
            eostabfile = '%s/%s' % (self.fdir, self.params['eostablefile'])
        nei = self.params['neibin']
        nrho = self.params['nrhobin']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(eostabfile, mode='r', shape=(nei, nrho, 4),
                          dtype=dtype, order='F')
        self.lnpg = table[:, :, 0]
        self.tgt = table[:, :, 1]
        self.lnne = table[:, :, 2]
        self.lnkr = table[:, :, 3]
        self.eosload = True
        if self.verbose:
            print(('*** Read EOS table from ' + eostabfile))

    def load_ent_table(self, eostabfile=None):
        '''
        Generates Entropy table from Bifrost EOS table
        '''
        self.enttab = np.zeros((self.params['neibin'], self.params['nrhobin']))
        for irho in range(1, self.params['nrhobin']):
            dinvrho = (1.0 / np.exp(self.lnrho[irho]) - 1.0 / np.exp(
                self.lnrho[irho - 1]))

            self.enttab[0, irho] = self.enttab[0, irho - 1] + 1.0 / \
                self.tgt[0, irho] * np.exp(self.lnpg[0, irho]) * dinvrho

            for iei in range(1, self.params['neibin']):
                dei = np.exp(self.lnei[iei]) - np.exp(self.lnei[iei - 1])
                self.enttab[iei, irho] = self.enttab[iei - 1, irho] + 1.0 / \
                    self.tgt[iei, irho] * dei
        for iei in range(1, self.params['neibin']):
            dei = np.exp(self.lnei[iei]) - np.exp(self.lnei[iei - 1])
            self.enttab[iei, 0] = self.enttab[iei - 1, 0] + \
                1.0 / self.tgt[iei, 0] * dei

        self.enttab = np.log(self.enttab - np.min(self.enttab) - 5.0e8)

    def load_rad_table(self, radtabfile=None):
        ''' Loads rhoei_radtab table. '''
        if radtabfile is None:
            radtabfile = '%s/%s' % (self.fdir,
                                    self.params['rhoeiradtablefile'])
        nei = self.params['neibin']
        nrho = self.params['nrhobin']
        nbins = self.params['nradbins']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(radtabfile, mode='r', shape=(nei, nrho, nbins, 3),
                          dtype=dtype, order='F')
        self.epstab = table[:, :, :, 0]
        self.temtab = table[:, :, :, 1]
        self.opatab = table[:, :, :, 2]
        self.radload = True
        if self.verbose:
            print(('*** Read rad table from ' + radtabfile))

    def get_table(self, out='ne', bine=None, order=1):

        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab',
                 'ent': 'enttab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
        if out in ['ent'] and not self.entload:
            if not self.eosload:
                raise ValueError("(EEE) tab_interp: EOS table not loaded!")
            if not self.entload:
                self.load_ent_table()
                self.entload = True
        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")
        quant = getattr(self, qdict[out])
        if out in ['opa eps temp'.split()]:
            if bin is None:
                print(("(WWW) tab_interp: radiation bin not set,"
                       " using first bin."))
                bin = 0
            quant = quant[..., bin]
        return quant

    def tab_interp(self, rho, ei, out='ne', bin=None, order=1):
        '''
        Interpolates the EOS/rad table for the required quantity in out.

        Parameters
        ----------
        rho  : ndarray
            Density in g/cm^3
        ei   : ndarray
            Internal energy in erg/g
        bin  : int, optional
            Radiation bin number for bin parameters
        order: int, optional
            Interpolation order (1: linear, 3: cubic)

        Returns
        -------
        output : array
            Same dimensions as input. Depeding on the selected option,
            could be:
            'nel'  : electron density [cm^-3]
            'tg'   : temperature [K]
            'pg'   : gas pressure [dyn/cm^2]
            'kr'   : Rosseland opacity [cm^2/g]
            'eps'  : scattering probability
            'opa'  : opacity
            'temt' : thermal emission
        '''

        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab',
                 'ent': 'enttab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
        if out in ['ent'] and not self.entload:
            if not self.eosload:
                raise ValueError("(EEE) tab_interp: EOS table not loaded!")
            if not self.entload:
                self.load_ent_table()
                self.entload = True
        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")
        quant = getattr(self, qdict[out])
        if out in ['opa', 'eps', 'temp']:
            if bin is None:
                print("(WWW) tab_interp: radiation bin not set, using first.")
                bin = 0
            quant = quant[:, :, bin]
        # warnings for values outside of table
        rhomin = np.min(rho)
        rhomax = np.max(rho)
        eimin = np.min(ei)
        eimax = np.max(ei)
        if rhomin < self.params['rhomin']:
            print('(WWW) tab_interp: density outside table bounds.' +
                  'Table rho min=%.3e, requested rho min=%.3e' %
                  (self.params['rhomin'], rhomin))
        if rhomax > self.params['rhomax']:
            print('(WWW) tab_interp: density outside table bounds. ' +
                  'Table rho max=%.1f, requested rho max=%.1f' %
                  (self.params['rhomax'], rhomax))
        if eimin < self.params['eimin']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei min=%.2f, requested Ei min=%.2f' %
                  (self.params['eimin'], eimin))
        if eimax > self.params['eimax']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei max=%.2f, requested Ei max=%.2f' %
                  (self.params['eimax'], eimax))
        # translate to table coordinates
        x = (np.log(ei) - self.lnei[0]) / self.dlnei
        y = (np.log(rho) - self.lnrho[0]) / self.dlnrho
        # interpolate quantity

        result = map_coordinates(
            quant, [x, y], order=order, mode='nearest')
        return (np.exp(result) if out != 'tg' else result)


class Opatab:
    """
    Class to loads opacity table and calculate the photoionization cross
    sections given by anzer & heinzel apj 622: 714-721, 2005, march 20
    they have big typos in their reported c's.... correct values to
    be found in rumph et al 1994 aj, 107: 2108, june 1994

    gaunt factors are set to 0.99 for h and 0.85 for heii,
    which should be good enough for the purposes of this code
    """
    def __init__(self, tabname=None, fdir='.',  dtype='f4',
                 verbose=True, lambd=100.0):
        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.lambd = lambd
        self.radload = False
        self.teinit = 4.0
        self.dte = 0.1
        # read table file and calculate parameters
        if tabname is None:
            tabname = os.path.join(fdir, 'ionization.dat')
        self.tabname = tabname
        # load table(s)
        self.load_opa_table()

    def hopac(self):
        ghi = 0.99
        o0 = 7.91e-18  # cm^2
        ohi = 0
        if self.lambd <= 912:
            ohi = o0 * ghi * (self.lambd / 912.0)**3
        return ohi

    def heiopac(self):
        c = [-2.953607e1, 7.083061e0, 8.678646e-1,
             -1.221932e0, 4.052997e-2, 1.317109e-1,
             -3.265795e-2, 2.500933e-3]
        ohei = 0
        if self.lambd <= 504:
            for i, cf in enumerate(c):
                ohei += cf * (np.log10(self.lambd))**i
            ohei = 10.0**ohei
        return ohei

    def heiiopac(self):
        gheii = 0.85
        o0 = 7.91e-18  # cm^2
        oheii = 0
        if self.lambd <= 228:
            oheii = 16 * o0 * gheii * (self.lambd / 912.0)**3
        return oheii

    def load_opa_table(self, tabname=None):
        ''' Loads ionizationstate table. '''
        if tabname is None:
            tabname = '%s/%s' % (self.fdir, 'ionization.dat')
        eostab = Rhoeetab(fdir=self.fdir)
        nei = eostab.params['neibin']
        nrho = eostab.params['nrhobin']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(tabname, mode='r', shape=(nei, nrho, 3), dtype=dtype,
                          order='F')
        self.ionh = table[:, :, 0]
        self.ionhe = table[:, :, 1]
        self.ionhei = table[:, :, 2]
        self.opaload = True
        if self.verbose:
            print('*** Read EOS table from ' + tabname)

    def tg_tab_interp(self, order=1):
        '''
        Interpolates the opa table to same format as tg table.
        '''
        self.load_opa1d_table()
        rhoeetab = Rhoeetab(fdir=self.fdir)
        tgTable = rhoeetab.get_table('tg')
        # translate to table coordinates
        x = (np.log10(tgTable) - self.teinit) / self.dte
        # interpolate quantity
        self.ionh = map_coordinates(self.ionh1d, [x], order=order)
        self.ionhe = map_coordinates(self.ionhe1d, [x], order=order)
        self.ionhei = map_coordinates(self.ionhei1d, [x], order=order)

    def h_he_absorb(self, lambd=None):
        '''
        Gets the opacities for a particular wavelength of light.
        If lambd is None, then looks at the current level for wavelength
        '''
        rhe = 0.1
        if lambd is not None:
            self.lambd = lambd
        self.tg_tab_interp()
        ion_h = self.ionh
        ion_he = self.ionhe
        ion_hei = self.ionhei
        ohi = self.hopac()
        ohei = self.heiopac()
        oheii = self.heiiopac()
        arr = (1 - ion_h) * ohi + rhe * ((1 - ion_he - ion_hei) *
                                         ohei + ion_he * oheii)
        arr[arr < 0] = 0
        return arr

    def load_opa1d_table(self, tabname=None):
        ''' Loads ionizationstate table. '''
        if tabname is None:
            tabname = '%s/%s' % (self.fdir, 'ionization1d.dat')
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(tabname, mode='r', shape=(41, 3), dtype=dtype,
                          order='F')
        self.ionh1d = table[:, 0]
        self.ionhe1d = table[:, 1]
        self.ionhei1d = table[:, 2]
        self.opaload = True
        if self.verbose:
            print('*** Read OPA table from ' + tabname)


class Cross_sect:
    """
    Reads data from Bifrost collisional cross section tables.

    Parameters
    ----------
    cross_tab - string or array of strings
        File names of the ascii cross table files.
    fdir - string, optional
        Directory where simulation files are. Must be a real path.
    verbose - bool, optional
        If True, will print out more diagnostic messages
    dtype - string, optional
        Data type for reading variables. Default is 32 bit float.

    Examples
    --------
    >>> a = cross_sect(['h-h-data2.txt','h-h2-data.txt'], fdir="/data/cb24bih")

    """
    def __init__(self, cross_tab=None, fdir='.', dtype='f4', verbose=True):
        '''
        Loads cross section tables and calculates collision frequencies and
        ambipolar diffusion.
        '''

        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        # read table file and calculate parameters
        cross_txt_list = ['h-h-data2.txt', 'h-h2-data.txt', 'he-he.txt',
                          'e-h.txt', 'e-he.txt', 'h2_molecule_bc.txt',
                          'h2_molecule_pj.txt', 'p-h-elast.txt', 'p-he.txt',
                          'proton-h2-data.txt']
        self.cross_tab_list = {}
        counter = 0
        if cross_tab is None:
            for icross_txt in cross_txt_list:
                os.path.isfile('%s/%s' % (fdir, icross_txt))
                self.cross_tab_list[counter] = '%s/%s' % (fdir, icross_txt)
                counter += 1
        else:
            for icross_txt in cross_tab:
                os.path.isfile('%s/%s' % (fdir, icross_txt))
                self.cross_tab_list[counter] = '%s/%s' % (fdir, icross_txt)
                counter += 1
        # load table(s)
        self.load_cross_tables()

    def load_cross_tables(self):
        '''
        Collects the information in the cross table files.
        '''
        uni = Bifrost_units()
        self.cross_tab = {}

        for itab in range(len(self.cross_tab_list)):
            self.cross_tab[itab] = read_cross_txt(self.cross_tab_list[itab])
            self.cross_tab[itab]['tg'] *= uni.ev_to_k

    def tab_interp(self, tg, itab=0, out='el', order=1):
        ''' Interpolates the cross section tables in the simulated domain.
            IN:
                tg  : Temperature [K]
                order: interpolation order (1: linear, 3: cubic)
            OUT:
                'se'  : Spin exchange cross section [a.u.]
                'el'  : Integral Elastic cross section [a.u.]
                'mt'  : momentum transfer cross section [a.u.]
                'vi'  : viscosity cross section [a.u.]
        '''

        if out in ['se el vi mt'.split()] and not self.load_cross_tables:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")

        finterp = interpolate.interp1d(self.cross_tab[itab]['tg'],
                                          self.cross_tab[itab][out])
        tgreg = tg * 1.0
        max_temp = np.max(self.cross_tab[itab]['tg'])
        tgreg[np.where(tg > max_temp)] = max_temp
        min_temp = np.min(self.cross_tab[itab]['tg'])
        tgreg[np.where(tg < min_temp)] = min_temp

        return finterp(tgreg)

###########
#  TOOLS  #
###########

def bifrost2d_to_rh15d(snaps, outfile, file_root, meshfile, fdir, writeB=False,
                       sx=slice(None), sz=slice(None), desc=None):
    """
    Reads a Bifrost 2D atmosphere are writes into RH 1.5D format,
    with the time dimension written in the y dimension (to maximise
    parallelism).
    Parameters
    ----------
    snaps : list or 1D array
        Numbers of snapshots to write.
    outfile: str
        Name of RH 1.5D atmosphere file to write.
    file_root: str
        Basename for bifrost files.
    meshfile : str
        Filename of mesh file (including full path).
    writeB : bool, optional
        If True, will also write magnetic field. Default is False.
    sx, sz : slice object, optional
        Slice objects for x and z dimensions, when not all points
        are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
        for every second point up to 100.
    desc : str
        Description.
    """
    from . import rh15d
    data = BifrostData(file_root, snap=snaps[0], meshfile=meshfile, fdir=fdir,
                       ghost_analyse=False)
    nz = len(data.z[sz])
    nx = max(len(data.x[sx]), len(data.y[sx]))
    ny = len(snaps)
    tgas = np.empty((nx, ny, nz), dtype='f')
    vz = np.empty_like(tgas)
    if writeB:
        Bz = np.empty_like(tgas)
        Bx = np.empty_like(tgas)
        By = np.empty_like(tgas)
    else:
        Bz = None
        Bx = None
        By = None
    ne = np.empty_like(tgas, dtype='d')
    if data.hion:
        nH = np.empty((6, ) + tgas.shape, dtype='f')
    else:
        nH = np.empty((1, ) + tgas.shape, dtype='f')

    # unit conversion to SI
    ul = data.params['u_l'] / 1.e2 # to metres
    ur = data.params['u_r']        # to g/cm^3  (for ne_rt_table)
    ut = data.params['u_t']        # to seconds
    uv = ul / ut
    ub = data.params['u_b'] * 1e-4 # to tgasesl
    ue = data.params['u_ee']       # to erg/g

    if not desc:
        desc = 'BIFROST snapshot from 2D sequence %s, sx=%s sy=1 sz=%s.' % \
                    (file_root, repr(sx), repr(sz))
        if data.hion:
            desc = 'hion ' + desc
    x = data.x[sx] * ul
    y = snaps
    z = data.z[sz] * (-ul)

    rdt = data.r.dtype
    cstagger.init_stagger(data.nz, data.dx, data.dy, data.z.astype(rdt),
                          data.zdn.astype(rdt), data.dzidzup.astype(rdt),
                          data.dzidzdn.astype(rdt))

    for i, s in enumerate(snaps):
        data.set_snap(s)
        tgas[:, i] = np.squeeze(data.tg)[sx, sz]
        rho = data.r[sx, sz]
        vz[:, i] = np.squeeze(cstagger.zup(data.pz)[sx, sz] / rho) * (-uv)
        if writeB:
            Bx[:, i] = np.squeeze(data.bx)[sx, sz] * ub
            By[:, i] = np.squeeze(-data.by)[sx, sz] * ub
            Bz[:, i] = np.squeeze(-data.bz)[sx, sz] * ub
        ne[:, i] = np.squeeze(data.get_electron_density(sx=sx, sz=sz)).to_value('1/m3')
        nH[:, :, i] = np.squeeze(data.get_hydrogen_pops(sx=sx, sz=sz)).to_value('1/m3')

    rh15d.make_xarray_atmos(outfile, tgas, vz, z, nH=nH, ne=ne, x=x, y=y,
                            append=False, Bx=Bx, By=By, Bz=Bz, desc=desc,
                            snap=snaps[0])


def read_idl_ascii(filename):
    ''' Reads IDL-formatted (command style) ascii file into dictionary '''
    li = 0
    params = {}
    # go through the file, add stuff to dictionary

    with open(filename) as fp:
        for line in fp:
            # ignore empty lines and comments
            line = line.strip()
            if not line:
                li += 1
                continue
            if line[0] == ';':
                li += 1
                continue
            line = line.split(';')[0].split('=')
            if len(line) != 2:
                print(('(WWW) read_params: line %i is invalid, skipping' % li))
                li += 1
                continue
            # force lowercase because IDL is case-insensitive
            key = line[0].strip().lower()
            value = line[1].strip()
            # instead of the insecure 'exec', find out the datatypes
            if value.find('"') >= 0:
                # string type
                value = value.strip('"')
                try:
                    if (value.find(' ') >= 0):
                        value2 = np.array(value.split())
                        if ((value2[0].upper().find('E') >= 0) or (
                                value2[0].find('.') >= 0)):
                            value = value2.astype(np.float)

                except:
                    value = value
            elif (value.find("'") >= 0):
                value = value.strip("'")
                try:
                    if (value.find(' ') >= 0):
                        value2 = np.array(value.split())
                        if ((value2[0].upper().find('E') >= 0) or (
                                value2[0].find('.') >= 0)):
                            value = value2.astype(np.float)
                except:
                    value = value
            elif (value.lower() in ['.false.', '.true.']):
                # bool type
                value = False if value.lower() == '.false.' else True
            elif (value.find('[') >= 0) and (value.find(']') >= 0):
                # list type
                value = eval(value)
            elif (value.upper().find('E') >= 0) or (value.find('.') >= 0):
                # float type
                value = float(value)
            else:
                # int type
                try:
                    value = int(value)
                except Exception:
                    print('(WWW) read_idl_ascii: could not find datatype in '
                          'line %i, skipping' % li)
                    li += 1
                    continue

            params[key] = value

    return params


def ionpopulation(rho, nel, tg, elem='h', lvl='1', dens=True):

    print('ionpopulation: reading species %s and level %s' % (elem, lvl))
    fdir = '.'
    try:
        tmp = find_first_match("*.idl", fdir)
    except IndexError:
        try:
            tmp = find_first_match("*idl.scr", fdir)
        except IndexError:
            try:
                tmp = find_first_match("mhd.in", fdir)
            except IndexError:
                tmp = ''
                print("(WWW) init: no .idl or mhd.in files found." +
                      "Units set to 'standard' Bifrost units.")
    uni = Bifrost_units(filename=tmp)

    totconst = 2.0 * uni.pi * uni.m_electron * uni.k_b / \
        uni.hplanck / uni.hplanck
    abnd = np.zeros(len(uni.abnddic))
    count = 0

    for ibnd in uni.abnddic.keys():
        abnddic = 10**(uni.abnddic[ibnd] - 12.0)
        abnd[count] = abnddic * uni.weightdic[ibnd] * uni.amu
        count += 1

    abnd = abnd / np.sum(abnd)
    phit = (totconst * tg)**(1.5) * 2.0 / nel
    kbtg = uni.ev_to_erg / uni.k_b / tg
    n1_n0 = phit * uni.u1dic[elem] / uni.u0dic[elem] * np.exp(
        - uni.xidic[elem] * kbtg)
    c2 = abnd[uni.atomdic[elem] - 1] * rho
    ifracpos = n1_n0 / (1.0 + n1_n0)

    if dens:
        if lvl == '1':
            return (1.0 - ifracpos) * c2
        else:
            return ifracpos * c2

    else:
        if lvl == '1':
            return (1.0 - ifracpos) * c2 * (uni.u_r / (uni.weightdic[elem] *
                                                       uni.amu))
        else:
            return ifracpos * c2 * (uni.u_r / (uni.weightdic[elem] *
                                               uni.amu))


def read_cross_txt(filename):
    ''' Reads IDL-formatted (command style) ascii file into dictionary '''
    li = 0
    params = {}
    count = 0
    # go through the file, add stuff to dictionary
    with open(filename) as fp:
        for line in fp:
            # ignore empty lines and comments
            line = line.strip()
            if len(line) < 1:
                li += 1
                continue
            if line[0] == ';':
                li += 1
                continue
            line = line.split(';')[0].split()
            if (len(line) < 2):
                print(('(WWW) read_params: line %i is invalid, skipping' % li))
                li += 1
                continue
            # force lowercase because IDL is case-insensitive
            temp = line[0].strip()
            cross = line[1].strip()

            # instead of the insecure 'exec', find out the datatypes
            if ((temp.upper().find('E') >= 0) or (temp.find('.') >= 0)):
                # float type
                temp = float(temp)
            else:
                # int type
                try:
                    temp = int(temp)
                except Exception:
                    print('(WWW) read_idl_ascii: could not find datatype in '
                          'line %i, skipping' % li)
                    li += 1
                    continue
            if not('tg' in params.keys()):
                params['tg'] = temp
            else:
                params['tg'] = np.append(params['tg'], temp)

            if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                # float type
                cross = float(cross)
            else:
                # int type
                try:
                    cross = int(cross)
                except Exception:
                    print('(WWW) read_idl_ascii: could not find datatype in '
                          'line %i, skipping' % li)
                    li += 1
                    continue
            if not('el' in params.keys()):
                params['el'] = cross
            else:
                params['el'] = np.append(params['el'], cross)

            if len(line) > 2:
                cross = line[2].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        print('(WWW) read_idl_ascii: could not find datatype'
                              'in line %i, skipping' % li)
                        li += 1
                        continue
                if not('mt' in params.keys()):
                    params['mt'] = cross
                else:
                    params['mt'] = np.append(params['mt'], cross)

            if len(line) > 3:
                cross = line[3].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        print('(WWW) read_idl_ascii: could not find datatype'
                              'in line %i, skipping' % li)
                        li += 1
                        continue
                if not hasattr(params, 'vi'):
                    params['vi'] = cross
                else:
                    params['vi'] = np.append(params['vi'], cross)

            if len(line) > 4:
                cross = line[4].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        print('(WWW) read_idl_ascii: could not find datatype'
                              'in line %i, skipping' % li)
                        li += 1
                        continue
                if not hasattr(params, 'se'):
                    params['se'] = cross
                else:
                    params['se'] = np.append(params['se'], cross)
            li += 1

    return params


def calc_grph(abundances, atomic_weights):
    """
    Calculate grams per hydrogen atom, given a mix of abundances
    and respective atomic weights.

    Parameters
    ----------
    abundances : 1D array
        Element abundances relative to hydrogen in log scale,
        where hydrogen is defined as 12.
    atomic_weights : 1D array
        Atomic weights for each element in atomic mass units.

    Returns
    -------
    grph : float
        Grams per hydrogen atom.
    """
    from astropy.constants import u as amu
    linear_abundances = 10.**(abundances - 12.)
    masses = atomic_weights * amu.to_value('g')
    return np.sum(linear_abundances * masses)


def subs2grph(subsfile):
    """
    Extract abundances and atomic masses from subs.dat, and calculate
    the number of grams per hydrogen atom.

    Parameters
    ----------
    subsfile : str
        File name of subs.dat.

    Returns
    -------
    grph : float
        Grams per hydrogen atom.
    """
    f = open(subsfile, 'r')
    nspecies = np.fromfile(f, count=1, sep=' ', dtype='i')[0]
    f.readline()  # second line not important
    ab = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    am = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    f.close()
    return calc_grph(ab, am)


def threadQuantity(task, numThreads, *args):
    # split arg arrays
    args = list(args)

    for index in range(np.shape(args)[0]):
        args[index] = np.array_split(args[index], numThreads)

    # make threadpool, task = task, with zipped args
    pool = ThreadPool(processes=numThreads)
    result = np.concatenate(pool.starmap(task, zip(*args)))
    return result


def threadQuantity_y(task, numThreads, *args):
    # split arg arrays
    args = list(args)

    for index in range(np.shape(args)[0]):
        if len(np.shape(args[index])) == 3:
            args[index] = np.array_split(args[index], numThreads, axis=1)
        else:
            args[index] = np.array_split(args[index], numThreads)
    # make threadpool, task = task, with zipped args
    pool = ThreadPool(processes=numThreads)
    result = np.concatenate(pool.starmap(task, zip(*args)), axis=1)
    return result


def threadQuantity_z(task, numThreads, *args):
    # split arg arrays
    args = list(args)

    for index in range(np.shape(args)[0]):
        if len(np.shape(args[index])) == 3:
            args[index] = np.array_split(args[index], numThreads, axis=2)
        else:
            args[index] = np.array_split(args[index], numThreads)

    # make threadpool, task = task, with zipped args
    pool = ThreadPool(processes=numThreads)
    result = np.concatenate(pool.starmap(task, zip(*args)), axis=2)
    return result


def find_first_match(name, path,incl_path=False):
    '''
    This will find the first match,
    name : string, e.g., 'patern*'
    incl_root: boolean, if true will add full path, otherwise, the name.
    path : sring, e.g., '.'
    '''
    originalpath=os.getcwd()
    os.chdir(path)
    for file in glob(name):
        if incl_path:
            os.chdir(originalpath)
            return os.path.join(path, file)
        else:
            os.chdir(originalpath)
            return file
    os.chdir(originalpath)
