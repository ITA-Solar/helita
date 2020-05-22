"""
Set of programs to read and interact with output from Bifrost
"""

import os
import warnings
from glob import glob
import numpy as np
try:
    from . import cstagger
except ImportError:
    print("(WWW) cstagger routines not imported, certain functions will be inaccesible")
from scipy import interpolate
from scipy.ndimage import map_coordinates
from .load_quantities import *
from .load_arithmetic_quantities import *

whsp = '  '

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
                 ghost_analyse=False, lowbus=False, numThreads=1, params_only=False):
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
        self.set_snap(snap,True,params_only=params_only)
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
        self.cross_sect = Cross_sect
        self.rhoeetab = Rhoeetab(fdir=fdir)

    def _set_snapvars(self,firstime=False):
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

    def set_snap(self, snap, firstime=False, params_only=False):
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

        self._read_params(firstime=firstime)
        # Read mesh for all snaps because meshfiles could differ
        self.__read_mesh(self.meshfile,firstime=firstime)
        # variables: lists and initialisation
        self._set_snapvars(firstime=firstime)
        # Do not call if params_only requested
        if(not params_only):
            self._init_vars(firstime=firstime)

    def _read_params(self,firstime=False):
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
            self.paramList.append(read_idl_ascii(file,firstime=firstime))

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
                    print("(WWW) read_params:"" %s not found, using "
                          "default of %.3e" % (unit, unit_def[unit]), 2*whsp,
                          end="\r", flush=True)
                    params[unit] = unit_def[unit]

        self.params = {}
        for key in self.paramList[0]:
            self.params[key] = np.array(
                [self.paramList[i][key] for i in range(
                    0, len(self.paramList))])

    def __read_mesh(self, meshfile, firstime=False):
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
            if self.verbose:
                if (firstime):
                    print('(WWW) Mesh file %s does not exist.' % meshfile)
            if self.dx == 0.0:
                self.dx = 1.0
            if self.dy == 0.0:
                self.dy = 1.0
            if self.dz == 0.0:
                self.dz = 1.0
            if self.verbose:
                if (firstime):
                    print(('(WWW) Creating uniform grid with [dx,dy,dz] = '
                        '[%f,%f,%f]') % (self.dx, self.dy, self.dz),
                        2 * whsp, end="\r", flush=True)
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

    def _init_vars(self, firstime=False, *args, **kwargs):
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
                    if firstime:
                        print('(WWW) init_vars: could not read '
                            'variable %s' % var)
        for var in self.auxxyvars:
            try:
                self.variables[var] = self._get_simple_var_xy(var, *args,
                                                              **kwargs)
                setattr(self, var, self.variables[var])
            except Exception:
                if self.verbose:
                    if firstime:
                        print('(WWW) init_vars: could not read '
                            'variable %s' % var)
        rdt = self.r.dtype
        if (self.nz > 1): 
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
            print('(get_var): reading ', var, whsp*6, end="\r", flush=True)

        if not hasattr(self, 'iix'):
            self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            self.set_domain_iiaxis(iinum=iiz, iiaxis='z')
        else:
            if (iix != slice(None)) and np.any(iix != self.iix):
                if self.verbose:
                    print('(get_var): iix ', iix, self.iix,
                        whsp*4, end="\r",flush=True)
                self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            if (iiy != slice(None)) and np.any(iiy != self.iiy):
                if self.verbose:
                    print('(get_var): iiy ', iiy, self.iiy, whsp*4,
                        end="\r",flush=True)
                self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            if (iiz != slice(None)) and np.any(iiz != self.iiz):
                if self.verbose:
                    print('(get_var): iiz ', iiz, self.iiz, whsp*4,
                        end="\r",flush=True)
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
                print('(get_var): setsnap ', snap, self.snap, whsp*6,
                    end="\r",flush=True)
            self.set_snap(snap)

        if var in self.simple_vars:  # is variable already loaded?
            val = self._get_simple_var(var, *args, **kwargs)
            if self.verbose:
                print('(get_var): reading simple ', np.shape(val), whsp*5,
                    end="\r",flush=True)
        elif var in self.auxxyvars:
            val = self._get_simple_var_xy(var, *args, **kwargs)
        elif var in self.compvars:  # add to variable list
            self.variables[var] = self._get_composite_var(var, *args, **kwargs)
            setattr(self, var, self.variables[var])
            val = self.variables[var]
        else:
            # Loading quantities
            val = load_quantities(self,var)
            # Loading arithmetic quantities
            if np.shape(val) is ():
                val = load_arithmetic_quantities(self,var)

        if var == '':
            print(help(self.get_var))
            print(self.description['ALL'])
            return None

            if np.shape(val) is ():
                raise ValueError(('get_var: do not know (yet) how to '
                              'calculate quantity %s. Note that simple_var '
                              'available variables are: %s.\nIn addition, '
                              'get_quantity can read others computed variables '
                              'see e.g. help(self.get_var) or get_var('')) for guidance'
                              '.' % (var, repr(self.simple_vars))))
            #val = self.get_quantity(var, *args, **kwargs)

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
        if var == '':
            print(help(self._get_simple_var))

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

    def _get_composite_var(self, var, *args, EOSTAB_QUANT=None, **kwargs):
        """
        Gets composite variables such as ux, uy, uz, ee, s tau (at 500),
        and other eos variables are in cgs except ne which is in SI.
        The electron density [m^-3] (ne), temperature [K] (tg),
        pressure [dyn/cm^2] (pg), Rosseland opacity [cm^2/g] (kr),
        scattering probability (eps), opacity (opa), thermal emission (temt)
        and entropy (ent). They will will load into memory.
        """

        if var == '':
            print(help(self._get_composite_var))

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
        print('a')

        # grph = 2.38049d-24 uni.GRPH
        # bk = 1.38e-16 uni.KBOLTZMANN
        # EV_TO_ERG=1.60217733E-12 uni.EV_TO_ERG
        if not hasattr(self, 'ne'):

            nel = self.get_var('ne')
        else:
            nel = self.ne
        print('b')
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
                abund = np.array(tabparams['abund']).astype('f')
                aweight = np.array(tabparams['aweight']).astype('f')
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
            print('Slicing and unit conversion...', whsp*4, end="\r",
                flush=True)
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
        if self.verbose:
            print('Write to file...', whsp*8, end="\r", flush=True)
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


def write_br_snap(rootname,r,px,py,pz,e,bx,by,bz):
    nx, ny, nz = r.shape
    data = np.memmap(rootname, dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,8))
    data[...,0] = r
    data[...,1] = px
    data[...,2] = py
    data[...,3] = pz
    data[...,4] = e
    data[...,5] = bx
    data[...,6] = by
    data[...,7] = bz
    data.flush()

def paramfile_br_update(infile, outfile, new_values):
    ''' Updates a given number of fields with values on a bifrost.idl file.
        These are given in a dictionary: fvalues = {field: value}.
        Reads from infile and writes into outfile.'''
    out = open(outfile, 'w')
    with open(infile) as fin:
        for line in fin:
            if line[0] == ';':
                out.write(line)
            elif line.find('=') < 0:
                out.write(line)
            else:
                ss = line.split('=')[0]
                ssv = ss.strip().upper()
                if ssv in list(new_values.keys()):
                    out.write('%s= %s\n' % (ss, str(new_values[ssv])))
                else:
                    out.write(line)
    return


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
            f.write(" ".join(map("{:.5f}".format, 1.0/dxidxup)) + "\n")
            f.write(" ".join(map("{:.5f}".format, 1.0/dxidxdn)) + "\n")
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
            self.params = read_idl_ascii(os.path.join(fdir,filename),firstime=True)
            try:
                self.u_l = self.params['u_l']
            except:
                print('(WWW) the filename does not have u_l.'
                    ' Default Solar Bifrost u_l has been selected')
                self.u_l = 1.0e8

            try:
                self.u_t = self.params['u_t']
            except:
                print('(WWW) the filename does not have u_t.'
                    ' Default Solar Bifrost u_t has been selected')
                self.u_t = 1.0e2

            try:
                self.u_r = self.params['u_r']
            except:
                print('(WWW) the filename does not have u_r.'
                    ' Default Solar Bifrost u_r has been selected')
                self.u_r = 1.0e-7

            try:
                self.gamma = self.params['gamma']
            except:
                print('(WWW) the filename does not have gamma.'
                    ' ideal gas has been selected')
                self.gamma = 1.667

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
        self.usi_r = 1e-4 # self.u_r * const.gram   # 1e-4
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
        self.hplancksi = aconst.h.to_value('J s') # Planck's constant [erg s]
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
            print(('*** Read parameters from ' + tabfile), whsp*4 ,end="\r",
                    flush=True)
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
            print('*** Read EOS table from ' + eostabfile, whsp*4, end="\r",
                flush=True)

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
            print('*** Read rad table from ' + radtabfile, whsp*4, end="\r",
                flush=True)

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
                print("(WWW) tab_interp: radiation bin not set,"
                       " using first bin.")
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
            print('*** Read EOS table from ' + tabname, whsp*4, end="\r",
                flush=True)

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
            print('*** Read OPA table from ' + tabname, whsp*4, end="\r",
                flush=True)


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
        self.load_cross_tables(firstime=True)

    def load_cross_tables(self,firstime=False):
        '''
        Collects the information in the cross table files.
        '''
        uni = Bifrost_units()
        self.cross_tab = {}

        for itab in range(len(self.cross_tab_list)):
            self.cross_tab[itab] = read_cross_txt(self.cross_tab_list[itab],firstime=firstime)
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
        rho = np.squeeze(data.r)[sx, sz]
        vz[:, i] = np.squeeze(cstagger.zup(data.pz))[sx, sz] / rho * (-uv)
        if writeB:
            Bx[:, i] = np.squeeze(data.bx)[sx, sz] * ub
            By[:, i] = np.squeeze(-data.by)[sx, sz] * ub
            Bz[:, i] = np.squeeze(-data.bz)[sx, sz] * ub
        ne[:, i] = np.squeeze(data.get_electron_density(sx=sx, sz=sz)).to_value('1/m3')
        nH[:, :, i] = np.squeeze(data.get_hydrogen_pops(sx=sx, sz=sz)).to_value('1/m3')

    rh15d.make_xarray_atmos(outfile, tgas, vz, z, nH=nH, ne=ne, x=x, y=y,
                            append=False, Bx=Bx, By=By, Bz=Bz, desc=desc,
                            snap=snaps[0])


def read_idl_ascii(filename,firstime=False):
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
                if firstime:
                    print('(WWW) read_params: line %i is invalid, skipping' % li)
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
                    if (firstime):
                        print('(WWW) read_idl_ascii: could not find datatype in'
                            ' line %i in file %s, %s, skipping %s' % (li,
                            filename, value, 4*whsp))
                    li += 1
                    continue

            params[key] = value

    return params


def read_cross_txt(filename,firstime=False):
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
            if (len(line) == 1):
                params['crossunits'] = float(line[0].strip())
                li += 1
                continue
            elif not('crossunits' in params.keys()):
                print('(WWW) read_cross: line %i is invalid, missing crossunits, file %s' % (li,filename))

            if (len(line) < 2):
                if (firstime):
                    print('(WWW) read_cross: line %i is invalid, skipping, file %s' % (li,filename))
                li += 1
                continue
            # force lowercase because IDL is case-insensitive
            temp = line[0].strip()
            cross = line[2].strip()

            # instead of the insecure 'exec', find out the datatypes
            if ((temp.upper().find('E') >= 0) or (temp.find('.') >= 0)):
                # float type
                temp = float(temp)
            else:
                # int type
                try:
                    temp = int(temp)
                except Exception:
                    if (firstime):
                        print('(WWW) read_cross: could not find datatype in '
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
                    if (firstime):
                        print('(WWW) read_cross: could not find datatype in '
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
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
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
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
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
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
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
