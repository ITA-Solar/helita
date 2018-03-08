"""
Set of programs to read and interact with output from Bifrost
"""

import numpy as np
import os
from glob import glob
from . import cstagger
import numba
import scipy as sp


class BifrostData(object):
    """
    Reads data from Bifrost simulations in native format.
    """
    snap = None

    def __init__(self, file_root, snap=None, meshfile=None, fdir='.',
                 verbose=True, cstagop=True, dtype='f4', big_endian=False,
                 ghost_analyse=False):
        """
        Loads metadata and initialises variables.

        Parameters
        ----------
        file_root - string
            Basename for all file names. Snapshot number will be added
            afterwards, and directory will be added before.
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

        Examples
        --------
        This reads snapshot 383 from simulation "cb24bih", whose file
        root is "cb24bih_", and is found at directory /data/cb24bih:

        >>> a = Bifrost.Data("cb24bih_", snap=383, fdir=""/data/cb24bih")

        Scalar variables do not need de-staggering and are available as
        memory map (only loaded to memory when needed), e.g.:

        >>> a.r.shape
        (504, 504, 496)

        Composite variables need to be obtained by get_var():

        >>> vx = a.get_var("ux")
        """
        self.fdir = fdir
        self.verbose = verbose
        self.cstagop = cstagop
        self.file_root = os.path.join(self.fdir, file_root)
        self.meshfile = meshfile
        self.ghost_analyse = ghost_analyse
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype

        self.set_snap(snap)

    def _set_snapvars(self):
        """
            Sets list of avaible variables
        """
        self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        self.auxvars = self.params['aux'][self.snapInd].split()
        if (self.do_mhd):
            self.snapvars += ['bx', 'by', 'bz']
        self.hionvars = []
        if 'do_hion' in self.params:
            if self.params['do_hion'][self.snapInd] > 0:
                self.hionvars = ['hionne', 'hiontg', 'n1',
                                 'n2', 'n3', 'n4', 'n5', 'n6', 'fion', 'nh2']
        self.compvars = ['ux', 'uy', 'uz', 's', 'ee']
        self.simple_vars = self.snapvars + self.auxvars + self.hionvars
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
                snap = int(tmp.split(self.file_root + '_')[1].split(".idl")[0])
            except IndexError:
                try:
                    tmp = sorted(glob("%s*idl.scr" % self.file_root))[0]
                    snap = -1
                except IndexError:
                    raise ValueError(("(EEE) set_snap: snapshot not defined "
                                      "and no .idl files found"))
        self.snap = snap

        # if not (isinstance(snap, np.int64) or isinstance(snap, int)):
        if np.size(snap) > 1:
            self.snap_str = []
            for num in snap:
                self.snap_str.append('_%03i' % int(num))
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
        if type(self.snap) is int:
            snap = [self.snap]
            snap_str = [self.snap_str]
        else:
            snap = self.snap
            snap_str = self.snap_str

        filename = []
        self.paramList = []

        for i, num in enumerate(snap):
            if (num < 0):
                filename.append(self.file_root + '.idl.scr')
            elif (num == 0):
                filename.append(self.file_root + '.idl')
            else:
                filename.append(self.file_root + snap_str[i] + '.idl')

        for file in filename:
            self.paramList.append(read_idl_ascii(file))

        # self.params = read_idl_ascii(filename)
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
                    order='F', mode='r', *args, **kwargs):

        self.iix = iix
        self.iiy = iiy
        self.iiz = iiz

        try:
            if ((snap is not None) and (snap != self.snap)):
                self.set_snap(snap)

        except ValueError:
            if ((snap is not None) and any(snap != self.snap)):
                self.set_snap(snap)

        def helper(var, *args, **kwargs):

            if var in ['x', 'y', 'z']:
                return getattr(self, var)

            if var in self.simple_vars:  # is variable already loaded?
                return self._get_simple_var(var, *args, **kwargs)
            elif var in self.auxxyvars:
                return self._get_simple_var_xy(var, *args, **kwargs)
            elif var in self.compvars:  # add to variable list
                self.variables[var] = self._get_composite_var(
                    var, *args, **kwargs)
                setattr(self, var, self.variables[var])
                return self.variables[var]
            else:
                # raise ValueError(
                    # ("get_var: could not read variable %s. Must be "
                    # "one of %s" (var, (self.simple_vars +
                    # self.compvars + self.auxxyvars))))
                return self._get_quantity(var, *args, **kwargs)

        # lengths for size of return array
        self.xLength = 0
        self.yLength = 0
        self.zLength = 0

        # indices for filling in the 4D return array
        self.xInd = 0
        self.yInd = 0
        self.zInd = 0

        for dim in ('iix', 'iiy', 'iiz'):
            if getattr(self, dim) is None:
                setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]))
                setattr(self, dim[2] + 'Ind', slice(None))
                setattr(self, dim, slice(None))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)
                if indSize > 1:
                    setattr(self, dim[2] + 'Ind', slice(None))

        snapLen = np.size(self.snap)
        value = np.empty([self.xLength, self.yLength, self.zLength, snapLen])

        for i in range(0, snapLen):
            self.snapInd = i
            self._set_snapvars()
            self._init_vars()

            if (np.size(self.iix) > 1 or np.size(self.iiy) > 1 or
                    np.size(self.iiz) > 1):
                axes = [0, -2, -1]
                helperCall = helper(var)

                for counter, dim in enumerate(['iix', 'iiy', 'iiz']):
                    if getattr(self, dim) != slice(None):
                        helperCall = helperCall.take(
                            getattr(self, dim), axis=axes[counter])
            else:
                helperCall = helper(var)[self.iix, self.iiy, self.iiz]

            value[self.xInd, self.yInd, self.zInd, i] = helperCall
        # self.params = self.paramList
        return value

    def set_domain_iiaxis(self, iinum=slice(None), iiaxis='x'):

        if iinum is None:
            iinum = slice(None)

        dim = 'ii' + iiaxis
        setattr(self, dim, iinum)
        setattr(self, iiaxis + 'Length', np.size(iinum))
        setattr(self, iiaxis + 'Ind', 0)

        if np.size(getattr(self, dim)) == 1:
            if getattr(self, dim) == slice(None):
                setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]))
                setattr(self, dim[2] + 'Ind', slice(None))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)
                if indSize > 1:
                    setattr(self, dim[2] + 'Ind', slice(None))
                elif indSize == 1:
                    temp = np.asarray(getattr(self, dim))
                    setattr(self, dim, temp.item())
        else:
            indSize = np.size(getattr(self, dim))
            setattr(self, dim[2] + 'Length', indSize)
            if indSize > 1:
                setattr(self, dim[2] + 'Ind', slice(None))
            elif indSize == 1:
                temp = np.asarray(getattr(self, dim))
                setattr(self, dim, temp.item())

    def get_var(self, var, snap=None, iix=None, iiy=None,
                iiz=None, *args, **kwargs):
        """
        Reads a given variable from the relevant files.

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
            if (iix is not None) and (iix != self.iix):
                self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            if (iiy is not None) and (iiy != self.iiy):
                self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            if (iiz is not None) and (iiz != self.iiz):
                self.set_domain_iiaxis(iinum=iiz, iiaxis='z')

        if var in ['x', 'y', 'z']:
            return getattr(self, var)

        if (snap is not None) and (snap != self.snap):
            self.set_snap(snap)

        if var in self.simple_vars:  # is variable already loaded?
            val = self._get_simple_var(var, *args, **kwargs)
        elif var in self.auxxyvars:
            val = self._get_simple_var_xy(var, *args, **kwargs)
        elif var in self.compvars:  # add to variable list
            self.variables[var] = self._get_composite_var(var, *args, **kwargs)
            setattr(self, var, self.variables[var])
            val = self.variables[var]
        else:
            # raise ValueError(
                # ("get_var: could not read variable %s. Must be "
                # "one of %s" % (var,
                # (self.simple_vars + self.compvars + self.auxxyvars))))
            val = self._get_quantity(var, *args, **kwargs)

        if np.shape(val) != (self.xLength, self.yLength, self.zLength):
            val[self.iix, self.iiy, self.iiz].reshape((
                        self.xLength, self.yLength, self.zLength))

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
        if (np.size(self.snap) > 1):
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

        if var in (self.snapvars):
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
                filename = '%s_.hion%s.snap' % (self.file_root, isnap)
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
        return np.memmap(filename, dtype=self.dtype, order=order, mode=mode,
                         offset=offset, shape=ss)

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
            mom = self.get_var('p' + var[1])
            if getattr(self, 'n' + var[1]) < 5 or not self.cstagop:
                # do not recentre for 2D cases (or close)
                return mom / self.get_var('r')
            else:  # will call xdn, ydn, or zdn to get r at cell faces
                return mom / cstagger.do(self.get_var('r'), var[1] + 'dn')
        elif var == 'ee':   # internal energy
            return self.get_var('e') / self.get_var('r')
        elif var == 's':   # entropy?
            entr = np.log(self.get_var(
                            'p')) - self.params['gamma'] * np.log(
                                    self.get_var('r'))
            return entr
        # else:
            # raise ValueError(('_get_composite_var: do not know (yet) how to'
            # 'get composite variable %s.' % var))

    def _get_quantity(self, quant, *args, **kwargs):
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
        Not all possibilities for quantities are shown here. But there are
        a few main categories:
        - DERIV_QUANT: allows to calculate derivatives of any variable.
                       It must start with d followed with the varname and
                       ending with dxdn etc, e.g., 'dbxdxdn'
        - CENTRE_QUANT: allows to center any vector. It must end with xc
                        etc, e.g., 'ixc',
        - MODULE_QUANT: allows to calculate the module of any vector.
                        It must start with 'mod' followed with the root
                        letter of varname, e.g., 'modb'
        - DIV_QUANT: allows to calculate the divergence of any vector.
                     It must start with div followed with the root letter
                     of the varname, e.g., 'divb'
        - SQUARE_QUANT: allows to calculate the squared modules for any
                        vector. It must end with 2 after the root lelter
                        of the varname, e.g. 'u2'.
        """
        quant = quant.lower()
        DERIV_QUANT = ['dxup', 'dyup', 'dzup', 'dxdn', 'dydn', 'dzdn']
        CENTRE_QUANT = ['xc', 'yc', 'zc']
        MODULE_QUANT = ['mod', 'h']
        DIV_QUANT = ['div']
        SQUARE_QUANT = ['2']
        RATIO_QUANT = 'rat'
        EOSTAB_QUANT = ['ne', 'tg', 'pg', 'kr', 'eps', 'opa', 'temt']
        PROJ_QUANT = ['par', 'per']
        CURRENT_QUANT = ['ix', 'iy', 'iz', 'wx', 'wy', 'wz']
        FLUX_QUANT = ['pfx', 'pfy', 'pfz', 'pfex', 'pfey', 'pfez', 'pfwx',
                      'pfwy', 'pfwz', 'hx', 'hy', 'hz', 'kx', 'ky', 'kz']
        PLASMA_QUANT = ['beta', 'va', 'cs', 's',
                        'mn', 'man', 'hp', 'vax', 'vay', 'vaz']

        if (np.size(self.snap) > 1):
            currSnap = self.snap[self.snapInd]
        else:
            currSnap = self.snap

        if (RATIO_QUANT in quant):
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
            return result / self.get_var(q)

        elif (quant[:3] in MODULE_QUANT) or (
                quant[-1] in MODULE_QUANT) or (quant[-1] in SQUARE_QUANT):
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

        elif quant[0] == 'd' and quant[-4:] in DERIV_QUANT:
            # Calculate derivative of quantity
            axis = quant[-3]
            q = quant[1:-4]  # base variable

            var = self.get_var(q)

            if getattr(self, 'n' + axis) < 5:  # 2D or close
                print('(WWW) get_quantity: DERIV_QUANT: '
                      'n%s < 5, derivative set to 0.0' % axis)
                return np.zeros_like(var)
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

            var = self.get_var(q)
            # 2D
            if getattr(self, 'n' + axis) < 5 or self.cstagop is False:
                return var
            else:
                if len(transf) == 2:
                    tmp = cstagger.do(var, transf[0])
                    return cstagger.do(tmp, transf[1])
                else:
                    return cstagger.do(var, transf[0])

        elif quant[:3] in DIV_QUANT:
            # Calculates divergence of vector quantity
            q = quant[3:]  # base variable
            varx = self.get_var(q + 'x')
            vary = self.get_var(q + 'y')
            varz = self.get_var(q + 'z')

            if getattr(self, 'nx') < 5:  # 2D or close
                result = np.zeros_like(varx)
            else:
                result = cstagger.ddxup(varx)
            if getattr(self, 'ny') > 5:
                result += cstagger.ddyup(vary)
            if getattr(self, 'nz') > 5:
                result += cstagger.ddzup(varz)
            return result

        elif quant in EOSTAB_QUANT:
            # unit conversion to SI
            # to g/cm^3  (for ne_rt_table)
            ur = self.params['u_r'][self.snapInd]
            ue = self.params['u_ee'][self.snapInd]        # to erg/g
            if 'do_hion' in self.params and quant == 'ne':
                if self.params['do_hion'][self.snapInd] > 0:
                    return self.get_var('hionne')
            rho = self.r
            rho = rho * ur
            ee = self.get_var('ee')
            ee = ee * ue
            if self.verbose:
                print(quant + ' interpolation...')

            fac = 1.0
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

            x1 = self.get_var(v1 + 'xc', self.snap)
            y1 = self.get_var(v1 + 'yc', self.snap)
            z1 = self.get_var(v1 + 'zc', self.snap)
            x2 = self.get_var(v2 + 'xc', self.snap)
            y2 = self.get_var(v2 + 'yc', self.snap)
            z2 = self.get_var(v2 + 'zc', self.snap)

            v2Mag = np.sqrt(x2**2 + y2**2 + z2**2)
            v2x, v2y, v2z = x2 / v2Mag, y2 / v2Mag, z2 / v2Mag
            # parX, parY, parZ = x1 * v2x, y1 * v2x, z1 * v2x
            parScal = x1 * v2x + y1 * v2y + z1 * v2z
            parX, parY, parZ = parScal * v2x, parScal * v2y, parScal * v2z
            result = np.abs(parScal)

            if quant[1:4] == 'per':
                perX = x1 - parX
                perY = y1 - parY
                perZ = z1 - parZ
                # print(np.min(x1*x1 + y1*y1 + z1*z1 - result**2))
                # result1 = np.sqrt(x1*x1 + y1*y1 + z1*z1 - result**2)
                v1Mag = np.sqrt(perX**2 + perY**2 + perZ**2)
                result = v1Mag
                # print(np.nanmax(np.abs(result - result1)))
            return result

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
                    derv = ['ddydn', 'ddzdn']
                elif axis == 'y':
                    varsn = ['x', 'z']
                    derv = ['ddzdn', 'ddxdn']
                elif axis == 'z':
                    varsn = ['y', 'x']
                    derv = ['ddxdn', 'ddydn']
                var = self.get_var(q + varsn[0])
                # 2D or close
                if (getattr(self, 'n' + varsn[0]) <
                        5) or (getattr(self, 'n' + varsn[1]) < 5):
                    return np.zeros_like(var)
                else:
                    return cstagger.do(var, derv[0]) - cstagger.do(
                            self.get_var(q + varsn[1]), derv[1])

        elif quant in FLUX_QUANT:
            axis = quant[-1]
            if axis == 'x':
                varsn = ['z', 'y']
            elif axis == 'y':
                varsn = ['x', 'z']
            elif axis == 'z':
                varsn = ['y', 'x']
            if 'pfw' in quant or len(quant) == 3:
                var = self.get_var('b' + axis + 'c') * (
                    self.get_var('u' + varsn[0] + 'c') *
                    self.get_var('b' + varsn[0] + 'c') +
                    self.get_var('u' + varsn[1] + 'c') *
                    self.get_var('b' + varsn[1] + 'c'))

            elif 'pfe' in quant or len(quant) == 3:
                var += self.get_var('u' + axis + 'c') * (
                    self.get_var('b' + varsn[0] + 'c')**2 +
                    self.get_var('b' + varsn[1] + 'c')**2)
            return var

        elif quant in PLASMA_QUANT:
            if quant in ['hp', 's', 'cs', 'beta']:
                var = self.get_var('p')
                if quant == 'hp':
                    if (getattr(self, 'nx') < 5):
                        return np.zeros_like(var)
                    else:
                        return 1. / (cstagger.do(var, 'ddzup') + 1e-12)
                elif quant == 'cs':
                    return np.sqrt(
                        self.params['gamma'][0] * var / self.get_var('r'))
                elif quant == 's':
                    return np.log(
                            var) - self.params['gamma'][0] * np.log(
                            self.get_var('r'))
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
                    return np.sqrt(self.get_var('b' + axis + 'c')**2 / var)

            if quant in ['hx', 'hy', 'hz', 'kx', 'ky', 'kz']:
                axis = quant[-1]
                var = self.get_var('p' + axis + 'c')
                if quant[0] == 'h':
                    return (self.get_var('e') + self.get_var('p')) / \
                        self.get_var('r') * var
                else:
                    return self.get_var('u2') * var * 0.5

        else:
            raise ValueError(('get_quantity: do not know (yet) how to '
                              'calculate quantity %s. Note that simple_var '
                              'available variables are: %s.\nIn addition, '
                              'get_quantity can read others computed variables'
                              ' see e.g. self._get_quantity? for guidance'
                              '.' % (quant, repr(self.simple_vars))))

    def write_rh15d(self, outfile, desc=None, append=True,
                    sx=slice(None), sy=slice(None), sz=slice(None)):
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
        # unit conversion to SI
        ul = self.params['u_l'][self.snapInd] / 1.e2  # to metres
        # to g/cm^3  (for ne_rt_table)
        ur = self.params['u_r'][self.snapInd]
        ut = self.params['u_t'][self.snapInd]         # to seconds
        uv = ul / ut
        ub = self.params['u_b'][self.snapInd] * 1e-4  # to Tesla
        ue = self.params['u_ee'][self.snapInd]        # to erg/g
        hion = False
        if 'do_hion' in self.params:
            if self.params['do_hion'][self.snapInd] > 0:
                hion = True
        if self.verbose:
            print('Slicing and unit conversion...')
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]
        rho = rho * ur
        # TIAGO: must get this at cell centres!
        if self.do_mhd:
            Bx = self.bx[sx, sy, sz]
            By = self.by[sx, sy, sz]
            Bz = self.bz[sx, sy, sz]
            # Change sign of Bz (because of height scale) and By
            # (to make right-handed system)
            Bx = Bx * ub
            By = -By * ub
            Bz = -Bz * ub
        else:
            Bx = By = Bz = None

        # TIAGO: must get this at cell centres!
        vz = self.get_var('uz')[sx, sy, sz]
        vz *= -uv
        x = self.x[sx] * ul
        y = self.y[sy] * (-ul)
        z = self.z[sz] * (-ul)
        # convert from rho to H atoms, ideally from subs.dat. Otherwise
        # default.
        if hion:
            print('Getting hion data...')
            ne = self.get_var('hionne')
            # slice and convert from cm^-3 to m^-3
            ne = ne[sx, sy, sz]
            ne = ne * 1.e6
            # read hydrogen populations (they are saved in cm^-3)
            nh = np.empty((6,) + temp.shape, dtype='Float32')
            for k in range(6):
                nv = self.get_var('n%i' % (k + 1))
                nh[k] = nv[sx, sy, sz]
            nh = nh * 1.e6
        else:
            ee = self.get_var('ee')[sx, sy, sz]
            ee = ee * ue
            if os.access('%s/subs.dat' % self.fdir, os.R_OK):
                grph = subs2grph('%s/subs.dat' % self.fdir)
            else:
                grph = 2.380491e-24
            nh = rho / grph * 1.e6       # from rho to nH in m^-3
            # interpolate ne from the EOS table
            if self.verbose:
                print('ne interpolation...')
            eostab = Rhoeetab(fdir=self.fdir)
            ne = eostab.tab_interp(rho, ee, order=1) * 1.e6  # cm^-3 to m^-3
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.file_root, repr(sx), repr(sy), repr(sz))
            if hion:
                desc = 'hion ' + desc
        # write to file
        if self.verbose:
            print('Write to file...')
        rh15d.make_xarray_atmos(outfile, temp, vz, nh, z, ne=ne, x=x, y=y,
                                append=append, Bx=Bx, By=By, Bz=Bz, desc=desc,
                                snap=self.snap)

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
        from .multi3dn import Multi3dAtmos
        # unit conversion to cgs and km/s
        ul = self.params['u_l'][self.snapInd]   # to cm
        ur = self.params['u_r'][self.snapInd]   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t'][self.snapInd]   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ue = self.params['u_ee'][self.snapInd]  # to erg/g
        nh = None
        hion = False
        if 'do_hion' in self.params:
            if self.params['do_hion'][self.snapInd] > 0:
                hion = True
        if self.verbose:
            print('Slicing and unit conversion...')
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]
        rho = rho * ur
        # Change sign of vz (because of height scale) and vy (to make
        # right-handed system)
        vx = self.get_var('ux')[sx, sy, sz]
        vx *= uv
        vy = self.get_var('uy')[sx, sy, sz]
        vy *= -uv
        vz = self.get_var('uz')[sx, sy, sz]
        vz *= -uv
        x = self.x[sx] * ul
        y = self.y[sy] * ul
        z = self.z[sz] * (-ul)
        # if Hion, get nH and ne directly
        if hion:
            print('Getting hion data...')
            ne = self.get_var('hionne')
            # slice and convert from cm^-3 to m^-3
            ne = ne[sx, sy, sz]
            ne = ne * 1.e6
            # read hydrogen populations (they are saved in cm^-3)
            nh = np.empty((6,) + temp.shape, dtype='Float32')
            for k in range(6):
                nv = self.get_var('n%i' % (k + 1))
                nh[k] = nv[sx, sy, sz]
            nh = nh * 1.e6
        else:
            ee = self.get_var('ee')[sx, sy, sz]
            ee = ee * ue
            # interpolate ne from the EOS table
            print('ne interpolation...')
            eostab = Rhoeetab(fdir=self.fdir)
            ne = eostab.tab_interp(rho, ee, order=1)
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
        if mesh is not None:
            fout2 = open(mesh, "w")
            fout2.write("%i\n" % nx)
            x.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % ny)
            y.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % nz)
            z.tofile(fout2, sep="  ", format="%11.5e")
            fout2.close()


class create_new_br_files():
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


class bifrost_units():
    import scipy.constants as const
    from astropy import constants as aconst
    """
    bifrost_units.py

    Created by Mikolaj Szydlarski on 2017-01-20.
    Copyright (c) 2014, ITA UiO - All rights reserved.
    """

    u_l = 1e8
    u_t = 1e2
    u_r = 1e-7
    u_u = u_l / u_t
    u_p = u_r * (u_l / u_t)**2          # Pressure [dyne/cm2]
    u_kr = 1 / (u_r * u_l)               # Rosseland opacity [cm2/g]
    u_ee = u_u**2
    u_e = u_r * u_ee
    u_te = u_e / u_t * u_l               # Box therm. em. [erg/(s ster cm2)]
    mu = 0.8
    u_n = 3.00e+10                  # Denisty number n_0 * 1/cm^3
    k_B = aconst.k_B.to('erg/K')  # 1.380658E-16 Boltzman's cst. [erg/K]
    m_H = const.m_n / const.gram  # 1.674927471e-24
    m_He = 6.65e-24
    m_p = mu * m_H   # Mass per particle
    m_e = const.m_e / const.gram  # 9.1093897E-28
    u_tg = (m_H / k_B) * u_ee
    u_tge = (m_e / k_B) * u_ee
    pi = const.pi
    u_b = u_u * np.sqrt(4. * pi * u_r)

    usi_l = u_l * const.centi  # 1e6
    usi_r = u_r * const.gram  # 1e-4
    usi_u = usi_l / u_t
    usi_p = usi_r * (usi_l / u_t)**2       # Pressure [N/m2]
    usi_kr = 1 / (usi_r * usi_l)            # Rosseland opacity [m2/kg]
    usi_ee = usi_u**2
    usi_e = usi_r * usi_ee
    usi_te = usi_e / u_t * usi_l            # Box therm. em. [J/(s ster m2)]
    ksi_B = aconst.k_B.to('J/K')  # 1.380658E-23 Boltzman's cst. [J/K]
    msi_H = const.m_n  # 1.674927471e-27
    msi_He = 6.65e-27
    msi_p = mu * msi_H  # Mass per particle
    usi_tg = (msi_H / ksi_B) * usi_ee
    msi_e = const.m_e  # 9.1093897e-31

    # Solar gravity
    gsun = 27400.0  # (cgs)

    # --- ideal gas
    gamma = 1.667

    # --- physical constants and other useful quantities
    CLIGHT = aconst.c.to('cm/s')  # 2.99792458E+10 Speed of light [cm/s]
    HPLANCK = aconst.h.to('erg s')  # 6.6260755E-27 Planck's constant [erg s]
    KBOLTZMANN = aconst.k_B.to('erg/K')  # 1.380658E-16 Boltzman's cst. [erg/K]
    AMU = aconst.u.to('g')  # 1.6605402E-24 Atomic mass unit [g]
    AMUSI = aconst.u.to('kg')  # 1.6605402E-27 Atomic mass unit [kg]
    M_ELECTRON = aconst.m_e.to('g')  # 9.1093897E-28 Electron mass [g]
    Q_ELECTRON = 4.80325E-10    # Electron charge [esu]
    QSI_ELECTRON = aconst.e  # 1.6021765e-19 Electron charge [C]
    RBOHR = aconst.a0.to('cm')  # 5.29177349E-9 Bohr radius [cm]
    E_RYDBERG = 2.1798741E-11  # Ion. pot. Hydrogen [erg]
    EH2DISS = 4.478          # H2 dissociation energy [eV]
    pie2_mec = 0.02654        # pi e^2 / m_e c [cm^2 Hz]
    # 5.670400e-5 Stefan-Boltzmann constant [erg/(cm^2 s K^4)]
    stefboltz = aconst.sigma_sb.to('erg/(cm2 s K4)')
    MION = m_H            # Ion mass [g]
    R_EI = 1.44E-7        # e^2 / kT = 1.44x10^-7 T^-1 cm

    # --- Unit conversions
    EV_TO_ERG = const.eV / const.erg  # 1.60217733E-12 One electronVolt [erg]
    EV_TO_J = const.eV  # 1.60217733E-19 One electronVolt [J]
    NM_TO_M = const.nano  # 1.0E-09
    CM_TO_M = const.centi  # 1.0E-02
    KM_TO_M = const.kilo  # 1.0E+03
    ERG_TO_JOULE = const.erg  # 1.0E-07
    G_TO_KG = const.gram  # 1.0E-03
    MICRON_TO_NM = 1.0E+03
    MEGABARN_TO_M2 = 1.0E-22
    ATM_TO_PA = const.atm  # 1.0135E+05 Atm to Pascal (N/m^2)
    DYNE_CM2_TO_PASCAL = 0.1
    K_TO_EV = 8.621738E-5    # KtoeV
    EV_TO_K = 11604.50520    # eVtoK
    ergd2wd = 0.1


class Rhoeetab:
    def __init__(self, tabfile=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True, radtab=False):
        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.eosload = False
        self.radload = False
        # read table file and calculate parameters
        if tabfile is None:
            tabfile = '%s/tabparam.in' % (fdir)
        self.param = self.read_tab_file(tabfile)
        # load table(s)
        self.load_eos_table()
        if radtab:
            self.load_rad_table()
        return

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
        return

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
        return

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
        return

    def get_table(self, out='ne', bine=None, order=1):
        import scipy.ndimage as ndimage
        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
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
        ''' Interpolates the EOS/rad table for the required quantity in out.

            IN:
                rho  : density [g/cm^3]
                ei   : internal energy [erg/g]
                bin  : (optional) radiation bin number for bin parameters
                order: interpolation order (1: linear, 3: cubic)

            OUT:
                depending on value of out:
                'nel'  : electron density [cm^-3]
                'tg'   : temperature [K]
                'pg'   : gas pressure [dyn/cm^2]
                'kr'   : Rosseland opacity [cm^2/g]
                'eps'  : scattering probability
                'opa'  : opacity
                'temt' : thermal emission
        '''
        import scipy.ndimage as ndimage
        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
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
        result = ndimage.map_coordinates(
            quant, [x, y], order=order, mode='nearest')
        return (np.exp(result) if out != 'tg' else result)


class Opatab:

    def __init__(self, tabname=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True, lambd=100.0):
        ''' Loads opacity table and calculates the photoionization cross
        sections given by anzer & heinzel apj 622: 714-721, 2005, march 20
        they have big typos in their reported c's.... correct values to
        be found in rumph et al 1994 aj, 107: 2108, june 1994

        gaunt factors are set to 0.99 for h and 0.85 for heii,
        which should be good enough for the purposes of this code
        '''
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
            tabname = '%s/ionization.dat' % (fdir)
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
        import scipy.ndimage as ndimage
        self.load_opa1d_table()
        rhoeetab = Rhoeetab(fdir=self.fdir)
        tgTable = rhoeetab.get_table('tg')
        # translate to table coordinates
        x = (np.log10(tgTable) - self.teinit) / self.dte
        # interpolate quantity
        self.ionh = ndimage.map_coordinates(self.ionh1d, [x], order=order)
        self.ionhe = ndimage.map_coordinates(self.ionhe1d, [x], order=order)
        self.ionhei = ndimage.map_coordinates(self.ionhei1d, [x], order=order)

    def h_he_absorb(self, lambd=None):
        '''
        Gets the opacities for a particular wavelength of light.
        If lambd is None, then looks at the current level for wavelength
        '''
        rhe = 0.1
        epsilon = 1.e-20
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


###########
#  TOOLS  #
###########
def read_idl_ascii(filename):
    ''' Reads IDL-formatted (command style) ascii file into dictionary '''
    li = 0
    params = {}
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
            line = line.split(';')[0].split('=')
            if (len(line) != 2):
                print(('(WWW) read_params: line %i is invalid, skipping' % li))
                li += 1
                continue
            # force lowercase because IDL is case-insensitive
            key = line[0].strip().lower()
            value = line[1].strip()
            # instead of the insecure 'exec', find out the datatypes
            if (value.find('"') >= 0):
                # string type
                value = value.strip('"')
            elif (value.find("'") >= 0):
                value = value.strip("'")
            elif (value.lower() in ['.false.', '.true.']):
                # bool type
                value = False if value.lower() == '.false.' else True
            elif (value.find('[') >= 0 and value.find(']') >= 0):
                # list type
                value = eval(value)
            elif ((value.upper().find('E') >= 0) or (value.find('.') >= 0)):
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
            li += 1
    return params


def subs2grph(subsfile):
    ''' From a subs.dat file, extract abundances and atomic masses to calculate
    grph, grams per hydrogen. '''
    from scipy.constants import atomic_mass as amu

    f = open(subsfile, 'r')
    nspecies = np.fromfile(f, count=1, sep=' ', dtype='i')[0]
    f.readline()  # second line not important
    ab = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    am = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    f.close()
    # linear abundances
    ab = 10.**(ab - 12.)
    # mass in grams
    am *= amu * 1.e3
    return np.sum(ab * am)


def ne_rt_table(rho, temp, order=1, tabfile=None):
    ''' Calculates electron density by interpolating the rho/temp table.
        Based on Mats Carlsson's ne_rt_table.pro.

        IN: rho (in g/cm^3),
            temp (in K),

        OPTIONAL: order (interpolation order 1: linear, 3: cubic),
                  tabfile (path of table file)

        OUT: electron density (in g/cm^3)

        '''
    import os
    import scipy.interpolate as interp
    import scipy.ndimage as ndimage
    from scipy.io.idl import readsav
    print('DEPRECATION WARNING: this method is deprecated in favour'
          ' of the Rhoeetab class.')
    if tabfile is None:
        tabfile = 'ne_rt_table.idlsave'
    # use table in default location if not found
    if not os.path.isfile(tabfile) and \
            os.path.isfile(os.getenv('TIAGO_DATA') + '/misc/' + tabfile):
        tabfile = os.getenv('TIAGO_DATA') + '/misc/' + tabfile
    tt = readsav(tabfile, verbose=False)
    lgrho = np.log10(rho)
    # warnings for values outside of table
    tmin = np.min(temp)
    tmax = np.max(temp)
    ttmin = np.min(5040. / tt['theta_tab'])
    ttmax = np.max(5040. / tt['theta_tab'])
    lrmin = np.min(lgrho)
    lrmax = np.max(lgrho)
    tlrmin = np.min(tt['rho_tab'])
    tlrmax = np.max(tt['rho_tab'])
    if tmin < ttmin:
        print(('(WWW) ne_rt_table: temperature outside table bounds. ' +
               'Table Tmin=%.1f, requested Tmin=%.1f' % (ttmin, tmin)))
    if tmax > ttmax:
        print(('(WWW) ne_rt_table: temperature outside table bounds. ' +
               'Table Tmax=%.1f, requested Tmax=%.1f' % (ttmax, tmax)))
    if lrmin < tlrmin:
        print(('(WWW) ne_rt_table: log density outside of table bounds. ' +
               'Table log(rho) min=%.2f, requested log(rho) min=%.2f' %
               (tlrmin, lrmin)))
    if lrmax > tlrmax:
        print(('(WWW) ne_rt_table: log density outside of table bounds. ' +
               'Table log(rho) max=%.2f, requested log(rho) max=%.2f' %
               (tlrmax, lrmax)))
    # Approximate interpolation (bilinear/cubic interpolation) with ndimage
    y = (5040. / temp - tt['theta_tab'][0]) / \
        (tt['theta_tab'][1] - tt['theta_tab'][0])
    x = (lgrho - tt['rho_tab'][0]) / (tt['rho_tab'][1] - tt['rho_tab'][0])
    result = ndimage.map_coordinates(
        tt['ne_rt_table'], [x, y], order=order, mode='nearest')
    return 10**result * rho / tt['grph']
