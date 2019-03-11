"""
Set of programs to read and interact with output from Bifrost
"""

import os
from glob import glob
import numpy as np
from . import cstagger


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

    Examples
    --------
    This reads snapshot 383 from simulation "cb24bih", whose file
    root is "cb24bih", and is found at directory /data/cb24bih:

    >>> a = Bifrost.Data("cb24bih", snap=383, fdir="/data/cb24bih")

    Scalar variables do not need de-staggering and are available as
    memory map (only loaded to memory when needed), e.g.:

    >>> a.r.shape
    (504, 504, 496)

    Composite variables need to be obtained by get_var():

    >>> vx = a.get_var("ux")
    """

    def __init__(self, file_root, snap=None, meshfile=None, fdir='.',
                 verbose=True, dtype='f4', big_endian=False,
                 ghost_analyse=False):
        """
        Loads metadata and initialises variables.
        """
        self.fdir = fdir
        self.verbose = verbose
        self.file_root = os.path.join(self.fdir, file_root)
        self.meshfile = meshfile
        self.ghost_analyse = ghost_analyse
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        self.hion = False

        self.set_snap(snap)

    def _set_snapvars(self):
        """
            Sets list of avaible variables
        """
        self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        self.auxvars = self.params['aux'].split()
        if self.do_mhd:
            self.snapvars += ['bx', 'by', 'bz']
        self.hionvars = []
        if 'do_hion' in self.params:
            if self.params['do_hion'] > 0:
                self.hion = True
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
        snap - integer
            Number of simulation snapshot to load.
        """
        if snap is None:
            try:
                tmp = sorted(glob("%s*idl" % self.file_root))[0]
                snap = int(tmp.split(self.file_root + '_')[1].split(".idl")[0])
            except IndexError:
                raise ValueError(("(EEE) set_snap: snapshot not defined and no"
                                  " .idl files found"))
        self.snap = snap
        self.snap_str = '_%03i' % snap

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
        if (self.snap < 0):
            filename = self.file_root + '.idl.scr'
        elif (self.snap == 0):
            filename = self.file_root + '.idl'
        else:
            filename = self.file_root + self.snap_str + '.idl'
        self.params = read_idl_ascii(filename)
        # assign some parameters as attributes
        for p in ['x', 'y', 'z', 'b']:
            try:
                setattr(self, 'n' + p, self.params['m' + p])
            except KeyError:
                raise KeyError(('read_params: could not find '
                                'm%s in idl file!' % p))
        for p in ['dx', 'dy', 'dz', 'do_mhd']:
            try:
                setattr(self, p, self.params[p])
            except KeyError:
                raise KeyError(('read_params: could not find '
                                '%s in idl file!' % p))
        try:
            if self.params['boundarychk'] == 1:
                self.nzb = self.nz + 2 * self.nb
            else:
                self.nzb = self.nz
        except KeyError:
            self.nzb = self.nz
        # check if units are there, if not use defaults and print warning
        unit_def = {'u_l': 1.e8, 'u_t': 1.e2, 'u_r': 1.e-7,
                    'u_b': 1.121e3, 'u_ee': 1.e12}
        for unit in unit_def:
            if unit not in self.params:
                print(("(WWW) read_params:"" %s not found, using "
                       "default of %.3e" % (unit, unit_def[unit])))
                self.params[unit] = unit_def[unit]

    def __read_mesh(self, meshfile):
        """
        Reads mesh file
        """
        if meshfile is None:
            meshfile = os.path.join(self.fdir, self.params['meshfile'].strip())
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
                self.z = np.concatenate((
                  self.z[0] - np.linspace(self.dz*self.nb, self.dz, self.nb),
                  self.z,
                  self.z[-1] + np.linspace(self.dz, self.dz*self.nb, self.nb)))
                self.zdn = np.concatenate((
                  self.zdn[0] - np.linspace(self.dz*self.nb, self.dz, self.nb),
                  self.zdn, (self.zdn[-1] +
                             np.linspace(self.dz, self.dz*self.nb, self.nb))))
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

    def get_var(self, var, snap=None, *args, **kwargs):
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
        if (snap is not None) and (snap != self.snap):
            self.set_snap(snap)
        if var in self.simple_vars:  # is variable already loaded?
            return self._get_simple_var(var, *args, **kwargs)
        elif var in self.auxxyvars:
            return self._get_simple_var_xy(var, *args, **kwargs)
        elif var in self.compvars:  # add to variable list
            self.variables[var] = self._get_composite_var(var, *args, **kwargs)
            setattr(self, var, self.variables[var])
            return self.variables[var]
        else:
            raise ValueError(
                ("get_var: could not read variable %s. Must be "
                 "one of %s" %
                 (var, (self.simple_vars + self.compvars + self.auxxyvars))))

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
        if self.snap < 0:
            filename = self.file_root
            fsuffix_b = '.scr'
        elif self.snap == 0:
            filename = self.file_root
            fsuffix_b = ''
        else:
            filename = self.file_root + self.snap_str
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
            isnap = self.params['isnap']
            if isnap <= -1:
                filename = filename + '.hion.snap.scr'
            elif isnap == 0:
                filename = filename + '.hion.snap'
            elif isnap > 0:
                filename = '%s.hion_%s.snap' % (self.file_root, isnap)
                if not os.path.isfile(filename):
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
            p = self._get_simple_var('p' + var[1], order='F')
            if getattr(self, 'n' + var[1]) < 5:
                return p / self.r   # do not recentre for 2D cases (or close)
            else:  # will call xdn, ydn, or zdn to get r at cell faces
                return p / cstagger.do(self.r, var[1] + 'dn')
        elif var == 'ee':   # internal energy
            return self.e / self.r
        elif var == 's':   # entropy?
            return np.log(self.p) - self.params['gamma'] * np.log(self.r)
        else:
            raise ValueError(('_get_composite_var: do not know (yet) how to'
                              'get composite variable %s.' % var))

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
        MODULE_QUANT = ['mod']
        DIV_QUANT = ['div']
        SQUARE_QUANT = ['2']

        if (quant[:3] in MODULE_QUANT) or (quant[-1] in SQUARE_QUANT):
            # Calculate module of vector quantity
            q = quant[3:]
            if q == 'b':
                if not self.do_mhd:
                    raise ValueError("No magnetic field available.")
            if getattr(self, 'nx') < 5:  # 2D or close
                result = getattr(self, q + 'x') ** 2
            else:
                result = self.get_quantity(q + 'xc') ** 2
            if getattr(self, 'ny') < 5:  # 2D or close
                result += getattr(self, q + 'y') ** 2
            else:
                result += self.get_quantity(q + 'yc') ** 2
            if getattr(self, 'nz') < 5:  # 2D or close
                result += getattr(self, q + 'z') ** 2
            else:
                result += self.get_quantity(q + 'zc') ** 2
            if quant[:3] in MODULE_QUANT:
                return np.sqrt(result)
            elif quant[-1] in SQUARE_QUANT:
                return result
        elif quant[0] == 'd' and quant[-4:] in DERIV_QUANT:
            # Calculate derivative of quantity
            axis = quant[-3]
            q = quant[1:-4]  # base variable
            try:
                var = getattr(self, q)
            except AttributeError:
                var = self.get_var(q)
            if getattr(self, 'n' + axis) < 5:  # 2D or close
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
            try:
                var = getattr(self, q)
            except AttributeError:
                var = self.get_var(q)
            if getattr(self, 'n' + axis) < 5:  # 2D or close
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
            try:
                varx = getattr(self, q + 'x')
                vary = getattr(self, q + 'y')
                varz = getattr(self, q + 'z')
            except AttributeError:
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
        else:
            raise ValueError(('get_quantity: do not know (yet) how to '
                              'calculate quantity %s. Note that simple_var '
                              'available variables are: %s.\nIn addition, '
                              'get_quantity can read others computed variables'
                              ' see e.g. help(self.get_quantity) for guidance'
                              '.' % (quant, repr(self.simple_vars))))

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
            ee = ee * self.params['u_ee']
            eostab = Rhoeetab(fdir=self.fdir)
            rho = self.r[sx, sy, sz] * self.params['u_r']   # to cm^-3
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
            shape = [6, ] + [np.mgrid[item].size for item in (sx, sy, sz)]
            nh = np.empty(shape, dtype='Float32')
            for k in range(6):
                nv = self.get_var('n%i' % (k + 1))
                nh[k] = nv[sx, sy, sz]
        else:
            rho = self.r[sx, sy, sz] * self.params['u_r']
            subsfile = os.path.join(self.fdir, 'subs.dat')
            tabfile = os.path.join(self.fdir, self.params['tabinputfile'].strip())
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
        ul = self.params['u_l'] / 1.e2  # to metres
        ur = self.params['u_r']         # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']         # to seconds
        uv = ul / ut
        ub = self.params['u_b'] * 1e-4  # to Tesla
        ue = self.params['u_ee']        # to erg/g
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
        ul = self.params['u_l']   # to cm
        ur = self.params['u_r']   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ue = self.params['u_ee']  # to erg/g
        nh = None
        hion = False
        if 'do_hion' in self.params:
            if self.params['do_hion'] > 0:
                hion = True
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
        self.lnrk = table[:, :, 3]
        self.eosload = True
        if self.verbose:
            print(('*** Read EOS table from ' + eostabfile))

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
    """
    Class to loads opacity table and calculate the photoionization cross
    sections given by anzer & heinzel apj 622: 714-721, 2005, march 20
    they have big typos in their reported c's.... correct values to
    be found in rumph et al 1994 aj, 107: 2108, june 1994

    gaunt factors are set to 0.99 for h and 0.85 for heii,
    which should be good enough for the purposes of this code
    """
    def __init__(self, tabname=None, fdir='.', big_endian=False, dtype='f4',
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
    hion = data.params['do_hion']
    if hion:
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
        if hion:
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
            elif value.find("'") >= 0:
                value = value.strip("'")
            elif value.lower() in ['.false.', '.true.']:
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
