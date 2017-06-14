"""
Set of programs to read and interact with output from Bifrost
"""

import numpy as np
import os

class BifrostData(object):

    """
    Class to hold data from Bifrost simulations in native format.
    """

    def __init__(self, snap, file_root='qsmag-by00_t', meshfile=None, fdir='.',
                 verbose=True, dtype='f4', big_endian=False):
        """
        Loads metadata and initialises variables.

        Parameters
        ----------
        snap - integer
            Snapshot number
        file_root - string, optional
            Basename for all file names. Snapshot number will be added
            afterwards, and directory will be added before.
        meshfile - string, optional
            File name (including full path) for file with mesh. If set
            to None (default), a uniform mesh will be created.
        fdir - string, optional
            Directory where simulation files are. Must be a real path.
        verbose - bool
            If True, will print out more diagnostic messages
        dtype - string, optional
            Data type for reading variables. Default is 32 bit float.
        big_endian - string, optional
            If True, will read variables in big endian. Default is False
            (reading in little endian).
        """
        self.snap = snap
        self.fdir = fdir
        self.file_root = os.path.join(self.fdir, file_root)
        self.snap_str = '_%03i' % snap
        self.verbose = verbose
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        # read .idl file
        self.__read_params()
        # read mesh file
        self.__read_mesh(meshfile)
        # variables: lists and initialisation
        self.auxvars = self.params['aux'].split()
        self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        self.snapevars = ['er', 'epx', 'epy', 'epz', 'ee']  # electron specific
        if (self.do_mhd):
            self.mhdvars = ['bx', 'by', 'bz']
        else:
            self.mhdvars = []
        self.hionvars = ['hionne', 'hiontg', 'n1',
                         'n2', 'n3', 'n4', 'n5', 'n6', 'fion', 'nh2']
        self.compvars = ['ux', 'uy', 'uz', 's', 'bxc', 'byc', 'bzc', 'rup',
                         'dxdbup', 'dxdbdn', 'dydbup', 'dydbdn', 'dzdbup',
                         'dzdbdn', 'modb', 'modp']   # composite variables
        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.auxxyvars.append('ixy1')
        self.vars2d = []
        # special case for the 2d variable, lives in a separate file
        for var in self.auxvars:
            if any(i in var for i in ('xy', 'yz', 'xz')):
                self.auxvars.remove(var)
                self.vars2d.append(var)
        self._init_vars()

    def __read_params(self):
        """
        Reads parameter file (.idl)
        """
        if (self.snap < 0):
            filename = self.file_root + self.snap_str + '.idl.src'
        elif (self.snap == 0):
            filename =self.file_root + '.idl'
        else:
            filename = self.file_root + self.snap_str + '.idl'

        self.params = read_idl_ascii(filename)

        # assign some parameters to root object
        try:
            self.nx = self.params['mx']
        except KeyError:
            raise KeyError('read_params: could not find nx in idl file!')
        try:
            self.ny = self.params['my']
        except KeyError:
            raise KeyError('read_params: could not find ny in idl file!')
        try:
            self.nz = self.params['mz']
        except KeyError:
            raise KeyError('read_params: could not find nz in idl file!')
        try:
            self.nb = self.params['mb']
        except KeyError:
            raise KeyError('read_params: could not find nb in idl file!')
        try:
            if self.params['boundarychk'] == 1:
                self.nzb = self.nz + 2 * self.nb
            else:
                self.nzb = self.nz
        except KeyError:
            self.nzb = self.nz
        try:
            self.dx = self.params['dx']
        except KeyError:
            raise KeyError('read_params: could not find dx in idl file!')

        try:
            self.dy = self.params['dy']
        except KeyError:
            raise KeyError('read_params: could not find dy in idl file!')

        try:
            self.dz = self.params['dz']
        except KeyError:
            raise KeyError('read_params: could not find dz in idl file!')

        try:
            self.do_mhd = self.params['do_mhd']
        except KeyError:
            raise KeyError('read_params: could not find do_mhd in idl file!')
        # check if units are there, if not use defaults and print warning
        unit_def = {'u_l': 1.e8, 'u_t': 1.e2, 'u_r': 1.e-7, 'u_b': 1.121e3,
                    'u_ee': 1.e12}
        for unit in unit_def:
            if unit not in self.params:
                print(("(WWW) read_params: %s not found, using default of %.3e" %
                       (unit, unit_def[unit])))
                self.params[unit] = unit_def[unit]

    def __read_mesh(self, meshfile):
        """
        Reads mesh file
        """
        if meshfile is None:
            meshfile = os.path.join(self.fdir, self.params['meshfile'].strip())
        if os.path.isfile(meshfile):
            f = open(meshfile, 'r')
            mx = int(f.readline().strip('\n').strip())
            assert mx == self.nx
            self.x = np.array([float(v)
                               for v in f.readline().strip('\n').split()])
            self.xdn = np.array([float(v) for v in
                                 f.readline().strip('\n').split()])
            self.dxidxup = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            self.dxidxdn = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            my = int(f.readline().strip('\n').strip())
            assert my == self.ny
            self.y = np.array([float(v) for v in
                               f.readline().strip('\n').split()])
            self.ydn = np.array([float(v) for v in
                                 f.readline().strip('\n').split()])
            self.dyidyup = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            self.dyidydn = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            mz = int(f.readline().strip('\n').strip())
            assert mz == self.nz
            self.z = np.array([float(v) for v in
                               f.readline().strip('\n').split()])
            self.zdn = np.array([float(v) for v in
                                 f.readline().strip('\n').split()])
            self.dzidzup = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            self.dzidzdn = np.array([float(v) for v in
                                     f.readline().strip('\n').split()])
            f.close()
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
            self.dxidxup = np.zeros(self.nx) + 1./self.dx
            self.dxidxdn = np.zeros(self.nx) + 1./self.dx
            # y
            self.y = np.arange(self.ny) * self.dy
            self.ydn = self.y - 0.5 * self.dy
            self.dyidyup = np.zeros(self.ny) + 1./self.dy
            self.dyidydn = np.zeros(self.ny) + 1./self.dy
            # z
            self.z = np.arange(self.nz) * self.dz
            self.zdn = self.z - 0.5 * self.dz
            self.dzidzup = np.zeros(self.nz) + 1./self.dz
            self.dzidzdn = np.zeros(self.nz) + 1./self.dz

    def getvar_xy(self, var, order='F', mode='r'):
        """
        Reads a given 2D variable from the _XY.aux file
        """
        import os
        if var in self.auxxyvars:
            fsuffix = '_XY.aux'
            idx = self.auxxyvars.index(var)
            filename = self.template + fsuffix
        else:
            raise ValueError('getvar_xy: variable %s not available. Available vars:'
                             % (var) + '\n' + repr(self.auxxyvars))
        # Now memmap the variable
        if not os.path.isfile(filename):
            raise IOError('getvar: variable %s should be in %s file, not found!' %
                          (var, filename))
        # size of the data type
        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * idx * dsize
        return np.memmap(filename, dtype=self.dtype, order=order, offset=offset,
                         mode=mode, shape=(self.nx, self.ny))

    def getvar(self, var, order='F', mode='r'):
        """
        Reads a given variable from the relevant files.
        """
        if var in ['x', 'y', 'z']:
            return getattr(self, var)
        # find filename template
        if self.snap < 0:
            filename = self.file_root
            fsuffix_b = '.scr'
        elif self.snap == 0:
            filename = self.file_root
            fsuffix_b = ''
        else:
            filename = self.file_root + self.snap_str
            fsuffix_b = ''
        # find in which file the variable is
        if var in self.compvars:
            return self._get_compvar(var)
        elif var in (self.snapvars + self.mhdvars):
            fsuffix_a = '.snap'
            idx = (self.snapvars + self.mhdvars).index(var)
            filename = filename + fsuffix_a + fsuffix_b
        elif var in self.auxvars:
            fsuffix_a = '.aux'
            idx = self.auxvars.index(var)
            filename = filename + fsuffix_a + fsuffix_b
        elif var in self.hionvars:
            idx = self.hionvars.index(var)
            isnap = self.params['isnap']
            if isnap <= -1:
                filename = filename + '.hion.snap.scr'
            elif isnap == 0:
                filename = filename + '.hion.snap'
            elif isnap > 0:
                filename = '%s_.hion%s.snap' % (self.file_root, isnap)
        else:
            raise ValueError('getvar: variable ' +
                             '%s not available. Available vars:' % (var) +
                             '\n' + repr(self.auxvars + self.snapvars +
                                         self.hionvars))
        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * self.nzb * idx * dsize
        return np.memmap(filename, dtype=self.dtype, order=order, offset=offset,
                         mode=mode, shape=(self.nx, self.ny, self.nzb))

    def _get_compvar(self, var):
        """
        Gets composite variables (will load into memory).
        """
        from . import cstagger
        # if rho is not loaded, do it (essential for composite variables)
        # rc is the same as r, but in C order (so that cstagger works)
        if not hasattr(self, 'rc'):
            self.rc = self.variables['rc'] = self.getvar('r', order='C')
            # initialise cstagger
            rdt = self.r.dtype
            cstagger.init_stagger(self.nzb, self.z.astype(rdt),
                                  self.zdn.astype(rdt))
        if var == 'ux':  # x velocity
            if not hasattr(self, 'px'):
                self.px = self.variables['px'] = self.getvar('px')
            if self.nx < 5:  # do not recentre for 2D cases (or close)
                return self.px / self.rc
            else:
                return self.px / cstagger.xdn(self.rc)
        elif var == 'uy':  # y velocity
            if not hasattr(self, 'py'):
                self.py = self.variables['py'] = self.getvar('py')
            if self.ny < 5:  # do not recentre for 2D cases (or close)
                return self.py / self.rc
            else:
                return self.py / cstagger.ydn(self.rc)
        elif var == 'uz':  # z velocity
            if not hasattr(self, 'pz'):
                self.pz = self.variables['pz'] = self.getvar('pz')
            return self.pz / cstagger.zdn(self.rc)
        elif var == 'ee':   # internal energy?
            if not hasattr(self, 'e'):
                self.e = self.variables['e'] = self.getvar('e')
            return self.e / self.r
        elif var == 's':   # entropy?
            if not hasattr(self, 'p'):
                self.p = self.variables['p'] = self.getvar('p')
            return np.log(self.p) - 1.667 * np.log(self.r)
        else:
            raise ValueError('getcompvar: composite var %s not found. Available:\n %s'
                             % (var, repr(self.compvars)))
        return

    def _init_vars(self):
        """
        Memmaps aux and snap variables, and maps them to methods.
        Also, sets file name[s] from which to read a data
        """
        self.variables = {}
        # snap variables
        for var in self.snapvars + self.mhdvars + self.auxvars:
            try:
                self.variables[var] = self.getvar(var)
                setattr(self, var, self.variables[var])
            except:
                print(('(WWW) init_vars: could not read variable %s' % var))
        for var in self.auxxyvars:
            try:
                self.variables[var] = self.getvar_xy(var)
                setattr(self, var, self.variables[var])
            except:
                print(('(WWW) init_vars: could not read variable %s' % var))

    def write_rh15d(self, outfile, sx=None, sy=None, sz=None, desc=None,
                    append=True):
        """
        Writes snapshot in RH 1.5D format
        """
        from . import rh15d
        # unit conversion to SI
        ul = self.params['u_l'] / 1.e2  # to metres
        ur = self.params['u_r']         # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']         # to seconds
        uv = ul / ut
        ub = self.params['u_b'] * 1e-4  # to Tesla
        ue = self.params['u_ee']        # to erg/g
        # slicing and unit conversion
        if sx is None:
            sx = [0, self.nx, 1]
        if sy is None:
            sy = [0, self.ny, 1]
        if sz is None:
            sz = [0, self.nzb, 1]
        hion = False
        if 'do_hion' in self.params:
            if self.params['do_hion'] > 0:
                hion = True
        print('Slicing and unit conversion...')
        temp = self.tg[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho = self.r[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho = rho * ur
        Bx = self.bx[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        By = self.by[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        Bz = self.bz[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        # Change sign of Bz (because of height scale) and By (to make
        # right-handed system)
        Bx = Bx * ub
        By = -By * ub
        Bz = -Bz * ub
        vz = self.getvar('uz')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                   sz[0]:sz[1]:sz[2]]
        vz *= -uv
        x = self.x[sx[0]:sx[1]:sx[2]] * ul
        y = self.y[sy[0]:sy[1]:sy[2]] * (-ul)
        z = self.z[sz[0]:sz[1]:sz[2]] * (-ul)
        # convert from rho to H atoms, ideally from subs.dat. Otherwise
        # default.
        if hion:
            print('Getting hion data...')
            ne = self.getvar('hionne')
            # slice and convert from cm^-3 to m^-3
            ne = ne[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
            ne = ne * 1.e6
            # read hydrogen populations (they are saved in cm^-3)
            nh = np.empty((6,) + temp.shape, dtype='Float32')
            for k in range(6):
                nv = self.getvar('n%i' % (k + 1))
                nh[k] = nv[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                           sz[0]:sz[1]:sz[2]]
            nh = nh * 1.e6
        else:
            ee = self.getvar('ee')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                       sz[0]:sz[1]:sz[2]]
            ee = ee * ue
            if os.access('%s/subs.dat' % self.fdir, os.R_OK):
                grph = subs2grph('%s/subs.dat' % self.fdir)
            else:
                grph = 2.380491e-24
            nh = rho / grph * 1.e6       # from rho to nH in m^-3
            # interpolate ne from the EOS table
            print('ne interpolation...')
            eostab = Rhoeetab(fdir=self.fdir)
            ne = eostab.tab_interp(rho, ee, order=1) * \
                1.e6  # from cm^-3 to m^-3
            # old method, using Mats's table
            # ne = ne_rt_table(rho, temp) * 1.e6  # from cm^-3 to m^-3
        # description
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.file_root, repr(sx), repr(sy), repr(sz))
            if hion:
                desc = 'hion ' + desc
        # write to file
        print('Write to file...')
        rh15d.make_hdf5_atmos(outfile, temp, vz, nh, z, ne=ne, x=x, y=y,
                              append=append, Bx=Bx, By=By, Bz=Bz, desc=desc,
                              snap=self.snap)

    def write_multi3d(self, outfile, mesh='mesh.dat', sx=None, sy=None,
                      sz=None, desc=None):
        """
        Writes snapshot in Multi3D format
        """
        from .multi3dn import Multi3dAtmos
        # unit conversion to cgs and km/s
        ul = self.params['u_l']   # to cm
        ur = self.params['u_r']   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t']   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ue = self.params['u_ee']  # to erg/g
        # slicing and unit conversion
        if sx is None:
            sx = [0, self.nx, 1]
        if sy is None:
            sy = [0, self.ny, 1]
        if sz is None:
            sz = [0, self.nzb, 1]
        nh = None
        hion = False
        if 'do_hion' in self.params:
            if self.params['do_hion'] > 0:
                hion = True
        print('Slicing and unit conversion...')
        temp = self.tg[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho = self.r[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        rho = rho * ur
        # Change sign of vz (because of height scale) and vy (to make
        # right-handed system)
        vx = self.getvar('ux')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                   sz[0]:sz[1]:sz[2]]
        vx *= uv
        vy = self.getvar('uy')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                   sz[0]:sz[1]:sz[2]]
        vy *= -uv
        vz = self.getvar('uz')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                   sz[0]:sz[1]:sz[2]]
        vz *= -uv
        x = self.x[sx[0]:sx[1]:sx[2]] * ul
        y = self.y[sy[0]:sy[1]:sy[2]] * ul
        z = self.z[sz[0]:sz[1]:sz[2]] * (-ul)
        # if Hion, get nH and ne directly
        if hion:
            print('Getting hion data...')
            ne = self.getvar('hionne')
            # slice and convert from cm^-3 to m^-3
            ne = ne[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
            ne = ne * 1.e6
            # read hydrogen populations (they are saved in cm^-3)
            nh = np.empty((6,) + temp.shape, dtype='Float32')
            for k in range(6):
                nv = self.getvar('n%i' % (k + 1))
                nh[k] = nv[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                           sz[0]:sz[1]:sz[2]]
            nh = nh * 1.e6
        else:
            ee = self.getvar('ee')[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                       sz[0]:sz[1]:sz[2]]
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



########################################
########################################
###############  OLD   #################
########################################
########################################
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
        self.lnrk = table[:, :, 3]
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
                  'Table rho min=%.3e, requested rho min=%.3e' % (self.params['rhomin'], rhomin))
        if rhomax > self.params['rhomax']:
            print('(WWW) tab_interp: density outside table bounds. ' +
                  'Table rho max=%.1f, requested rho max=%.1f' % (self.params['rhomax'], rhomax))
        if eimin < self.params['eimin']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei min=%.2f, requested Ei min=%.2f' % (self.params['eimin'], eimin))
        if eimax > self.params['eimax']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei max=%.2f, requested Ei max=%.2f' % (self.params['eimax'], eimax))
        # translate to table coordinates
        x = (np.log(ei) - self.lnei[0]) / self.dlnei
        y = (np.log(rho) - self.lnrho[0]) / self.dlnrho
        # interpolate quantity
        result = ndimage.map_coordinates(
            quant, [x, y], order=order, mode='nearest')
        return (np.exp(result) if out != 'tg' else result)


###########
#  TOOLS  #
###########
def read_idl_ascii(filename):
    ''' Reads IDL-formatted (command style) ascii file into dictionary '''
    li = 0
    params = {}
    with open(filename) as docfile:
        # doc = docfile.read()
        # go through the file, add stuff to dictionary
        for line in docfile:
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
                print(('(WWW) read_params: line %i is invalid, continuing' % li))
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
                except:
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
               'Table log(rho) min=%.2f, requested log(rho) min=%.2f' % (tlrmin, lrmin)))
    if lrmax > tlrmax:
        print(('(WWW) ne_rt_table: log density outside of table bounds. ' +
               'Table log(rho) max=%.2f, requested log(rho) max=%.2f' % (tlrmax, lrmax)))

    ## Tiago: this is for the real thing, global fit 2D interpolation:
    ## (commented because it is TREMENDOUSLY SLOW)
    # x = np.repeat(tt['rho_tab'],  tt['theta_tab'].shape[0])
    # y = np.tile(  tt['theta_tab'],  tt['rho_tab'].shape[0])
    # 2D grid interpolation according to method (default: linear interpolation)
    # result = interp.griddata(np.transpose([x,y]), tt['ne_rt_table'].ravel(),
    #                         (lgrho, 5040./temp), method=method)
    #
    # if some values outside of the table, use nearest neighbour
    # if np.any(np.isnan(result)):
    #    idx = np.isnan(result)
    #    near = interp.griddata(np.transpose([x,y]), tt['ne_rt_table'].ravel(),
    #                            (lgrho, 5040./temp), method='nearest')
    #    result[idx] = near[idx]
    ## Tiago: this is the approximate thing (bilinear/cubic interpolation) with
    ## ndimage
    y = (5040. / temp - tt['theta_tab'][0]) / \
        (tt['theta_tab'][1] - tt['theta_tab'][0])
    x = (lgrho - tt['rho_tab'][0]) / (tt['rho_tab'][1] - tt['rho_tab'][0])
    result = ndimage.map_coordinates(
        tt['ne_rt_table'], [x, y], order=order, mode='nearest')
    return 10**result * rho / tt['grph']
