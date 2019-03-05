"""
Set of programs and tools to read the outputs from RH, 1.5D version
"""
import os
import warnings
import datetime
import numpy as np
import xarray as xr
import h5py
from io import StringIO
from astropy import units


class Rh15dout:
    """
    Class to load and manipulate output from RH 1.5D.
    """
    def __init__(self, fdir='.', verbose=True, autoread=True):
        self.files = []
        self.params = {}
        self.verbose = verbose
        self.fdir = fdir
        if autoread:
            for outfile in ["output_aux", "output_indata"]:
                OUTFILE = os.path.join(self.fdir, "%s.hdf5" % (outfile))
                self.read_groups(OUTFILE)
            RAYFILE = os.path.join(self.fdir, "output_ray.hdf5")
            self.read_ray(RAYFILE)

    def read_groups(self, infile):
        ''' Reads indata file, group by group. '''
        if not os.path.isfile(infile):   # See if netCDF file exists
            infile = os.path.splitext(infile)[0] + '.ncdf'
        if not os.path.isfile(infile):
            return
        f = h5py.File(infile, "r")
        GROUPS = [g for g in f.keys() if type(f[g]) == h5py._hl.group.Group]
        f.close()
        for g in GROUPS:
            setattr(self, g, xr.open_dataset(infile, group=g, autoclose=True))
            self.files.append(getattr(self, g))
        if self.verbose:
            print(('--- Read %s file.' % infile))

    def read_ray(self, infile=None):
        ''' Reads ray file. '''
        if infile is None:
            infile = '%s/output_ray.hdf5' % self.fdir
            if not os.path.isfile(infile):  # See if netCDF file exists
                infile = os.path.splitext(infile)[0] + '.ncdf'
        if not os.path.isfile(infile):
            return
        self.ray = xr.open_dataset(infile, autoclose=True)
        self.files.append(self.ray)
        if self.verbose:
            print(('--- Read %s file.' % infile))

    def close(self):
        ''' Closes the open files '''
        for f in self.files:
            f.close()

    def __del__(self):
        self.close()


class HDF5Atmos:
    """
    Class to load and manipulate RH 1.5D input atmosphere files in HDF5.
    """
    def __init__(self, infile):
        self.file = read_hdf5(self, infile)
        self.closed = False

    def close(self):
        try:
            self.file.close()
            self.closed = True
        except RuntimeError:
            print('(WWW) HDF5Atmos: input file already closed.')

    def read(self, infile):
        if not self.closed:
            self.close()
        self.file = read_hdf5(self, infile)

    def write_multi(self, outfile, xi, yi, nti=0, writeB=False,
                    write_dscale=False, zcut=0, depth_optimise=False):
        '''
        Writes MULTI atmosphere file from a column of the 3D model,
        in RH 1.5D HDF5 format. Also writes the binary XDR file with magnetic
        fields, if writeB is true.
        '''
        from .multi import watmos_multi
        from .rh import write_B
        writeB = writeB and self.params['has_B']
        # if only total H available, will have to use rhpy (which is sometimes
        # risky...)
        if self.params['nhydr'] == 1:
            try:
                import rhpy
            except ImportError:
                raise ValueError("This function depents on rhpy, which is not"
                                 " installed in this system.")
            nh = rhpy.nh_lte(self.temperature[nti, xi, yi, zcut:].astype('Float64'),
                             self.electron_density[
                                   nti, xi, yi, zcut:].astype('Float64'),
                             self.hydrogen_populations[
                                   nti, 0, xi, yi, zcut:].astype('Float64'))
        elif self.params['nhydr'] == 6:
            nh = self.hydrogen_populations[nti, :, xi, yi, zcut:]
        else:
            raise ValueError("(EEE) write_multi: found %i hydrogen levels."
                             " For multi, need 6 or 1 " % self.params['nhydr'])
        M_TO_CM3 = (units.m**-3).to('1 / cm3')
        M_TO_KM = units.m.to('km')
        temp = self.temperature[nti, xi, yi, zcut:].copy()
        ne = self.electron_density[nti, xi, yi, zcut:].copy() / M_TO_CM3
        if len(self.z.shape) > 2:
            self.z = self.z[:, xi, yi]
        z = self.z[nti, zcut:].copy() * M_TO_KM * 1.e5    # in cm
        vz = self.velocity_z[nti, xi, yi, zcut:].copy() * M_TO_KM
        nh = nh / M_TO_CM3
        if writeB:
            bx = self.B_x[nti, xi, yi, zcut:].copy()
            by = self.B_y[nti, xi, yi, zcut:].copy()
            bz = self.B_z[nti, xi, yi, zcut:].copy()
        else:
            bx = by = bz = None
        if depth_optimise:
            rho = self.hydrogen_populations[
                nti, 0, xi, yi, zcut:] * 2.380491e-24 / M_TO_CM3
            res = depth_optim(z, temp, ne, vz, rho, nh=nh, bx=bx, by=by, bz=bz)
            z, temp, ne, vz, rho, nh = res[:6]
            if writeB:
                bx, by, bz = res[6:]
        watmos_multi(outfile, temp, ne, z * 1e-5, vz=vz, nh=nh,
                     write_dscale=write_dscale,
                     id='%s txy-slice: (t,x,y) = (%i,%i,%i)' %
                     (self.params['description'], nti, xi, yi))
        if writeB:
            write_B('%s.B' % outfile, bx, by, bz)
            print(('--- Wrote magnetic field to %s.B' % outfile))

    def write_multi_3d(self, outfile, nti=0, sx=None, sy=None, sz=None,
                       big_endian=False):
        ''' Writes atmosphere in multi_3d format (the same as the
            pre-Jorrit multi3d) '''
        from . import multi
        ul = units.m.to('cm')
        uv = (units.m / units.s).to('km / s')
        # slicing and unit conversion
        if sx is None:
            sx = [0, self.nx, 1]
        if sy is None:
            sy = [0, self.ny, 1]
        if sz is None:
            sz = [0, self.nz, 1]
        if self.params['nhydr'] > 1:
            nh = np.mean(self.hydrogen_populations[nti, :, sx[0]:sx[1]:sx[2],
                                                   sy[0]:sy[1]:sy[2],
                                                   sz[0]:sz[1]:sz[2]], axis=1) / (ul**3)
        else:
            nh = self.hydrogen_populations[nti, 0, sx[0]:sx[1]:sx[2],
                                           sy[0]:sy[1]:sy[2],
                                           sz[0]:sz[1]:sz[2]] / (ul**3)
        rho = nh * 2.380491e-24  # nH to rho [g cm-3]
        x = self.x[sx[0]:sx[1]:sx[2]] * ul
        y = self.y[sy[0]:sy[1]:sy[2]] * ul
        z = self.z[nti, sz[0]:sz[1]:sz[2]] * ul
        ne = self.electron_density[nti, sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                   sz[0]:sz[1]:sz[2]] / (ul**3)
        temp = self.temperature[nti, sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                                sz[0]:sz[1]:sz[2]]
        vz = self.velocity_z[nti, sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                             sz[0]:sz[1]:sz[2]] * uv
        # write to file
        multi.write_atmos3d(outfile, x, y, z, ne, temp, vz, rho=rho,
                            big_endian=big_endian)


class DataHolder:
    def __init__(self):
        pass


class AtomFile:
    """
    Class to hold data from an RH or MULTI atom file.

    Parameters
    ----------
    filename: str
        String with atom file name.
    format: str
        Can be 'RH' (default) or 'MULTI'.
    """
    def __init__(self, filename, format='RH'):
        self.read_atom(filename, format)

    @staticmethod
    def read_atom_levels(data, format='RH'):
        """
        Reads levels part of atom file
        """
        tmp = []
        dtype=[('energy', 'f8'), ('g_factor', 'f8'),('label', '|U30'),
               ('stage', 'i4'), ('level_no','i4')]
        if format.upper() == "RH":
            extra_cols = 2
        elif format.upper() == "MULTI":
            extra_cols = 1
            dtype = dtype[:-1]
        else:
            raise ValueError("Format must be RH or MULTI")
        for line in data:
            buf = line.split("'")
            assert len(buf) == 3
            tmp.append(tuple(buf[0].split() +
                        [buf[1].strip()] + buf[2].split()[:extra_cols]))
        return np.array(tmp, dtype=dtype)

    def read_atom(self, filename, format='RH'):
        self.format = format.upper()
        data = []
        counter = 0
        with open(filename, 'r') as atom_file:
            for line in atom_file:
                tmp = line.strip()
                # clean up comments and blank lines
                if not tmp:
                    continue
                if tmp[0] in ['#', '*']:
                    continue
                data.append(tmp)
        self.element = data[counter]
        counter += 1
        if self.format == 'RH':
            self.units = {'level_energies': units.Unit('J m / cm'),
                          'line_wavelength': units.Unit('nm'),
                          'line_stark': units.Unit('m'),
                          'continua_photoionisation': units.Unit('m2'),
                          'continua_wavelength': units.Unit('nm'),
                          'collision_cross_sections': units.Unit('m3')}
        elif self.format == 'MULTI':
            self.units = {'level_energies': units.Unit('J m / cm'),
                          'line_wavelength': units.Unit('Angstrom'),
                          'line_stark': units.Unit('cm'),
                          'continua_photoionisation': units.Unit('cm2'),
                          'continua_wavelength': units.Unit('Angstrom'),
                          'collision_cross_sections': units.Unit('cm3')}
            self.abund = data[counter].split()[0]
            self.atomic_weight = data[counter].split()[1]
            counter += 1
        else:
            raise ValueError("Unsupported atom format " + format)
        nlevel, nline, ncont, nfixed = np.array(data[counter].split(), dtype='i')
        self.nlevel = nlevel
        self.nline = nline
        self.ncont = ncont
        self.nfixed = nfixed
        counter += 1
        # read levels
        self.levels = self.read_atom_levels(data[counter:counter + nlevel],
                                             self.format)
        counter += nlevel
        # read lines
        tmp = StringIO('\n'.join(data[counter:counter + nline]))
        if self.format == "RH":
            data_type = [('level_start', 'i4'), ('level_end', 'i4'),
                         ('f_value', 'f8'), ('type', 'U10'), ('nlambda', 'i'),
                         ('symmetric', 'U10'), ('qcore', 'f8'), ('qwing', 'f8'),
                         ('vdApprox', 'U10'), ('vdWaals', 'f8', (4,)),
                         ('radiative_broadening', 'f8'),
                         ('stark_broadening', 'f8')]
        elif self.format == "MULTI":
            data_type = [('level_start', 'i4'), ('level_end', 'i4'),
                         ('f_value', 'f8'), ('nlambda', 'i'),
                         ('qwing', 'f8'), ('qcore', 'f8'), ('iw', 'i4'),
                         ('radiative_broadening', 'f8'),
                         ('vdWaals', 'f8', (1,)), ('stark_broadening', 'f8'),
                         ('type', 'U10')]
        self.lines = np.genfromtxt(tmp, dtype=data_type)
        counter += nline
        # read continua
        self.continua = []
        for _ in range(ncont):
            line = data[counter].split()
            counter += 1
            result = {}
            result['level_start'] = int(line[0])
            result['level_end'] = int(line[1])
            result['edge_cross_section'] = float(line[2])
            result['nlambda'] = int(line[3])
            if self.format == "RH":
                result['wavelength_dependence'] = line[4].upper()
                result['wave_min'] = float(line[5])
            elif self.format == "MULTI":
                if float(line[4]) > 0:
                    result['wavelength_dependence'] = "HYDROGENIC"
                else:
                    result['wavelength_dependence'] = "EXPLICIT"
            if result['wavelength_dependence'] == 'EXPLICIT':
                tmp = '\n'.join(data[counter:counter + result['nlambda']])
                counter += result['nlambda']
                result['cross_section'] = np.genfromtxt(StringIO(tmp))
            self.continua.append(result)
        # read fixed transitions
        self.fixed_transitions = []
        for _ in range(nfixed):
            line = data[counter].split()
            counter += 1
            result = {}
            result['level_start'] = int(line[0])
            result['level_end'] = int(line[1])
            result['strength'] = float(line[2])
            result['trad'] = float(line[3])
            result['trad_option'] = line[4]
            self.fixed_transitions.append(result)
        # read collisions
        ### IN MULTI FORMAT COLLISIONS START WITH GENCOL
        ### Also in MULTI, must merge together lines that are written in
        ### free format (ie, not prefixed by OMEGA, CE, etc...)
        self.collision_temperatures = []
        self.collision_tables = []
        # Keys for rates given as function of temperature
        COLLISION_KEYS_TEMP = ['OHMEGA', 'OMEGA', 'CE', 'CI', 'CP', 'CH',
                               'CH0', 'CH+', 'CR', 'TEMP']
        # Keys for rates written as single line
        COLLISION_KEYS_LINE = ['AR85-CEA', 'AR85-CHP', 'AR85-CHH', 'SHULL82',
                               'BURGESS', 'SUMMERS']
        COLLISION_KEYS_OTHER = ['AR85-CDI', 'BADNELL']
        ALL_KEYS = (COLLISION_KEYS_TEMP + COLLISION_KEYS_LINE +
                        COLLISION_KEYS_OTHER)
        SINGLE_KEYS = ['GENCOL', 'END']

        if self.format == 'MULTI':   # merge lines in free FORMAT
            collision_data = []
            while counter < len(data):
                line = data[counter]
                key = data[counter].split()[0].upper().strip()
                if key in ALL_KEYS:
                    tmp = line
                    while True:
                        counter += 1
                        key = data[counter].split()[0].upper().strip()
                        if key in ALL_KEYS + SINGLE_KEYS:
                            collision_data.append(tmp)
                            break
                        else:
                            tmp += '  '  + data[counter]
                elif key in SINGLE_KEYS:
                    collision_data.append(line)
                    counter += 1
        else:
            collision_data = data[counter:]

        unread_lines = False
        counter = 0
        while counter < len(collision_data) - 1:
            line = collision_data[counter].split()
            key = line[0].upper()
            result = {}
            if key == 'END':
                break
            elif key == 'TEMP':
                temp_tmp = np.array(line[2:]).astype('f')
                self.collision_temperatures.append(temp_tmp)
            # Collision rates given as function of temperature
            elif key in COLLISION_KEYS_TEMP:
                assert self.collision_temperatures, ('No temperature block'
                         ' found before %s table' % (key))
                ntemp = len(self.collision_temperatures[-1])
                result = {'type': key, 'level_start': int(line[1]),
                          'level_end': int(line[2]),
                          'temp_index': len(self.collision_temperatures) - 1,
                          'data': np.array(line[3:3 + ntemp]).astype('d')}  # this will not work in MULTI
                assert len(result['data']) == len(temp_tmp), ('Inconsistent '
                    'number of points between temperature and collision table')
            elif key in COLLISION_KEYS_LINE:
                if key == "SUMMERS":
                    result = {'type': key, 'data': float(line[1])}
                else:
                    result = {'type': key, 'level_start': int(line[1]),
                              'level_end': int(line[2]),
                              'data': np.array(line[2:]).astype('f')}
            elif key in ["AR85-CDI", "BADNELL"]:
                assert len(line) >= 4, '%s must have >3 elements' % key
                result = {'type': key, 'level_start': int(line[1]),
                              'level_end': int(line[2])}
                if key == "BADNELL":
                    rows = 2
                else:
                    rows = int(line[3])
                if self.format == 'MULTI':  # All values in one line
                    tmp = np.array(line[4:]).astype('d')
                    assert tmp.shape[0] % rows == 0, ('Inconsistent number of'
                                                 ' data points for %s' % key)
                    result['data'] = tmp.reshape((rows, tmp.shape[0] // rows))
                    counter += 1
                else:  # For RH, values written in matrix form
                    tmp = data[counter + 1: counter + 1 + rows]
                    result['data'] = np.array([l.split() for l in tmp]).astype('d')
                    counter += rows
            elif key == "GENCOL":
                pass
            else:
                unread_lines = True

            if result:
                self.collision_tables.append(result)
            counter += 1

        if unread_lines:
            warnings.warn("Some lines in collision section were not understood",
                          RuntimeWarning)


def read_hdf5(inclass, infile):
    """
    Reads HDF5/netCDF4 file into inclass, instance of any class.
    Variables are read into class attributes, dimensions and attributes
    are read into params dictionary.
    """
    if not os.path.isfile(infile):
        raise IOError('read_hdf5: File %s not found' % infile)
    f = h5py.File(infile, mode='r')
    if 'params' not in dir(inclass):
        inclass.params = {}
    # add attributes
    attrs = [a for a in f.attrs]
    for att in f.attrs:
        try:
            inclass.params[att] = f.attrs[att]
        except OSError:  # catch errors where h5py cannot read UTF-8 strings
            pass
    # add variables and groups
    for element in f:
        name = element.replace(' ', '_')    # sanitise string for spaces
        if type(f[element]) == h5py._hl.dataset.Dataset:
            setattr(inclass, name, f[element])
            # special case for netCDF dimensions, add them to param list
            if 'NAME' in f[element].attrs:
                if f[element].attrs['NAME'][:20] == b'This is a netCDF dim':
                    inclass.params[element] = f[element].shape[0]
        if type(f[element]) == h5py._hl.group.Group:
            setattr(inclass, name, DataHolder())
            cur_class = getattr(inclass, name)
            cur_class.params = {}
            for variable in f[element]:   # add group variables
                vname = variable.replace(' ', '_')
                setattr(cur_class, vname, f[element][variable])
            for att in f[element].attrs:  # add group attributes
                cur_class.params[att] = f[element].attrs[att]
    return f


def make_xarray_atmos(outfile, T, vz, z, nH=None, x=None, y=None, Bz=None, By=None,
                      Bx=None, rho=None, ne=None, vx=None, vy=None, vturb=None,
                      desc=None, snap=None, boundary=None, append=False):
    """
    Creates HDF5 input file for RH 1.5D using xarray.

    Parameters
    ----------
    outfile : string
        Name of destination. If file exists it will be wiped.
    T : n-D array
        Temperature in K. Its shape will determine the output
        dimensions. Shape is generally (nt, nx, ny, nz), but any
        dimensions except nz can be omitted. Therefore the array can
        be 1D, 2D, or 3D, 4D but ultimately will always be saved as 4D.
    vz : n-D array
        Line of sight velocity in m/s. Same shape as T.
    z : n-D array
        Height in m. Can have same shape as T (different height scale
        for each column) or be only 1D (same height for all columns).
    nH : n-D array, optional
        Hydrogen populations in m^-3. Shape is (nt, nhydr, nx, ny, nz),
        where nt, nx, ny can be omitted but must be consistent with
        the shape of T. nhydr can be 1 (total number of protons) or
        more (level populations). If nH is not given, rho must be given!
    ne : n-D array, optional
        Electron density in m^-3. Same shape as T.
    rho : n-D array, optional
        Density in kg m^-3. Same shape as T. Only used if nH is not given.
    vx : n-D array, optional
        x velocity in m/s. Same shape as T. Not in use by RH 1.5D.
    vy : n-D array, optional
        y velocity in m/s. Same shape as T. Not in use by RH 1.5D.
    vturb : n-D array, optional
        Turbulent velocity (Microturbulence) in km/s. Not usually needed
        for MHD models, and should only be used when a depth dependent
        microturbulence is needed (constant microturbulence can be added
        in RH).
    Bx : n-D array, optional
        Magnetic field in x dimension, in Tesla. Same shape as T.
    By : n-D array, optional
        Magnetic field in y dimension, in Tesla. Same shape as T.
    Bz : n-D array, optional
        Magnetic field in z dimension, in Tesla. Same shape as T.
    x : 1-D array, optional
        Grid distances in m. Same shape as first index of T.
    y : 1-D array, optional
        Grid distances in m. Same shape as second index of T.
    x : 1-D array, optional
        Grid distances in m. Same shape as first index of T.
    snap : array-like, optional
        Snapshot number(s).
    desc : string, optional
        Description of file
    boundary : Tuple, optional
        Tuple with [bottom, top] boundary conditions. Options are:
        0: Zero, 1: Thermalised, 2: Reflective.
    append : boolean, optional
        If True, will append to existing file (if any).
    """
    data = {'temperature': [T, 'K'],
            'velocity_z': [vz, 'm / s'],
            'velocity_y': [vy, 'm / s'],
            'velocity_x': [vx, 'm / s'],
            'electron_density': [ne, '1 / m3'],
            'hydrogen_populations': [nH, '1 / m3'],
            'density': [rho, 'kg / m3'],
            'B_x': [Bx, 'T'],
            'B_y': [By, 'T'],
            'B_z': [Bz, 'T'],
            'velocity_turbulent': [vturb, 'm / s'],
            'x': [x, 'm'],
            'y': [y, 'm'],
            'z': [z, 'm']}
    VARS4D = ['temperature', 'B_x', 'B_y', 'B_z', 'density', 'velocity_x',
              'velocity_y', 'velocity_z', 'velocity_turbulent', 'density',
              'electron_density']
    # Remove variables not given
    data = {key: data[key] for key in data if data[key][0] is not None}
    if (nH is None) and (rho is None):
        raise ValueError("Missing nH or rho. Need at least one of them")
    if (append and not os.path.isfile(outfile)):
        append = False
    idx = [None] * (4 - len(T.shape)) + [Ellipsis]  # empty axes for 1D/2D/3D
    for var in data:
        if var not in ['x', 'y']:  # these are always 1D
            data[var][0] = data[var][0][idx]
    if len(data['temperature'][0].shape) != 4:
        raise ValueError('Invalid shape for T')
    nt, nx, ny, nz = data['temperature'][0].shape
    if boundary is None:
        boundary = [1, 0]
    if snap is None:
        data['snapshot_number'] = [np.arange(nt, dtype='i4'), '']
    else:
        data['snapshot_number'] = [np.array([snap], dtype='i4'), '']
    if not append:
        variables = {}
        coordinates = {}
        for v in data:
            if v in VARS4D:
                variables[v] = (('snapshot_number', 'x', 'y', 'depth'),
                                data[v][0], {'units': data[v][1]})
            elif v == 'hydrogen_populations':
                variables[v] = (('snapshot_number', 'nhydr', 'x', 'y', 'depth'),
                                data[v][0], {'units': data[v][1]})
            elif v == 'z':
                dims = ('snapshot_number', 'depth')
                if len(data[v][0].shape) == 1:  # extra dim for nt dependency
                    data[v][0] = data[v][0][None, :]
                elif len(data[v][0].shape) == 4:
                    dims = ('snapshot_number', 'x', 'y', 'depth')
                coordinates[v] = (dims, data[v][0], {'units': data[v][1]})
            elif v in ['x', 'y', 'snapshot_number']:
                coordinates[v] = ((v), data[v][0], {'units': data[v][1]})

        attrs = {"comment": ("Created with make_xarray_atmos "
                             "on %s" % datetime.datetime.now()),
                 "boundary_top": boundary[1], "boundary_bottom": boundary[0],
                 "has_B": int(Bz is not None), "description": str(desc),
                 "nx": nx, "ny": ny, "nz": nz, "nt": nt}
        data = xr.Dataset(variables, coordinates, attrs)
        data.to_netcdf(outfile, mode='w', format='NETCDF4',
                       unlimited_dims=('snapshot_number'))
    else:  # use h5py to append existing file
        rootgrp = h5py.File(outfile, mode='a')
        nti = int(rootgrp.attrs['nt'])
        #rootgrp.attrs['nt'] = nti + nt  # add appended number of snapshots
        for var in data:
            if var in VARS4D + ['hydrogen_populations', 'z', 'snapshot_number']:
                rootgrp[var].resize(nti + nt, axis=0)
                rootgrp[var][nti:nti + nt] = data[var][0][:]
        rootgrp.close()


def depth_optim(height, temp, ne, vz, rho, nh=None, bx=None, by=None, bz=None,
                tmax=5e4):
    """
    Performs depth optimisation of one single column (as per multi_3d).

        IN:
            height   [cm]
            temp     [K]
            ne       [cm-3]
            vz       [any]
            rho      [g cm-3]
            nh       [any] (optional)
            bx,by,bz [any] (optional)
            tmax     [K] maximum temperature of the first point

    """
    from scipy.integrate import cumtrapz
    import scipy.interpolate as interp
    import astropy.constants as const
    ndep = len(height)
    # calculate optical depth from H-bf only
    taumax = 100
    grph = 2.26e-24   # grams per hydrogen atom
    crhmbf = 2.9256e-17
    ee = constants.e.si.value * 1e7
    bk = constants.k_B.cgs.value
    xhbf = 1.03526e-16 * ne * crhmbf / temp**1.5 * \
        np.exp(0.754 * ee / bk / temp) * rho / grph
    tau = np.concatenate(([0.], cumtrapz(xhbf, -height)))
    idx = (tau < taumax) & (temp < tmax)
    # find maximum variance of T, rho, and tau for each depth
    tt = temp[idx]
    rr = rho[idx]
    ta = tau[idx]
    tdiv = np.abs(np.log10(tt[1:]) - np.log10(tt[:-1])) / np.log10(1.1)
    rdiv = np.abs(np.log10(rr[1:]) - np.log10(rr[:-1])) / np.log10(1.1)
    taudiv = np.abs(np.log10(ta[1:]) - np.log10(ta[:-1])) / 0.1
    taudiv[0] = 0.
    aind = np.concatenate(
        ([0.], np.cumsum(np.max(np.array([tdiv, rdiv, taudiv]), axis=0))))
    aind *= (ndep - 1) / aind[-1]
    # interpolate new height so it is constant in aind2
    nheight = interp.splev(np.arange(ndep), interp.splrep(
        aind, height[idx], k=3, s=0), der=0)
    # interpolate quantities for new depth scale
    ntemp = np.exp(interp.splev(nheight, interp.splrep(height[::-1], np.log(temp[::-1]),
                                                       k=3, s=0), der=0))
    nne = np.exp(interp.splev(nheight, interp.splrep(height[::-1], np.log(ne[::-1]),
                                                     k=3, s=0), der=0))
    nrho = np.exp(interp.splev(nheight, interp.splrep(height[::-1], np.log(rho[::-1]),
                                                      k=3, s=0), der=0))
    nvz = interp.splev(nheight, interp.splrep(height[::-1], vz[::-1],
                                              k=3, s=0), der=0)
    result = [nheight, ntemp, nne, nvz, nrho]
    if nh is not None:
        for k in range(nh.shape[0]):
            nh[k] = np.exp(interp.splev(nheight,
                                        interp.splrep(height[::-1],
                                                      np.log(nh[k, ::-1]), k=3,
                                                      s=0), der=0))
        result += [nh]
    if bx is not None:
        nbx = interp.splev(nheight, interp.splrep(
            height[::-1], bx[::-1], k=3, s=0), der=0)
        nby = interp.splev(nheight, interp.splrep(
            height[::-1], by[::-1], k=3, s=0), der=0)
        nbz = interp.splev(nheight, interp.splrep(
            height[::-1], bz[::-1], k=3, s=0), der=0)
        result += [nbx, nby, nbz]
    return result


def make_wave_file(outfile, start=None, end=None, step=None, new_wave=None,
                   ewave=None, air=True):
    """
    Writes RH wave file (in xdr format). All wavelengths should be in nm.

    Parameters
    ----------
    start: number
        Starting wavelength.
    end: number
        Ending wavelength (non-inclusive)
    step: number
        Wavelength separation
    new_wave: 1D array
        Alternatively to start/end, one can specify an array of
        wavelengths here.
    outfile: string
        Name of file to write.
    ewave: 1-D array, optional
        Array of existing wavelengths. Program will make discard points
        to make sure no step is enforced using these points too.
    air: boolean, optional
        If true, will at the end convert the wavelengths into vacuum
        wavelengths.
    """
    import xdrlib
    from specutils.utils.wcs_utils import air_to_vac
    if new_wave is None:
        new_wave = np.arange(start, end, step)
        if None in [start, end, step]:
            raise ValueError('Must specify either new_wave, or start, end, '
                             'step. Stopping.')
    if step is None:
        step = np.median(np.diff(new_wave))
    if ewave is not None:  # ensure step is kept at most times
        keepers = []
        for w in new_wave:
            if np.min(np.abs(w - ewave)) > step * 0.375:
                keepers.append(w)
        new_wave = np.array(keepers)
    if air:
        # RH uses Edlen (1966) to convert from vacuum to air
        new_wave = air_to_vac(new_wave * units.nm, method='edlen1966',
                              scheme='iteration').value

    # write file
    p = xdrlib.Packer()
    nw = len(new_wave)
    p.pack_int(nw)
    p.pack_farray(nw, new_wave.astype('d'), p.pack_double)
    f = open(outfile, 'wb')
    f.write(p.get_buffer())
    f.close()
    print(("Wrote %i wavelengths to file." % nw))


def read_wave_file(infile):
    """
    Reads RH wavelength file.

    Parameters
    ----------
    infile : str
        Name of wavelength file to read.

    Returns
    -------
    wave : array
        Wavelength from file.
    """
    import xdrlib
    import io
    from .rh import read_xdr_var
    f = io.open(infile, 'rb')
    buf = xdrlib.Unpacker(f.read())
    f.close()
    nw = read_xdr_var(buf, 'i')
    return read_xdr_var(buf, ('d', (nw,)))


def clean_var(data, only_positive=True):
    """
    Cleans a 2D or 3D variable filled with NaNs and other irregularities.
    """
    from ..utils import utilsfast
    data = np.ma.masked_invalid(data, copy=False)
    if only_positive:
        data = np.ma.masked_less(data, 0., copy=False)
    tmp = np.abs(data)
    thres = tmp.mean() + tmp.std() * 4  # points more than 4 std away
    data = np.ma.masked_where(tmp > thres, data, copy=False)
    if data.ndim == 2:
        data = data[..., np.newaxis]
    for k in range(data.shape[-1]):
        tmp = data[..., k].astype("d")
        tmp[data[..., k].mask] = np.nan
        data[..., k] = utilsfast.replace_nans(tmp, 15, 0.1, 3, "localmean")
    return np.squeeze(data)
