"""
Set of programs and tools to read the outputs from RH, 1.5D version
"""
import os
import re
import warnings
import datetime
import numpy as np
import xarray as xr
import h5py
import netCDF4
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
        GROUPS = netCDF4.Dataset(infile).groups.keys()
        for g in GROUPS:
            setattr(self, g, xr.open_dataset(infile, group=g, lock=None))
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
        self.ray = xr.open_dataset(infile, lock=None)
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
        self.filename = filename
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
            ncol = 15
            data_type = [('level_start', 'i4'), ('level_end', 'i4'),
                         ('f_value', 'f8'), ('type', 'U10'), ('nlambda', 'i'),
                         ('symmetric', 'U10'), ('qcore', 'f8'), ('qwing', 'f8'),
                         ('vdApprox', 'U10'), ('vdWaals', 'f8', (4,)),
                         ('radiative_broadening', 'f8'),
                         ('stark_broadening', 'f8')]
        elif self.format == "MULTI":
            ncol = 11
            data_type = [('level_start', 'i4'), ('level_end', 'i4'),
                         ('f_value', 'f8'), ('nlambda', 'i'),
                         ('qwing', 'f8'), ('qcore', 'f8'), ('iw', 'i4'),
                         ('radiative_broadening', 'f8'),
                         ('vdWaals', 'f8', (1,)), ('stark_broadening', 'f8'),
                         ('type', 'U10')]
        self.lines = np.loadtxt(tmp, dtype=data_type, ndmin=1, usecols=range(ncol))
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
        self.COLLISION_KEYS_TEMP = ['OHMEGA', 'OMEGA', 'CE', 'CI', 'CP', 'CH',
                               'CH0', 'CH+', 'CR', 'TEMP']
        # Keys for rates written as single line
        self.COLLISION_KEYS_LINE = ['AR85-CEA', 'AR85-CHP', 'AR85-CHH', 'SHULL82',
                               'BURGESS', 'SUMMERS']
        self.COLLISION_KEYS_OTHER = ['AR85-CDI', 'BADNELL']
        self.ALL_KEYS = (self.COLLISION_KEYS_TEMP + self.COLLISION_KEYS_LINE +
                        self.COLLISION_KEYS_OTHER)
        SINGLE_KEYS = ['GENCOL', 'END']

        if self.format == 'MULTI':   # merge lines in free FORMAT
            collision_data = []
            while counter < len(data):
                line = data[counter]
                key = data[counter].split()[0].upper().strip()
                if key in self.ALL_KEYS:
                    tmp = line
                    while True:
                        counter += 1
                        key = data[counter].split()[0].upper().strip()
                        if key in self.ALL_KEYS + SINGLE_KEYS:
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
        while counter < len(collision_data):
            line = collision_data[counter].split()
            key = line[0].upper()
            result = {}
            if key == 'END':
                break
            elif key == 'TEMP':
                temp_tmp = np.array(line[2:]).astype('float64')
                self.collision_temperatures.append(temp_tmp)
            # Collision rates given as function of temperature
            elif key in self.COLLISION_KEYS_TEMP:
                assert self.collision_temperatures, ('No temperature block'
                         ' found before %s table' % (key))
                ntemp = len(self.collision_temperatures[-1])
                result = {'type': key, 'level_start': int(line[1]),
                          'level_end': int(line[2]),
                          'temp_index': len(self.collision_temperatures) - 1,
                          'data': np.array(line[3:3 + ntemp]).astype('d')}  # this will not work in MULTI
                assert len(result['data']) == len(temp_tmp), ('Inconsistent '
                    'number of points between temperature and collision table')
            elif key in self.COLLISION_KEYS_LINE:
                if key == "SUMMERS":
                    result = {'type': key, 'data': float(line[1])}
                else:
                    result = {'type': key, 'level_start': int(line[1]),
                              'level_end': int(line[2]),
                              'data': np.array(line[3:]).astype('float64')}
            elif key in ["AR85-CDI", "BADNELL"]:
                assert len(line) >= 4, '%s must have >3 elements' % key
                result = {'type': key, 'level_start': int(line[1]),
                              'level_end': int(line[2])}
                if key == "BADNELL":
                    rows = 2
                else:
                    rows = int(line[3])
                if self.format == 'MULTI':  # All values in one line
                    tmp = np.array(line[4:]).astype('float64')
                    assert tmp.shape[0] % rows == 0, ('Inconsistent number of'
                                                 ' data points for %s' % key)
                    result['data'] = tmp.reshape((rows, tmp.shape[0] // rows))
                    counter += 1
                else:  # For RH, values written in matrix form
                    tmp = collision_data[counter + 1: counter + 1 + rows]
                    result['data'] = np.array([l.split() for l in tmp]).astype('float64')
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
    
    def write_yaml(self, output_filename):
        """
        Writes the content of the atom file object
        to "output_filename" in yaml format.
        """
        try:
            import yaml
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Need yaml module to write YAML files")

        if self.nfixed != 0:
            raise NotImplementedError(output_filename 
             + " not written: Writing of fixed transitions to YAML is not implemented.")
        UNITS = dict()
        # Levels:
        UNITS['energy'] = 'cm^-1'
        # Lines:
        UNITS['vmicro_char'] = 'km / s'
        UNITS['natural_broadening'] = 's^-1'
        UNITS['vdW_broadening'] = {'ABO_σ':'a_0^2'}
        UNITS['vdW_broadening']['ABO_α'] = 'm/m'
        UNITS['vdW_broadening']['RR_α'] = {'h':'1.0e-8*cm^3/s', 'he':'1.0e-9*cm^3/s'}
        # Collisions:
        UNITS['coll_temp'] = 'K'
        UNITS['coll_data'] = {'Omega':'m/m'}
        COLL_GROUP_1 = ['CE', 'CI']
        COLL_GROUP_2 = ['CP', 'CH', 'CH0', 'CH+', 'CR']
        if self.format == 'RH':
            # Continua (EXPLICIT):
            UNITS['radiative_bound_free'] = {'cross_section':['nm', 'm^2']}
            # Continua (HYDROGENIC):
            UNITS['radiative_bound_free']['σ_peak'] = 'm^2'
            UNITS['radiative_bound_free']['λ_min'] = 'nm'
            # Collisions:
            UNITS['coll_data']['group_1'] = 's^-1 * K^-1/2 * m^3'
            UNITS['coll_data']['group_2'] = 's^-1 * m^3'
        elif self.format == 'MULTI':
            # Continua (EXPLICIT):
            UNITS['radiative_bound_free'] = {'cross_section':['Å', 'cm^2']}
            # Continua (HYDROGENIC):
            UNITS['radiative_bound_free']['σ_peak'] = 'cm^2'
            UNITS['radiative_bound_free']['λ_min'] = 'Å' 
            # Collisions:
            UNITS['coll_data']['group_1'] = 's^-1 * K^-1/2 * cm^3'
            UNITS['coll_data']['group_2'] = 's^-1 * cm^3'

        ###
        ### Sorting and renaming things to make writing to file easier: 
        ###

        def __dict_to_yaml(dict):
            if type(dict) in [float, int, str]:
                return f"{dict}\n"
            else:
                return yaml.dump(dict,
                                 default_flow_style=True, sort_keys=False,
                                 allow_unicode=True, width=200)

        # LEVELS:
        levels_dict = dict()
        def __level_str(level_i, type=self.format):
            if (type == 'RH'):
                level_i += 1
            num = str(level_i)
            return 'lev'+ num

        for i in range(self.nlevel):
            values = self.levels[i]
            if self.format=='RH':
                num = values[4] + 1
            else:
                num = i + 1
            key = 'lev'+str(num)
            energy_dict = {'value': float(values[0]), 'unit': str(UNITS['energy'])}
            g = values[1]
            label = values[2]
            if self.format =='RH':
                stage = values[3] + 1
            else:
                stage = values[3]
            levels_dict[key] = {'energy': energy_dict, 'g': int(g), 
                                'stage': int(stage), 'label': str(label)}

        # LINES:
        lines_list = []
        if self.format == 'RH':
            ityp = 3
            igr = 10
            ibs = 11
            idWt = 8
            ivdW = 9
            inw = 4
            iqw = 7
            iqc = 6
        elif self.format == 'MULTI':
            ityp = 10
            igr = 7 
            ibs = 9
            idWt = None
            ivdW = 8
            inw = 3
            iqw = 4
            iqc = 5
        for i in range(self.nline):
            line_dict = dict()
            line_i = self.lines[i]
            line_dict['transition'] = [__level_str(line_i[0]), 
                                       __level_str(line_i[1])]
            line_dict['f_value'] = float(line_i[2])
            if line_i[ityp] == 'PRD':
                line_dict['type_profile'] = 'PRD'
            else:
                line_dict['type_profile'] = (line_i[ityp][0] 
                                                + line_i[ityp][1:].lower())
            line_dict['γ_rad'] = {'value': float(line_i[igr]), 
                                  'unit': UNITS['natural_broadening']}
            line_dict['broadening_stark'] = {'coefficient': float(line_i[ibs])}
            if self.format=='RH':
                vdWtype = line_i[idWt]
                vdWval = line_i[ivdW]
                if vdWtype == 'UNSOLD':
                    line_dict['broadening_vanderwaals'] = {
                        'type': 'Unsold', 
                        'h_coefficient': float(vdWval[0]), 
                        'he_coefficient': float(vdWval[2])}
                elif vdWtype == 'PARAMTR':
                    line_dict['broadening_vanderwaals'] = {
                        'type': 'Ridder_Rensbergen', 
                        'h':{'α': {'value': float(vdWval[0]), 
                                   'unit': UNITS['vdW_broadening']['RR_α']['h']},
                             'β': vdWval[1]},
                        'he': {'α': {'value': float(vdWval[2]), 
                                     'unit': UNITS['vdW_broadening']['RR_α']['he']},
                               'β': vdWval[3]}
                    }
                elif vdWtype == 'BARKLEM':
                    line_dict['broadening_vanderwaals'] = [{
                        'type': 'ABO', 
                        'σ': {'value': float(vdWval[0]), 
                              'unit': UNITS['vdW_broadening']['ABO_σ']}, 
                        'α': {'value': float(vdWval[1]), 
                              'unit': UNITS['vdW_broadening']['ABO_α']}
                    }]
                    line_dict['broadening_vanderwaals'] += [{
                        'type': 'Unsold',
                        'he_coefficient': vdWval[2]
                    }]
                else:
                    raise NotImplementedError(
                            'vdWtype not recognized: %s'%vdWtype)
            else:
                coeff = line_i[ivdW][0]
                if coeff >= 20:
                    # Recipe from MULTI
                    sigma = int(coeff) * 2.80028E-21
                    alpha = coeff - int(coeff)
                    line_dict['broadening_vanderwaals'] = [
                        {'type': 'ABO', 
                         'σ': {'value': float(sigma), 
                               'unit': str(UNITS['vdW_broadening']['ABO_σ'])}, 
                         'α': {'value': float(alpha), 
                               'unit': str(UNITS['vdW_broadening']['ABO_α'])}}]
                else:
                    line_dict['broadening_vanderwaals'] = {
                            'type': 'Unsold', 
                            'h_coefficient': coeff, 
                            'he_coefficient': 0.0}
            line_dict['wavelengths'] = {'type': self.format, 
                                        'nλ': int(line_i[inw]), 
                                        'qwing': float(line_i[iqw]), 
                                        'qcore': float(line_i[iqc]), 
                                        'vmicro_char': {
                                            'value': 8.0,
                                            'unit': UNITS['vmicro_char']}
            }
            if self.format=='RH': 
                line_dict['wavelengths']['vmicro_char']['value'] = 2.5 
                line_dict['wavelengths']['asymmetric'] = ((line_i[5])=='ASYMM')
            lines_list += [line_dict]

        # CONTINUA:
        # self.continua directly used in writing to file later
        for continuum in self.continua:
            assert (continuum['wavelength_dependence'] in ['EXPLICIT', 'HYDROGENIC']), (
                'Wavelength dependence type not understood: %s' %continuum)['wavelength_dependence']

        # COLLISIONS:
        # self.collision_tables sorted by transition here:
        nc = len(self.collision_tables)
        collisions_list = []
        transitions_list = []
        summers = None
        for i in range(nc):
            collision = self.collision_tables[i]

            transition_data = dict() # Store the collision table here
            if collision['type'] == 'SUMMERS':
                summers = collision['data']
                continue
            else:
                collision['data'] = collision['data'].tolist()
            # Store the name in lowercase or uppercase:
            if collision['type'] in ['OMEGA','OHMEGA', 'OHM']: # MULTI
                transition_data['type'] = 'Omega'
            elif collision['type'] in ['TEMP', 'BURGESS', 
                                       'BADNELL', 'SHULL82']:
                transition_data['type'] = (
                    collision['type'][0] + collision['type'][1:].lower())
            else:
                transition_data['type'] = collision['type']
            # The tables with temperature dependence:
            if collision['type'] in self.COLLISION_KEYS_TEMP:
                # Store temperature:
                transition_data['temperature'] = {
                    'value': self.collision_temperatures[collision['temp_index']], 
                    'unit': UNITS['coll_temp']} 
                # Store data:
                transition_data['data'] = {'value':collision['data']} 
                # Store data unit:
                if transition_data['type'] == 'Omega':
                    transition_data['data']['unit'] = (
                        UNITS['coll_data']['Omega'])
                elif transition_data['type'] in COLL_GROUP_1:
                    transition_data['data']['unit'] = (
                        UNITS['coll_data']['group_1'])
                elif transition_data['type'] in COLL_GROUP_2:
                    transition_data['data']['unit'] = (
                        UNITS['coll_data']['group_2'])
            # Store these in dictionaries:
            elif collision['type'] in self.COLLISION_KEYS_LINE:
                if collision['type'] == 'SHULL82':
                    transition_data['data'] = { 
                                'a_col': collision['data'][0], 
                                't_col': collision['data'][1], 
                                'a_rad': collision['data'][2], 
                                'x_rad': collision['data'][3], 
                                'a_di': collision['data'][4], 
                                'b_di': collision['data'][5], 
                                't0': collision['data'][6], 
                                't1': collision['data'][7]
                    }
                elif collision['type'] in ['AR85-CEA', 'BURGESS']:
                    transition_data['data'] = {
                        'coefficient': collision['data'][0]}
                elif collision['type'] in ['AR85-CHP', 'AR85-CHH']:
                    transition_data['data'] = { 
                                't1': collision['data'][0], 
                                't2': collision['data'][1], 
                                'a': collision['data'][2], 
                                'b': collision['data'][3], 
                                'c': collision['data'][4], 
                                'd': collision['data'][5], 
                    }
                else: 
                    transition_data['data'] = collision['data']
            # Store these in nested lists/arrays -- as they are:
            elif collision['type'] in self.COLLISION_KEYS_OTHER:
                transition_data['data'] = collision['data']
            else:
                raise NotImplementedError(
                    f"Collision data type not understood! type: {collision['type']}")
            # Place the data in correct transition:
            tr = [__level_str(collision['level_start']), 
                  __level_str(collision['level_end'])]
            if tr in transitions_list:
                # append data to that transition
                idx = transitions_list.index(tr)
                collisions_list[idx]['data'] += [transition_data]
            else: 
                # Make new transition entry
                transitions_list += [tr]
                coll_dict = {'transition':tr, 'data':[transition_data]}
                collisions_list += [coll_dict]

        ###
        ### Write to file:
        ###

        output_file = open(output_filename, 'w')
        output_file.write("%YAML 1.1\n---\n")
        output_file.write(
            (f"# Automatically converted to YAML from {self.filename.split('/')[-1]}"
            " using helita.sim.rh15d.AtomFile.\n\n")
        )
        tab2 = 2 * ' '
        tab4 = 4 * ' '
        tab6 = 6 * ' '
        tab8 = 8 * ' '

        # Header:
        element = self.element[0].upper()
        if len(self.element) > 1: 
            if ' ' in self.element: # Removes roman numerals if separated
                words = self.element.split() 
                element += words[0][1:].lower()
            else:
                element += self.element[1:].lower()

        output_file.write('element:\n')
        output_file.write(tab2 + 'symbol: %s'%(element) +'\n')

        # Atomic levels:
        output_file.write('\natomic_levels:\n')
        for lev, value in levels_dict.items():
            output_file.write(tab2 + f"{lev}: {__dict_to_yaml(value)}")
        # Radiative bound-bound:
        output_file.write('\nradiative_bound_bound:\n')
        if self.nline == 0:
            output_file.write(tab2 + '[]\n')
        for line in lines_list:
            up, lo = line['transition']
            output_file.write(f"{tab2}- transition: [{up}, {lo}]\n")
            del line['transition']
            for key, val in line.items():
                output_file.write(f"{tab4}{key}: {__dict_to_yaml(val)}")

        # Radiative bound-free:
        output_file.write('\nradiative_bound_free:\n')
        if self.ncont == 0:
            output_file.write(tab2 + '[]\n')
        for continuum in self.continua:
            up = __level_str(continuum['level_start'])
            lo = __level_str(continuum['level_end'])
            output_file.write(f"{tab2}- transition: [{up}, {lo}]\n")

            if continuum['wavelength_dependence'] == 'EXPLICIT':
                output_file.write(tab4 + 'cross_section: \n')
                u_wave, u_sigma = UNITS['radiative_bound_free']['cross_section']
                output_file.write(f"{tab6}unit: [{u_wave}, {u_sigma}]\n{tab6}value: \n")
                for val in continuum['cross_section'].tolist():
                    output_file.write(f"{tab8}- {__dict_to_yaml(val)}")          
                
            elif continuum['wavelength_dependence'] == 'HYDROGENIC':
                output_file.write(tab4 + 'cross_section_hydrogenic: \n')
                val = {'value': float(continuum['edge_cross_section']), 
                        'unit': str(UNITS['radiative_bound_free']['σ_peak'])} 
                output_file.write(f"{tab6}σ_peak: {__dict_to_yaml(val)}")
                val = {'value': float(continuum['wave_min']), 
                        'unit': str(UNITS['radiative_bound_free']['λ_min'])}
                output_file.write(f"{tab6}λ_min: {__dict_to_yaml(val)}")
                output_file.write(f"{tab6}nλ: {continuum['nlambda']}\n")

        # Collisional:
        output_file.write('\ncollisional:\n')
        if len(collisions_list) == 0:
            output_file.write(tab2 + '[]\n')
        for collisions in collisions_list:
            up, lo = collisions['transition']
            output_file.write(f"{tab2}- transition: [{up}, {lo}]\n")
            output_file.write(tab4 + 'data: \n')
            for data in collisions['data']:
                output_file.write(tab6 + '- type: ' + str(data['type']) +'\n')
                # If summers:
                if (data['type'] in ['Badnell', 'Shull82']) and (summers != None):
                    output_file.write(
                        tab8 + 'scaling_summers: '+ str(summers) +'\n')
                # If temperature:
                if 'temperature' in data.keys():
                    output_file.write(tab8 + 'temperature: \n')
                    output_file.write(
                        tab8 + tab2 + 'unit: ' + data['temperature']['unit'] +'\n')
                    output_file.write((f"{tab8}  value: "
                                       f"{data['temperature']['value'].tolist()}\n"))

                ## Data:
                output_file.write(tab8 + 'data: \n')

                # Temp dependent, dict with newlines, nested list:
                if data['type'].upper() in self.COLLISION_KEYS_TEMP:
                    if 'unit' in data['data'].keys():
                        output_file.write(
                            tab8 + tab2 + 'unit: ' + data['data']['unit'] +'\n')
                    output_file.write((f"{tab8}  value: "
                                       f"{data['data']['value']}\n"))

                # Single line versions - dictionaries on one line
                elif data['type'].upper() in self.COLLISION_KEYS_LINE:
                    output_file.write(f"{tab8}  {__dict_to_yaml(data['data'])}")
                
                # Nested lists
                else: # self.COLLISION_KEYS_OTHER
                    assert data['type'].upper() in self.COLLISION_KEYS_OTHER, (
                                    'Data type not in ALL_KEYS? %s'%data['type'])
                    for elem in data['data']:
                        output_file.write(f"{tab8}  - {elem}\n")
        
        output_file.close()
                


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
