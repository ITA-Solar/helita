"""
Set of programs and tools to read the outputs from RH (Han's version)
"""
import os
import sys
import io
import xdrlib
import numpy as np


class Rhout:
    """
    Reads outputs from RH.

    Currently the reading the following output files is supported:
     - input.out
     - geometry.out
     - atmos.out
     - spectrum.out (no Stokes)
     - spectrum_XX (no Stokes, from solveray)
     - brs.out
     - J.dat
     - opacity.out (no Stokes)

    These output files are NOT supported:
      - Atom (atom, collrate, damping, pops, radrate)
      - Flux
      - metals
      - molecule

    Parameters
    ----------
    fdir : str, optional
        Directory with output files.
    verbose : str, optional
        If True, will print more details.

    Notes
    -----
    In general,  the way to read all the XDR files should be:

     Modify read_xdr_file so that it returns only xdata.
     Then, on each read_xxx, read the necessary header variables,
     rewind (xdata.set_position(0)), then read the variables in order.
     This allows the flexibility of derived datatypes, and appending to dictionary
     (e.g. as in readatmos for all the elements and etc.). It also allows one to
     read directly into attribute of the class (with setattr(self,'aa',<data>))
    """
    def __init__(self, fdir='.', verbose=True):
        ''' Reads all the output data from a RH run.'''
        self.verbose = verbose
        self.fdir = fdir
        self.read_input('{0}/input.out'.format(fdir))
        self.read_geometry('{0}/geometry.out'.format(fdir))
        self.read_atmosphere('{0}/atmos.out'.format(fdir))
        self.read_spectrum('{0}/spectrum.out'.format(fdir))
        if os.path.isfile('{0}/spectrum_1.00'.format(fdir)):
            self.read_ray('{0}/spectrum_1.00'.format(fdir))

    def read_input(self, infile='input.out'):
        ''' Reads RH input.out file. '''
        data = read_xdr_file(infile)
        self.input = {}
        input_vars = [('magneto_optical', 'i'), ('PRD_angle_dep', 'i'),
                      ('XRD', 'i'), ('start_solution', 'i'),
                      ('stokes_mode', 'i'), ('metallicity', 'd'),
                      ('backgr_pol', 'i'), ('big_endian', 'i')]
        for v in input_vars:
            self.input[v[0]] = read_xdr_var(data, v[1:])
        close_xdr(data, infile, verbose=self.verbose)

    def read_geometry(self, infile='geometry.out'):
        ''' Reads RH geometry.out file. '''
        data = read_xdr_file(infile)
        self.geometry = {}
        geom_type = ['ONE_D_PLANE', 'TWO_D_PLANE',
                     'SPHERICAL_SYMMETRIC', 'THREE_D_PLANE']
        type = read_xdr_var(data, ('i',))
        if type not in list(range(4)):
            raise ValueError('read_geometry: invalid geometry type {0} in {1}'.
                             format(type, infile))
        nrays = read_xdr_var(data, ('i',))
        self.nrays = nrays
        self.geometry_type = geom_type[type]
        # read some parameters and define structure to be read
        if self.geometry_type == 'ONE_D_PLANE':
            ndep = read_xdr_var(data, ('i',))
            self.ndep = ndep
            geom_vars = [('xmu', 'd', (nrays,)), ('wmu', 'd', (nrays,)),
                         ('height', 'd', (ndep,)), ('cmass', 'd', (ndep,)),
                         ('tau500', 'd', (ndep,)), ('vz', 'd', (ndep,))]
        elif self.geometry_type == 'TWO_D_PLANE':
            nx = read_xdr_var(data, ('i',))
            nz = read_xdr_var(data, ('i',))
            self.nx = nx
            self.nz = nz
            geom_vars = [('angleSet', 'i'), ('xmu', 'd', (nrays,)),
                         ('ymu', 'd', (nrays,)), ('wmu', 'd', (nrays,)),
                         ('x', 'd', (nx,)), ('z', 'd', (nz,)),
                         ('vx', 'd', (nx, nz)), ('vz', 'd', (nx, nz))]
        elif self.geometry_type == 'THREE_D_PLANE':
            nx = read_xdr_var(data, ('i',))
            ny = read_xdr_var(data, ('i',))
            nz = read_xdr_var(data, ('i',))
            self.nx = nx
            self.ny = ny
            self.nz = nz
            geom_vars = [('angleSet', 'i'), ('xmu', 'd', (nrays,)),
                         ('ymu', 'd', (nrays,)), ('wmu', 'd', (nrays,)),
                         ('dx', 'd'), ('dy', 'd'),
                         ('z', 'd', (nz,)), ('vx', 'd', (nx, ny, nz)),
                         ('vy', 'd', (nx, ny, nz)), ('vz', 'd', (nx, ny, nz))]
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            nradius = read_xdr_var(data, ('i',))
            ncore = read_xdr_var(data, ('i',))
            self.nradius = nradius
            self.ncore = ncore
            geom_vars = [('radius', 'd'), ('xmu', 'd', (nrays,)),
                         ('wmu', 'd', (nrays,)), ('r', 'd', (nradius,)),
                         ('cmass', 'd', (nradius,)), ('tau500', 'd', (nradius,)),
                         ('vr', 'd', (nradius,))]
        # read data
        for v in geom_vars:
            self.geometry[v[0]] = read_xdr_var(data, v[1:])
        close_xdr(data, infile, verbose=self.verbose)

    def read_atmosphere(self, infile='atmos.out'):
        ''' Reads RH atmos.out file '''
        if not hasattr(self, 'geometry'):
            em = ('read_atmosphere: geometry data not loaded, '
                  'call read_geometry() first!')
            raise ValueError(em)
        data = read_xdr_file(infile)
        self.atmos = {}
        nhydr = read_xdr_var(data, ('i',))
        nelem = read_xdr_var(data, ('i',))
        self.atmos['nhydr'] = nhydr
        self.atmos['nelem'] = nelem
        # read some parameters and define structure to be read
        if self.geometry_type == 'ONE_D_PLANE':
            ndep = self.ndep
            atmos_vars = [('moving', 'i'), ('T', 'd', (ndep,)),
                          ('n_elec', 'd', (ndep,)), ('vturb', 'd', (ndep,)),
                          ('nh', 'd', (ndep, nhydr)), ('id', 's')]
        elif self.geometry_type == 'TWO_D_PLANE':
            nx, nz = self.nx, self.nz
            atmos_vars = [('moving', 'i'), ('T', 'd', (nx, nz)),
                          ('n_elec', 'd', (nx, nz)), ('vturb', 'd', (nx, nz)),
                          ('nh', 'd', (nx, nz, nhydr)), ('id', 's')]
        elif self.geometry_type == 'THREE_D_PLANE':
            nx, ny, nz = self.nx, self.ny, self.nz
            atmos_vars = [('moving', 'i'), ('T', 'd', (nx, ny, nz)),
                          ('n_elec', 'd', (nx, ny, nz)
                           ), ('vturb', 'd', (nx, ny, nz)),
                          ('nh', 'd', (nx, ny, nz, nhydr)), ('id', 's')]
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            nradius = self.nradius
            atmos_vars = [('moving', 'i'), ('T', 'd', (nradius,)),
                          ('n_elec', 'd', (nradius,)), ('vturb', 'd', (nradius,)),
                          ('nh', 'd', (nradius, nhydr)), ('id', 's')]
        # read data
        for v in atmos_vars:
            self.atmos[v[0]] = read_xdr_var(data, v[1:])
        # read elements into nested dictionaries
        self.elements = {}
        for v in range(nelem):
            el = read_xdr_var(data, ('s',)).strip()
            weight = read_xdr_var(data, ('d',))
            abund = read_xdr_var(data, ('d',))
            self.elements[el] = {'weight': weight, 'abund': abund}
        # read stokes data, if present
        self.stokes = False
        if self.geometry_type != 'SPHERICAL_SYMMETRIC':
            try:
                stokes = read_xdr_var(data, ('i',))
            except EOFError or IOError:
                if self.verbose:
                    print('(WWW) read_atmos: no Stokes data in atmos.out,'
                          ' skipping.')
                return
            self.stokes = True
            ss = self.atmos['T'].shape
            stokes_vars = [('B', 'd', ss), ('gamma_B', 'd', ss),
                           ('chi_B', 'd', ss)]
            for v in stokes_vars:
                self.atmos[v[0]] = read_xdr_var(data, v[1:])
        close_xdr(data, infile, verbose=self.verbose)

    def read_spectrum(self, infile='spectrum.out'):
        ''' Reads RH spectrum.out file '''
        if not hasattr(self, 'geometry'):
            em = ('read_spectrum: geometry data not loaded, '
                  'call read_geometry() first!')
            raise ValueError(em)
        if not hasattr(self, 'atmos'):
            em = ('read_spectrum: atmos data not loaded, '
                  'call read_atmos() first!')
            raise ValueError(em)
        data = read_xdr_file(infile)
        profs = {}
        self.spec = {}
        nspect = read_xdr_var(data, ('i',))
        self.spec['nspect'] = nspect
        nrays = self.nrays
        self.wave = read_xdr_var(data, ('d', (nspect,)))
        if self.geometry_type == 'ONE_D_PLANE':
            ishape = (nrays, nspect)
        elif self.geometry_type == 'TWO_D_PLANE':
            ishape = (self.nx, nrays, nspect)
        elif self.geometry_type == 'THREE_D_PLANE':
            ishape = (self.nx, self.ny, nrays, nspect)
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            ishape = (nrays, nspect)
        self.imu = read_xdr_var(data, ('d', ishape))
        self.spec['vacuum_to_air'] = read_xdr_var(data, ('i',))
        self.spec['air_limit'] = read_xdr_var(data, ('d',))
        if self.stokes:
            self.stokes_Q = read_xdr_var(data, ('d', ishape))
            self.stokes_U = read_xdr_var(data, ('d', ishape))
            self.stokes_V = read_xdr_var(data, ('d', ishape))
        close_xdr(data, infile, verbose=self.verbose)
        # read as_rn, if it exists
        if os.path.isfile('asrs.out'):
            data = read_xdr_file('asrs.out')
            if self.atmos['moving'] or self.stokes or self.input['PRD_angle_dep']:
                self.spec['as_rn'] = read_xdr_var(data, ('i', (nrays, nspect)))
            else:
                self.spec['as_rn'] = read_xdr_var(data, ('i', (nspect,)))
            close_xdr(data, 'asrs.out', verbose=self.verbose)

    def read_ray(self, infile='spectrum_1.00'):
        ''' Reads spectra for single ray files (e.g. mu=1). '''
        if not hasattr(self, 'geometry'):
            em = ('read_spectrum: geometry data not loaded,'
                  ' call read_geometry() first!')
            raise ValueError(em)
        if not hasattr(self, 'spec'):
            em = ('read_spectrum: spectral data not loaded, '
                  'call read_spectrum() first!')
            raise ValueError(em)
        data = read_xdr_file(infile)
        nspect = self.spec['nspect']
        self.ray = {}
        if self.geometry_type == 'ONE_D_PLANE':
            self.muz = read_xdr_var(data, ('d',))
            ishape = (nspect,)
            sshape = (self.ndep,)
        elif self.geometry_type == 'TWO_D_PLANE':
            self.mux = read_xdr_var(data, ('d',))
            self.muz = read_xdr_var(data, ('d',))
            ishape = (self.nx, nspect)
            sshape = (self.nx, self.nz)
        elif self.geometry_type == 'THREE_D_PLANE':
            self.mux = read_xdr_var(data, ('d',))
            self.muy = read_xdr_var(data, ('d',))
            ishape = (self.nx, self.ny, nspect)
            sshape = (self.nx, self.ny, self.nz)
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            self.muz = read_xdr_var(data, ('d',))
            ishape = (nspect,)
            sshape = (self.nradius,)
        # read intensity
        self.int = read_xdr_var(data, ('d', ishape))
        # read absorption and source function if written
        ns = read_xdr_var(data, ('i',))
        if ns > 0:
            nshape = (ns,) + sshape
            self.ray['chi'] = np.zeros(nshape, dtype='d')
            self.ray['S'] = np.zeros(nshape, dtype='d')
            self.ray['wave_idx'] = np.zeros(ns, dtype='l')
            for i in range(ns):
                self.ray['wave_idx'][i] = read_xdr_var(data, ('i',))
                self.ray['chi'][i] = read_xdr_var(data, ('d', sshape))
                self.ray['S'][i] = read_xdr_var(data, ('d', sshape))
        if self.stokes:
            self.ray_stokes_Q = read_xdr_var(data, ('d', ishape))
            self.ray_stokes_U = read_xdr_var(data, ('d', ishape))
            self.ray_stokes_V = read_xdr_var(data, ('d', ishape))
        close_xdr(data, infile, verbose=self.verbose)

    def read_brs(self, infile='brs.out'):
        ''' Reads the file with the background opacity record settings,
            in the old (xdr) format. '''
        if not hasattr(self, 'geometry'):
            em = ('read_brs: geometry data not loaded, call read_geometry()'
                  ' first!')
            raise ValueError(em)
        if not hasattr(self, 'spec'):
            em = ('read_brs: spectrum data not loaded, call read_spectrum()'
                  ' first!')
            raise ValueError(em)
        data = read_xdr_file(infile)
        atmosID = read_xdr_var(data, ('s',)).strip()
        nspace = read_xdr_var(data, ('i',))
        nspect = read_xdr_var(data, ('i',))
        if nspect != self.spec['nspect']:
            em = ('(EEE) read_brs: nspect in file different from atmos. '
                  'Aborting.')
            raise ValueError(em)
        self.brs = {}
        if self.atmos['moving'] or self.stokes:
            ishape = (2, self.nrays, nspect)
        else:
            ishape = (nspect,)
        self.brs['hasline'] = read_xdr_var(
            data, ('i', (nspect,))).astype('Bool')
        self.brs['ispolarized'] = read_xdr_var(
            data, ('i', (nspect,))).astype('Bool')
        self.brs['backgrrecno'] = read_xdr_var(data, ('i', ishape))
        close_xdr(data, infile, verbose=self.verbose)

    def read_j(self, infile='J.dat'):
        ''' Reads the mean radiation field, for all wavelengths. '''
        if not hasattr(self, 'geometry'):
            em = 'read_j: geometry data not loaded, call read_geometry() first!'
            raise ValueError(em)
        if not hasattr(self, 'spec'):
            em = 'read_j: spectrum data not loaded, call read_spec() first!'
            raise ValueError(em)
        data_file = open(infile, 'r')
        nspect = self.spec['nspect']
        if self.geometry_type == 'ONE_D_PLANE':
            rec_len = self.ndep * 8
            ishape = (nspect, self.ndep)
        elif self.geometry_type == 'TWO_D_PLANE':
            rec_len = (self.nx * self.nz) * 8
            ishape = (nspect, self.nx, self.nz)
        elif self.geometry_type == 'THREE_D_PLANE':
            rec_len = (self.nx * self.ny * self.nz) * 8
            ishape = (nspect, self.nx, self.ny, self.nz)
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            rec_len = self.nradius * 8
            ishape = (nspect, self.nradius)
        self.J = np.zeros(ishape)
        for i in range(nspect):
            # point background file to position and read
            data_file.seek(i * rec_len)
            self.J[i] = read_file_var(data_file, ('d', ishape[1:]))
        data_file.close()

    def read_opacity(self, infile_line='opacity.out', infile_bg='background.dat',
                     imu=0):
        ''' Reads RH atmos.out file '''
        if not hasattr(self, 'geometry'):
            em = ('read_opacity: geometry data not loaded,'
                  ' call read_geometry() first!')
            raise ValueError(em)
        if not hasattr(self, 'spec'):
            em = ('read_opacity: spectrum data not loaded,'
                  ' call read_spec() first!')
            raise ValueError(em)
        if not hasattr(self.atmos, 'brs'):
            self.read_brs()
        data_line = read_xdr_file(infile_line)
        file_bg = open(infile_bg, 'r')
        nspect = self.spec['nspect']
        if self.geometry_type == 'ONE_D_PLANE':
            as_rec_len = 2 * self.ndep * 8
            bg_rec_len = self.ndep * 8
            ishape = (nspect, self.ndep)
        elif self.geometry_type == 'TWO_D_PLANE':
            as_rec_len = 2 * (self.nx * self.nz) * 8
            bg_rec_len = (self.nx * self.nz) * 8
            ishape = (nspect, self.nx, self.nz)
        elif self.geometry_type == 'THREE_D_PLANE':
            as_rec_len = 2 * (self.nx * self.ny * self.nz) * 8
            bg_rec_len = (self.nx * self.ny * self.nz) * 8
            ishape = (nspect, self.nx, self.ny, self.nz)
        elif self.geometry_type == 'SPHERICAL_SYMMETRIC':
            as_rec_len = 2 * self.nradius * 8
            bg_rec_len = self.nradius * 8
            ishape = (nspect, self.nradius)
        # create arrays
        chi_as = np.zeros(ishape)
        eta_as = np.zeros(ishape)
        chi_c = np.zeros(ishape)
        eta_c = np.zeros(ishape)
        scatt = np.zeros(ishape)
        # NOTE: this will not work when a line is polarised.
        #       For those cases these arrays must be read per wavelength, and will
        #       have different sizes for different wavelengths.
        if np.sum(self.brs['ispolarized']):
            em = ('read_opacity: Polarized line(s) detected, cannot continue'
                  ' with opacity extraction')
            raise ValueError(em)
        # get record numbers
        if self.atmos['moving'] or self.stokes or self.input['PRD_angle_dep']:
            as_index = self.spec['as_rn'][imu] * as_rec_len
            bg_index = self.brs['backgrrecno'][1, imu] * bg_rec_len
        else:
            as_index = self.spec['as_rn'] * as_rec_len
            bg_index = self.brs['backgrrecno'] * bg_rec_len
        # Read arrays
        for i in range(nspect):
            if as_index[i] >= 0:  # avoid non-active set lines
                # point xdr buffer to position and read
                data_line.set_position(as_index[i])
                chi_as[i] = read_xdr_var(data_line, ('d', ishape[1:]))
                eta_as[i] = read_xdr_var(data_line, ('d', ishape[1:]))
            # point background file to position and read
            file_bg.seek(bg_index[i])
            chi_c[i] = read_file_var(file_bg, ('d', ishape[1:]))
            eta_c[i] = read_file_var(file_bg, ('d', ishape[1:]))
            scatt[i] = read_file_var(file_bg, ('d', ishape[1:]))
        self.chi_as = chi_as
        self.eta_as = eta_as
        self.chi_c = chi_c
        self.eta_c = eta_c
        self.scatt = scatt
        close_xdr(data_line, infile_line, verbose=False)
        file_bg.close()

    def get_contrib_imu(self, imu, type='total', op_file='opacity.out',
                        bg_file='background.dat', j_file='J.dat'):
        ''' Calculates the contribution function for intensity, for a
            particular ray, defined by imu.

            type can be: \'total\', \'line, or \'continuum\'

            The units of self.contribi are J m^-2 s^-1 Hz^-1 sr^-1 km^-1

            NOTE: This only calculates the contribution function for
                  the quadrature rays (ie, often not for disk-centre)
                  For rays calculated with solve ray, one must use
                  get_contrib_ray

        '''
        type = type.lower()
        if not hasattr(self, 'geometry'):
            em = ('get_contrib_imu: geometry data not loaded,'
                  ' call read_geometry() first!')
            raise ValueError(em)
        if not hasattr(self, 'spec'):
            em = ('get_contrib_imu: spectrum data not loaded,'
                  ' call read_spec() first!')
            raise ValueError(em)
        self.read_opacity(infile_line=op_file, infile_bg=bg_file, imu=imu)
        self.read_j(infile=j_file)
        mu = self.geometry['xmu'][imu]
        # Calculate optical depth
        ab = (self.chi_c + self.chi_as)
        self.tau = get_tau(self.geometry['height'], mu, ab)
        # Calculate source function
        if type == 'total':
            self.S = (self.eta_as + self.eta_c + self.J * self.scatt) / ab
        elif type == 'line':
            self.S = self.eta_as / ab
        elif type == 'continuum':
            self.S = (self.eta_c + self.J * self.scatt) / ab
        else:
            raise ValueError('get_contrib_imu: invalid type!')
        # Calculate contribution function
        self.contribi = get_contrib(
            self.geometry['height'], mu, self.tau, self.S)
        return

    def get_contrib_ray(self, inray='ray.input', rayfile='spectrum_1.00'):
        ''' Calculates the contribution function for intensity, for a
            particular ray

            The units of self.contrib are J m^-2 s^-1 Hz^-1 sr^-1 km^-1
        '''
        inray = self.fdir + '/' + inray
        rayfile = self.fdir + '/' + rayfile
        if not hasattr(self, 'ray'):
            self.read_ray(infile=rayfile)
        if 'wave_idx' not in list(self.ray.keys()):
            em = ('get_contrib_ray: no chi/source function written to '
                  'ray file, aborting.')
            raise ValueError(em)
        # read mu from ray.input file
        mu = np.loadtxt(inray, dtype='f')[0]
        if not (0 <= mu <= 1.):
            em = 'get_contrib_ray: invalid mu read: %f' % mu
            raise ValueError(em)
        idx = self.ray['wave_idx']
        # Calculate optical depth
        self.tau = get_tau(self.geometry['height'], mu, self.ray['chi'])
        # Calculate contribution function
        self.contrib = get_contrib(self.geometry['height'], mu, self.tau,
                                   self.ray['S'])
        return


class RhAtmos:
    """
    Reads input atmosphere from RH. Currently only 2D format supported.

    Parameters
    ----------
    format : str, optional
        Atmosphere format. Currently only '2D' (default) supported.
    filename : str, optional
        File to read.
    verbose : str, optional
        If True, will print more details.
    """
    def __init__(self, format="2D", filename=None, verbose=True):
        ''' Reads RH input atmospheres. '''
        self.verbose = verbose
        if format.lower() == "2d":
            if filename is not None:
                self.read_atmos2d(filename)
        else:
            raise NotImplementedError("Format %s not yet supported" % format)

    def read_atmos2d(self, filename):
        """
        Reads input 2D atmosphere
        """
        data = read_xdr_file(filename)
        self.nx = read_xdr_var(data, ('i',))
        self.nz = read_xdr_var(data, ('i',))
        self.nhydr = read_xdr_var(data, ('i',))
        self.hboundary = read_xdr_var(data, ('i',))
        self.bvalue = read_xdr_var(data, ('i', (2, )))
        nx, nz, nhydr = self.nx, self.nz, self.nhydr
        atmos_vars = [('dx', 'd', (nx,)), ('z', 'd', (nz,)),
                      ('T', 'd', (nx, nz)), ('ne', 'd', (nx, nz)),
                      ('vturb', 'd', (nx, nz)), ('vx', 'd', (nx, nz)),
                      ('vz', 'd', (nx, nz)), ('nh', 'd', (nx, nz, nhydr))
                      ]
        for v in atmos_vars:
            setattr(self, v[0], read_xdr_var(data, v[1:]))

    def write_atmos2d(self, filename, dx, z, T, ne, vturb, vx, vz, nh,
                      hboundary, bvalue):
        nx, nz = T.shape
        nhydr = nh.shape[-1]
        assert T.shape == ne.shape
        assert ne.shape == vturb.shape
        assert vturb.shape == nh.shape[:-1]
        assert dx.shape[0] == nx
        assert z.shape[0] == nz
        # Pack as double
        p = xdrlib.Packer()
        p.pack_int(nx)
        p.pack_int(nz)
        p.pack_int(nhydr)
        p.pack_int(hboundary)
        p.pack_int(bvalue[0])
        p.pack_int(bvalue[1])
        p.pack_farray(nx, dx.ravel().astype('d'), p.pack_double)
        p.pack_farray(nz, z.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz, T.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz, ne.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz, vturb.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz, vx.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz, vz.ravel().astype('d'), p.pack_double)
        p.pack_farray(nx * nz * nhydr, nh.T.ravel().astype('d'), p.pack_double)
        # Write to file
        f = open(filename, 'wb')
        f.write(p.get_buffer())
        f.close()


#############################################################################
# TOOLS
#############################################################################
class EmptyData:
    def __init__(self):
        pass


def read_xdr_file(filename):  # ,var,cl=None,verbose=False):
    """
    Reads data from XDR file.

    Because of the way xdrlib works, this reads the whole file to
    memory at once. Avoid with very large files.

    Parameters
    ----------
    filename : string
        File to read.

    Returns
    -------
    result   : xdrlib.Unpacker object
    """
    try:
        f = io.open(filename, 'rb')
        data = f.read()
        f.close()
    except IOError as e:
        raise IOError(
            'read_xdr_file: problem reading {0}: {1}'.format(filename, e))
    # return XDR data
    return xdrlib.Unpacker(data)


def close_xdr(buf, ofile='', verbose=False):
    """
    Closes the xdrlib.Unpacker object, gives warning if not all data read.

    Parameters
    ----------
    buf : xdrlib.Unpacker object
        data object.
    ofile : string, optional
        Original file from which data was read.
    verbose : bool, optional
        Whether to print warning or not.
    """
    try:
        buf.done()
    except:  # .done() will raise error if data remaining
        if verbose:
            print(('(WWW) close_xdr: {0} not all data read!'.format(ofile)))


def read_xdr_var(buf, var):
    """
    Reads a single variable/array from a xdrlib.Unpack buffer.

    Parameters
    ----------

    buf:  xdrlib.Unpack object
        Data buffer.
    var: tuple with (type[,shape]), where type is 'f', 'd', 'i', 'ui',
             or 's'. Shape is optional, and if true is shape of array.
        Type and shape of variable to read

    Returns
    -------
    out :  int/float or array
        Resulting variable.
    """
    assert len(var) > 0
    if var[0] not in ['f', 'd', 'i', 'ui', 's']:
        raise ValueError('read_xdr_var: data type'
                         ' {0} not currently supported'.format(var[0]))
    fdict = {'f': buf.unpack_float,
             'd': buf.unpack_double,
             'i': buf.unpack_int,
             'ui': buf.unpack_uint,
             's': buf.unpack_string}
    func = fdict[var[0]]
    # Single or array?
    if len(var) == 1:
        # this is because RH seems to write the size of the string twice
        if var[0] == 's':
            buf.unpack_int()
        out = func()
    else:
        nitems = np.prod(var[1])
        out = np.array(buf.unpack_farray(nitems, func)).reshape(var[1][::-1])
        # invert order of indices, to match IDL's
        out = np.transpose(out, list(range(len(var[1])))[::-1])
    return out


def read_file_var(buf, var):
    ''' Reads a single variable/array from a file buffer.

    IN:
       buf:  open file object
       var:  tuple with (type[,shape]), where type is 'f', 'd', 'i', 'ui',
             or 's'. Shape is optional, and if true is shape of array.
    OUT:
       variable/array

    '''
    assert len(var) > 0
    if len(var) == 1:
        out = np.fromfile(buf, dtype=var, count=1)
    elif len(var) == 2:
        out = np.fromfile(buf, dtype=var[0], count=var[1][0])
    else:
        nitems = np.prod(var[1])
        out = np.array(np.fromfile(buf, dtype=var[0], count=nitems)).\
            reshape(var[1][::-1])
        out = np.transpose(out, list(range(len(var[1])))[::-1])
    return out


def get_tau(x, mu, chi):
    ''' Calculates the optical depth, given x (height), mu (cos[theta]) and
        chi, absorption coefficient. Chi can be n-dimensional, as long as
        last index is depth.
    '''
    # With scipy, this could be done in one line with
    # scipy.integrate.quadrature.cumtrapz, but we are avoiding scipy to keep
    # these tools more independent
    if len(x) != chi.shape[-1]:
        raise ValueError('get_tau: x and chi have different sizes!')
    path = x / mu
    npts = len(x)
    # bring depth to first index, to allow n-d algebra
    chi_t = np.transpose(chi)
    tau = np.zeros(chi_t.shape)
    for i in range(1, npts):
        tau[i] = tau[i - 1] + 0.5 * \
            (chi_t[i - 1] + chi_t[i]) * (path[i - 1] - path[i])
    return tau.T


def get_contrib(z, mu, tau_in, S):
    ''' Calculates contribution function using x, mu, tau, and the source
        function. '''
    # Tau truncated at 100 (large enough to be useless)
    tau = tau_in.copy()
    tau[tau_in > 100.] = 100.
    # Calculate dtau (transpose to keep n-D generic form), and dx
    dtau = np.zeros(tau_in.shape[::-1])
    tt = np.transpose(tau_in)
    dtau[1:] = tt[1:] - tt[:-1]
    dtau = np.transpose(dtau)
    dx = np.zeros(z.shape)
    dx[1:] = (z[1:] - z[:-1]) / mu
    dx[0] = dx[1]
    # Calculate contribution function
    contrib = S * np.exp(-tau) * (- dtau / dx) / mu
    # convert from m^-1 to km^-1, units are now: J m^-2 s^-1 Hz^-1 sr^-1 km^-1
    contrib *= 1.e3
    return contrib


def write_B(outfile, Bx, By, Bz):
    ''' Writes a RH magnetic field file. Input B arrays can be any rank, as
        they will be flattened before write. Bx, By, Bz units should be T.'''
    if (Bx.shape != By.shape) or (By.shape != Bz.shape):
        raise TypeError('writeB: B arrays have different shapes!')
    n = np.prod(Bx.shape)
    # Convert into spherical coordinates
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    gamma_B = np.arccos(Bz / B)
    chi_B = np.arctan(By / Bx)
    # Pack as double
    p = xdrlib.Packer()
    p.pack_farray(n, B.ravel().astype('d'), p.pack_double)
    p.pack_farray(n, gamma_B.ravel().astype('d'), p.pack_double)
    p.pack_farray(n, chi_B.ravel().astype('d'), p.pack_double)
    # Write to file
    f = open(outfile, 'wb')
    f.write(p.get_buffer())
    f.close()
    return
