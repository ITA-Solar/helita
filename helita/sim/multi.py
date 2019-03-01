"""
Set of routines to interface with MULTI (1D or _3D)
"""
import numpy as np
import os


class Multi_3dOut:
    """
    Class that reads and deals with output from multi_3d
    """
    def __init__(self, outfile=None, basedir='.', atmosid='', length=4,
                 verbose=False, readall=False):
        self.verbose = verbose
        out3dfiles = ['cmass3d', 'dscal2', 'height3d', 'Iv3d', 'taulg3d',
                      'x3d', 'xnorm3d']
        c3dfiles = ['n3d', 'b3d']
        if outfile is None:
            outfile = '%s/out3d.%s' % (basedir, atmosid)
        else:
            basedir = os.path.split(outfile)[0]
            atmosid = os.path.split(mm)[1].split('out3d.')[1]
        out3dfiles = ['%s/%s.%s' % (basedir, s, atmosid) for s in out3dfiles]
        c3dfiles = ['%s/%s.%s' % (basedir, s, atmosid) for s in c3dfiles]
        self.read_out3d(outfile, length=length)
        # read all output files
        if readall:
            for f in out3dfiles:
                if os.path.isfile(f):
                    self.read_out3d(f, length=length)
            for f in c3dfiles:
                if os.path.isfile(f):
                    self.read_c3d(f, length=length,
                                  mode=(os.path.split(f)[1].split('.' +
                                                                  atmosid)[0]))

    def check_basic(self):
        """
        Checks to see if basic input parameters have been read from out3d.
        """
        basic = ['nx', 'ny', 'ndep', 'mq', 'nrad', 'nqtot']
        for p in basic:
            if p not in dir(self):
                raise ValueError('(EEE) %s has not been read. Make sure '
                                 'out3d was read.' % p)

    def read_out3d(self, outfile, length=4):
        """ Reads out3d file. """
        from ..io.fio import fort_read

        # find out endianness
        test = np.fromfile(outfile, dtype='<i', count=1)[0]
        be = False if test == 16 else True
        file = open(outfile, 'r')
        readon = True
        arrays_xyz = ['taulg3d', 'cmass3d', 'dscal2', 'xnorm3d', 'x3d',
                      'height3d']
        while readon:
            try:
                itype, isize, cname = fort_read(file, 0, ['i', 'i', '8c'],
                                                big_endian=be, length=length)
                cname = cname.strip()
                if self.verbose:
                    print(('--- reading ' + cname))
                if cname == 'id':
                    self.id = fort_read(file, 0, ['80c'])[0].strip()
                elif cname == 'dim':
                    aa = fort_read(file, isize, 'i', big_endian=be,
                                   length=length)
                    self.nx, self.ny, self.ndep, self.mq, self.nrad = aa[:5]
                    if isize == 5:
                        self.version = 1
                    else:
                        self.version = 2
                        self.nq = aa[5:]
                        self.nqtot = np.sum(self.nq) + self.nrad
                    self.nxyz = self.nx * self.ny * self.ndep
                elif cname == 'q':
                    self.check_basic()
                    aa = fort_read(file, self.mq * self.nrad, 'f',
                                   big_endian=be, length=length)
                    self.q = np.transpose(aa.reshape(self.nrad, self.mq))
                elif cname == 'xl':
                    self.check_basic()
                    self.xl = fort_read(file, self.nqtot, 'd', big_endian=be,
                                        length=length)
                elif cname in arrays_xyz:
                    self.check_basic()
                    aa = fort_read(file, self.nxyz, 'f', big_endian=be,
                                   length=length)
                    setattr(self, cname,
                            np.transpose(aa.reshape(self.ndep, self.ny,
                                                   self.nx)))
                elif cname == 'Iv':
                    self.check_basic()
                    aa = fort_read(file, isize, 'f', big_endian=be,
                                   length=length)
                    self.Iv = np.transpose(aa.reshape(self.ny, self.nx,
                                                     self.nqtot))
                elif cname == 'n3d':  # might be brokenp...
                    self.check_basic()
                    self.nk = isize // (self.nx * self.ny * self.ndep)
                    aa = fort_read(file, isize, 'f', big_endian=be,
                                   length=length)
                    self.n3d = np.transpose(aa.reshape(self.nk, self.ndep,
                                                      self.ny, self.nx))
                elif cname == 'nk':
                    self.nk = fort_read(file, 1, 'i', big_endian=be,
                                        length=length)[0]
                else:
                    print(('(WWW) read_out3d: unknown label found: %s. '
                           'Aborting.' % cname))
                    break
            except EOFError:
                readon = False
        if self.verbose:
            print(('--- Read %s.' % outfile))

    def read_c3d(self, outfile, length=4, mode='n3d'):
        ''' Reads the 3D cube output file, like n3d or b3d. '''
        self.check_basic()
        self.nk = os.path.getsize(outfile) // (self.nxyz * 4)
        setattr(self, mode, np.memmap(outfile, dtype='Float32', mode='r',
                                      order='F', shape=(self.nx, self.ny,
                                                        self.ndep, self.nk)))
        if self.verbose:
            print('--- Read ' + outfile)


class Atmos3d:
    def __init__(self, infile, big_endian=False):
        ''' Reads multi_3d/old multi3d atmos3d file '''
        self.big_endian = big_endian
        self.read(infile, big_endian=big_endian)
        return

    def read(self, infile, big_endian, length=4):
        from ..io.fio import fort_read
        file = open(infile, 'r')
        types = {4: 'f', 5: 'd'}  # precision, float or double
        # read header stuff
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        nx, ny, nz = fort_read(file, 3, 'i', big_endian=big_endian,
                               length=length)
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # x [cm]
        itype, isize, lx1, lx2 = fort_read(file, 4, 'i', big_endian=big_endian,
                                           length=length)
        prec = types[itype]
        self.x = fort_read(file, nx, prec, big_endian=big_endian,
                           length=length)
        # y [cm]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        self.y = fort_read(file, ny, prec, big_endian=big_endian,
                           length=length)
        # z [cm]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        self.z = fort_read(file, nz, prec, big_endian=big_endian,
                           length=length)
        # electron density [cm-3]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                       length=length)
        self.ne = np.transpose(aa.reshape((nz, ny, nx)))
        # temperature [K]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                       length=length)
        self.temp = np.transpose(aa.reshape((nz, ny, nx)))
        # vx [km/s]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                       length=length)
        self.vx = np.transpose(aa.reshape((nz, ny, nx)))
        # vy [km/s]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                       length=length)
        self.vy = np.transpose(aa.reshape((nz, ny, nx)))
        # vz [km/s]
        fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                       length=length)
        self.vz = np.transpose(aa.reshape((nz, ny, nx)))
        # reading rho, if written to file
        last = fort_read(file, 16, 'b', big_endian=big_endian, length=length)
        if len(last) != 0:
            # rho [g cm-3]
            aa = fort_read(file, nx * ny * nz, prec, big_endian=big_endian,
                           length=length)
            self.rho = np.transpose(aa.reshape((nz, ny, nx)))
        file.close()
        return


    def write_rh15d(self, outfile, sx=None, sy=None, sz=None, desc=None):
        ''' Writes atmos into rh15d NetCDF format. '''
        from . import rh15d
        if not hasattr(self, 'rho'):
            raise UnboundLocalError('Current atmosphere has no rho, '
                                    'cannot convert to rh15d format.')
        # slicing and unit conversion
        if sx is None:
            sx = [0, self.nx, 1]
        if sy is None:
            sy = [0, self.ny, 1]
        if sz is None:
            sz = [0, self.nz, 1]
        temp = self.temp[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                         sz[0]:sz[1]:sz[2]]
        rho = self.rho[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2], sz[0]:sz[1]:sz[2]]
        ne = self.ne[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                     sz[0]:sz[1]:sz[2]] * 1.e6
        vz = self.vz[sx[0]:sx[1]:sx[2], sy[0]:sy[1]:sy[2],
                     sz[0]:sz[1]:sz[2]] * 1.e3
        z = self.z[sz[0]:sz[1]:sz[2]] * 1e-2
        nh = rho / 2.380491e-24 * 1.e6       # from rho to nH in m^-3
        # write to file
        rh15d.make_xarray_atmos(outfile, temp, vz, z, nH=nh, ne=ne,
                                append=False, desc=desc, snap=0)


def watmos_multi(filename, temp, ne, z=None, logtau=None, vz=None, vturb=None,
                 cmass=None, nh=None, id='Model', scale='height', logg=4.44,
                 write_dscale=False, spherical=False):
    """
    Writes file with atmosphere in MULTI format.

    Parameters
    ----------
    filename : str
        Name of file to write.
    temp : 1D array
        Temperature in K.
    ne : 1D array
        Electron density per cm^-3
    z : 1D array, optional
        Height scale in km
    logtau : 1D array, optional
        Log of optical depth at 500 nm scale
    vz : 1D array, optional
        Line of sight velocity in km/s. Positive is upflow.
    vturb : 1D array, optional
        Turbulent velocity in km/s.
    cmass : 1D array, optional
        Column mass scale in g cm^-2
    nh : 2D array, optional
        Hydrogen populations per cm^-3. Shape must be (6, nheight),
        so always 6 levels.
    id : str, optional
        Model ID string
    scale : str, optional
        Type of scale to use. Options are:
            'height' - HEIGHT scale (default)
            'tau'    - TAU(5000) scale
            'mass'   - column mass scale
        Must supply z, logtau, or cmass accordingly.
    logg : float, optional
        Log of gravity. Default is solar (4.44)
    write_dscale : bool, optional
        If True, will write DSCALE file.
        Height scale in km
    spherical : bool, optional
        If True, will write model in spherical geometry

    Returns
    -------
    None. Writes file to disk.
    """
    if scale.lower() == 'height':
        if z is None:
            raise ValueError('watmos_multi: height scale selected '
                             'but z not given!')
        scl = z
        desc = 'HEIGHT (KM)'
    elif scale.lower() == 'tau':
        if logtau is None:
            raise ValueError('watmos_multi: tau scale selected but '
                             'tau not given!')
        scl = logtau
        desc = 'LG TAU(5000)'
    elif scale.lower() == 'mass':
        if cmass is None:
            raise ValueError('watmos_multi: mass scale selected but '
                             'column mass not given!')
        scl = cmass
        desc = 'LOG COLUMN MASS'
    else:
        raise ValueError('watmos_multi: invalid scale: {0}'.format(scale))
    f = open(filename, 'w')
    ndep = len(temp)
    # write 'header'
    f.write(' {0}\n*\n'.format(id))
    f.write(' {0} scale\n'.format(scale).upper())
    f.write('* LG G\n')
    f.write('{0:6.2f}\n'.format(logg))
    if spherical:
        f.write('* Nradius   Ncore    Ninter\n')
        f.write('{0:5d}        8         0\n'.format(ndep))
    else:
        f.write('* NDEP\n')
        f.write('{0:5d}\n'.format(ndep))
    f.write('*  {0}     TEMPERATURE        NE             V            '
            'VTURB\n'.format(desc))
    if vz is None:
        vz = np.zeros(ndep, dtype='f')
    if vturb is None:
        vturb = np.zeros(ndep, dtype='f')
    elif type(vturb) == type(5):   # constant vturb
        vturb = np.zeros(ndep, dtype='f') + vturb
    # write atmosphere
    for i in range(ndep):
        # astype hack to get over numpy bug
        f.write('{0:15.6E}{1:15.6E}{2:15.6E}{3:15.6E}{4:15.6E}'
                '\n'.format(scl[i].astype('d'), temp[i].astype('d'),
                            ne[i].astype('d'), vz[i].astype('d'),
                            vturb[i].astype('d')))
    # if nh given
    if nh is not None:
        if nh.shape != (6, ndep):
            raise ValueError('watmos_multi: nh has incorrect shape. Must be '
                             '6 H levels!')
        f.write('*\n* Hydrogen populations\n')
        f.write('*    nh(1)       nh(2)       nh(3)       nh(4)       nh(5)   '
                'np\n')
        for i in range(ndep):
            ss = ''
            for j in range(nh.shape[0]):
                ss += '{0:12.4E}'.format(nh[j, i].astype('d'))
            f.write(ss + '\n')
    f.close()
    print('--- Wrote multi atmosphere to ' + filename)
    if write_dscale:
        f = open(filename + '.dscale', 'w')
        f.write(' {0}\n*\n'.format(id))
        f.write(' {0} scale\n'.format(scale).upper())
        # setting the second element to zero will force it to be calculated
        # in DPCONV. Will it work for height scale?
        f.write('{0:5d}    {1:.5f}\n'.format(ndep, 0.))
        for i in range(ndep):
            f.write('{0:15.6E}\n'.format(scl[i].astype('d')))
        f.close()
        print(('--- Wrote dscale to ' + filename + '.dscale'))


def write_atmos3d(outfile, x, y, z, ne, temp, vz, vx=None, vy=None, rho=None,
                  big_endian=False, length=4, prec='Float32'):
    """
    Writes file with atmos3d atmosphere (format of 'old' multi3d and multi_3d).

    Parameters
    ----------
    outfile : str
        Name of file to write.
    x, y, z : 1D arrays
        Arrays with x, y, and z  scales in cm.
    ne : 3D array, C order
        Electron density per cm^-3. Shape must be (nx, ny, nz).
    temp : 3D array, C order
        Temperature in K. Shape must be (nx, ny, nz).
    vz : 3D array, C order
        Velocity in height axis in km/s. Positive is upflow.
        Shape must be (nx, ny, nz).
    vx : 3D array, C order, optional
        Velocity in x axis in km/s. Shape must be (nx, ny, nz).
        If not given, zeros will be used.
    vy : 3D array, C order, optional
        Velocity in y axis in km/s. Shape must be (nx, ny, nz).
        If not given, zeros will be used.
    rho : 3D array, C order, optional
        Density in g cm^-3. Shape must be (nx, ny, nz).
        If not given, zeros will be used.
    big_endian : bool, optional
        Endianness of output file. Default is False (little endian).
    length : int, optional
        Length of fortran format pad. Should be 4 (default) in most cases.
    prec : str, optional
        Precision ('Float32' or 'Float64')

    Returns
    -------
    None. Writes file to disk.
    """
    import os
    from ..io.fio import fort_write

    if os.path.isfile(outfile):
        raise IOError('(EEE) write_atmos3d: file %s already exists, refusing '
                      'to overwrite.' % outfile)
    f = open(outfile, 'w')
    # Tiago note: these should be fortran longs. However, in 64-bit systems the
    #             size of a long in python is 8 bytes, where fortran longs are
    #             still 4 bytes. Hence, it is better to keep all longs as ints,
    #             as sizeof(int) = 4
    nx = len(x)
    ny = len(y)
    nz = len(z)
    ii = 3
    ir = 5 if prec in ['Float64', 'd'] else 4
    ll = length
    be = big_endian
    if vx is None:
        vx = np.zeros(vz.shape, dtype=prec)
    if vy is None:
        vy = np.zeros(vz.shape, dtype=prec)
    fort_write(f, 0, [ii, 3, 'dim     '], big_endian=be, length=ll)
    fort_write(f, 0, [nx, ny, nz], big_endian=be, length=ll)
    fort_write(f, 0, [ir, nx, 'x grid  '], big_endian=be, length=ll)
    fort_write(f, x.size, x.astype(prec), big_endian=be, length=ll)
    fort_write(f, 0, [ir, nx, 'y grid  '], big_endian=be, length=ll)
    fort_write(f, y.size, y.astype(prec), big_endian=be, length=ll)
    fort_write(f, 0, [ir, nx, 'z grid  '], big_endian=be, length=ll)
    fort_write(f, z.size, z.astype(prec), big_endian=be, length=ll)
    fort_write(f, 0, [ir, nx, 'nne     '], big_endian=be, length=ll)
    fort_write(f, ne.size, np.transpose(ne).astype(prec), big_endian=be,
               length=ll)
    fort_write(f, 0, [ir, nx, 'temp    '], big_endian=be, length=ll)
    fort_write(f, temp.size, np.transpose(temp).astype(prec), big_endian=be,
               length=ll)
    fort_write(f, 0, [ir, nx, 'vel x   '], big_endian=be, length=ll)
    fort_write(f, vx.size, np.transpose(vx).astype(prec), big_endian=be,
               length=ll)
    fort_write(f, 0, [ir, nx, 'vel y   '], big_endian=be, length=ll)
    fort_write(f, vy.size, np.transpose(vy).astype(prec), big_endian=be,
               length=ll)
    fort_write(f, 0, [ir, nx, 'vel z   '], big_endian=be, length=ll)
    fort_write(f, vz.size, np.transpose(vz).astype(prec), big_endian=be,
               length=ll)
    if rho is not None:
        fort_write(f, 0, [ir, nx, 'rho     '], big_endian=be, length=ll)
        fort_write(f, rho.size, np.transpose(rho).astype(prec), big_endian=be,
                   length=ll)
    f.close()
    print(('Wrote %s' % outfile))
