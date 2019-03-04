"""
Set of routines to read and work with input and output from Multi3D
"""
import os
import numpy as np
import scipy.io
import astropy.units as u


class Geometry:
    """
    class def for geometry
    """
    def __init__(self):
        self.nx = -1
        self.ny = -1
        self.nz = -1
        self.nmu = -1
        self.x = None
        self.y = None
        self.z = None
        self.mux = None
        self.muy = None
        self.muz = None
        self.wmu = None


class Atom:
    """
    class def for atom
    """
    def __init__(self):
        self.nrad = -1
        self.nrfix = -1
        self.ncont = -1
        self.nline = -1
        self.nlevel = -1
        self.id = None
        self.crout = None
        self.label = None
        self.ion = None
        self.ilin = None
        self.icon = None
        self.abnd = -1e10
        self.awgt = -1e10
        self.ev = None
        self.g = None
        self.n = None
        self.nstar = None
        self.totn = None
        self.dopfac = None

class Atmos:
    """
    class def for atmos
    """
    def __init__(self):
        self.ne = None
        self.tg = None
        self.vx = None
        self.vy = None
        self.vz = None
        self.r = None
        self.nh = None
        self.vturb = None
        self.x500 = None


class Spectrum:
    """
    class def for spectrum
    """
    def __init__(self):
        self.nnu = -1
        self.maxal = -1
        self.maxac = -1
        self.nu = None
        self.wnu = None
        self.ac = None
        self.al = None
        self.nac = None
        self.nal = None


class Cont:
    """
    class def for continuum
    """
    def __init__(self):
        self.f_type = None
        self.j = -1
        self.i = -1
        self.nnu = -1
        self.ntrans = -1
        self.ired = -1
        self.iblue = -1
        self.nu0 = -1.0 * u.Hz
        self.numax = -1.0
        self.alpha0 = -1.0
        self.alpha = None
        self.nu = None
        self.wnu = None


class Line:
    """
    class def for spectral line
    """
    def __init__(self):
        self.profile_type = None
        self.ga = -1.0
        self.gw = -1.0
        self.gq = -1.0
        self.lambda0 = -1.0
        self.nu0 = -1.0 * u.Hz
        self.Aji = -1.0
        self.Bji = -1.0
        self.Bij = -1.0
        self.f = -1.0
        self.qmax = -1.0
        self.Grat = -1.0
        self.ntrans = -1
        self.j = -1
        self.i = -1
        self.nnu = -1
        self.ired = -1
        self.iblue = -1
        self.nu = None
        self.q = None
        self.wnu = None
        self.wq = None
        self.adamp = None


class Transition:
    """
    class to hold transition info for IO
    """
    def __init__(self):
        self.i = -1
        self.j = -1
        self.isline = False
        self.iscont = False
        self.kr = -1
        self.nnu = -1
        self.nu = None
        self.l = None
        self.dl = None
        self.ired = -1
        self.ff = -1
        self.ang = -1


class Multi3dOut:
    """
    Reads and handles multi3d output

    Parameters
    ----------
    inputfile : str, optional
        Name of multi3d input file. Default is 'multi3d.input'
    directory : str, optional
        Directory with output files. Default is current directory.
    printinfo : bool, optional
        If True (default), will print more verbose output.

    Examples
    --------

    >>> data = Multi3dOut(directory='./output')
    >>> data.readall()

    Now select transition (by upper / lower level):

    >>> data.set_transition(3, 2)
    >>> emergent_intensity = data.readvar('ie')
    >>> source_function = data.readvar('snu')
    >>> tau1_height = data.readvar('zt1')

    Wavelength for the selected transition is saved in data.d.l, e.g.:

    >>> plt.plot(data.d.l, emergent_intensity[0, 0])
    """

    def __init__(self, inputfile="multi3d.input", directory='./', printinfo=True):
        """
        initializes object, default directory to look for files is ./
        default input options file name is multi3d.input
        """
        self.inputfile = inputfile
        self.directory = directory
        self.theinput = None
        self.outnnu = -1
        self.outff = -1
        self.sp = Spectrum()
        self.geometry = Geometry()
        self.atom = Atom()
        self.atmos = Atmos()
        self.d = Transition()
        self.inttype = np.int32
        self.floattype = np.float64
        self.printinfo = printinfo

    def readall(self):
        """
        reads multi3d.input file and all the out_* files
        """
        self.readinput()
        self.readpar()
        self.readnu()
        self.readn()
        self.readatmos()
        self.readrtq()

    def readinput(self):
        """
        reads input from self.inputfile into a dict.
        """
        fname = os.path.join(self.directory, self.inputfile)
        try:
            lines = [line.strip() for line in open(fname)]
            if self.printinfo:
                print("reading " + fname)
        except Exception as e:
            print(e)
            return
        tmp = []
        # Remove IDL comments ;
        for line in lines:
            head, sep, tail = line.partition(';')
            tmp.append(head)
        # Remove blank lines
        tmp = filter(None, tmp)
        self.theinput = dict()

        for line in tmp:
            head, sep, tail = line.partition("=")
            tail = tail.strip()
            head = head.strip()
            # Checks which type the values are
            try:
                int(tail)
                tail = int(tail)
            except ValueError:
                try:
                    float(tail)
                    tail = float(tail)
                except ValueError:
                    # First and last tokens are quotes
                    tail = tail[1:-1]
            # special items, multiple float values in a string
            if head in ["muxout", "muyout", "muzout"]:
                temp = []
                for item in tail.split():
                    temp.append(float(item))
                    self.theinput[head] = temp
            else:
                # simple str
                self.theinput[head] = tail
        # set xn,ny,nz here as they are now known
        self.geometry.nx = self.theinput["nx"]
        self.geometry.ny = self.theinput["ny"]
        self.geometry.nz = self.theinput["nz"]


    def readpar(self):
        """
        reads the out_par file
        """
        fname = os.path.join(self.directory, "out_par")
        f = scipy.io.FortranFile(fname, 'r')
        if self.printinfo:
            print("reading " + fname)
        # geometry struct
        self.geometry.nmu = int(f.read_ints(dtype=self.inttype))
        self.geometry.nx = int(f.read_ints(dtype=self.inttype))
        self.geometry.ny = int(f.read_ints(dtype=self.inttype))
        self.geometry.nz = int(f.read_ints(dtype=self.inttype))
        self.geometry.x = f.read_reals(dtype=self.floattype)
        self.geometry.y = f.read_reals(dtype=self.floattype)
        self.geometry.z = f.read_reals(dtype=self.floattype)
        self.geometry.mux = f.read_reals(dtype=self.floattype)
        self.geometry.muy = f.read_reals(dtype=self.floattype)
        self.geometry.muz = f.read_reals(dtype=self.floattype)
        self.geometry.wmu = f.read_reals(dtype=self.floattype)
        self.sp.nnu = int(f.read_ints(dtype=self.inttype))
        self.sp.maxac = int(f.read_ints(dtype=self.inttype))
        self.sp.maxal = int(f.read_ints(dtype=self.inttype))
        self.sp.nu = f.read_reals(dtype=self.floattype)
        self.sp.wnu = f.read_reals(dtype=self.floattype)
        # next two need reform
        self.sp.ac = f.read_ints(dtype=self.inttype)
        self.sp.al = f.read_ints(dtype=self.inttype)
        self.sp.nac = f.read_ints(dtype=self.inttype)
        self.sp.nal = f.read_ints(dtype=self.inttype)
        # atom struct
        self.atom.nrad = int(f.read_ints(dtype=self.inttype))
        self.atom.nrfix = int(f.read_ints(dtype=self.inttype))
        self.atom.ncont = int(f.read_ints(dtype=self.inttype))
        self.atom.nline = int(f.read_ints(dtype=self.inttype))
        self.atom.nlevel = int(f.read_ints(dtype=self.inttype))
        ss = [self.atom.nlevel, self.atom.nlevel]
        self.atom.id = (f.read_record(dtype='S20'))[0].strip()
        self.atom.crout = (f.read_record(dtype='S20'))[0].strip()
        self.atom.label = f.read_record(dtype='S20').tolist()
        self.atom.ion = f.read_ints(dtype=self.inttype)
        self.atom.ilin = f.read_ints(dtype=self.inttype).reshape(ss)
        self.atom.icon = f.read_ints(dtype=self.inttype).reshape(ss)
        self.atom.abnd = f.read_reals(dtype=self.floattype)[0]
        self.atom.awgt = f.read_reals(dtype=self.floattype)[0]
        self.atom.ev = f.read_reals(dtype=self.floattype)
        self.atom.g = f.read_reals(dtype=self.floattype)
        self.sp.ac.resize([self.sp.nnu, self.atom.ncont])
        self.sp.al.resize([self.sp.nnu, self.atom.nline])

        # cont info
        self.cont = [Cont() for i in range(self.atom.ncont)]
        for c in self.cont:
            c.bf_type = f.read_record(dtype='S20')[0].strip()
            c.i = int(f.read_ints(dtype=self.inttype))
            c.j = int(f.read_ints(dtype=self.inttype))
            c.ntrans = int(f.read_ints(dtype=self.inttype))
            c.nnu = int(f.read_ints(dtype=self.inttype))
            c.ired = int(f.read_ints(dtype=self.inttype))
            c.iblue = int(f.read_ints(dtype=self.inttype))
            c.nu0 = f.read_reals(dtype=self.floattype)[0] * u.Hz
            c.numax = f.read_reals(dtype=self.floattype)[0]
            c.alpha0 = f.read_reals(dtype=self.floattype)[0]
            c.alpha = f.read_reals(dtype=self.floattype)[0]
            c.nu = f.read_reals(dtype=self.floattype)
            c.wnu = f.read_reals(dtype=self.floattype)

        #line info
        self.line = [Line() for i in range(self.atom.nline)]
        for l in self.line:
            l.profile_type = f.read_record(dtype='S72')[0].strip()
            l.ga = f.read_reals(dtype=self.floattype)[0]
            l.gw = f.read_reals(dtype=self.floattype)[0]
            l.gq = f.read_reals(dtype=self.floattype)[0]
            l.lambda0 = f.read_reals(dtype=self.floattype)[0]
            l.nu0 = f.read_reals(dtype=self.floattype)[0] * u.Hz
            l.Aji = f.read_reals(dtype=self.floattype)[0]
            l.Bji = f.read_reals(dtype=self.floattype)[0]
            l.Bij = f.read_reals(dtype=self.floattype)[0]
            l.f = f.read_reals(dtype=self.floattype)[0]
            l.qmax = f.read_reals(dtype=self.floattype)[0]
            l.Grat = f.read_reals(dtype=self.floattype)[0]
            l.ntrans = int(f.read_ints(dtype=self.inttype))
            l.j = int(f.read_ints(dtype=self.inttype))
            l.i = int(f.read_ints(dtype=self.inttype))
            l.nnu = int(f.read_ints(dtype=self.inttype))
            l.ired = int(f.read_ints(dtype=self.inttype))
            l.iblue = int(f.read_ints(dtype=self.inttype))
            l.nu = f.read_reals(dtype=self.floattype)
            l.q = f.read_reals(dtype=self.floattype)
            l.wnu = f.read_reals(dtype=self.floattype)
            l.wq = f.read_reals(dtype=self.floattype)
        f.close()

    def readn(self):
        """
        reads populations as numpy memmap
        """
        if self.theinput is None:
            self.readinput()

        fname = os.path.join(self.directory, "out_pop")
        nlevel = self.atom.nlevel
        nx, ny, nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        gs = nx * ny * nz * nlevel * 4
        self.atom.n = np.memmap(fname, dtype='float32', mode='r',
                                shape=(nx, ny, nz, nlevel), order='F')
        self.atom.nstar = np.memmap(fname, dtype='float32', mode='r', order='F',
                                    shape=(nx, ny, nz, nlevel), offset=gs, )

        self.atom.ntot = np.memmap(fname, dtype='float32', mode='r', order='F',
                                   shape=(nx, ny, nz), offset=gs * 2)
        if self.printinfo:
            print("reading " + fname)

    def readatmos(self):
        """
        reads atmosphere as numpy memmap
        """
        if self.theinput is None:
            self.readinput()
        fname = os.path.join(self.directory, "out_atm")
        nhl = 6
        nx, ny, nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        s = (nx, ny, nz)
        gs = nx * ny * nz * 4
        self.atmos.ne = np.memmap(fname, dtype='float32', mode='r',
                                  shape=s, order='F')
        self.atmos.tg = np.memmap(fname, dtype='float32', mode='r',
                                  shape=s, offset=gs, order='F')
        self.atmos.vx = np.memmap(fname, dtype='float32', mode='r',
                                  shape=s, offset=gs*2, order='F')
        self.atmos.vy = np.memmap(fname, dtype='float32', mode='r',
                                  shape=s, offset=gs*3, order='F')
        self.atmos.vz = np.memmap(fname, dtype='float32', mode='r',
                                  shape=s, offset=gs*4, order='F')
        self.atmos.rho = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s, offset=gs*5, order='F')
        self.atmos.nh = np.memmap(fname, dtype='float32', mode='r', order='F',
                                  shape=(nx, ny, nz, nhl), offset=gs * 6)
        #self.atmos.vturb = np.memmap(fname, dtype='float32', mode='r',
        #                             shape=s ,offset=gs*12, order='F' )
        if self.printinfo:
            print("reading " + fname)

    def readrtq(self):
        """
        reads out_rtq as numpy memmap
        """
        if self.theinput is None:
            self.readinput()
        if self.sp is None:
            self.readpar()
        fname = os.path.join(self.directory, "out_rtq")
        nx, ny, nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        s = (nx, ny, nz)
        gs = nx * ny * nz * 4
        self.atmos.x500 = np.memmap(fname, dtype='float32', mode='r',
                                    shape=s, order='F')
        self.atom.dopfac = np.memmap(fname, dtype='float32', mode='r',
                                     shape=s, offset=gs, order='F')
        i = 2
        for l in self.line:
            l.adamp = np.memmap(fname, dtype='float32', mode='r',
                                shape=s, offset=gs * i, order='F')
            i += 1

    def readnu(self):
        """
        reads the out_nu file
        """
        fname = os.path.join(self.directory, "out_nu")
        f = scipy.io.FortranFile(fname, 'r')
        if self.printinfo:
            print("reading " + fname)
        self.outnnu = int(f.read_ints(dtype=self.inttype))
        self.outff = f.read_ints(dtype=self.inttype)

    def set_transition(self, i, j, fr=-1, ang=0):
        """
        Sets parameters of transition to read
        """
        from astropy.constants import c
        cc = c.to('Angstrom / s')

        self.d.i = i-1
        self.d.j = j-1
        self.d.isline = self.atom.ilin[self.d.i, self.d.j] != 0
        self.d.iscont = self.atom.icon[self.d.i, self.d.j] != 0

        if self.d.isline:
            self.d.kr = self.atom.ilin[self.d.i, self.d.j] - 1
            self.d.nnu = self.line[self.d.kr].nnu
            self.d.nu = np.copy(self.line[self.d.kr].nu) * u.Hz
            self.d.l = (cc / self.d.nu).to('Angstrom')
            self.d.dl = (cc * (1.0 / self.d.nu -
                               1.0 / self.line[self.d.kr].nu0)).to('Angstrom')
            self.d.ired = self.line[self.d.kr].ired
        elif self.d.iscont:
            self.d.kr = self.atom.icon[self.d.i, self.d.j] - 1
            self.d.nnu = self.cont[self.d.kr].nnu
            self.d.nu = np.copy(self.cont[self.d.kr].nu) * u.Hz
            self.d.l = (cc / self.d.nu).to('Angstrom')
            self.d.dl = None
            self.d.ired = self.cont[self.d.kr].ired
        else:
            raise RuntimeError('upper and lower level %i, %i are not connected'
                               ' with a radiative transition.' % (i, j))
        if fr == -1:
            self.d.ff = -1
        else:
            self.d.ff = self.d.ired + fr
        self.d.ang = ang

        if self.printinfo:
            print('transition parameters are set to:')
            print(' i   =', self.d.i)
            print(' j   =', self.d.j)
            print(' kr  =', self.d.kr)
            print(' ff  =', self.d.ff)
            print(' ang =', self.d.ang)

    def readvar(self, var, all_vars=False):
        """
        Reads output variable
        """
        allowed_names = ['chi', 'ie', 'jnu', 'zt1', 'st', 'xt', 'cf', 'snu',
                         'chi_c', 'scatt', 'therm']
        if var.lower() not in allowed_names:
            raise ValueError("%s is not an valid variable name, must be one"
                             "of '%s.'" % (var, "', '".join(allowed_names)))
        mx = "{:+.2f}".format(self.theinput['muxout'][self.d.ang])
        my = "{:+.2f}".format(self.theinput['muyout'][self.d.ang])
        mz = "{:+.2f}".format(self.theinput['muzout'][self.d.ang])
        mus = '_' + mx + '_' + my + '_' + mz + '_allnu'
        fname = os.path.join(self.directory, var + mus)
        if os.path.isfile(fname):
            if self.printinfo:
                print('reading from ' + fname)
        else:
            raise IOError('%s does not exist' % fname)
        sg = self.geometry
        if var in ('ie', 'zt1'):
            if all_vars:
                shape = (sg.nx, sg.ny, self.outnnu)
                offset = 0
            elif self.d.ff == -1:
                shape = (sg.nx, sg.ny, self.d.nnu)
                offset = (4 * sg.nx * sg.ny *
                          np.where(self.outff == self.d.ired)[0])[0]
            else:
                shape = (sg.nx, sg.ny)
                offset = (4 * sg.nx * sg.ny *
                          np.where(self.outff == self.d.ff)[0])[0]
        else:
            if all_vars:
                shape = (sg.nx, sg.ny, sg.nz, self.outnnu)
                offset = 0
            elif self.d.ff == -1:
                shape = (sg.nx, sg.ny, sg.nz, self.d.nnu)
                offset = (4 * sg.nx * sg.ny * sg.nz *
                          np.where(self.outff == self.d.ired)[0])[0]
            else:
                shape = (sg.nx, sg.ny, sg.nz)
                offset = (4 * sg.nx * sg.ny * sg.nz *
                          np.where(self.outff == self.d.ff)[0])[0]
        return np.memmap(fname, dtype='float32', mode='r',
                         shape=shape, order='F', offset=offset)


class Multi3dAtmos:
    """
    Class to read/write input atmosphere for Multi3D.

    Parameters
    ----------
    infile : str
        Name of file to read.
    nx, ny, nz : ints
        Number of points in x, y, and z dimensions.
    mode : str, optional
        Access mode. Can be either 'r' (read), 'w' (write, deletes existing),
        or 'w+' (write, update).
    nhydr : int, optional
        Number of hydrogen levels. Default is 6.
    dp : bool, optional
        If True, will write in double precision (float64). Otherwise,
        will write in single precision (float32, default).
    big_endian : bool, optional
        Endianness of output file. Default is False (little endian).
    read_nh : bool, optional
        If True, will read/write hydrogen populations. Default is False.
    read_vturb : bool, optional
        If True, will read/write turbulent velocity. Default is False.
    """
    def __init__(self, infile, nx, ny, nz, mode='r', **kwargs):
        if os.path.isfile(infile) or (mode == "w+"):
            self.open_atmos(infile, nx, ny, nz, mode=mode, **kwargs)

    def open_atmos(self, infile, nx, ny, nz, mode="r", nhydr=6, dp=False,
                   big_endian=False, read_nh=False, read_vturb=False):
        """Reads/writes multi3d atmosphere into parent object."""
        dtype = ["<", ">"][big_endian] + ["f4", "f8"][dp]
        ntot = nx * ny * nz * np.dtype(dtype).itemsize
        mm = mode
        self.ne = np.memmap(infile, dtype=dtype, mode=mm, offset=0,
                            shape=(nx, ny, nz), order="F")
        self.temp = np.memmap(infile, dtype=dtype, mode=mm, offset=ntot,
                              shape=(nx, ny, nz), order="F")
        self.vx = np.memmap(infile, dtype=dtype, mode=mm, offset=ntot * 2,
                            shape=(nx, ny, nz), order="F")
        self.vy = np.memmap(infile, dtype=dtype, mode=mm, offset=ntot * 3,
                            shape=(nx, ny, nz), order="F")
        self.vz = np.memmap(infile, dtype=dtype, mode=mm, offset=ntot * 4,
                            shape=(nx, ny, nz), order="F")
        self.rho = np.memmap(infile, dtype=dtype, mode=mm, offset=ntot * 5,
                             shape=(nx, ny, nz), order="F")
        offset = ntot * 6
        if read_nh:
            self.nh = np.memmap(infile, dtype=dtype, mode=mm, offset=offset,
                                shape=(nx, ny, nz, nhydr), order="F")
            offset += ntot * nhydr
        if read_vturb:
            self.vturb = np.memmap(infile, dtype=dtype, mode=mm, order="F",
                                   offset=offset, shape=(nx, ny, nz))
