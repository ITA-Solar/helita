"""
Set of routines to read output from multi3d MPI (new version!)
"""
from ..io.fio import fort_read
import numpy as np
import os


class m3dprof:
    def __init__(self, dir='.', bswp=False, precision='f8', nopop=True,
                 nortq=True, noatm=True, line=None, v1d=False, angle=False,
                 contrib=False, dcint=True):
        ''' Reads multi3d output data in out3d. '''
        self.dtype = precision
        self.bswp = bswp
        self.dir = dir
        self.v1d = v1d
        # specifying sizes of data types
        self.l = 'i4'  # size of fortran long (4 bytes = 32 bits)
        self.fl = 'f4'  # size of float
        self.d = 'f8'  # size of double
        # Read standard output files
        self.read_par()
        if not nopop:
            self.read_pop()
        if not nortq:
            self.read_rtq()
        if not noatm:
            self.read_atm()
        # Reads transition dependent data?
        if line is not None:
            self.read_line(line, angle=angle, contrib=contrib, dcint=dcint)
        return

    def read_par(self, hdrlen=4, filename=None):
        ''' Reads out_par file '''
        if filename is None:
            filename = self.dir + '/out_par'
        file = open(filename, 'r')
        bswp = self.bswp
        # Skip file types and sizes
        fort_read(file, 4, self.l, big_endian=bswp)
        nk, nrad, nline, nrfix, nmu, nx, ny, ndep = fort_read(
            file, 8, self.l, big_endian=bswp)
        self.nk = nk
        self.nrad = nrad
        self.nline = nline
        self.nrfix = nrfix
        self.nmu = nmu
        self.nx = nx
        self.ny = ny
        self.ndep = ndep
        fort_read(file, 4, self.l, big_endian=bswp)
        uu = fort_read(file, nx + ny + ndep, self.fl, big_endian=bswp)
        self.widthx = uu[:nx]
        self.widthy = uu[nx:nx + ny]
        self.height = uu[nx + ny:]
        fort_read(file, 4, self.l, big_endian=bswp)
        uu = fort_read(file, 4 * nmu, self.fl, big_endian=bswp)
        self.xmu = uu[:nmu]
        self.ymu = uu[nmu:2 * nmu]
        self.zmu = uu[2 * nmu:3 * nmu]
        self.wmu = uu[3 * nmu:]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.nq = fort_read(file, nrad, self.l, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        # Read directly to string
        self.atomid = file.read(hdrlen * 3)[hdrlen:hdrlen * 2]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.abnd, self.awgt, self.grph, self.qnorm = fort_read(
            file, 4, self.fl, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        self.ev = fort_read(file, nk, self.fl, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        self.g = fort_read(file, nk, self.fl, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        self.label = np.zeros(nk, dtype='S20')
        file.read(hdrlen)
        for i in range(nk):
            self.label[i] = file.read(20)
        file.read(hdrlen)
        fort_read(file, 4, self.l, big_endian=bswp)
        self.ion = fort_read(file, nk, self.l, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        uu = fort_read(file, nrad * 2, self.l, big_endian=bswp)
        self.jrad = uu[:nrad]
        self.irad = uu[nrad:]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.krad = np.reshape(
            fort_read(file, nk * nk, self.l, big_endian=bswp), (nk, nk))
        fort_read(file, 4, self.l, big_endian=bswp)
        uu = fort_read(file, nrad * 5 + nline, self.fl, big_endian=bswp)
        self.f = uu[:nrad]
        self.ga = uu[nrad:nrad * 2]
        self.gw = uu[nrad * 2:nrad * 3]
        self.gq = uu[nrad * 3:nrad * 4]
        self.alamb = uu[nrad * 4:nrad * 5]
        self.a = uu[nrad * 5:]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.ktrans = fort_read(file, nrad, self.l, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        self.b = np.reshape(
            fort_read(file, nk * nk, self.fl, big_endian=bswp), (nk, nk))
        if nrfix >= 1:
            fort_read(file, 4, self.l, big_endian=bswp)
            uu = fort_read(file, nrfix * 4, self.l, big_endian=bswp)
            self.jfx = uu[:nrfix]
            self.ifx = uu[nrfix:nrfix * 2]
            self.ipho = uu[nrfix * 2:nrfix * 3]
            self.itrad = uu[nrfix * 3:]
            fort_read(file, 4, self.l, big_endian=bswp)
            uu = fort_read(file, nrfix * 2, self.fl, big_endian=bswp)
            self.a0 = uu[:nrfix]
            self.trad = uu[nrfix:]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.q = np.reshape(fort_read(file, np.max(self.nq) * nrad, self.fl, big_endian=bswp),
                            (nrad, np.max(self.nq)))
        self.q = np.transpose(self.q, axes=(1, 0))
        fort_read(file, 4, self.l, big_endian=bswp)
        uu = fort_read(file, nrad * 2, self.l, big_endian=bswp)
        self.qmax = uu[:nrad]
        self.q0 = uu[nrad:]
        fort_read(file, 4, self.l, big_endian=bswp)
        self.alfac = np.reshape(fort_read(file, np.max(self.nq) * (nrad - nline), self.fl,
                                          big_endian=bswp), (nrad - nline, np.max(self.nq)))
        self.alfac = np.transpose(self.alfac, axes=(1, 0))
        fort_read(file, 4, self.l, big_endian=bswp)
        self.frq = np.reshape(fort_read(file, (np.max(self.nq) + 1) * (nrad - nline), self.fl,
                                        big_endian=bswp), (nrad - nline, (np.max(self.nq) + 1)))
        self.frq = np.transpose(self.frq, axes=(1, 0))
        fort_read(file, 4, self.l, big_endian=bswp)
        self.ind = fort_read(file, nrad, self.l, big_endian=bswp)
        fort_read(file, 4, self.l, big_endian=bswp)
        ee, em, hh, cc, bk, uu, hce, hc2, hck, ek, pi = fort_read(
            file, 11, self.fl, big_endian=bswp)
        self.ee = ee
        self.em = em
        self.hh = hh
        self.cc = cc
        self.bk = bk
        self.uu = uu
        self.hce = hce
        self.hc2 = hc2
        self.hck = hck
        self.ek = ek
        self.pi = pi
        # Converting to km
        self.height /= 1.e5
        self.widthx /= 1.e5
        self.widthy /= 1.e5
        file.close()
        return

    def read_pop(self, filename=None):
        ''' Reads out_pop file (n, nstar, totn)
            Now with np.fromfile instead of scipy.io.fread --Tiago, 20101004
        '''
        if filename is None:
            filename = self.dir + '/out_pop'
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * self.nk)
        #tmp   = fread(file,self.nx*self.ny*self.ndep*self.nk,dtype,dtype,bswp)
        self.n = np.transpose(np.reshape(tmp, (self.nk, self.ndep, self.ny, self.nx)),
                              axes=(3, 2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * self.nk)
        #tmp      =fread(file,self.nx*self.ny*self.ndep*self.nk,dtype,dtype,bswp)
        self.nstar = np.transpose(np.reshape(tmp, (self.nk, self.ndep, self.ny, self.nx)),
                                  axes=(3, 2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        #tmp      = fread(file,self.nx*self.ny*self.ndep,dtype,dtype,bswp)
        self.totn = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                 axes=(2, 1, 0))
        tmp = False  # just make sure garbage collection doesn't miss this.
        file.close()
        return


    def read_rtq(self, filename=None):
        ''' Reads out_rtq file (radiative transfer quantities) '''
        if filename is None:
            filename = self.dir + '/out_rtq'
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.xnorm = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                  axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.dnyd = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                 axes=(2, 1, 0))

        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.tau = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx *
                          self.ny * self.ndep * self.nline)
        self.adamp = np.transpose(np.reshape(tmp, (self.nline, self.ndep, self.ny,
                                                   self.nx)), axes=(3, 2, 1, 0))
        tmp = False  # just make sure garbage collection doesn't miss this.
        file.close()
        return

    def read_atm(self, filename=None):
        ''' Reads out_atm file (atmosphere file) '''
        if filename is None:
            filename = self.dir + '/out_atm'
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.ne = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                               axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.temp = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                 axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.vx = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                               axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.vy = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                               axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.vz = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                               axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.rho = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                axes=(2, 1, 0))
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * 6)
        self.nh = np.transpose(np.reshape(tmp, (6, self.ndep, self.ny, self.nx)),
                               axes=(3, 2, 1, 0))
        tmp = False  # just make sure garbage collection doesn't miss this.
        file.close()
        return

    def read_line(self, lid, angle=False, contrib=False, jny=False,
                  sfopa=False, dcint=True):
        ''' Reads transition dependent output. This includes intensities
        (disk-centre, angle, jny), contribution functions, source functions,
        opacities.'''
        from specutils.utils.wcs_utils import vac_to_air
        from astropy import units as u
        self.lid = lid
        suffix = '%03i_%03i' % (self.irad[lid], self.jrad[lid])
        intfile = self.dir + '/iv_' + suffix  # disk-centre intensity
        angfile = self.dir + '/ie_' + suffix  # angle intensity
        confile = self.dir + '/cf_' + suffix  # contribution functions
        jnyfile = self.dir + '/jn_' + suffix  # angle averaged intensity
        lsffile = self.dir + '/sl_' + suffix  # line source function
        tsffile = self.dir + '/st_' + suffix  # total source function
        xttfile = self.dir + '/xt_' + suffix  # total opacity
        # Construct wavelength scale, converted to air wavelengths
        self.wave = self.alamb[lid] * (1 - self.q[:self.nq[lid], lid] *
                                       self.qnorm / self.cc * 1e5)
        self.wave = vac_to_air(self.wave[::-1].astype('d') * u.angstrom,
                               method='Ciddor1996').value
        # Convert to nm
        self.wave /= 10.
        if self.v1d:  # for 1D case, duplicate velocities
            self.wave = np.concatenate((self.wave[:-1],
                                        -(self.wave[::-1] - 2 * self.wave[-1])))
        if dcint:
            self.read_int(intfile)
        if angle:
            self.read_ang(angfile)
        if contrib:
            self.read_con(confile)
        if jny:
            self.read_jny(jnyfile)
        if sfopa:
            self.read_lsf(lsffile)
            self.read_tsf(tsffile)
            self.read_xtt(xttfile)
        return

    def read_int(self, filename):
        ''' Reads the disk-centre intensity.
            Units are erg s-1 cm-2 Hz-1 ster-1. '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_int: no line set, run .read_line() first!')
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.nq[self.lid])
        self.int_xy = np.transpose(np.reshape(tmp, (self.nq[self.lid], self.ny,
                                                    self.nx)), axes=(2, 1, 0))
        # Spatially averaged disk-centre intensity
        self.int = np.mean(np.mean(self.int_xy, axis=0), axis=0)
        if self.v1d:  # for 1D case, duplicate values
            self.int = np.concatenate((self.int[::-1], self.int[1:]))
        file.close()
        return

    def read_ang(self, filename):
        ''' Reads the angle-dependent intensity.
            Units are erg s-1 cm-2 Hz-1 ster-1. '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_ang: no line set, run .read_line() first!')
        nqf = self.nq[self.lid]
        tmp = np.fromfile(file, dtype, self.nx * self.ny * nqf * self.nmu)
        self.angle_xy = np.transpose(np.reshape(tmp, (nqf, self.nmu, self.ny,
                                                self.nx)),axes=(1, 3, 2, 0))
        file.close()
        # Now must average int into unique mu angles.
        # hack to identify mu angles with 6 decimal places (avoids duplicates)
        aa = [float('%.6f' % (a)) for a in self.zmu]
        self.mus = np.sort(np.array(list(set(aa))))
        # Use indexing of [mu,nx,ny,nqf]
        tmp = np.zeros((len(self.mus), self.nx, self.ny, nqf))
        for i in range(len(self.mus)):
            diff = abs(self.zmu - self.mus[i])
            aa = self.angle_xy[np.where(diff < 1e-4)[0], :, :, :]
            tmp[i, :, :, :] = np.mean(aa, axis=0)
        # Now get the spatially averaged mu-dependent profiles
        self.angle = np.mean(np.mean(tmp, axis=1), axis=1)
        if self.v1d:  # for 1D case, duplicate values
            tmp = np.zeros((self.angle.shape[0], self.angle.shape[1] * 2 - 1))
            for i in range(len(self.mus)):
                tmp[i] = np.concatenate(
                    (self.angle[i, ::-1], self.angle[i, 1:]))
            self.angle = tmp.copy()
        return

    def read_con(self, filename):
        ''' Reads the contribution function for disk-centre intensity. '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_con: no line set, run .read_line() first!')
        nqf = self.nq[self.lid]
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * nqf)
        self.contrbi = np.transpose(np.reshape(tmp, (nqf, self.ndep, self.ny,
                                               self.nx)), axes=(3, 2, 1, 0))
        file.close()
        if self.v1d:  # for 1D case, duplicate values
            # Transpose to concatenate on first index
            self.rawcontrbi = self.contrbi.copy()
            self.contrbi = np.transpose(self.contrbi)
            self.contrbi = np.transpose(np.concatenate((self.contrbi[::-1],
                                                        self.contrbi[1:])))
        # calculate contribution function for the equivalent width (sum)
        self.contrib_eqw_xy = np.sum(self.contrbi, axis=-1)
        self.contrib_eqw = np.sum(
            np.mean(np.mean(self.contrbi, axis=0), axis=0), axis=1)
        return

    def read_jny(self, filename):
        ''' Reads the angle-averaged intensity. '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_jny: no line set, run .read_line() first!')
        nqf = self.nq[self.lid]
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * nqf)
        self.jny = np.transpose(np.reshape(tmp, (nqf, self.ndep, self.ny, self.nx)),
                                axes=(3, 2, 1, 0))
        file.close()
        if self.v1d:  # for 1D case, duplicate values
            # Transpose to concatenate on first index
            self.rawjny = self.jny.copy()
            self.jny = np.transpose(self.jny)
            self.jny = np.transpose(np.concatenate(
                (self.jny[::-1], self.jny[1:])))
        return

    def read_xtt(self, filename):
        ''' Reads the total opacity (line + continuum) '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_xtt: no line set, run .read_line() first!')
        nqf = self.nq[self.lid]
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * nqf)
        self.xt = np.transpose(np.reshape(tmp, (nqf, self.ndep, self.ny,
                                                self.nx)), axes=(3, 2, 1, 0))
        file.close()
        if self.v1d:  # for 1D case, duplicate values
            # Transpose to concatenate on first index
            self.rawxt = self.xt.copy()
            self.xt = np.transpose(self.xt)
            self.xt = np.transpose(np.concatenate(
                (self.xt[::-1], self.xt[1:])))
        return

    def read_tsf(self, filename):
        ''' Reads the total source function (line + continuum) '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        if 'lid' not in dir(self):
            raise ValueError('read_tsf: no line set, run .read_line() first!')
        nqf = self.nq[self.lid]
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep * nqf)
        self.st = np.transpose(np.reshape(tmp, (nqf, self.ndep, self.ny,
                                                self.nx)), axes=(3, 2, 1, 0))
        file.close()
        if self.v1d:  # for 1D case, duplicate values
            # Transpose to concatenate on first index
            self.rawst = self.st.copy()
            self.st = np.transpose(self.st)
            self.st = np.transpose(np.concatenate(
                (self.st[::-1], self.st[1:])))
        return

    def read_lsf(self, filename):
        ''' Reads the line source function (constant over the line profile) '''
        file = open(filename, 'r')
        if self.bswp:
            dtype = '>' + self.dtype
        else:
            dtype = '<' + self.dtype
        tmp = np.fromfile(file, dtype, self.nx * self.ny * self.ndep)
        self.sl = np.transpose(np.reshape(tmp, (self.ndep, self.ny, self.nx)),
                                          axes=(2, 1, 0))
        file.close()
        return


class Multi3dout:
    """
    For latest Multi3d (RH solver).
    OBSOLETE. WILL NEED TO SWAP BY LATEST PACKAGE FROM JOHAN/JORRIT.
    Also, does not work.
    """
    def __init__(self, fdir='.', precision='f4', verbose=True):
        self.fdir = fdir
        self.verbose = verbose
        self.read_input()
        self.read_par()

    def read_input(self, filename=None):
        if filename is None:
            filename = os.path.join(self.fdir, 'multi3d.input')
        try:
            lines = [line.strip() for line in open(filename)]  # Read into list
            if self.verbose:
                print(('--- Read %s file.' % filename))
        except Exception as e:
            print(e)
            return
        ll = []
        size = len(lines)
        for i in range(size):
            head, sep, tail = lines[i].partition(';')
            ll.append(head)
        lll = [_f for _f in ll if _f]  # Remove blank lines
        self.input = {}
        size = len(ll)
        for i in range(size):
            head, sep, tail = ll[i].partition("=")
            tail = tail.strip()
            head = head.strip()
            # Checks which type the values are
            try:
                tail = int(tail)
            except:
                try:
                    tail = float(tail)
                except:
                    pass
            if(head == "muxout"):
                temp = tail.split()
                self.muxout = []
                for i in range(len(temp)):
                    temp_var = str(temp[i]).replace("'", "")
                    self.muxout.append("+" + str(temp[i]).replace("'", ""))
            elif(head == "muyout"):
                temp = tail.split()
                self.muyout = []
                for i in range(len(temp)):
                    self.muyout.append("+" + str(temp[i]).replace("'", ""))
            elif(head == "muzout"):
                temp = tail.split()
                self.muzout = []
                for i in range(len(temp)):
                    self.muzout.append("+" + str(temp[i]).replace("'", ""))
            else:
                self.input[head] = tail
        try:
            self.nx = self.input["nx"]
        except:
            self.nx = None
        try:
            self.ny = self.input["nx"]
        except:
            self.ny = None
        try:
            self.nz = self.input["nz"]
        except:
            self.nz = None

    def read_par(self, filename=None, itype="int32", ftype="float64"):
        if filename is None:
            filename = os.path.join(self.fdir, 'out_par')
        fobj = open(filename, "rb")
        data = np.fromfile(fobj, dtype=itype, count=3 * 4 + 1)
        self.nmu = data[1]
        nx = data[4]
        ny = data[7]
        nz = data[10]
        try:
            assert nx == self.nx
            assert ny == self.ny
            assert nz == self.nz
        except AssertionError:
            print("read_par: inconsistent dimensions, aborting.")
            return

        def _read_fort_rec(f, dtype="int32", count=2):
            __ = np.fromfile(f, dtype=dtype, count=count)

        self.widthx = np.fromfile(fobj, dtype=ftype, count=nx)
        _read_fort_rec(fobj)
        self.widthy = np.fromfile(fobj, dtype="float64", count=ny)
        _read_fort_rec(fobj)
        self.height = np.fromfile(fobj, dtype="float64", count=nz)
        _read_fort_rec(fobj)
        self.mux = np.fromfile(fobj, dtype="float64", count=self.nmu)
        _read_fort_rec(fobj)
        self.muy = np.fromfile(fobj, dtype="float64", count=self.nmu)
        _read_fort_rec(fobj)
        self.muz = np.fromfile(fobj, dtype="float64", count=self.nmu)
        _read_fort_rec(fobj)
        self.wmu = np.fromfile(fobj, dtype="float64", count=self.nmu)
        # Record crap from Fortran
        data = np.fromfile(fobj, dtype="int32", count=1)
        data = np.fromfile(fobj, dtype="int32", count=3 * 3 + 1)
        self.nnu = data[1]
        self.maxac = data[4]
        self.maxal = data[7]
        data = np.fromfile(fobj, dtype="float64", count=self.nnu)
        self.nu = data[0:self.maxac * self.nnu]
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="float64", count=self.nnu)
        self.wnu = data[0:self.maxac * self.nnu]
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.maxac * self.nnu)
        self.ac = data[0:self.maxac *
                       self.nnu].reshape((self.maxac, self.nnu), order="FORTRAN")
        # Reading starts to be wrong from this point on
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.maxal * self.nnu)
        self.al = data[0:self.maxal *
                       self.nnu].reshape((self.maxal, self.nnu), order="FORTRAN")
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.nnu)
        self.nac = data[0:self.nnu].reshape((self.nnu), order="FORTRAN")
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.nnu)
        self.nal = data[0:self.nnu].reshape((self.nnu), order="FORTRAN")
        # atom
        data = np.fromfile(fobj, dtype="int32", count=1)  # Fortran record
        data = np.fromfile(fobj, dtype="int32", count=3 * 5 + 1)
        self.nrad = data[1]
        self.nrfix = data[4]
        self.ncont = data[7]
        self.nline = data[10]
        self.nlevel = data[13]
        # Record crap from Fortran
        data = np.fromfile(fobj, dtype="a20", count=1)
        self.id = str(data[0]).replace(" ", "")
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="a20", count=1)
        self.crout = str(data[0]).replace(" ", "")
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="a20", count=self.nlevel)
        self.label = [s.rstrip() for s in data]  # data.replace(" ", "")
        _read_fort_rec(fobj)
        ion = np.fromfile(fobj, dtype="int32", count=self.nlevel)
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.nlevel**2)
        self.ilin = data[0:self.nlevel **
                         2].reshape((self.nlevel, self.nlevel), order="FORTRAN")
        _read_fort_rec(fobj)
        data = np.fromfile(fobj, dtype="int32", count=self.nlevel**2)
        self.icon = data[0:self.nlevel **
                         2].reshape((self.nlevel, self.nlevel), order="FORTRAN")
        _read_fort_rec(fobj)
        self.abnd = np.fromfile(fobj, dtype="float64", count=1)[0]
        _read_fort_rec(fobj)
        self.awgt = np.fromfile(fobj, dtype="float64", count=1)[0]
        _read_fort_rec(fobj)
        self.ev = np.fromfile(fobj, dtype="float64", count=self.nlevel)
        _read_fort_rec(fobj)
        self.g = np.fromfile(fobj, dtype="float64", count=self.nlevel)
        # read cont info
        self.bf_type = []
        self.cont_j = []
        self.cont_i = []
        self.cont_ntrans = []
        self.cont_nnu = []
        self.cont_ired = []
        self.cont_iblue = []
        self.cont_nu0 = []
        self.cont_numax = []
        self.cont_alpha0 = []
        self.cont_alpha = []
        self.cont_nu = []
        self.cont_wnu = []
        for i in range(self.ncont):
            _read_fort_rec(fobj)
            self.bf_type.append(np.fromfile(fobj, dtype="a20", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_j.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_i.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_ntrans.append(np.fromfile(
                fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_nnu.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_ired.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_iblue.append(np.fromfile(
                fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_nu0.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_numax.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_alpha0.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.cont_alpha.append(np.fromfile(
                fobj, dtype="float64", count=self.cont_nnu[i]))
            _read_fort_rec(fobj)
            self.cont_nu.append(np.fromfile(
                fobj, dtype="float64", count=self.cont_nnu[i]))
            _read_fort_rec(fobj)
            self.cont_wnu.append(np.fromfile(
                fobj, dtype="float64", count=self.cont_nnu[i]))

        self.bf_type = [s.rstrip() for s in self.bf_type]  # remove whitespaces
        # Read line info
        self.line_profiletype = []
        self.line_ga = []
        self.line_gw = []
        self.line_gq = []
        self.line_lambda0 = []
        self.line_nu0 = []
        self.line_Aij = []
        self.line_Bji = []
        self.line_Bij = []
        self.line_f = []
        self.line_qmax = []
        self.line_Grat = []
        self.line_ntrans = []
        self.line_j = []
        self.line_i = []
        self.line_nnu = []
        self.line_ired = []
        self.line_iblue = []
        self.line_nu = []
        self.line_q = []
        self.line_wnu = []
        self.line_wq = []
        for i in range(self.nline):
            _read_fort_rec(fobj)
            self.line_profiletype.append(
                np.fromfile(fobj, dtype="a20", count=1)[0])
            #
            # TODO: Unknown why the record header is much bigger here!
            # Usually its just count=2, but here it was count=15.... :(
            # - Johan PB
            data = np.fromfile(fobj, dtype="int32", count=15)
            self.line_ga.append(np.fromfile(fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_gw.append(np.fromfile(fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_gq.append(np.fromfile(fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_lambda0.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_nu0.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_Aij.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_Bji.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_Bij.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_f.append(np.fromfile(fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_qmax.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_Grat.append(np.fromfile(
                fobj, dtype="float64", count=1)[0])
            _read_fort_rec(fobj)
            self.line_ntrans.append(np.fromfile(
                fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_j.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_i.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_nnu.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_ired.append(np.fromfile(fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_iblue.append(np.fromfile(
                fobj, dtype="int32", count=1)[0])
            _read_fort_rec(fobj)
            self.line_nu.append(np.fromfile(
                fobj, dtype="float64", count=self.line_nnu[i]))
            _read_fort_rec(fobj)
            self.line_q.append(np.fromfile(
                fobj, dtype="float64", count=self.line_nnu[i]))
            _read_fort_rec(fobj)
            self.line_wnu.append(np.fromfile(
                fobj, dtype="float64", count=self.line_nnu[i]))
            _read_fort_rec(fobj)
            self.line_wq.append(np.fromfile(
                fobj, dtype="float64", count=self.line_nnu[i]))
        self.line_profiletype = [
            s.rstrip() for s in self.line_profiletype]  # remove whitespaces
        fobj.close()

    def read_ie(self, filename, startf=None, dtype="f", nnu=101):
        # startf should be 115 for Halpha
        if startf is None:
            raise NotImplementedError("Must specify starting frequency index.")
        offset = self.nx * self.ny * startf * np.dtype(dtype).itemsize
        self.ie = np.memmap(filename, dtype=dtype, mode="r", offset=offset,
                            shape=(nnu, self.nx, self.ny))


class Multi3dAtmos:
    def __init__(self, infile, nx, ny, nz, mode="r"):
        if os.path.isfile(infile) or (mode == "w+"):
            self.open_atmos(infile, nx, ny, nz, mode=mode)

    def open_atmos(self, infile, nx, ny, nz, nhydr=6, mode="r", dp=False,
                   big_endian=False, read_nh=False, read_vturb=False):
        """
        Reads/writes multi3d atmosphere into parent object.

        Parameters
        ----------
        infile : string
            Name of file to read.
        """
        dtype = ["<", ">"][big_endian] + ["f4", "f8"][dp]
        ntot = nx * ny * nz * np.dtype(dtype).itemsize
        mm = mode
        # bullshit fort_read with header/footer not needed here.
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
