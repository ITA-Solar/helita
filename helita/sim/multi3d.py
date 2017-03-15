"""
Set of routines to read output from multi3d
"""
from ..io.fio import fort_read, fort_write
from ..utils.waveconv import waveconv
import numpy as np


class out3d:
    def __init__(self, dir='.', bswp=False, hdrlen=4):
        ''' Reads multi3d output data in out3d. '''
        self.read(dir, bswp, hdrlen)
        return

    def read(self, dir, bswp, hdrlen):
        ''' Reads out3d file '''
        file = open(dir + '/out3d', 'r')
        fort_read(file, 4, 'l', big_endian=bswp)  # Skip file types and sizes
        nk, nrad, nline, nrfix, nmu, nx, ny, ndep = fort_read(file, 8, 'l',
                                                              big_endian=bswp)
        self.nk = nk
        self.nrad = nrad
        self.nline = nline
        self.nrfix = nrfix
        self.nmu = nmu
        self.nx = nx
        self.ny = ny
        self.ndep = ndep
        fort_read(file, 4, 'l', big_endian=bswp)
        uu = fort_read(file, nx + ny + ndep, 'f', big_endian=bswp)
        self.widthx = uu[:nx]
        self.widthy = uu[nx:nx + ny]
        self.height = uu[nx + ny:]
        fort_read(file, 4, 'l', big_endian=bswp)
        uu = fort_read(file, 4 * nmu, 'f', big_endian=bswp)
        self.xmu = uu[:nmu]
        self.ymu = uu[nmu:2 * nmu]
        self.zmu = uu[2 * nmu:3 * nmu]
        self.wmu = uu[3 * nmu:]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.nq = fort_read(file, nrad, 'l', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        # Read directly to string
        self.atomid = file.read(hdrlen * 3)[hdrlen:hdrlen * 2]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.abnd, self.awgt, self.grph, self.qnorm = fort_read(
            file, 4, 'f', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        self.ev = fort_read(file, nk, 'f', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        self.g = fort_read(file, nk, 'f', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        self.label = np.zeros(nk, dtype='S20')
        file.read(hdrlen)
        for i in range(nk):
            self.label[i] = file.read(20)
        file.read(hdrlen)
        fort_read(file, 4, 'l', big_endian=bswp)
        self.ion = fort_read(file, nk, 'l', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        uu = fort_read(file, nrad * 2, 'l', big_endian=bswp)
        self.jrad = uu[:nrad]
        self.irad = uu[nrad:]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.krad = np.reshape(
            fort_read(file, nk * nk, 'l', big_endian=bswp), (nk, nk))
        fort_read(file, 4, 'l', big_endian=bswp)
        uu = fort_read(file, nrad * 5 + nline, 'f', big_endian=bswp)
        self.f = uu[:nrad]
        self.ga = uu[nrad:nrad * 2]
        self.gw = uu[nrad * 2:nrad * 3]
        self.gq = uu[nrad * 3:nrad * 4]
        self.alamb = uu[nrad * 4:nrad * 5]
        self.a = uu[nrad * 5:]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.ktrans = fort_read(file, nrad, 'l', big_endian=bswp)
        fort_read(file, 4, 'l', big_endian=bswp)
        self.b = np.reshape(
            fort_read(file, nk * nk, 'f', big_endian=bswp), (nk, nk))
        if nrfix >= 1:
            fort_read(file, 4, 'l', big_endian=bswp)
            uu = fort_read(file, nrfix * 4, 'l', big_endian=bswp)
            self.jfx = uu[:nrfix]
            self.ifx = uu[nrfix:nrfix * 2]
            self.ipho = uu[nrfix * 2:nrfix * 3]
            self.itrad = uu[nrfix * 3:]
            fort_read(file, 4, 'l', big_endian=bswp)
            uu = fort_read(file, nrfix * 2, 'f', big_endian=bswp)
            self.a0 = uu[:nrfix]
            self.trad = uu[nrfix:]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.dnyd = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                         big_endian=bswp), (ndep, ny, nx))
        # Fix axis order to be the same as in IDL:
        self.dnyd = np.transpose(self.dnyd, axes=(2, 1, 0))

        self.adamp = np.zeros((nx, ny, ndep, nline), dtype='Float32')
        for l in range(nline):
            fort_read(file, 4, 'l', big_endian=bswp)
            tmp = fort_read(file, nx * ny * ndep, 'f', big_endian=bswp)
            self.adamp[:, :, :, l] = np.transpose(np.reshape(tmp, (ndep, ny, nx)),
                                                  axes=(2, 1, 0))
        self.n = np.zeros((nx, ny, ndep, nk), dtype='Float32')
        for k in range(nk):
            fort_read(file, 4, 'l', big_endian=bswp)
            tmp = fort_read(file, nx * ny * ndep, 'f', big_endian=bswp)
            self.n[:, :, :, k] = np.transpose(
                np.reshape(tmp, (ndep, ny, nx)), axes=(2, 1, 0))
        self.nstar = np.zeros((nx, ny, ndep, nk), dtype='Float32')
        for k in range(nk):
            fort_read(file, 4, 'l', big_endian=bswp)
            tmp = fort_read(file, nx * ny * ndep, 'f', big_endian=bswp)
            self.nstar[:, :, :, k] = np.transpose(np.reshape(tmp, (ndep, ny, nx)),
                                                  axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.totn = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                         big_endian=bswp), (ndep, ny, nx))
        self.totn = np.transpose(self.totn, axes=(2, 1, 0))

        fort_read(file, 4, 'l', big_endian=bswp)
        self.q = np.reshape(fort_read(file, np.max(self.nq) * nrad, 'f',
                                      big_endian=bswp), (nrad, np.max(self.nq)))
        self.q = np.transpose(self.q, axes=(1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        uu = fort_read(file, nrad * 2, 'l', big_endian=bswp)
        self.qmax = uu[:nrad]
        self.q0 = uu[nrad:]
        fort_read(file, 4, 'l', big_endian=bswp)
        self.alfac = np.reshape(fort_read(file, np.max(self.nq) * (nrad - nline),
                                          'f', big_endian=bswp),
                                (nrad - nline, np.max(self.nq)))
        self.alfac = np.transpose(self.alfac, axes=(1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.frq = np.reshape(fort_read(file, (np.max(self.nq) + 1) * (nrad - nline),
                                        'f', big_endian=bswp),
                              (nrad - nline, (np.max(self.nq) + 1)))
        self.frq = np.transpose(self.frq, axes=(1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.ind = fort_read(file, nrad, 'l', big_endian=bswp)

        fort_read(file, 4, 'l', big_endian=bswp)
        self.temp = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                         big_endian=bswp), (ndep, ny, nx))
        self.temp = np.transpose(self.temp, axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.nne = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                        big_endian=bswp), (ndep, ny, nx))
        self.nne = np.transpose(self.nne, axes=(2, 1, 0))
        self.nh = np.zeros((nx, ny, ndep, 6), dtype='Float32')
        for k in range(6):
            fort_read(file, 4, 'l', big_endian=bswp)
            tmp = fort_read(file, nx * ny * ndep, 'f', big_endian=bswp)
            self.nh[:, :, :, k] = np.transpose(np.reshape(tmp, (ndep, ny, nx)),
                                              axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.rho = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                        big_endian=bswp), (ndep, ny, nx))
        self.rho = np.transpose(self.rho, axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.vx = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                       big_endian=bswp), (ndep, ny, nx))
        self.vx = np.transpose(self.vx, axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.vy = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                       big_endian=bswp), (ndep, ny, nx))
        self.vy = np.transpose(self.vy, axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        self.vz = np.reshape(fort_read(file, nx * ny * ndep, 'f',
                                       big_endian=bswp), (ndep, ny, nx))
        self.vz = np.transpose(self.vz, axes=(2, 1, 0))
        fort_read(file, 4, 'l', big_endian=bswp)
        ee, em, hh, cc, bk, uu, hce, hc2, hck, ek, pi = fort_read(
            file, 11, 'f', big_endian=bswp)
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


class m3dprof:
    def __init__(self, lid, dir='.', bswp=False, irc=8, angle=True,
                 contrib=False, o3dswp=False, v1d=False):
        ''' Reads line profiles and contribution functions from multi3d '''
        self.bswp = bswp
        self.v1d = v1d
        self.irc = irc  # record length? parameter in init_const.f
        if not o3dswp:
            o3d = out3d(dir, self.bswp)
        else:
            o3d = out3d(dir, not self.bswp)
        self.o3d = o3d
        # Get transition id
        let = [l for l in 'abcdefghijklmnopqrstuvwxyz']
        if lid > o3d.nrad - 1:
            print('(EEE) m3dprof: lid too high! (max %i)' % o3d.nrad)
            return
        self.nqf = o3d.nq[lid]
        l1 = o3d.irad[lid]
        l2 = o3d.jrad[lid]
        suffix = let[l1 - 1] + let[l2 - 1]
        imufile = dir + '/imu_' + suffix
        intfile = dir + '/int_' + suffix
        confile = dir + '/con_' + suffix
        # Construct wavelength scale, converted to air wavelengths
        self.wave = o3d.alamb[
            lid] * (1 - o3d.q[:o3d.nq[lid], lid] * o3d.qnorm / o3d.cc * 1e5)
        # Better use waveconv_regner (though not the best), because it's
        # the same that is used internally in lte.x
        self.wave = waveconv_regner(
            self.wave[::-1].astype('d'), mode='vac2air')
        # Convert to nm
        self.wave /= 10.
        if self.v1d:  # for 1D case, duplicate velocities
            self.wave = np.concatenate((self.wave[:-1],
                                       -(self.wave[::-1] - 2 * self.wave[-1])))
        # Read several quantities
        self.read_int(intfile)
        if angle:   # angle-dependent intensities
            self.read_imu(imufile)
        if contrib:  # contribution functions
            self.read_con(confile)
        return

    def read_int(self, filename):
        ''' Reads vertical (disk-centre) intensity '''
        file = open(filename, 'r')
        o3d = self.o3d
        fort_read(file, 4, 'l', big_endian=self.bswp)
        # swap nx with ny (to give same order as lte.x)
        self.int_xy = np.transpose(np.reshape(fort_read(file,
                                                o3d.nx * o3d.ny * self.nqf, 'f',
                                                big_endian=self.bswp),
                                   (o3d.ny, o3d.nx, self.nqf)), [1, 0, 2])
        # Spatially averaged disk-centre intensity
        self.int = np.mean(np.mean(self.int_xy, axis=0), axis=0)
        if self.v1d:  # for 1D case, duplicate values
            self.int = np.concatenate((self.int[::-1], self.int[1:]))
        file.close()
        return

    def read_imu(self, filename):
        ''' Reads mu-dependent intensity '''
        # New definition, use indexing of [mu,nx,ny,nqf]:
        o3d = self.o3d
        self.rawangle = np.zeros((o3d.nmu, o3d.nx, o3d.ny, self.nqf))
        file = open(filename, 'r')
        if self.bswp:
            dt = '>d'
        else:
            dt = '<d'
        aa = np.fromfile(file, dt, self.nqf * o3d.nx *
                        o3d.ny * o3d.nmu * self.irc / 2)
        aa = np.transpose(np.reshape(aa, (o3d.nmu * self.irc / 2, o3d.ny,
                                          o3d.nx, self.nqf)), axes=(3, 2, 1, 0))
        # New definition, use indexing of [mu,nx,ny,nqf]
        for i in range(o3d.nmu):
            self.rawangle[i] = np.transpose(aa[:, ..., i * self.irc / 2],
                                            axes=(1, 2, 0))
        file.close()
        # Now must average int into unique mu angles
        self.mus = np.sort(np.array(list(set(o3d.zmu))))
        # New definition, use indexing of [mu,nx,ny,nqf]
        self.angle_xy = np.zeros((len(self.mus), o3d.nx, o3d.ny, self.nqf))
        for i in range(len(self.mus)):
            aa = self.rawangle[np.where(o3d.zmu == self.mus[i])[0], :, :, :]
            self.angle_xy[i, :, :, :] = np.mean(aa, axis=0)
        # Now get the spatially averaged mu-dependent profiles
        self.angle = np.mean(np.mean(self.angle_xy, axis=1), axis=1)
        if self.v1d:  # for 1D case, duplicate values
            self.angle2 = np.zeros(
                (self.angle.shape[0], self.angle.shape[1] * 2 - 1))
            for i in range(len(self.mus)):
                self.angle2[i] = np.concatenate(
                    (self.angle[i, ::-1], self.angle[i, 1:]))
            self.angle = self.angle2
        return

    def read_con(self, filename):
        '''  Reads contribution functions:

        self.contribi[nx,ny,nz,nqf] : contrib. function for intensity (mu=1)
        self.contribi[nx,ny,nz,nz,nqf] : contrib. function for flux
        self.contribr[nx,ny,nz,nz,nqf] : contrib. function for rel. intensity

        and the spatially averaged and wavelength summed contrib. functions:

        self.contrib_int[nz] : for intensity
        self.contrib_flx[nz] : for flux
        self.contrib_rel[nz] : for relative intensity

        '''
        file = open(filename, 'r')
        o3d = self.o3d
        fort_read(file, 4, 'l', big_endian=self.bswp)
        fort_read(file, 4, 'l', big_endian=self.bswp)
        self.contrbi = np.zeros((o3d.nx, o3d.ny, o3d.ndep, self.nqf))
        self.contrbf = np.zeros((o3d.nx, o3d.ny, o3d.ndep, self.nqf))
        self.contrbr = np.zeros((o3d.nx, o3d.ny, o3d.ndep, self.nqf))
        for j in range(o3d.ny):
            for i in range(o3d.nx):
                fort_read(file, 4, 'l', big_endian=self.bswp)
                uu = fort_read(file, 3 * self.nqf * o3d.ndep,
                               'f', big_endian=self.bswp)
                contrbi, contrbf, contrbr = np.reshape(
                    uu, (3, self.nqf * o3d.ndep))
                self.contrbi[i, j] = np.reshape(contrbi, (o3d.ndep, self.nqf))
                self.contrbf[i, j] = np.reshape(contrbf, (o3d.ndep, self.nqf))
                self.contrbr[i, j] = np.reshape(contrbr, (o3d.ndep, self.nqf))
        # Spatially averaged and wavelength summed contribution functions
        self.contrib_int = np.sum(
            np.mean(np.mean(self.contrbi, axis=0), axis=0), axis=1)
        self.contrib_flx = np.sum(
            np.mean(np.mean(self.contrbf, axis=0), axis=0), axis=1)
        self.contrib_rel = np.sum(
            np.mean(np.mean(self.contrbr, axis=0), axis=0), axis=1)
        file.close()
        return

    def read_xtra(self, filename, dim=[], dtype='d', it=1):
        '''
        Reads extra information written to filename. Use for testing
        purposes (eg., read rotated source function datacubes). Dimensions
        of the array are given in dim, data type in dtype. If quantity is
        written for several iterations, then set it to the number of iterations
        '''
        file = open(filename, 'r')
        if not dim:
            raise ValueError
        if it == 1:
            dim = np.array(dim)[::-1]
            xtra = np.reshape(fort_read(file, np.prod(dim), dtype,
                                        big_endian=self.bswp), dim)
            # Invert axes
            xtra = np.transpose(xtra, axes=list(range(xtra.ndim))[::-1])
        else:
            xtra = np.empty((it,) + dim, dtype=dtype)
            dim = np.array(dim)[::-1]
            for i in range(it):
                tmp = np.reshape(fort_read(file, np.prod(dim), dtype,
                                           big_endian=self.bswp), dim)
                xtra[i] = np.transpose(tmp, axes=list(range(tmp.ndim))[::-1])
        file.close()
        return xtra

    def read_n(self, filename='n', it=None):
        ''' Reads the population file n, for all the iterations.
            Only useful for debug'''
        if not os.path.isfile('olog'):
            raise IOError('Could not find olog file. Check path?')
        elif not os.path.isfile(filename):
            raise IOError('Could not find n file. Check path?')
        nx = self.o3d.nx
        ny = self.o3d.ny
        nz = self.o3d.ndep
        nk = self.o3d.nk
        if it is None:
            it = read_iter('olog')
        nf = open(filename, 'r')
        print('--- Reading %i iterations from n file' % it)
        self.fulln = np.empty(self.o3d.np.shape + (it,), dtype='Float64')
        for i in range(it):
            tmp = fort_read(nf, (nx + 1) * (ny + 1) * nz *
                            nk, 'd', big_endian=self.bswp)
            self.fulln[:, :, :, :, i] = np.transpose(np.reshape(tmp, (nk, nz, ny + 1, nx + 1)),
                                                    axes=(3, 2, 1, 0))[:-1, :-1]
        nf.close()
        return


############################################################################
################               TOOLS                ########################
############################################################################
def params_equal(p1, p2, exclude=[None]):
    ''' Checks if parameter dictionaries are the same
    (given the exclude list). Returns True if they match, False otherwise.'''
    for a in list(p1.keys()):
        if a not in exclude:
            try:
                if p1[a] != p2[a]:
                    print('--- check_params: unmatching parameters:')
                    print(a, p1[a], p2[a])
                    return False
            except ValueError:  # tweak for arrays
                if p1[a].all() != p2[a].all():
                    print('--- check_params: unmatching parameters:')
                    print(a, p1[a], p2[a])
                    return False
    return True


def mkatom_sh(sh, sh1file, outfile):
    ''' Builds a multi3d atom file for an arbitrary S_H, using an existing
        atom file for S_H=1.

        IN: sh (float), sh1file (file with S_H=1), outfile

    --Tiago, 20090424
    '''
    import os

    fout = open(outfile, 'w')
    s1 = '* COLLISIONAL EXCITATIONS AND IONIZATIONS BY HI (DRAWIN *   1.00000)'
    s2 = '* COLLISIONAL IONIZATIONS BY HI (DRAWIN *   1.00000)'
    act = False
    for line in file(sh1file):
        if line in [s1 + '\n', s2 + '\n']:
            line = line[:-12] + ' %.3f )\n' % sh
        if act:
            # Multiply collision rates by S_H
            uu = line.split()
            line = '%4i%4i' % (int(uu[0]), int(uu[1]))
            for i in range(len(uu) - 2):
                line += ' %.2E' % (float(uu[2 + i]) * sh)
            line += '\n'
        if line[0] != '*' and (line.find('CH ') >= 0 or line.find('CHI') >= 0):
            act = True
        else:
            act = False
        fout.write(line)
    fout.close()
    print('*** Wrote S_H=%.3f in %s' % (sh, outfile))
    return


def read_iter(ologfile):
    """
    Hack to read how many iterations were runp. Parses olog to extract it.
    """
    it = 0
    for line in file(ologfile):
        if line[:17] == ' *******ITERATION':
            it = int(line.split()[-1])
    return it
