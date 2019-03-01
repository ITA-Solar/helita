import os
import numpy as np


class MuramAtmos:
    """
    Class to read MURaM atmosphere

    Parameters
    ----------
    fdir : str, optional
        Directory with snapshots.
    template : str, optional
        Template for snapshot number.
    verbose : bool, optional
        If True, will print more information.
    dtype : str or numpy.dtype, optional
        Datatype of read data.
    big_endian : bool, optional
        Endianness of output file. Default is False (little endian).
    prim : bool, optional
        Set to True if moments are written instead of velocities.
    """
    def __init__(self, fdir='.', template=".020000", verbose=True, dtype='f4',
                 big_endian=False, prim=False):
        self.prim = prim
        self.fdir = fdir
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        self.read_header("%s/Header%s" % (fdir, template))
        self.read_atmos(fdir, template)
        self.snap = 0

    def read_header(self, headerfile):
        tmp = np.loadtxt(headerfile)
        self.nx, self.nz, self.ny = tmp[:3].astype("i")
        self.dx, self.dz, self.dy, self.time, self.dt, self.vamax = tmp[3:9]
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.z = np.arange(self.nz) * self.dz

    def read_atmos(self, fdir, template):
        ashape = (self.nx, self.nz, self.ny)
        file_T = "%s/eosT%s" % (fdir, template)
        if os.path.isfile(file_T):
            self.tg = np.memmap(file_T, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_press = "%s/eosP%s" % (fdir, template)
        if os.path.isfile(file_press):
            self.pressure = np.memmap(file_press, mode="r", shape=ashape,
                                      dtype=self.dtype,
                                      order="F").transpose((0, 2, 1))
        file_rho = "%s/result_prim_0%s" % (fdir, template)
        if os.path.isfile(file_rho):
            self.rho = np.memmap(file_rho, mode="r", shape=ashape,
                                 dtype=self.dtype,
                                 order="F").transpose((0, 2, 1))
        file_vx = "%s/result_prim_1%s" % (fdir, template)
        if os.path.isfile(file_vx):
            self.vx = np.memmap(file_vx, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_vz = "%s/result_prim_2%s" % (fdir, template)
        if os.path.isfile(file_vz):
            self.vz = np.memmap(file_vz, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_vy = "%s/result_prim_3%s" % (fdir, template)
        if os.path.isfile(file_vy):
            self.vy = np.memmap(file_vy, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_ei = "%s/result_prim_4%s" % (fdir, template)
        if os.path.isfile(file_ei):
            self.ei = np.memmap(file_ei, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_Bx = "%s/result_prim_5%s" % (fdir, template)
        if os.path.isfile(file_Bx):
            self.bx = np.memmap(file_Bx, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_Bz = "%s/result_prim_6%s" % (fdir, template)
        if os.path.isfile(file_Bz):
            self.bz = np.memmap(file_Bz, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_By = "%s/result_prim_7%s" % (fdir, template)
        if os.path.isfile(file_By):
            self.by = np.memmap(file_By, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F").transpose((0, 2, 1))
        file_tau = "%s/tau%s" % (fdir, template)
        if os.path.isfile(file_tau):
            self.tau = np.memmap(file_tau, mode="r", shape=ashape,
                                 dtype=self.dtype,
                                 order="F").transpose((0, 2, 1))
        file_Qtot = "%s/Qtot%s" % (fdir, template)
        if os.path.isfile(file_Qtot):
            self.qtot = np.memmap(file_Qtot, mode="r", shape=ashape,
                                  dtype=self.dtype,
                                  order="F").transpose((0, 2, 1))
        bfact = np.sqrt(4 * np.pi)
        self.bx = self.bx * bfact
        self.by = self.by * bfact
        self.bz = self.bz * bfact
        # from moments to velocities
        if self.prim:
            self.vx /= self.rho
            self.vy /= self.rho
            self.vz /= self.rho

    def write_rh15d(self, outfile, desc=None, append=True, writeB=False,
                    sx=slice(None), sy=slice(None), sz=slice(None),
                    wght_per_h=1.4271):
        ''' Writes RH 1.5D NetCDF snapshot '''
        from . import rh15d
        import scipy.constants as ct
        from .bifrost import Rhoeetab
        # unit conversion to SI
        ul = 1.e-2  # to metres
        ur = 1.e3   # from g/cm^3 to kg/m^3
        ut = 1.     # to seconds
        uv = ul / ut
        ub = 1.e-4  # to Tesla
        ue = 1.      # to erg/g
        # slicing and unit conversion (default slice of None selects all)
        if sx != slice(None):
            sx = slice(sx[0], sx[1], sx[2])
        if sy != slice(None):
            sy = slice(sy[0], sy[1], sy[2])
        if sz != slice(None):
            sz = slice(sz[0], sz[1], sz[2])
        print('Slicing and unit conversion...')
        temp = self.tg[sx, sy, sz]
        rho = self.rho[sx, sy, sz]
        rho = rho * ur
        if writeB:
            Bx = self.bx[sx, sy, sz]
            By = self.by[sx, sy, sz]
            Bz = self.bz[sx, sy, sz]
            Bx = Bx * ub
            By = By * ub
            Bz = Bz * ub
        else:
            Bx, By, Bz = [None] * 3
        vx = self.vx[sx, sy, sz] * ul
        vy = self.vy[sx, sy, sz] * ul
        vz = self.vz[sx, sy, sz] * ul
        x = self.x[sx] * ul
        y = self.y[sy] * ul
        z = self.z[sz] * ul
        # convert from rho to H atoms
        nh = rho / (wght_per_h * ct.atomic_mass)  # from rho to nH
        # description
        if desc is None:
            desc = 'MURAM shapshot sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.fdir, repr(sx), repr(sy), repr(sz))
        # write to file
        print('Write to file...')
        rh15d.make_xarray_atmos(outfile, temp, vz, z, nH=nh, x=x, y=y, vx=vx,
                                vy=vy, rho=rho, append=append, Bx=Bx, By=By,
                                Bz=Bz, desc=desc, snap=self.snap)
