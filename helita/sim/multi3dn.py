"""
Set of routines to read output from multi3d MPI (new version!)
"""
from ..io.fio import fort_read
import numpy as np
import os


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
        infile : str
            Name of file to read.
        nx, ny, nz : ints
            Number of points in x, y, and z dimensions.
        nhydr : int, optional
            Number of hydrogen levels. Default is 6.
        mode : str, optional
            Access mode. Can be either 'r' (read), 'w' (write, deletes existing),
            or 'w+' (write, update).
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
