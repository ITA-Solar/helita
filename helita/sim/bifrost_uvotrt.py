"""
Set of programs to read and interact with output from BifrostData
simulations focus on optically thin radiative transfer and similar
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii
from .bifrost import subs2grph, bifrost_units
from . import cstagger
from glob import glob
import scipy as sp
import time
import imp
import pickle
import scipy.ndimage as ndimage

try:
    imp.find_module('pycuda')
    found = True
except ImportError:
    found = False

datapath = 'PYTHON/br_int/br_ioni/data/'

units = bifrost_units()

class UVOTRTData(BifrostData):

    """
    Class that operates radiative transfer form BifrostData simulations
    in native format.
    """

    def __init__(self, *args, **kwargs):
        super(UVOTRTData, self).__init__(*args, **kwargs)

    def get_intny(self, spline, nlamb=141, axis=2, rend_opacity=False,
                  dopp_width_range=5e1, azimuth=None,
                  altitude=None, ooe=False, stepsize=0.01, *args, **kwargs):
        """
        Calculates intensity profiles from the simulation quantiables.

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the
            format, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must be
            name, e.g., 'fe_8_108.073'. If you dont have any, rung form
            helita.sim.atom_tools create_goftne_tab (very simple to run).
        nlamb - integer
            Number of points along the wavelenght axis.
        dopp_width_range - float number
            It selects the wavelength range. The value means.
        axis - integer number: 0 = x, 1 = y, 2 = z
            It allows to chose the LOS integration axis
        rend_opacity - allows to calcuate opacities.
            Right now uses a very obsolete table; so I suggest to wait until
            further improvements.
        azimuth -  This allows to trace rays for any angle. In this cases uses
            none sa modules. In this case, axis parameter is not need to
            be used.
        altitude -  This allows to trace rays for any angle. In this
            cases uses none sa modules.
            In this case, axis parameter is not need to be used.
        ooe -  True or False. Default is False.
            If true, uses Olluri out of equilibrium ionization output
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 2D spatial from the simulation
            and spectra.

        Notes
        -----
            uses cuda
        """

        if found:

            if os.environ.get('CUDA_LIB', 'null') == 'null':
                os.environ['CUDA_LIB'] = os.environ['BIFROST'] + 'CUDA/'

            if azimuth is not None or altitude is not None:  # For any angle
                if ooe:
                    choice = 'tdi'
                else:
                    choice = 'static'
                if azimuth is not None:
                    azimuth = 90.0
                if altitude is not None:
                    altitude = 0.0
            else:  # For a specific axis, this is setup with axis =[0,1,2]
                if ooe:
                    choice = 'satdi'
                else:
                    choice = 'sastatic'
                azimuth = 90.0
                altitude = 0.0

            opts = int_options()
            opts.infile = self.file_root
            opts.snap = self.snap
            opts.choice = choice
            opts.simdir = self.fdir
            data_dir = (opts.simdir if opts.simdir else askdirectory(
                title='Simulation Directory')).rstrip('/')

            acont_filenames = [os.path.relpath(
                i, os.path.dirname('')) for i in glob(
                    os.environ[
                        'BIFROST'] + datapath + spline + '.opy')]
            channel = 0

            if len(acont_filenames) == 0:
                raise ValueError(
                    "(EEE) get_intny: GOFT file does not exist", spline)
            snap_range = (self.snap, self.snap)
            # + '_' + '%%0%ii' % np.max((len(str(self.snap)),3))
            template = opts.infile

            # from br_ioni import RenderGUI

            if opts.rendtype == 'tdi':  # OOE along any angle
                from br_ioni import TDIEmRenderer
                tdi_paramfile_abs = (
                    opts.tdiparam if opts.tdiparam else askopenfilename(
                        title='OOE Ionization Paramfile'))
                tdi_paramfile = os.path.relpath(tdi_paramfile_abs, data_dir)
                s = TDIEmRenderer(
                    data_dir=data_dir, paramfile=tdi_paramfile, snap=opts.snap,
                    cstagop=self.cstagop)
            else:
                # Statistical Equibilibrium along specific axis: x, y or z
                if opts.rendtype == 'sastatic':
                    from br_ioni import SAStaticEmRenderer
                    s = SAStaticEmRenderer(snap_range, acont_filenames,
                                           template, data_dir=data_dir,
                                           snap=opts.snap,
                                           cstagop=self.cstagop)
                else:
                    # OOE along specific axis, i.e., x, y or z
                    if opts.rendtype == 'satdi':
                        from br_ioni import SATDIEmRenderer
                        tdi_paramfile_abs = (
                            opts.tdiparam if (
                                opts.tdiparam) else askopenfilename(
                                    title='OOE Ionization Paramfile'))
                        tdi_paramfile = os.path.relpath(
                            tdi_paramfile_abs, data_dir)

                        s = SATDIEmRenderer(
                            data_dir=data_dir, paramfile=tdi_paramfile,
                            snap=opts.snap, cstagop=self.cstagop)
                    else:  # Statistical Equibilibrium along any direction
                        from br_ioni import StaticEmRenderer
                        s = StaticEmRenderer(snap_range, acont_filenames,
                                             template, data_dir=data_dir,
                                             snap=opts.snap,
                                             cstagop=self.cstagop)

            rend_reverse = False
            gridsplit = 128

            return s.il_render(channel, azimuth, -altitude, axis,
                               rend_reverse, gridsplit=gridsplit, nlamb=nlamb,
                               dopp_width_range=dopp_width_range,
                               opacity=rend_opacity, verbose=self.verbose,
                               stepsize=stepsize)
        else:
            print('I am so sorry... but you need pycuda:\n' +
                  '1, install latest CUDA at \n' +
                  'https://developer.nvidia.com/cuda-downloads\n ' +
                  '2, pycuda: https://wiki.tiker.net/PyCuda/Installation\n' +
                  'no warranty that this will work on non-NVIDIA')

    def save_intny(self, spline, snap=None, nlamb=141, axis=2,
                   rend_opacity=False, dopp_width_range=5e1, azimuth=None,
                   altitude=None, ooe=False, stepsize=0.01, *args, **kwargs):
        """
        Calculate and dave profiles in a binary file.
        """
        nlines = np.size(spline)
        if (snap is None):
            snap = self.snap

        nt = np.size(self.snap)
        t0 = time.time()
        if (axis == 0):
            axis_txt='_yz_'
        elif (axis == 1):
            axis_txt='_xz_'
        else:
            axis_txt='_xy_'
            
        for it in range(0,nt):
            self.set_snap(snap[it])
            t1 = time.time()
            if (nlines == 1):
                intny = self.get_intny(spline, nlamb=nlamb, axis=axis,
                                       rend_opacity=rend_opacity,
                                       dopp_width_range=dopp_width_range,
                                       azimuth=azimuth,
                                       altitude=altitude, ooe=ooe,
                                       stepsize=stepsize)

                # make file
                savedFile = open(spline+axis_txt+"it"+str(self.snap)+".bin", "wb")
                # write to file
                nx = np.shape(intny[0])[0]
                ny = np.shape(intny[0])[1]
                nwvl = np.shape(intny[0])[2]
                savedFile.write(np.array((nwvl, nx, ny)))
                savedFile.write(intny[1])
                savedFile.write(intny[0])
                savedFile.close()
                if (self.verbose):
                    print('done ', spline, ' it=', it,
                          ',time used:', time.time() - t0)
            else:
                for iline in range(0, nlines):
                    t2 = time.time()
                    intny = self.get_intny(spline[iline], nlamb=nlamb,
                                           axis=axis,
                                           rend_opacity=rend_opacity,
                                           dopp_width_range=dopp_width_range,
                                           azimuth=azimuth,
                                           altitude=altitude, ooe=ooe,
                                           stepsize=stepsize)

                    # make file
                    savedFile = open(spline[iline]+axis_txt+"it"+str(self.snap)+".bin", "wb")
                    # write to file
                    nx = np.shape(intny[0])[0]
                    ny = np.shape(intny[0])[1]
                    nwvl = np.shape(intny[0])[2]
                    savedFile.write(np.array((nwvl, nx, ny)))
                    savedFile.write(intny[1])
                    savedFile.write(intny[0])
                    savedFile.close()
                    print(it, spline[iline], time.time() - t2)
                    if (self.verbose):
                        print('done ', spline[iline], ' it=',
                              it, ',time used:', time.time() - t0)
        print(it, time.time() - t1)

    def get_int(self, spline, axis=2, rend_opacity=False, azimuth=None,
                altitude=None, ooe=False, *args, **kwargs):
        """
        Calculates intensity from the simulation quantiables.

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the f
            ormat, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must
            be name, e.g., 'fe_8_108.073'.
        axis - integer number: 0 = x, 1 = y, 2 = z
            allows to chose the LOS integration axis (see azimuth and altitude)
        rend_opacity - allows to calcuate opacities.
            Right now uses a very obsolete table; so I suggest to wait until
            further improvements.
        azimuth -  This allows to trace rays for any angle. In this cases
            uses none sa modules.
            In this case, axis parameter is not need to be used.
        altitude -  This allows to trace rays for any angle. In this cases
            uses none sa modules.
            In this case, axis parameter is not need to be used.
        ooe -  True or False. Default is False.
            If true, uses Olluri out of equilibrium ionization output
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 2D spatial from the simulation
            and spectra.

        Notes
        -----
            uses cuda
        """

        if found:

            if os.environ.get('CUDA_LIB', 'null') == 'null':
                os.environ['CUDA_LIB'] = os.environ['BIFROST'] + 'CUDA/'

            # Calculation settings
            if azimuth is not None or altitude is not None:  # For any angle
                if ooe:
                    choice = 'tdi'
                else:
                    choice = 'static'
                if azimuth is not None:
                    azimuth = 90.0
                if altitude is not None:
                    altitude = 0.0
            else:  # For a specific axis, this is setup with axis =[0,1,2]
                if ooe:
                    choice = 'satdi'
                else:
                    choice = 'sastatic'
                azimuth = 90.0
                altitude = 0.0

            opts = int_options()
            opts.infile = self.file_root
            opts.snap = self.snap
            opts.choice = choice
            opts.simdir = self.fdir
            data_dir = (opts.simdir if opts.simdir else askdirectory(
                title='Simulation Directory')).rstrip('/')

            acont_filenames = [os.path.relpath(
                i, os.path.dirname('')) for i in glob(
                os.environ['BIFROST'] + datapath + spline + '.opy')]
            channel = 0
            if len(acont_filenames) == 0:
                raise ValueError(
                    "(EEE) get_int: GOFT file does not exist", spline)

            snap_range = (self.snap, self.snap)
            # + '_' + '%%0%ii' % np.max((len(str(self.snap)),3))
            template = opts.infile

            # from br_ioni import RenderGUI

            if opts.rendtype == 'tdi':  # OOE along any angle
                from br_ioni import TDIEmRenderer
                tdi_paramfile_abs = (
                    opts.tdiparam if opts.tdiparam else askopenfilename(
                        title='OOE Ionization Paramfile'))
                tdi_paramfile = os.path.relpath(tdi_paramfile_abs, data_dir)
                s = TDIEmRenderer(data_dir=data_dir,
                                  paramfile=tdi_paramfile, snap=opts.snap)
            else:
                # Statistical Equibilibrium along specific axis: x, y or z
                if opts.rendtype == 'sastatic':
                    from br_ioni import SAStaticEmRenderer
                    s = SAStaticEmRenderer(
                        snap_range, acont_filenames, template,
                        data_dir=data_dir, snap=opts.snap)
                else:
                    # OOE along specific axis, i.e., x, y or z
                    if opts.rendtype == 'satdi':
                        from br_ioni import SATDIEmRenderer
                        tdi_paramfile_abs = (
                            opts.tdiparam if (
                                opts.tdiparam) else askopenfilename(
                                    title='OOE Ionization Paramfile'))
                        tdi_paramfile = os.path.relpath(
                            tdi_paramfile_abs, data_dir)

                        s = SATDIEmRenderer(
                            data_dir=data_dir, paramfile=tdi_paramfile,
                            snap=opts.snap)
                    else:  # Statistical Equibilibrium for any angle
                        from br_ioni import StaticEmRenderer
                        s = StaticEmRenderer(snap_range, acont_filenames,
                                             template, data_dir=data_dir,
                                             snap=opts.snap)

            rend_reverse = False
            gridsplit = 64

            return s.i_render(channel, azimuth, -altitude, axis,
                              reverse=rend_reverse, gridsplit=gridsplit,
                              tau=None, opacity=rend_opacity,
                              verbose=self.verbose, fw=None, stepsize=stepsize)
        else:
            print('I am so sorry... but you need pycuda:\n' +
                  '1, install latest CUDA at \n' +
                  'https://developer.nvidia.com/cuda-downloads\n ' +
                  '2, pycuda: https://wiki.tiker.net/PyCuda/Installation no' +
                  'warranty that this will work on non-NVIDIA')

    def get_emstgu(self, spline, order=1, axis=2,
                   vel_axis=np.linspace(- 20, 20, 20),
                   tg_axis=np.linspace(4, 9, 25)):
        """
        Calculates emissivity (EM) as a funtion of temparature and velocity,
        i.e., VDEM.

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the
            format, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must
            be name, e.g., 'fe_8_108.073'.
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 3D spatial from the simulation

        Notes
        -----
            Uses cuda
        """
        ems = self.get_emiss(spline, axis=axis, order=order)
        tg = self.get_var('tg')
        if axis == 0:
            vel = self.get_var('ux')
            nx = self.ny
            ny = self.nz
            ems = np.transpose(ems, (1, 2, 0))
            tg = np.transpose(tg, (1, 2, 0))
            vel = np.transpose(vel, (1, 2, 0))
        elif axis == 1:
            vel = self.get_var('uy')
            nx = self.nx
            ny = self.nz
            ems = np.transpose(ems, (0, 2, 1))
            tg = np.transpose(tg, (0, 2, 1))
            vel = np.transpose(vel, (0, 2, 1))
        else:
            vel = self.get_var('uz')
            nx = self.nx
            ny = self.ny

        nvel = len(vel_axis)
        ntg = len(tg_axis)
        vdem = np.zeros((ntg, nvel, nx, ny))

        for itg in range(0, ntg - 1):
            ems_temp = ems * 1.0
            print('itg =', itg)
            loc = np.where(np.log10(tg) < tg_axis[itg])
            ems_temp[loc] = 0.0
            loc = np.where(np.log10(tg) > tg_axis[itg + 1])
            ems_temp[loc] = 0.0
            for ivel in range(0, nvel - 1):
                ems_temp_v = ems_temp * 1.0
                loc = np.where(vel < vel_axis[ivel])
                ems_temp_v[loc] = 0.0
                loc = np.where(vel > vel_axis[ivel + 1])
                ems_temp_v[loc] = 0.0
                for ix in range(0, nx):
                    for iy in range(0, ny):
                        vdem[itg, ivel, ix, iy] = np.sum(ems_temp_v[ix, iy, :])

        return vdem

    def get_vdem(self, axis=2, vel_axis=np.linspace(- 20, 20, 20),
                 tg_axis=np.linspace(4, 9, 25), zcut=None):
        """
        Calculates emissivity (EM) as a funtion of temparature and velocity,
        i.e., VDEM.

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the
            format, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must
            be name, e.g., 'fe_8_108.073'.
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 3D spatial from the simulation

        Notes
        -----
            Uses cuda
        """
        ems = self.get_dem(axis=axis, zcut=zcut)
        tg = self.get_var('tg')
        if axis == 0:
            vel = self.get_var('ux')
            nx = self.ny
            ny = self.nz
            dx = self.dy * units.u_l
            dy = self.dzidzdn * units.u_l
            ems = np.transpose(ems, (1, 2, 0))
            tg = np.transpose(tg, (1, 2, 0))
            vel = np.transpose(vel, (1, 2, 0))
        elif axis == 1:
            vel = self.get_var('uy')
            nx = self.nx
            ny = self.nz
            dx = self.dx * units.u_l
            dy = self.dzidzdn * units.u_l
            ems = np.transpose(ems, (0, 2, 1))
            tg = np.transpose(tg, (0, 2, 1))
            vel = np.transpose(vel, (0, 2, 1))
        else:
            vel = self.get_var('uz')
            nx = self.nx
            ny = self.ny
            dx = self.dx * units.u_l
            dy = self.dy * units.u_l

        nvel = len(vel_axis)
        ntg = len(tg_axis)
        vdem = np.zeros((ntg, nvel, nx, ny))

        for itg in range(0, ntg - 1):
            ems_temp = ems * 1.0
            print('itg =', itg)
            loc = np.where(np.log10(tg) < tg_axis[itg])
            ems_temp[loc] = 0.0
            loc = np.where(np.log10(tg) > tg_axis[itg + 1])
            ems_temp[loc] = 0.0
            for ivel in range(0, nvel - 1):
                ems_temp_v = ems_temp * 1.0
                loc = np.where(vel < vel_axis[ivel])
                ems_temp_v[loc] = 0.0
                loc = np.where(vel > vel_axis[ivel + 1])
                ems_temp_v[loc] = 0.0
                for ix in range(0, nx):
                    for iy in range(0, ny):
                        vdem[itg, ivel, ix, iy] = np.sum(
                            ems_temp_v[ix, iy, :] * dx * dy)

        return vdem

    def get_dem(self, axis=2, zcut=None, *args, **kwargs):
        """
        Calculates emissivity (EM).

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the
            format, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must
            be name, e.g., 'fe_8_108.073'.
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 3D spatial from the simulation

        Notes
        -----
            Uses cuda
        """
        # mem = np.memmap(data_dir + '/' + acontfile, dtype='float32')
        tg = self.get_var('tg')
        en = self.get_var('ne') / 1e6  # from SI to cgs
        rho = self.get_var('r')
        if axis == 0:
            ds = self.dx * units.u_l
        elif axis == 1:
            ds = self.dy * units.u_l
        else:
            ds = self.dzidzdn * units.u_l

        nh = rho * units.u_r / units.GRPH
        dem = nh * 0.0

        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                dem[ix, iy, :] = ds * nh[ix, iy, :]

        if (zcut is not None):
            for iz in range(0, self.nz):
                if self.z[iz] < zcut:
                    izcut = iz
            dem[:, :, izcut:] = 0.0
        print(np.max(en), np.max(dem))
        return en * dem

    def get_emiss(self, spline, axis=2, order=1, *args, **kwargs):
        """
        Calculates emissivity (EM).

        Parameters
        ----------
        spline - string
            Name of the spectral line to calculate. In order to know the
            format, $BIFROST/PYTHON/br_int/br_ioni/data
            contains files with the G(T,ne), usually name.opy. spline must
            be name, e.g., 'fe_8_108.073'.
        Returns
        -------
        array - ndarray
            Array with the dimensions of the 3D spatial from the simulation

        Notes
        -----
            Uses cuda
        """
        # mem = np.memmap(data_dir + '/' + acontfile, dtype='float32')

        CC = units.CLIGHT.value * units.CM_TO_M  # 2.99792458e8 m/s
        CCA = CC * 1e10  # AA/s
        nspline = np.size(spline)

        if nspline > 1:
            acont_filename = [os.path.relpath(
                i, os.path.dirname('')) for i in glob(
                    os.environ['BIFROST'] + datapath + spline[0] + '.opy')]
            for ispline in range(1, nspline):
                acont_filename += [os.path.relpath(
                    i, os.path.dirname('')) for i in glob(os.environ[
                        'BIFROST'] + datapath + spline[ispline] + '.opy')]
        else:
            acont_filename = [os.path.relpath(
                i, os.path.dirname('')) for i in glob(
                    os.environ['BIFROST'] + datapath + spline + '.opy')]
        filehandler = open(acont_filename[0], 'rb')
        ion = pickle.load(filehandler)
        filehandler.close()

        ntgbin = len(ion.Gofnt['temperature'])
        nedbin = len(ion.Gofnt['press'])
        tgmin = np.min(np.log10(ion.Gofnt['temperature']))
        tgrange = np.max(np.log10(ion.Gofnt['temperature'])) - tgmin
        enmin = np.min(ion.Gofnt['press'])
        enrange = np.max(ion.Gofnt['press']) - enmin

        tg_axis = np.linspace(tgmin, tgmin + tgrange, ntgbin)
        press_axis = np.linspace(enmin, enmin + enrange, nedbin)
        self.ny0 = (CCA / ion.Gofnt['wvl'])
        self.awgt = ion.mass

        self.acont_table = np.transpose(ion.Gofnt['gofnt'])
        self.acont_table = np.array(self.acont_table)
        for itab in range(1, len(acont_filename)):
            if self.verbose:
                if (itab == 1):
                    print(('(WWW) get_emiss: is summing G(T,ne) tables'))
                print(('(WWW) get_emiss: Adding ...%s G(T,ne) table' %
                       acont_filename[itab][-19:-5]))
            filehandler = open(acont_filename[itab], 'rb')
            ion = pickle.load(filehandler)
            filehandler.close()
            acont_tab_temp = np.transpose(ion.Gofnt['gofnt'])
            acont_tab_temp = np.array(self.acont_table)
            self.acont_table += acont_tab_temp

        tg = self.get_var('tg')
        en = self.get_var('ne')

        # warnings for values outside of table
        enminv = np.min(tg * en)
        enmaxv = np.max(tg * en)
        tgminv = np.min(np.log10(tg))
        tgmaxv = np.max(np.log10(tg))
        if enminv < enmin:
            print('(WWW) tab_interp: electron density outside table bounds.' +
                  'Table ne min=%.3e, requested ne min=%.3e' %
                  (enmin, enminv))
        if enmaxv > enmin + enrange:
            print('(WWW) tab_interp: electron density outside table bounds. ' +
                  'Table ne max=%.1f, requested ne max=%.1f' %
                  (enmin + enrange, enmaxv))
        if tgminv < tgmin:
            print('(WWW) tab_interp: tg outside of table bounds. ' +
                  'Table tg min=%.2f, requested tg min=%.2f' %
                  (tgmin, tgminv))
        if tgmaxv > tgmin + tgrange:
            print('(WWW) tab_interp: tg outside of table bounds. ' +
                  'Table tg max=%.2f, requested tg max=%.2f' %
                  (tgmin + tgrange, tgmaxv))

        # translate to table coordinates
        y = (np.log10(tg) - tg_axis[0]) / (tg_axis[1] - tg_axis[0])
        x = (en * tg - press_axis[0]) / (press_axis[1] - press_axis[0])
        g = ndimage.map_coordinates(
            self.acont_table, [x, y], order=order, mode='nearest')

        rho = self.get_var('r')
        if axis == 0:
            ds = self.dx * units.u_l
        elif axis == 1:
            ds = self.dy * units.u_l
        else:
            ds = self.dzidzdn * units.u_l
        nh = rho * units.u_r / units.GRPH

        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                g[ix, iy, :] *= ds * nh[ix, iy, :]

        return en * g


# Calculation settings
class int_options:

    def __init__(self):
        self.rendtype = 'sastatic'
        self.tdiparam = False
        self.simdir = ''
        self.snap = 1
        self.infile = ''
