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
import struct
from sunpy.io.special import read_genx
try:
    imp.find_module('pycuda')
    found = True
except ImportError:
    found = False

datapath = 'PYTHON/br_int/br_ioni/data/'
genxpath = 'IDL/data/lines/emissivity_functions/'

units = bifrost_units()

class UVOTRTData(BifrostData):

    """
    Class that operates radiative transfer form BifrostData simulations
    in native format.
    """

    def __init__(self, *args, **kwargs):
        super(UVOTRTData, self).__init__(*args, **kwargs)


    def load_intny_module(self, axis=2, azimuth=None,
                  altitude=None, ooe=False, *args, **kwargs):

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
                self.intcudamod = TDIEmRenderer(
                    data_dir=data_dir, paramfile=tdi_paramfile, snap=opts.snap,
                    cstagop=self.cstagop)
            else:
                # Statistical Equibilibrium along specific axis: x, y or z
                if opts.rendtype == 'sastatic':
                    from br_ioni import SAStaticEmRenderer
                    self.intcudamod = SAStaticEmRenderer(snap_range,
                                           template,axis=axis, data_dir=data_dir,
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

                        self.intcudamod = SATDIEmRenderer(axis=axis, 
                            data_dir=data_dir, paramfile=tdi_paramfile,
                            snap=opts.snap, cstagop=self.cstagop)
                    else:  # Statistical Equibilibrium along any direction
                        from br_ioni import StaticEmRenderer
                        self.intcudamod = StaticEmRenderer(snap_range,
                                             template, data_dir=data_dir,
                                             snap=opts.snap,
                                             cstagop=self.cstagop)

        else:
            print('I am so sorry... but you need pycuda:\n' +
                  '1, install latest CUDA at \n' +
                  'https://developer.nvidia.com/cuda-downloads\n ' +
                  '2, pycuda: https://wiki.tiker.net/PyCuda/Installation\n' +
                  'no warranty that this will work on non-NVIDIA')

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
        rend_reverse = False
        gridsplit = 128

        if not hasattr(self,'intcudamod'):
            self.load_intny_module(axis=axis, azimuth=azimuth,
                                   altitude=altitude, ooe=ooe)


        acont_filenames = [os.path.relpath(
            i, os.path.dirname('')) for i in glob(
                os.environ[
                    'BIFROST'] + datapath + spline + '.opy')]
        channel = 0

        if len(acont_filenames) == 0:
            raise ValueError(
                "(EEE) get_intny: GOFT file does not exist", spline)

        self.intcudamod.save_accontfiles(acont_filenames)

        return self.intcudamod.il_render(channel, azimuth, -altitude, axis,
                           rend_reverse, gridsplit=gridsplit, nlamb=nlamb,
                           dopp_width_range=dopp_width_range,
                           opacity=rend_opacity, verbose=self.verbose,
                           stepsize=stepsize)

    def load_intny(self, spfilebin):
        """
        Calculate and dave profiles in a binary file.
        """
        # open  file
        loadFile = open(spfilebin, "rb")
        # write to file
        data = loadFile.read()
        loadFile.close()
        (nwvl, temp, nx, temp, ny) = struct.unpack('iiiii',data[:4*5])
        intny={}
        intny[1] = np.zeros((nwvl))
        intny[0] = np.zeros((nwvl, nx, ny))
        for iwvl in range(nwvl):
            intny[1][iwvl] = struct.unpack('f',data[4*6+iwvl*4:4*6+4*iwvl+4])[0]
            for iix in range(nx):
                for iiy in range(ny):
                    const = 4*6 + 4*nwvl + iix*nwvl*ny*4 + iiy*nwvl*4 + iwvl*4
                    intny[0][iwvl,iix,iiy] = struct.unpack(
                        'f',data[const: const+ 4])[0]
        self.wvl = intny[1]
        return intny

    def save_intny(self, spline, snap=None, nlamb=141, axis=2,
                   rend_opacity=False, dopp_width_range=5e1, azimuth=None,
                   altitude=None, ooe=False, stepsize=0.001, *args, **kwargs):
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
                savedFile = open(spline+axis_txt+"it"+str(self.snap[it])+".bin", "wb")
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
                    savedFile = open(spline[iline]+axis_txt+"it"+str(self.snap[it])+".bin", "wb")
                    # write to file
                    nx = np.shape(intny[0])[0]
                    ny = np.shape(intny[0])[1]
                    nwvl = np.shape(intny[0])[2]
                    savedFile.write(np.array((nwvl, nx, ny)))
                    savedFile.write(intny[1])
                    savedFile.write(intny[0])
                    savedFile.close()
                    if (self.verbose):
                        print('done ', spline[iline], ' it=',
                              it, ',time used:', time.time() - t0)

    def get_int(self, spline, axis=2, rend_opacity=False, azimuth=None,
                altitude=None, ooe=False, stepsize=0.001, *args, **kwargs):
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
                        snap_range, template,
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
                        s = StaticEmRenderer(snap_range,
                                             template, data_dir=data_dir,
                                             snap=opts.snap)

            rend_reverse = False
            gridsplit = 350

            s.save_accontfiles(acont_filenames)
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

    def read_synlines(self, spline=None):
        '''
        reads the Source function for specific lines over a range of log T,
        '''
        if spline is None:
            spline = "*"
        synlines = [os.path.relpath(i, os.path.dirname('')) for i in glob(
                        os.environ['BIFROST'] + genxpath + spline +  '.genx')]

        self.synlinfiles = {}
        for ifile in range(0, np.size(synlines)):
            self.synlinfiles[ifile] = synlines[ifile]

        self.sglin = {}
        for ilines in self.synlinfiles.keys():
            print(ilines)
            # Read in precomputed response functions for 108 171 and 284, from
            # muse_resp.pro in IDL
            self.sglin[ilines] = read_genx(self.synlinfiles[ilines])
        self.sglin_ntg = np.size(self.sglin[0]['TEMP'])

    def calc_synvdemint(self):
        '''
        Calculate profile intensity for each velocity bin
        '''

        if not hasattr(self,'vdem'):
            vdem = self.get_vdem()
        self.synvdemint = {}
        for ilines in self.synlinfiles.keys():
            thisg = np.interp(self.tg_axis,
                              np.log10(self.sglin[ilines]['TEMP']),
                              self.sglin[ilines]['G'])
            self.synvdemint[ilines] = {}
            self.synvdemint[ilines]['intvtg'] = (self.vdem.reshape(
                self.nlnt, self.ndop * self.nx * self.ny) * thisg.reshape(
                self.nlnt, 1)).reshape(
                self.nlnt, self.ndop, self.nx, self.ny)
            self.synvdemint[ilines]['intens'] = np.sum(np.sum(
                self.synvdemint[ilines]['intvtg'], axis=0), axis=0)

    def trapez(self,xvec,yvec):
        '''
        performs trapezoidal integration.

        Input:
            xvec: vector of n elements
            yvec: vector with same number of elements as yvec
        Output:
            float number of the trapezoidal integration
        '''
        nxv=np.size(xvec)
        nyv=np.size(yvec)
        if nxv != nyv:
          print('trapez: different number of elements in x and y array')

        integrand=(yvec[:-1]+yvec[1:])*(xvec[1:]-xvec[:-1])*0.5
        return np.sum(integrand)

    def calc_profmoments(self,intny):
        '''
        calculates 0, 1st and 2nd moment of intensity profiles
        '''
        dny = intny[1]
        nwvl = np.size(dny)
        ddny = np.outer(dny,np.ones((nwvl)))

        prof_mom = {}

        nsx = np.shape(intny[0])[1]
        nsy = np.shape(intny[0])[2]
        prof_mom['itot'] = np.zeros((nsx,nsy))
        prof_mom['utot'] = np.zeros((nsx,nsy))
        prof_mom['wtot'] = np.zeros((nsx,nsy))

        for iix in range(nsx):
            for iiy in range(nsy):
                prof_mom['itot'][iix,iiy] = self.trapez(
                                    dny, intny[0][:,iix,iiy])
                prof_mom['utot'][iix,iiy] = self.trapez(
                    dny, np.inner(
                    intny[0][:,iix,iiy], ddny)) / prof_mom['itot'][iix,iiy]
                prof_mom['wtot'][iix,iiy] = self.trapez(
                    dny,np.inner(intny[0][:,iix,iiy], ddny**2)) / (
                        prof_mom['itot'][iix,iiy] - prof_mom['utot'][iix,iiy]**2)
        prof_mom['utot'] *= self.ny2vel
        prof_mom['wtot'] *= self.ny2vel * self.ny2vel

        return prof_mom

    def get_lambda(self,spline):
        '''
        '''
        acont_filename = os.path.relpath(
            os.environ['BIFROST'] + datapath + spline + '.opy')
        filehandler = open(acont_filename, 'rb')
        iontab = pickle.load(filehandler)
        self.wvl0 = iontab.Gofnt['wvl']
        self.lambd = self.wvl0 / (
            self.wvl * self.wvl0 / 1.0e13 + units.CLIGHT.value /
                1.0e5 ) * units.CLIGHT.value / 1.0e5
        self.ny2vel = self.wvl0 * 1.e-13
        self.wvldop = self.wvl * self.ny2vel

    def calc_vdemmoments(self):
        '''
        Calculate moments for each lines in sglin using the true VDEM raster
        '''

        if not hasattr(self,'synvdemint'):
            self.calc_synvdemint()

        for ilines in self.synlinfiles.keys():

            trthisprofile = np.transpose(
                self.synvdemint[ilines]['intvtg'], (1, 0, 2, 3)).reshape(
                self.ndop, self.nlnt * self.nx * self.ny)

            self.synprofras[ilines]['intens'] = np.sum(np.sum(
                self.synvdemint[ilines]['intvtg'], axis=0), axis=0)  # ph/s/pix

            self.synprofras[ilines]['firstmom'] = (np.sum(np.sum(np.reshape(
                trthisprofile.reshape(
                    self.ndop, self.nlnt * self.nx * self.ny) * self.dopaxis.reshape(
                    self.ndop, 1), (
                    self.ndop, self.nlnt, self.nx, self.ny)), axis=0),
                    axis=0) / self.synprofsolras[ilines]['intens']).reshape(
                    self.nslits * np.size(self.ioffset), self.ny)

            self.synprofras[ilines]['secondmom'] = np.sqrt(np.sum(np.sum(np.reshape(
                trthisprofile.reshape(
                    self.ndop, self.nlnt * self.nx * self.ny) * self.dopaxis.reshape(
                    self.ndop, 1)**2, (
                    self.ndop, self.nlnt, self.nx, self.ny)), axis=0),
                    axis=0) / self.synprofsolras[ilines]['intens']).reshape(
                    self.nslits * np.size(self.ioffset), self.ny)

            thirdmom = (np.sum(np.sum(np.reshape(
                trthisprofile.reshape(
                    self.ndop, self.nlnt * self.nx * self.ny) * self.dopaxis.reshape(
                    self.ndop, 1)**3, (
                    self.ndop, self.nlnt, self.nx, self.ny)), axis=0),
                    axis=0) / self.synprofsolras[ilines]['intens']).reshape(
                    self.nslits * np.size(self.ioffset) , self.ny)

            sign = np.zeros((self.nx, self.ny))
            sign[:, :] = 1
            sign[np.where(thirdmom < 0)] = -1.

            self.synprofras[ilines]['thirdmom'] = (
                np.abs(thirdmom))**(1. / 3.) * sign

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
                vdem[itg, ivel, :, :] = np.sum(ems_temp_v,axis=2)

        return vdem


    def load_vdem_npz(self, vdemfile):
        '''
        reads the ground true VDEM. The npz file can be
            generated using bifrost.py for Bifrost simulations.
            It needs to be called prior doing the inversions:
            VDEMSInversionCode, or all_comb
        VDEM function shape (T, v, nx, ny).
        '''

        vdem_str = np.load(vdemfile)
        self.vdem = vdem_str['vdem']
        self.vel_axis = vdem_str['vel_axis']
        self.tg_axis = vdem_str['tg_axis']

        self.nlnt = np.size(self.tg_axis)
        self.ndop = np.size(self.vel_axis)

    def get_vdem(self, axis=2, vel_axis=np.linspace(- 20, 20, 21),
                 tg_axis=np.linspace(4, 9, 25), zcut=None,
                 save_vdem = None):
        """
        Calculates DEM as a funtion of temparature and velocity,
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
            Array with the dimensions of the 2D spatial from the simulation
            and temperature and velocity bins.
        """
        ems = self.get_dem(axis=axis, zcut=zcut)
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
                vdem[itg, ivel, :, :] = np.sum(ems_temp_v,axis=2)

        self.vdem = vdem
        self.tg_axis = tg_axis
        self.vel_axis = vel_axis
        self.nlnt = np.size(tg_axis)
        self.ndop = np.size(vel_axis)

        if save_vdem is not None:
            np.savez('%s_tg=%.1f-%.1f_%.1f_vel=%i_%i_it=%i.npz' % (
                     save_vdem, tg_axis[0], max(tg_axis), tg_axis[1]-tg_axis[0],
                     max(vel_axis*units.u_u),
                     (vel_axis[1]-vel_axis[0])*units.u_u,
                     self.snap[0]), tg_axis=tg_axis, vel_axis=vel_axis*10,
                     vdem=vdem)

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
        tg = self.get_var('tg')
        en = self.get_var('ne') / 1e6  # from SI to cgs
        rho = self.get_var('r')
        if axis == 0:
            ds = self.dx * units.u_l
        elif axis == 1:
            ds = self.dy * units.u_l
        else:
            ds = self.dz1d * units.u_l

        nh = rho * units.u_r / units.GRPH

        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                nh[ix, iy, :] = ds * nh[ix, iy, :]

        if (zcut is not None):
            for iz in range(0, self.nz):
                if self.z[iz] < zcut:
                    izcut = iz
            nh[:, :, izcut:] = 0.0
        return en * nh

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
            ds = self.dz1d * units.u_l
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
