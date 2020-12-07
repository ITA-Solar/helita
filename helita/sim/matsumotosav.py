import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav

class Matsumotosav:
    """
    Class to read Matsumoto's sav file atmosphere. 
    Snapshots from a MHD simulation ( Matsumoto 2018 )
    https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.3328M/abstract

    Parameters
    ----------
    fdir : str, optional
        Directory with snapshots.
    rootname : str, optional
        Template for snapshot number.

    verbose : bool, optional
        If True, will print more information.
    """
    def __init__(self, rootname, it, fdir='./', verbose=True):
        self.fdir     = fdir        
        self.rootname = rootname
        self.savefile = rsav(fdir+rootname+'{:06d}'.format(it)+'.sav')
        self.it       = it
        self.time     = self.savefile['v']['time'][0]
        self.grav     = self.savefile['v']['gx'][0]
        self.gamma    = self.savefile['v']['gm'][0]
        self.x        = self.savefile['v']['x'][0]/1e8 # Mm
        self.y        = self.savefile['v']['y'][0]/1e8
        self.z        = self.savefile['v']['z'][0]/1e8
        
        self.dx       = self.savefile['v']['dx'][0]/1e8
        self.dy       = self.savefile['v']['dy'][0]/1e8
        self.dz       = self.savefile['v']['dz'][0]/1e8
        
        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)
        self.genvar()
        self.units()

    def get_var(self,var,it=None, iix=None, iiy=None, iiz=None, layout=None, cgs=True): 
        '''
        Reads the variables from a snapshot (it).

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        it - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot.
        cgs- logic 
            converts into cgs units. 
            
        Axes: 
        -----
            x and y axes horizontal plane
            z-axis is vertical axis
        
        Variable list: 
        --------------
            ro          -- Density (g/cm^3) [nx, ny, nz]
            temperature -- Temperature (K) [nx, ny, nz]
            vx          -- component x of the velocity (cm/s) [nx, ny, nz]
            vy          -- component y of the velocity (cm/s) [nx, ny, nz]
            vz          -- component z of the velocity (cm/s) [nx, ny, nz]
            bx          -- component x of the magnetic field (G) [nx, ny, nz]
            by          -- component y of the magnetic field (G) [nx, ny, nz]
            bz          -- component z of the magnetic field (G) [nx, ny, nz]
            pressure    -- Pressure (dyn/cm^2)  [nx, ny, nz]
        
        '''
        if var == '':
            print(help(self.get_var))
            print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
            for ii in self.varn: 
                print('use ', ii,' for ',self.varn[ii])
            return None


        if it != None: 
            self.it = it
            self.savefile = rsav(self.fdir+self.rootname+'{:06d}'.format(self.it)+'.sav')
        
        if (cgs): 
            varu=var.replace('x','')
            varu=varu.replace('y','')
            varu=varu.replace('z','')
            if (var in self.varn.keys()) and (varu in self.uni.keys()): 
                cgsunits = self.uni[varu]
            else: 
                cgsunits = 1.0
        else: 
            cgsunits = 1.0
        
        if var in self.varn.keys(): 
            varname=self.varn[var]
        else:
            varname=var

        self.data = self.savefile['v'][varname][0] * cgsunits
                        
        return self.data.T

    def get_ems(self,iter=None,layout=None, wght_per_h=1.4271, unitsnorm = 1e27, axis=2): 
        
        rho = self.get_var('rho',it=iter,layout=layout) 
        
        nh = rho / (wght_per_h * ct.atomic_mass * 1e3)  # from rho to nH and added unitsnorm
        if axis == 0:
            ds = self.dx * self.uni['l']
            oper = 'ijk,i->ijk'
        elif axis == 1:
            ds = self.dy * self.uni['l']
            oper = 'ijk,j->ijk'
        else:
            ds = self.dz * self.uni['l']
            oper = 'ijk,k->ijk'

        en = nh + 2.*nh*(wght_per_h-1.) # this may need a better adjustment.             
        nh = np.einsum(oper,nh,ds)

        return en * (nh / unitsnorm)
    
    def units(self): 
        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.uni['proton'] = 1.67262158e-24 #gr
        self.uni['kboltz'] = 1.380658e-16 
        self.uni['c']      = 299792.458 * 1e5 #cm/s
        self.uni['gamma']  = self.gamma 
        self.uni['tg']     = 1.0 # K
        self.uni['l']      = 1.0e8 # Mm -> cm
        self.uni['rho']    = 1.0 # gr cm^-3 
        self.uni['n']      = 1.0 # cm^-3
        self.uni['u']      = 1.0 # cm/s
        self.uni['b']      = 1.0 # Gauss
        self.uni['t']      = 1.0 # seconds
   
    def genvar(self): 
        '''
        Dictionary of original variables which will allow to convert to cgs. 
        '''
        self.varn={}
        self.varn['rho']= 'ro'
        self.varn['tg'] = 'te'
        self.varn['pg'] = 'pr'
        self.varn['ux'] = 'vx'
        self.varn['uy'] = 'vy'
        self.varn['uz'] = 'vz'
        self.varn['bx'] = 'bx'
        self.varn['by'] = 'by'
        self.varn['bz'] = 'bz'


