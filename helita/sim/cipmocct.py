import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav

class Cipmocct:
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
    def __init__(self, rootname, it, fdir='./', verbose=True):
        
        self.rootname = rootname
        self.fdir = fdir
        self.it = it 
        params = rsav(self.fdir+'params_'+rootname+'.sav')
        
        self.x = params['x2']
        self.y = params['x3']
        self.z = params['x1']
        
        self.dx = self.x-np.roll(self.x,1) 
        self.dx[0] = self.dx[1]
        
        self.dy = self.y-np.roll(self.y,1) 
        self.dy[0] = self.dy[1]
        
        self.dz = self.z-np.roll(self.z,1) 
        self.dz[0] = self.dz[1]
        
        self.nx = len(params['x2'])
        self.ny = len(params['x3'])
        self.nz = len(params['x1'])
        
        self.time =  params['time'] # No uniforme (array)


    def get_var(self,var,it=None, iix=None, iiy=None, iiz=None, layout=None): 
        '''
        Reads the variables from a snapshot (it).

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        it - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot.
        
        Axes: 
        -----
            x-axis is along the loop
            y and z axes are perperdicular to the loop
        
        Variable list: 
        --------------
            ro_cube       -- Density (multipy by self.u_normro to get in g/cm^3)
            te_cube       -- Temperature (multipy by self.u_normte to get in K)
            vx_cube       -- component x of the velocity (multipy by self.u_normcs to get in cm/s) 
            vy_cube       -- component y of the velocity (multipy by self.u_normcs to get in cm/s)
            vz_cube       -- component z of the velocity (multipy by self.u_normcs to get in cm/s)
            bx_cube       -- component x of the magnetic field (multipy by self.u_normb to get in G)
            by_cube       -- component y of the magnetic field (multipy by self.u_normb to get in G)
            bz_cube       -- component z of the magnetic field (multipy by self.u_normb to get in G) 
        '''
        if var == '':
            print(help(self.get_var))
            return None
        
        if it != None: 
            self.it = it
        
        itname = '_'+inttostring(self.it)
        
        varfile = rsav(self.fdir+'vars_'+self.rootname+itname+'.sav')
        self.data = varfile[var]
        
        return self.data

    def get_ems(self,iter=None,layout=None, wght_per_h=1.4271, unitsnorm = 1e27, axis=2): 
        
        rho = self.get_var('ro_cube',it=iter,layout=layout) * self.u_normro 
        
        nh = rho / (wght_per_h * ct.atomic_mass * 1e3)  # from rho to nH and added unitsnorm
        if axis == 0:
            ds = self.dx * self.u_rad
            oper = 'ijk,i->ijk'
        elif axis == 1:
            ds = self.dy * self.u_rad
            oper = 'ijk,j->ijk'
        else:
            ds = self.dz * self.u_rad
            oper = 'ijk,k->ijk'

        print(np.shape(nh),oper,np.shape(ds))
        en = nh + 2.*nh*(wght_per_h-1.) # this may need a better adjustment.             
        nh = np.einsum(oper,nh,ds)

        return en * (nh / unitsnorm)
    
    def units(self): 
        self.u_proton = 1.67262158e-24 #gr
        self.u_kboltz = 1.380658e-16 
        self.u_c = 299792.458 * 1e5 #cm/s
        self.u_gamma = 5./3.
        self.u_normte = 1.0e6 # K
        self.u_normn = 1.0e9 # cm^-3
        self.u_fact = 2
        self.u_rad = 1000.*self.u_fact*1.0e5 # for having a 2000 km wide loop
        self.u_normro = self.u_normn * self.u_proton/2. # gr cm^-3 
        self.u_normcs = np.sqrt(2*self.u_gamma*self.u_kboltz/self.u_proton*self.u_normte) # cm/s
        self.u_normb = self.u_normcs*np.sqrt(self.u_normro) # Gauss
        self.u_normj = self.u_normb/self.u_rad*self.u_c # current density
        self.u_normt = self.u_rad/self.u_normcs # seconds


def inttostring(ii,ts_size=4):

  str_num = str(ii)

  for bb in range(len(str_num),ts_size,1):
    str_num = '0'+str_num
  
  return str_num


