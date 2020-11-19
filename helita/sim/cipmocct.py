import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav

class Cipmocct:
    """
    Class to read cipmocct atmosphere

    Parameters
    ----------
    fdir : str, optional
        Directory with snapshots.
    rootname : str
        rootname of the file (wihtout params or vars).
    verbose : bool, optional
        If True, will print more information.
    it : integer 
        snapshot number 
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
        self.units()
        self.genvar()

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
            x-axis is along the loop
            y and z axes are perperdicular to the loop
        
        Variable list: 
        --------------
            ro_cube       -- Density (multipy by self.uni['rho'] to get in g/cm^3)
            te_cube       -- Temperature (multipy by self.uni['tg'] to get in K)
            vx_cube       -- component x of the velocity (multipy by self.uni['u'] to get in cm/s) 
            vy_cube       -- component y of the velocity (multipy by self.uni['u'] to get in cm/s)
            vz_cube       -- component z of the velocity (multipy by self.uni['u'] to get in cm/s)
            bx_cube       -- component x of the magnetic field (multipy by self.uni['b'] to get in G)
            by_cube       -- component y of the magnetic field (multipy by self.uni['b'] to get in G)
            bz_cube       -- component z of the magnetic field (multipy by self.uni['b'] to get in G) 
        '''
        if var == '':
            print(help(self.get_var))
            print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
            for ii in self.varn: 
                print('use ', ii,' for ',self.varn[ii])
            return None
        
        if it != None: 
            self.it = it
       
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
            
        itname = '_'+inttostring(self.it)
        
        varfile = rsav(self.fdir+'vars_'+self.rootname+itname+'.sav')
        self.data = varfile[varname] * cgsunits
        
        return self.data

    def get_ems(self,iter=None,layout=None, wght_per_h=1.4271, unitsnorm = 1e27, axis=2): 
        '''
        Computes emission meassure in cgs and normalized to unitsnorm
        '''
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

        print(np.shape(nh),oper,np.shape(ds))
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
        self.uni['gamma']  = 5./3.
        self.uni['tg']     = 1.0e6 # K
        self.uni['fact']   = 2
        self.uni['l']      = 1000.*self.uni['fact']*1.0e5 # for having a 2000 km wide loop
        self.uni['n']      = 1.0e9 # cm^-3
        self.uni['rho']    = self.uni['n'] * self.uni['proton'] /2. # gr cm^-3 
        self.uni['u']      = np.sqrt(2*self.uni['gamma']*self.uni['kboltz']/self.uni['proton']*self.uni['tg']) # cm/s
        self.uni['b']      = self.uni['u']*np.sqrt(self.uni['rho']) # Gauss
        self.uni['t']      = 1.0 # seconds
        self.uni['j']      = self.uni['b']/self.uni['l']*self.uni['c'] # current density
   
    def genvar(self): 
        '''
        Dictionary of original variables which will allow to convert to cgs. 
        '''
        self.varn={}
        self.varn['rho']= 'ro_cube'
        self.varn['tg'] = 'te_cube'
        self.varn['ux'] = 'vx_cube'
        self.varn['uy'] = 'vy_cube'
        self.varn['uz'] = 'vz_cube'
        self.varn['bx'] = 'bx_cube'
        self.varn['by'] = 'by_cube'
        self.varn['bz'] = 'bz_cube'
        
def inttostring(ii,ts_size=4):

  str_num = str(ii)

  for bb in range(len(str_num),ts_size,1):
    str_num = '0'+str_num
  
  return str_num


