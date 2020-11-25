import os
import numpy as np
import scipy.constants as ct
import radynpy as rd

class radyn:
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
    def __init__(self, filename, fdir='./', verbose=True):
        
        self.filename = filename
        self.fdir = fdir
        self.rdobj = rd.cdf.LazyRadynData(filename)
        self.x = 0.0
        self.y = 0.0
        self.z = self.rdobj.__getattr__('zm')
        
        self.dx = 1.0
        self.dy = 1.0
        self.dz = np.copy(self.z)
        self.nt = np.shape(self.z)[0]
        self.nz = np.shape(self.z)[1]
        for it in range(0,self.nt):
            self.dz[it,:] = np.roll(self.z[it,:],1)-self.dz[it,:] 
            self.dz[it,:] = self.dz[it,1]
        
        self.nx = np.shape(self.x)
        self.ny = np.shape(self.y)
        
        self.time =  self.rdobj.__getattr__('time')
        self.units()
        self.genvar()

    def get_var(self,var,it=None, iix=None, iiy=None, iiz=None, layout=None, cgs=True): 
        '''
        Reads the variables from a snapshot (it).

        Parameters
        ----------
        var - string
            Name of the variable to read.
        it - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot.
        cgs- logic 
            converts into cgs units.         
        Axes: 
        -----
            z-axis is along the loop
            x and y axes have only one grid. 
        
        Information about radynpy library: 
        --------------
        '''
        if var == '':
            print(help(self.get_var))
            print(self.rdobj.var_info('*'))
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
                    
        self.data = self.rdobj.__getattr__(varname) * cgsunits
        
        return self.data

    def get_ems(self,iter=None,layout=None, wght_per_h=1.4271, unitsnorm = 1e27, axis=2): 
        '''
        Computes emission meassure in cgs and normalized to unitsnorm
        '''
        rho = self.get_var('rho',it=iter,layout=layout) 
        nel = self.get_var('nel',it=iter,layout=layout) 

        
        nh = rho / (wght_per_h * ct.atomic_mass * 1e3)  # from rho to nH and added unitsnorm
        ds = self.dz * self.uni['l']
        
        return nel * (nh * ds / unitsnorm)
    
    def units(self): 
        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.uni['proton'] = 1.67262158e-24 #gr
        self.uni['kboltz'] = 1.380658e-16 
        self.uni['c']      = 299792.458 * 1e5 #cm/s
        self.uni['gamma']  = 5./3.
        self.uni['tg']     = 1
        self.uni['l']      = 1
        self.uni['n']      = 1
        self.uni['rho']    = 1
        self.uni['u']      = 1
        self.uni['b']      = 1
        self.uni['t']      = 1.0 # seconds
        self.uni['j']      = 1
   
    def genvar(self): 
        '''
        Dictionary of original variables which will allow to convert to cgs. 
        '''
        self.varn={}
        self.varn['rho']= 'd1'
        self.varn['tg'] = 'tg1'
        self.varn['ux'] = 'ux'
        self.varn['uy'] = 'uy'
        self.varn['uz'] = 'vz1'
        self.varn['bx'] = 'bx'
        self.varn['by'] = 'by'
        self.varn['bz'] = 'bz'
        self.varn['nel']= 'ne1'



