import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav
from .load_quantities import *
from .load_noeos_quantities import *
from .load_arithmetic_quantities import *
from .tools import *

class Laresav:
  """
  Class to read Lare3D sav file atmosphere

  Parameters
  ----------
  fdir : str, optional
      Directory with snapshots.
  rootname : str, optional
      Template for snapshot number.
  verbose : bool, optional
      If True, will print more information.
  """
  def __init__(self, snap, fdir='.', sel_units = 'cgs', verbose=True):

    self.fdir     = fdir
    try: 
        self.savefile = rsav(os.path.join(self.fdir,'{:03d}'.format(snap)+'.sav'))
    except: 
        self.savefile = rsav(os.path.join(self.fdir,'{:04d}'.format(snap)+'.sav'))

    self.rootname = self.savefile['d']['filename'][0]
    self.snap = snap 
    self.sel_units = sel_units
    self.verbose = verbose
    self.uni = Laresav_units()
    
    self.time     = self.savefile['d']['time'][0].copy()
    self.time_prev= self.savefile['d']['time_prev'][0].copy()
    self.timestep = self.savefile['d']['timestep'][0].copy()
    self.dt       = self.savefile['d']['dt'][0].copy()

    self.visc_heating= self.savefile['d']['visc_heating'][0].copy()
    self.visc3_heating= self.savefile['d']['visc3_heating'][0].copy()

    self.x       = self.savefile['d']['x'][0].copy().byteswap('=').newbyteorder('=')
    self.y       = self.savefile['d']['y'][0].copy().byteswap('=').newbyteorder('=')
    self.z       = self.savefile['d']['z'][0].copy().byteswap('=').newbyteorder('=')
    
    if self.sel_units=='cgs': 
        self.x *= self.uni.uni['l']
        self.y *= self.uni.uni['l']
        self.z *= self.uni.uni['l']

    #GRID            STRUCT    -> <Anonymous> Array[1]
    
    self.nx = len(self.x)
    self.ny = len(self.y)
    self.nz = len(self.z)

    if self.nx > 1:
        self.dx1d = np.gradient(self.x) 
    else: 
        self.dx1d = np.zeros(self.nx)

    if self.ny > 1:            
        self.dy1d = np.gradient(self.y) 
    else:
        self.dy1d = np.zeros(self.ny)

    if self.nz > 1:
        self.dz1d = np.gradient(self.z)
    else:
        self.dz1d = np.zeros(self.nz)
        
    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

    self.genvar()

  def get_var(self,var, *args, snap=None, iix=None, iiy=None, iiz=None, layout=None, **kargs): 
    '''
    Reads the variables from a snapshot (snap).

    Parameters
    ----------
    var - string
        Name of the variable to read. Must be Bifrost internal names.
    snap - integer, optional
        Snapshot number to read. By default reads the loaded snapshot;
        if a different number is requested, will load that snapshot.
        
    Axes: 
    -----
        x and y axes horizontal plane
        z-axis is vertical axis, top corona is last index and positive. 
    
    Variable list: 
    --------------
        rho         -- Density (multipy by self.uni['r'] to get in g/cm^3) [nx, ny, nz]
        energy      -- Energy (multipy by self.uni['e'] to get in erg) [nx, ny, nz]
        temperature -- Temperature (multipy by self.uni['tg'] to get in K) [nx, ny, nz]
        vx          -- component x of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        vy          -- component y of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        vz          -- component z of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        bx          -- component x of the magnetic field (multipy by self.uni['b'] to get in G) [nx+1, ny, nz]
        by          -- component y of the magnetic field (multipy by self.uni['b'] to get in G) [nx, ny+1, nz]
        bz          -- component z of the magnetic field (multipy by self.uni['b'] to get in G) [nx, ny, nz+1]
        jx          -- component x of the current [nx+1, ny+1, nz+1]
        jy          -- component x of the current [nx+1, ny+1, nz+1]
        jz          -- component x of the current [nx+1, ny+1, nz+1]
        pressure    -- Pressure (multipy by self.uni['pg'])  [nx, ny, nz]
        eta         -- eta (?) [nx, ny, nz]
    
    '''

    if var in self.varn.keys(): 
        varname=self.varn[var]
    else:
        varname=var
        
    if snap != None: 
      self.snap = snap
      try: 
        self.savefile = rsav(os.path.join(self.fdir,'{:03d}'.format(snap)+'.sav'))
      except: 
        self.savefile = rsav(os.path.join(self.fdir,'{:04d}'.format(snap)+'.sav'))

    try: 

      if self.sel_units == 'cgs': 
        varu=var.replace('x','')
        varu=varu.replace('y','')
        varu=varu.replace('z','')
        if (var in self.varn.keys()) and (varu in self.uni.uni.keys()): 
          cgsunits = self.uni.uni[varu]
        else: 
          cgsunits = 1.0
      else: 
        cgsunits = 1.0

      
      self.data = (self.savefile['d'][varname][0].T).copy().byteswap('=').newbyteorder('=') * cgsunits
      
      if (np.shape(self.data)[0]>self.nx): 
          self.data = ((self.data[1:,:,:] + self.data[:-1,:,:]) / 2).copy()
      
      if (np.shape(self.data)[1]>self.ny): 
          self.data = ((self.data[:,1:,:] + self.data[:,:-1,:]) / 2).copy()
      
      if (np.shape(self.data)[2]>self.nz): 
          self.data = ((self.data[:,:,1:] + self.data[:,:,:-1]) / 2).copy()

    except: 
      # Loading quantities
      if self.verbose: 
        print('Loading composite variable',end="\r",flush=True)
      self.data = load_noeos_quantities(self,var, **kargs)


      if np.shape(self.data) == ():
        self.data = load_quantities(self,var,PLASMA_QUANT='', CYCL_RES='',
                COLFRE_QUANT='', COLFRI_QUANT='', IONP_QUANT='',
                EOSTAB_QUANT='', TAU_QUANT='', DEBYE_LN_QUANT='',
                CROSTAB_QUANT='', COULOMB_COL_QUANT='', AMB_QUANT='', 
                HALL_QUANT='', BATTERY_QUANT='', SPITZER_QUANT='', 
                KAPPA_QUANT='', GYROF_QUANT='', WAVE_QUANT='', 
                FLUX_QUANT='', CURRENT_QUANT='', COLCOU_QUANT='',  
                COLCOUMS_QUANT='', COLFREMX_QUANT='', **kargs)

        # Loading arithmetic quantities
        if np.shape(self.data) == ():
          if self.verbose: 
            print('Loading arithmetic variable',end="\r",flush=True)
          self.data = load_arithmetic_quantities(self,var, **kargs) 
    
    if var == '': 

      print(help(self.get_var))
      print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
      for ii in self.varn: 
          print('use ', ii,' for ',self.varn[ii])
      print(self.description['ALL']) 

      return None
   
    return self.data


  def genvar(self): 
    '''
    Dictionary of original variables which will allow to convert to cgs. 
    '''
    self.varn={}
    self.varn['rho']= 'rho'
    self.varn['tg'] = 'temperature'
    self.varn['e']  = 'energy'
    self.varn['pg']  = 'pressure'
    self.varn['ux'] = 'vx'
    self.varn['uy'] = 'vy'
    self.varn['uz'] = 'vz'
    self.varn['bx'] = 'bx'
    self.varn['by'] = 'by'
    self.varn['bz'] = 'bz'
    self.varn['jx'] = 'jx'
    self.varn['jy'] = 'jy'
    self.varn['jz'] = 'jz'


  def trans2comm(self,varname,snap=None): 
    '''
    Transform the domain into a "common" format. All arrays will be 3D. The 3rd axis 
    is: 

      - for 3D atmospheres:  the vertical axis
      - for loop type atmospheres: along the loop 
      - for 1D atmosphere: the unique dimension is the 3rd axis. 
      At least one extra dimension needs to be created artifically. 

    All of them should obey the right hand rule 

    In all of them, the vectors (velocity, magnetic field etc) away from the Sun. 

    If applies, z=0 near the photosphere. 

    Units: everything is in cgs. 
    
    If an array is reverse, do ndarray.copy(), otherwise pytorch will complain. 

    '''

    self.sel_units = 'cgs'

    self.trans2commaxes

    var = self.get_var(varname,snap=snap).copy()

    #var = transpose(var,(X,X,X))
    # also velocities. 

    return var

  def trans2commaxes(self): 

    if self.transunits == False:
      #self.x =  # including units conversion 
      #self.y = 
      #self.z =
      #self.dx = 
      #self.dy = 
      #self.dz =
      self.transunits = True

  def trans2noncommaxes(self): 

    if self.transunits == True:
      # opposite to the previous function 
      self.transunits = False

class Laresav_units(object): 

    def __init__(self,verbose=False):

        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.verbose=verbose
        self.uni['b']      = 2.0 # Gauss
        self.uni['l']      = 1.0e8 # Mm -> cm
        self.uni['gamma']  = 5./3.
        self.uni['rho']    = 1.67e-15 # gr cm^-3 

        globalvars(self)
        
        mu0=4.e-7*np.pi
        
        self.uni['u']      = self.uni['b']*1e-3 / np.sqrt(mu0 * self.uni['rho']*1e3) * 1e2 # cm/s
        self.uni['tg']     = (self.uni['u']*1e-2)**2 * self.msi_h / self.ksi_b # K
        self.uni['t']      = self.uni['l'] / self.uni['u']  # seconds
        
        # Units and constants in SI
        convertcsgsi(self)

        self.uni['n']      = self.uni['rho'] / self.m_p / 2. # cm^-3