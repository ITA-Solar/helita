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
    self.savefile = rsav(os.path.join(self.fdir,'{:03d}'.format(it)+'.sav'))

    self.rootname = self.savefile['d']['filename'][0]
    self.snap = snap 
    self.sel_units = sel_units

    self.time     = self.savefile['d']['time'][0]
    self.time_prev= self.savefile['d']['time_prev'][0]
    self.timestep = self.savefile['d']['timestep'][0]
    self.dt       = self.savefile['d']['dt'][0]

    self.visc_heating= self.savefile['d']['visc_heating'][0]
    self.visc3_heating= self.savefile['d']['visc3_heating'][0]

    self.x       = self.savefile['d']['x'][0]
    self.y       = self.savefile['d']['y'][0]
    self.z       = self.savefile['d']['z'][0]

    self.dx = self.x-np.roll(self.x,1) 
    self.dx[0] = self.dx[1]
    
    self.dy = self.y-np.roll(self.y,1) 
    self.dy[0] = self.dy[1]
    
    self.dz = self.z-np.roll(self.z,1) 
    self.dz[0] = self.dz[1]
    
    #GRID            STRUCT    -> <Anonymous> Array[1]
    
    self.nx = len(self.x)
    self.ny = len(self.y)
    self.nz = len(self.z)

    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

    self.genvar()
    self.units()

  def get_var(self,var,snap=None, iix=None, iiy=None, iiz=None, layout=None): 
    '''
    Reads the variables from a snapshot (snap).

    Parameters
    ----------
    var - string
        Name of the variable to read. Must be Bifrost internal names.
    snap - integer, optional
        Snapshot number to read. By default reads the loaded snapshot;
        if a different number is requested, will load that snapshot.
    cgs- logic 
        converts into cgs units. 
        
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
      self.savefile = rsav(os.path.join(self.fdir,'{:03d}'.format(snap)+'.sav'))

    try: 

      if self.sel_units == 'cgs': 
        varu=var.replace('x','')
        varu=varu.replace('y','')
        varu=varu.replace('z','')
        if (var in self.varn.keys()) and (varu in self.uni.keys()): 
          cgsunits = self.uni[varu]
        else: 
          cgsunits = 1.0
      else: 
        cgsunits = 1.0

      
      self.data = self.savefile['d'][varname][0].T * cgsunits
      
      if (np.shape(self.data)[0]>self.nx): 
          self.data = (self.data[1:,:,:] + self.data[:-1,:,:]) / 2
      
      if (np.shape(self.data)[1]>self.ny): 
          self.data = (self.data[:,1:,:] + self.data[:,:-1,:]) / 2
      
      if (np.shape(self.data)[2]>self.nz): 
          self.data = (self.data[:,:,1:] + self.data[:,:,:-1]) / 2

    except: 
      # Loading quantities
      if self.verbose: 
        print('Loading composite variable',end="\r",flush=True)
      self.data = load_noeos_quantities(self,var)


      if np.shape(self.data) == ():
        self.data = load_quantities(self,var,PLASMA_QUANT='',
                CYCL_RES='', COLFRE_QUANT='', COLFRI_QUANT='',
                IONP_QUANT='', EOSTAB_QUANT='', TAU_QUANT='',
                DEBYE_LN_QUANT='', CROSTAB_QUANT='',
                COULOMB_COL_QUANT='', AMB_QUANT='')

        # Loading arithmetic quantities
        if np.shape(self.data) == ():
          if self.verbose: 
            print('Loading arithmetic variable',end="\r",flush=True)
          self.data = load_arithmetic_quantities(self,var) 
    
    if var != '': 

      print(help(self.get_var))
      print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
      for ii in self.varn: 
          print('use ', ii,' for ',self.varn[ii])
      print(self.description['ALL']) 

      return None
   
    return self.data

  
  def units(self): 
    '''
    Units and constants in cgs
    '''
    self.uni={}
    self.uni['gamma']  = 5./3.
    self.uni['tg']     = 5.77e9 # K
    self.uni['l']      = 1.0e8 # Mm -> cm
    self.uni['rho']    = 1.67e-9 # gr cm^-3 
    self.uni['u']      = 6.9e8 # cm/s
    self.uni['b']      = 100.0 # Gauss
    self.uni['t']      = 0.145 # seconds
    
    # Units and constants in SI

    convertcsgsi(self)

    globalvars(self)

    self.uni['n']      = self.uni['rho'] / self.m_p / 2. # cm^-3

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
      - for 1D atmosphere: the unic dimension is the 3rd axis. 

    All of them should obey the right hand rule 

    In all of them, the vectors (velocity, magnetic field etc) away from the Sun. 
    
    For 1D models, first axis could be time. 

    Units: everything is in cgs. 

    '''

    self.sel_units = 'cgs'

    if self.transunits == False:
      self.transunits == True
      #self.x =  # including units conversion 
      #self.y = 
      #self.z =
      #self.dx = 
      #self.dy = 
      #self.dz =

    var = get_var(varname,snap=snap)

    #var = transpose(var,(X,X,X))
    # also velocities. 

    return var


