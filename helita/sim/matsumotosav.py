import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav
from .load_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from .load_noeos_quantities import *

class Matsumotosav:
  """
  Class to read Matsumoto's sav file atmosphere. 
  Snapshots from a MHD simulation ( Matsumoto 2018 )
  https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.3328M/abstract

  Parameters
  ----------
  fdir : str, optional
      Directory with snapshots.
  rootname : str
      Template for snapshot number.
  it : integer
      Snapshot number to read. By default reads the loaded snapshot;
      if a different number is requested, will load that snapshot.
  verbose : bool, optional
      If True, will print more information.
  """
  def __init__(self, rootname, snap, fdir='.', sel_units = 'cgs', verbose=True):

    self.fdir     = fdir        
    self.rootname = rootname
    self.savefile = rsav(os.path.join(fdir,rootname+'{:06d}'.format(snap)+'.sav'))
    self.snap     = snap
    self.sel_units= sel_units

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
        
    Axes: 
    -----
        y and z axes horizontal plane
        x-axis is vertical axis, top corona is first index and negative. 
    
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

    if snap != None: 
        self.snap = snap
        self.savefile = rsav(os.path.join(self.fdir,self.rootname+'{:06d}'.format(self.snap)+'.sav'))

    if var in self.varn.keys(): 
        varname=self.varn[var]
    else:
        varname=var

    if var != '':

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

        
        self.data = self.savefile['v'][varname][0].T * cgsunits
        
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
                      
      return self.data
    
    else: 

      print(help(self.get_var))
      print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
      for ii in self.varn: 
          print('use ', ii,' for ',self.varn[ii])
      print(self.description['ALL']) 

      return None

  
  def units(self): 
    '''
    Units and constants in cgs
    '''
    self.uni={}
    self.uni['tg']     = 1.0 # K
    self.uni['l']      = 1.0e8 # Mm -> cm
    self.uni['rho']    = 1.0 # gr cm^-3 
    self.uni['n']      = 1.0 # cm^-3
    self.uni['u']      = 1.0 # cm/s
    self.uni['b']      = 1.0 # Gauss
    self.uni['t']      = 1.0 # seconds

    # Units and constants in SI
    convertcsgsi(self)

    globalvars(self)


    self.uni['gamma']  = self.gamma 

 
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

