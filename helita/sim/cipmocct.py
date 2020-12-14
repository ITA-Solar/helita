import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav
from .load_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from .load_noeos_quantities import *

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
  snap : integer 
      snapshot number 
  """

  def __init__(self, rootname, snap, fdir='.', sel_units = 'cgs', verbose=True):
      
    self.rootname = rootname
    self.fdir = fdir
    self.snap = snap 
    self.sel_units = sel_units
    self.verbose = verbose
    self.uni = Cipmocct_units()
    
    params = rsav(os.path.join(self.fdir,'params_'+rootname+'.sav'))
    
    self.x = params['x1']
    self.y = params['x2']
    self.z = params['x3']
    
    self.nx = len(params['x1'])
    self.ny = len(params['x2'])
    self.nz = len(params['x3'])

    if self.sel_units=='cgs': 
        self.x *= self.uni.uni['l']
        self.y *= self.uni.uni['l']
        self.z *= self.uni.uni['l']

    if self.nx > 1:
        self.dx1d = np.gradient(self.x) 
        self.dx = self.dx1d
    else: 
        self.dx1d = np.zeros(self.nx)
        self.dx = self.dx1d
        
    if self.ny > 1:            
        self.dy1d = np.gradient(self.y) 
        self.dy = self.dy1d
    else:
        self.dy1d = np.zeros(self.ny)
        self.dy = self.dy1d
        
    if self.nz > 1:
        self.dz1d = np.gradient(self.z)
        self.dz = self.dz1d
    else:
        self.dz1d = np.zeros(self.nz)
        self.dz = self.dz1d
        
    self.time =  params['time'] # No uniforme (array)

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

    
    if snap != None: 
        self.snap = snap

    if var in self.varn.keys(): 
        varname=self.varn[var]
    else:
        varname=var

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

      itname = '_'+inttostring(self.snap,ts_size=4)
    
      varfile = rsav(self.fdir+'vars_'+self.rootname+itname+'.sav')
      self.data = varfile[varname] * cgsunits

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
          self.data = load_arithmetic_quantities(self, var, **kargs) 
    
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
    self.varn['rho']= 'ro_cube'
    self.varn['tg'] = 'te_cube'
    self.varn['ux'] = 'vx_cube'
    self.varn['uy'] = 'vy_cube'
    self.varn['uz'] = 'vz_cube'
    self.varn['bx'] = 'bx_cube'
    self.varn['by'] = 'by_cube'
    self.varn['bz'] = 'bz_cube'


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

    var = self.get_var(varname,snap=snap)

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

class Cipmocct_units(object): 

    def __init__(self):

        '''
        Units and constants in cgs
        '''
        self.uni={}

        self.uni['gamma']  = 5./3.
        self.uni['tg']     = 1.0e6 # K
        self.uni['fact']   = 2
        self.uni['l']      = 1000.*self.uni['fact']*1.0e5 # for having a 2000 km wide loop
        self.uni['n']      = 1.0e9 # cm^-3    
        self.uni['t']      = 1.0 # seconds

        # Units and constants in SI
        convertcsgsi(self)

        globalvars(self)

        self.uni['rho']    = self.uni['n'] * self.uni['proton'] /2. # gr cm^-3 
        self.uni['u']      = np.sqrt(2*self.uni['gamma']*self.k_b/self.m_p*self.uni['tg']) # cm/s
        self.uni['b']      = self.uni['u']*np.sqrt(self.uni['rho']) # Gauss
        self.uni['j']      = self.uni['b']/self.uni['l']*self.uni['c'] # current density


