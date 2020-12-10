import os
import numpy as np
import scipy.constants as ct
import radynpy as rd
from .load_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from .load_noeos_quantities import *

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
  def __init__(self, filename, fdir='.', sel_units = 'cgs', verbose=True):
    
    self.filename = filename
    self.fdir = fdir
    self.rdobj = rd.cdf.LazyRadynData(filename)
    self.x = 0.0
    self.y = 0.0
    self.z = self.rdobj.__getattr__('zm')
    self.sel_units= sel_units
    
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

    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

    self.units()
    self.genvar()

  def get_var(self, var, iix=None, iiy=None, iiz=None, layout=None): 
    '''
    Reads the variables from a snapshot (it).

    Parameters
    ----------
    var - string
        Name of the variable to read.
 
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

    if var in self.varn.keys(): 
      varname=self.varn[var]
    else:
      varname=var

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

      self.data = self.rdobj.__getattr__(varname) * cgsunits
    except: 

      self.data = load_quantities(self,var,PLASMA_QUANT='',
                  CYCL_RES='', COLFRE_QUANT='', COLFRI_QUANT='',
                  IONP_QUANT='', EOSTAB_QUANT='', TAU_QUANT='',
                  DEBYE_LN_QUANT='', CROSTAB_QUANT='',
                  COULOMB_COL_QUANT='', AMB_QUANT='')

      if np.shape(self.data) == ():
        if self.verbose: 
          print('Loading arithmetic variable',end="\r",flush=True)
        self.data = load_arithmetic_quantities(self,var)    

    return self.data
  

  def units(self): 
    '''
    Units and constants in cgs
    '''
    self.uni={}

    self.uni['tg']     = 1.0
    self.uni['l']      = 1.0
    self.uni['n']      = 1.0
    self.uni['rho']    = 1.0
    self.uni['u']      = 1.0
    self.uni['b']      = 1.0
    self.uni['t']      = 1.0 # seconds
    self.uni['j']      = 1.0
        
    # Units and constants in SI
    convertcsgsi(self)

    globalvars(self)
 

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
    self.varn['ne']= 'ne1'


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

