import os
import numpy as np
import scipy.constants as ct
import radynpy as rd
from .load_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from .load_noeos_quantities import *
from math import ceil, floor
from scipy.sparse import coo_matrix
import torch
import imp
try:
    imp.find_module('pycuda')
    found = True
except ImportError:
    found = False
    

class radyn(object):
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
  def __init__(self, filename, *args, fdir='.', 
               sel_units = 'cgs', verbose=True, **kwargs):
        
    self.filename = filename
    self.fdir = fdir
    self.rdobj = rd.cdf.LazyRadynData(filename)
    self.x = np.array([0.0])
    self.y = np.array([0.0])
    self.z = np.flip(self.rdobj.__getattr__('zm'))
    self.sel_units= sel_units
    self.verbose = verbose
    self.snap = None
    self.uni = Radyn_units()
    
    self.dx = np.array([1.0])
    self.dy = np.array([1.0])
    self.dz = np.copy(self.z)
    self.nt = np.shape(self.z)[0]
    self.nz = np.shape(self.z)[1]
    for it in range(0,self.nt):
      self.dz[it,:] = np.gradient(self.z[it,:])
    self.dz1d = self.dz
    self.dx1d = np.array([1.0])
    self.dy1d = np.array([1.0])
    
    self.nx = np.shape(self.x)
    self.ny = np.shape(self.y)
    
    self.time =  self.rdobj.__getattr__('time')

    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

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

      self.data = self.rdobj.__getattr__(varname) * cgsunits
    except: 

      self.data = load_quantities(self,var,PLASMA_QUANT='', CYCL_RES='',
                COLFRE_QUANT='', COLFRI_QUANT='', IONP_QUANT='',
                EOSTAB_QUANT='', TAU_QUANT='', DEBYE_LN_QUANT='',
                CROSTAB_QUANT='', COULOMB_COL_QUANT='', AMB_QUANT='', 
                HALL_QUANT='', BATTERY_QUANT='', SPITZER_QUANT='', 
                KAPPA_QUANT='', GYROF_QUANT='', WAVE_QUANT='', 
                FLUX_QUANT='', CURRENT_QUANT='', COLCOU_QUANT='',  
                COLCOUMS_QUANT='', COLFREMX_QUANT='')

      if np.shape(self.data) == ():
        if self.verbose: 
          print('Loading arithmetic variable',end="\r",flush=True)
        self.data = load_arithmetic_quantities(self,var)    

    if var == '': 

      print(help(self.get_var))
      print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
      for ii in self.varn: 
          print('use ', ii,' for ',self.varn[ii])
      print(self.description['ALL']) 
      print('\n radyn obj is self.rdobj, self.rdobj.var_info is as follows')
      print(self.rdobj.var_info)
    
      return None

    self.trans2noncommaxes()
    
    return self.data

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


  def trans2comm(self,varname,snap=0, **kwargs): 
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

    for key, value in kwargs.items():
      if key == 'dx': 
        if hasattr(self,'trans_dx'):
            if value != self.trans_dx: 
              self.transunits = False
      if key == 'dz': 
        if hasattr(self,'trans_dz'):
            if value != self.trans_dz: 
              self.transunits = False

    if self.snap != snap: 
        self.snap=snap
        self.transunits = False
    
    var = self.get_var(varname)[self.snap]
    
    self.trans2commaxes(**kwargs)
    
    if not hasattr(self,'trans_loop_width'):   
        self.trans_loop_width=1.0
    if not hasattr(self,'trans_sparse'):   
        self.trans_sparse=3e7
            
    for key, value in kwargs.items():
      if key == 'loop_width': 
        self.trans_loop_width = value
      if key == 'unscale': 
        do_expscale = value
      if key == 'sparse': 
        self.trans_sparse = value

    # Semicircular loop
    s = self.rdobj.zm[snap]
    good = s >=0.0
    s = s[good]
    var = var[good]
    ## GSK -- smax was changed 12th March 2021. See comment in trans2commaxes
    #smax = self.rdobj.cdf['zll'][self.snap]
    smax = np.max(self.rdobj.__getattr__('zm'))
    R = 2*smax/np.pi

    ## JMS we are assuming here that self.z.min() = 0
    ## GSK: This isn't true, if you mean the minimum height in RADYN. Z can go sub-photosphere (~60km)
    shape = (ceil(self.x_loop.max()/self.trans_dx), 1, ceil(self.z_loop.max()/self.trans_dx))
    
    # In the RADYN model in the corona, successive grid points may be several pixels away from each other.
    # In this case, need to refine loop.
    do_expscale = False
    for key, value in kwargs.items():
        if key == 'unscale': 
            do_expscale = value
            
    if self.gridfactor > 1:
        if do_expscale: 
            ss, var= refine(s, np.log(var),factor=self.gridfactor, unscale=np.exp)
        else: 
            ss, var= refine(s, var,factor=self.gridfactor)
    else:
        ss = s
    omega = ss/R
    
    # Arc lengths (in radians)
    dA=  np.abs(omega[1:]-omega[0:-1])
    dA = dA.tolist()
    dA.insert(0,dA[0])
    dA.append(dA[-1])
    dA = np.array(dA)
    dA = 0.5*(dA[1:]+dA[0:-1])
    #dA = R*dA*(loop_width*dx)
    dA = 0.5*dA*((R+0.5*self.trans_loop_width*self.trans_dx)**2-(R-0.5*self.trans_loop_width*self.trans_dx)**2)
    
    #Componnets of velocity in the x and z directions
    if varname == 'ux': 
        var = -var*np.sin(omega)
    if varname == 'uz': 
        var =  var*np.cos(omega)

    xind = np.floor(self.x_loop/self.trans_dx).astype(np.int64)
    zind = np.clip(np.floor(self.z_loop/self.trans_dz).astype(np.int64),0,shape[2]-1)

    # Define matrix with column coordinate corresponding to point along loop
    # and row coordinate corresponding to position in Cartesian grid
    col = np.arange(len(self.z_loop),dtype=np.int64)
    row = xind*shape[2]+zind

    if self.trans_sparse:
        M = coo_matrix((dA/(self.trans_dx*self.trans_dz), (row,col)),shape=(shape[0]*shape[2], len(ss)), dtype=np.float)
        M = M.tocsr()
    else:
        M = np.zeros(shape=(shape[0]*shape[2], len(ss)), dtype=np.float)
        M[row,col] = dA/(self.dx1d*self.dz1d.max())  #weighting by area of arc segment
   
    # The final quantity at each Cartesian grid cell is an area-weighted 
    # average of values from loop passing through this grid cell
    # This arrays are not actually used for VDEM extraction
    var = (M@var).reshape(shape)

    self.x = np.linspace(self.x_loop.min(),self.x_loop.max(),np.shape(var)[0])
    self.z = np.linspace(self.z_loop.min(),self.z_loop.max(),np.shape(var)[-1])
    
    self.dx1d = np.gradient(self.x)
    self.dy1d = 1.0
    self.dz1d = np.gradient(self.z)
        
    return var

  def trans2commaxes(self, **kwargs): 

    if self.transunits == False:

      if not hasattr(self,'trans_dx'):   
        self.trans_dx=3e7
      if not hasattr(self,'trans_dz'):   
        self.trans_dz=3e7

      for key, value in kwargs.items():
        if key == 'dx':
          self.trans_dx = value
        if key == 'dz':
          self.trans_dz = value
            
      # Semicircular loop    
      self.zorig = self.rdobj.__getattr__('zm')[self.snap]
      s = np.copy(self.zorig)
      good = s >=0.0
      s = s[good]
      ##JMS -- Sometimes zll is slightly different to the max of zm which causes problems on the assumption of a 1/4 loop. 
      ##        max(zm) fix the problem
      #smax = self.rdobj.cdf['zll'][self.snap]
      smax = np.max(self.rdobj.__getattr__('zm'))
      R = 2*smax/np.pi
      x = np.cos(s/R)*R
      z = np.sin(s/R)*R
    
      shape = (ceil(x.max()/self.trans_dx), ceil(z.max()/self.trans_dz))
    
      # In the RADYN model in the corona, successive grid points may be several pixels away from each other.
      # In this case, need to refine loop.
      maxdl = np.abs(z[1:]-z[0:-1]).max()
      self.gridfactor = ceil(2*maxdl/np.min([self.trans_dx,self.trans_dz]))
            
      if self.gridfactor > 1:
        ss, self.x_loop = refine(s,x, factor=self.gridfactor)
        ss, self.z_loop = refine(s,z, factor=self.gridfactor)
      else:
        self.z_loop = z
        self.x_loop = x
      
      self.y = np.array([0.0])
        
      self.dx1d_loop = np.gradient(self.x_loop)
      self.dy1d = 1.0
      self.dz1d_loop = np.gradient(self.z_loop)
        
      self.transunits = True

  def trans2noncommaxes(self): 

    if self.transunits == True:
      self.x = np.array([0.0])
      self.y = np.array([0.0])
      self.z = self.rdobj.__getattr__('zm')

      self.dx = np.array([1.0])
      self.dy = np.array([1.0])
      self.dz = np.copy(self.z)
      self.nz = np.shape(self.z)[1]
      for it in range(0,self.nt):
        self.dz[it,:] = np.gradient(self.z[it,:])
      self.dz1d = self.dz
      self.dx1d = np.array([1.0])
      self.dy1d = np.array([1.0])

      self.nx = np.shape(self.x)
      self.ny = np.shape(self.y)
      self.transunits = False

    
class Radyn_units(object): 

    def __init__(self,verbose=False):
        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.verbose = verbose 
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
