import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav

#import radynpy as rd
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
    

class preft(object):
  """
  Class to read preft atmosphere

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
  def __init__(self, filename, snap, *args, fdir='.', 
               sel_units = 'cgs', verbose=True, **kwargs):
        
    self.filename = filename
    self.fdir = fdir

    a = readsav(filename, python_dict=True)
    #self.la = a['la']
    self.snap = snap
    la = a['la']
    l = a['la'][snap]
    
    #self.zmax = (a['la'][0][3][:,2]).max()
    
    self.extent = (0, (a['la'][0][3][:,0]).max()-(a['la'][0][3][:,0]).min(),
                   0, (a['la'][0][3][:,1]).max()-(a['la'][0][3][:,1]).min(),
                   0, (a['la'][0][3][:,2]).max()-(a['la'][0][3][:,2]).min())
    
    self.obj = {'time':np.array([lal[0] for lal in la]),
                's':np.stack([lal[1] for lal in la]),
                'ux':np.stack([lal[2][:,0] for lal in la]),
                'uy':np.stack([lal[2][:,1] for lal in la]),
                'uz':np.stack([lal[2][:,2] for lal in la]),
                'x':np.stack([lal[3][:,0] for lal in la]),
                'y':np.stack([lal[3][:,1] for lal in la]),
                'z':np.stack([lal[3][:,2] for lal in la]),            
                'rho':np.stack([lal[4] for lal in la]),
                'p':np.stack([lal[5] for lal in la]),
                'tg':np.stack([lal[6] for lal in la]),
                'ne':np.stack([lal[7] for lal in la]),
                'b':np.stack([lal[8] for lal in la]),
                'units':l[9]}
    
    #= {'time':l[0],'s':l[1],'ux':l[2][:,0],'uy':l[2][:,1],'uz':l[2][:,2],
    #            'x':l[3][:,0], 'y':l[3][:,1], 'z':l[3][:,2], 
    #            'rho':l[4],'p':l[5],'tg':l[6],'ne':l[7],'b':l[8], 'units':l[9]}
    
    self.x = l[3][:,0].copy()
    self.x-= self.x.min() #np.array([0.0])
    self.y = l[3][:,1].copy()
    self.y-= self.y.min() #np.array([0.0])
    self.z = l[3][:,2].copy() #np.array([0.0])
    #self.z = np.flip(self.rdobj.__getattr__('zm'))
    self.sel_units= sel_units
    self.verbose = verbose
    #self.snap = None
    self.uni = PREFT_units()
    
    #self.dx = np.array([1.0])
    #self.dy = np.array([1.0])
    #self.dz = np.copy(self.z)
    self.nt = [1] #np.shape(self.z)[0]
    self.nz = np.shape(self.z)[0]
    #for it in range(0,self.nt):
    #self.dz[it,:] = np.gradient(self.z[it,:])
    self.dx = np.gradient(self.x)
    self.dy = np.gradient(self.y)
    self.dz = np.gradient(self.z)
    
    self.dz1d = self.dz
    self.dx1d = np.array([1.0])
    self.dy1d = np.array([1.0])
    
    self.nx = np.shape(self.x)
    self.ny = np.shape(self.y)
    
    #self.time =  self.rdobj.__getattr__('time')

    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

    self.genvar()
    

  def get_var(self, var, snap=None, iix=None, iiy=None, iiz=None, layout=None): 
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

    if snap == None:
        snap = self.snap
    if var in self.varn.keys(): 
      varname=self.varn[var]
    else:
      varname=var

    #print(var,varname,'try')
    #print(self.obj.keys())
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
        
        #print(varname)
        self.data = self.obj[varname][snap]*cgsunits  
    
    #self.rdobj.__getattr__(varname) * cgsunits
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
      #print('\n radyn obj is self.rdobj, self.rdobj.var_info is as follows')
      #print(self.rdobj.var_info)
    
      return None

    #self.trans2noncommaxes()
    
    return self.data

  def genvar(self): 
    '''
    Dictionary of original variables which will allow to convert to cgs. 
    '''
    self.varn={}
    self.varn['rho']= 'rho'
    self.varn['tg'] = 'tg'
    self.varn['ux'] = 'ux'
    self.varn['uy'] = 'uy'
    self.varn['uz'] = 'uz'
    self.varn['bx'] = 'bx'
    self.varn['by'] = 'by'
    self.varn['bz'] = 'bz'
    self.varn['ne'] = 'ne'


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

    if not hasattr(self,'trans_dx'):   
        self.trans_dx=3e7
    if not hasattr(self,'trans_dy'):   
        self.trans_dy=3e7
    if not hasattr(self,'trans_dz'):   
        self.trans_dz=3e7

    
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
    
    #var = self.get_var(varname)
    
    #What does this do?
    #self.trans2commaxes(var, **kwargs)
    
    if not hasattr(self,'trans_loop_width'):   
        self.trans_loop_width=1.0
    if not hasattr(self,'trans_sparse'):   
        self.trans_sparse=True
            
    for key, value in kwargs.items():
      if key == 'loop_width': 
        self.trans_loop_width = value
      if key == 'unscale': 
        do_expscale = value
      if key == 'sparse': 
        self.trans_sparse = value
    
    ## GSK -- smax was changed 12th March 2021. See comment in trans2commaxes
    ##smax = s.max() 

    shape = (ceil(self.extent[1]/self.trans_dx), 
             ceil(self.extent[3]/self.trans_dy), 
             ceil(self.extent[5]/self.trans_dz)+1)
    
    # In the PREFT model in the corona, successive grid points may be several pixels away from each other.
    # In this case, need to refine loop.
    #do_expscale = False
    #for key, value in kwargs.items():
    #    if key == 'unscale': 
    #        do_expscale = value
            
    #if self.gridfactor > 1:
    #    if do_expscale: 
    #        ss, var= refine(s, np.log(var),factor=self.gridfactor, unscale=np.exp)
    #    else: 
    #        ss, var= refine(s, var,factor=self.gridfactor)
    #else:
    #    ss = s
    #var_copy = var.copy()

    x_loop  = self.obj['x'][self.snap] 
    y_loop  = self.obj['y'][self.snap] 
    z_loop  = self.obj['z'][self.snap] 
    s_loop  = self.obj['s'][self.snap]
    x_loop -= x_loop.min()
    y_loop -= y_loop.min()
    z_loop -= z_loop.min()

    var = self.get_var(varname,snap=self.snap)

    print(s_loop, var)
    x_loop, y_loop, z_loop, var = self.trans2commaxes(x_loop, y_loop, z_loop, var, s_loop,  **kwargs)
    
    # Arc lengths (in radians)
    dA = np.ones(var.shape)*self.trans_dx*self.trans_dy*self.trans_dz
    xind = np.floor(x_loop/self.trans_dx).astype(np.int64)
    yind = np.floor(y_loop/self.trans_dy).astype(np.int64)
    zind = np.clip(np.floor(z_loop/self.trans_dz).astype(np.int64),0,shape[2]-1)

    # Make copies of loops with an x-offset 
    for xoffset in range(-shape[0],shape[0],10):
        this_snap = self.snap + xoffset + shape[0]
        x_loop  = self.obj['x'][this_snap]
        y_loop  = self.obj['y'][this_snap]
        z_loop  = self.obj['z'][this_snap]
        s_loop  = self.obj['s'][this_snap]
        x_loop -= x_loop.min()
        y_loop -= y_loop.min()
        z_loop -= z_loop.min()
        this_var = self.get_var(varname,snap=this_snap)
        print(this_snap, s_loop.shape)
        
        x_loop, y_loop, z_loop, this_var = self.trans2commaxes(x_loop, y_loop, z_loop, s_loop, this_var, **kwargs)

        
        xind = np.concatenate((xind, np.floor((x_loop+xoffset*self.trans_dx)/self.trans_dx).astype(np.int64)))
        yind = np.concatenate((yind, np.floor(y_loop/self.trans_dy).astype(np.int64)))
        zind = np.concatenate((zind, np.clip(np.floor(z_loop/self.trans_dz).astype(np.int64),0,shape[2]-1)))
        
        dA = np.concatenate((dA, np.ones(var.shape)*self.trans_dx*self.trans_dy*self.trans_dz))
        var = np.concatenate((var,  this_var))

    good = (xind>=0)*(xind<shape[0])
    good*= (yind>=0)*(yind<shape[1])
    good*= (zind>=0)*(zind<shape[2])
    xind = xind[good]
    yind = yind[good]
    zind = zind[good]
    dA   = dA[good]
    var = var[good]
    
    # Define matrix with column coordinate corresponding to point along loop
    # and row coordinate corresponding to position in Cartesian grid
    col = np.arange(len(xind),dtype=np.int64)
    row = (xind*shape[1]+yind)*shape[2]+zind

    if self.trans_sparse:
        M = coo_matrix((dA/(self.trans_dx*self.trans_dy*self.trans_dz), 
                        (row,col)),shape=(shape[0]*shape[1]*shape[2], len(xind)), dtype=np.float)
        M = M.tocsr()
    else:
        M = np.zeros(shape=(shape[0]*shape[1]*shape[2], len(ss)), dtype=np.float)
        M[row,col] = dA/(self.dx1d*self.dz1d.max())  #weighting by area of arc segment
   
    # The final quantity at each Cartesian grid cell is an volume-weighted 
    # average of values from loop passing through this grid cell
    # This arrays are not actually used for VDEM extraction
    var = (M@var).reshape(shape)

    self.x = np.linspace(self.x_loop.min(),self.x_loop.max(),np.shape(var)[0])
    self.y = np.linspace(self.y_loop.min(),self.y_loop.max(),np.shape(var)[1])
    self.z = np.linspace(self.z_loop.min(),self.z_loop.max(),np.shape(var)[2])
    
    self.dx1d = np.gradient(self.x)
    self.dy1d = np.gradient(self.y)
    self.dz1d = np.gradient(self.z)
        
    return var

  def trans2commaxes(self, x, y, z, s, var, **kwargs): 

    if self.transunits == False:

      if not hasattr(self,'trans_dx'):   
        self.trans_dx=3e7
      if not hasattr(self,'trans_dy'):   
        self.trans_dy=3e7
      if not hasattr(self,'trans_dz'):   
        self.trans_dz=3e7

      for key, value in kwargs.items():
        if key == 'dx':
          self.trans_dx = value
        if key == 'dy':
          self.trans_dy = value
        if key == 'dz':
          self.trans_dz = value
            
      # Semicircular loop    
      #s = self.obj['s'] #np.copy(self.zorig)
      #good = (s >=0.0)
      #s = s[good]
         
      #x = self.x 
      #y = self.y
      #z = self.z 
    
      #shape = (ceil(x.max()/self.trans_dx),ceil(y.max()/self.trans_dy), ceil(self.zmax/self.trans_dz))
    
      # In the RADYN model in the corona, successive grid points may be several pixels away from each other.
      # In this case, need to refine loop.
      maxdl = np.sqrt((z[1:]-z[0:-1])**2+ (x[1:]-x[0:-1])**2 + (y[1:]-y[0:-1])**2 ).max()
      gridfactor = ceil(2*maxdl/np.min([self.trans_dx,self.trans_dy,self.trans_dz]))
    
      do_expscale = False
      for key, value in kwargs.items():
        if key == 'unscale': 
            do_expscale = value

      if gridfactor > 1:
        print(s,x)
        ss, x_loop = refine(s,x, factor=gridfactor)
        ss, y_loop = refine(s,y, factor=gridfactor)
        ss, z_loop = refine(s,z, factor=gridfactor)
        if do_expscale: 
            ss, var_loop = refine(s, np.log(var),factor=gridfactor, unscale=np.exp)
        else: 
            ss, var_loop = refine(s, var,factor=gridfactor)
      else:
        x_loop = x.copy()
        y_loop = y.copy()
        z_loop = z.copy()
        var_loop = var.copy()

      self.dx1d_loop = np.gradient(x_loop)
      self.dy1d_loop = np.gradient(y_loop)
      self.dz1d_loop = np.gradient(z_loop)
    
      self.transunits = True
      return x_loop, y_loop, z_loop, var_loop
        

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

    
class PREFT_units(object): 

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
