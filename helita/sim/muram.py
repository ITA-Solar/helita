import os
import numpy as np
import scipy.constants as ct
from astropy.io import fits
from .tools import *
from .load_quantities import *
from .load_arithmetic_quantities import *
from .bifrost import Rhoeetab 

class MuramAtmos:
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

  def __init__(self, fdir='.', template=".020000", verbose=True, dtype='f4',
               big_endian=False, prim=False):

    self.prim = prim
    self.fdir = fdir
    self.verbose = verbose
    self.sel_units  = sel_units
    # endianness and data type
    if big_endian:
        self.dtype = '>' + dtype
    else:
        self.dtype = '<' + dtype
    self.read_header("%s/Header%s" % (fdir, template))
    
    #self.read_atmos(fdir, template)
    self.units()
    # Snapshot number
    self.snap = int(template[1:])
    self.filename=''
    self.siter = template
    self.file_root = template

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load
    tabfile = os.path.join(self.fdir, 'tabparam.in')

    if os.access(tabfile, os.R_OK):
        self.rhoee = Rhoeetab(tabfile=tabfile,fdir=fdir,radtab=False)

    self.genvar()

      
  def read_header(self, headerfile):
    tmp = np.loadtxt(headerfile)
    dims = tmp[:3].astype("i")
    deltas = tmp[3:6]
    if len(tmp) == 10: # Old version of MURaM, deltas stored in km
        self.uno['l'] = 1e5 # JMS What is this for? 
    time= tmp[7]
    self.order = tmp[-3:].astype(int)
    dims = dims[self.order]
    deltas = deltas[self.order]

    self.x = np.arange(dims[0])*deltas[0]
    self.y = np.arange(dims[1])*deltas[1]
    self.z = np.arange(dims[2])*deltas[2]
    self.dx, self.dy, self.dz = deltas[0], deltas[1], deltas[2]
    self.nx, self.ny, self.nz = dims[0], dims[1], dims[2]


  def read_atmos(self, fdir, template):
    ashape = (self.nx, self.nz, self.ny)
    file_T = "%s/eosT%s" % (fdir, template)
    #When 0-th dimension is vertical, 1st is x, 2nd is y
    # when 1st dimension is vertical, 0th is x. 
    # remember to swap names
    
    bfact = np.sqrt(4 * np.pi)
    if os.path.isfile(file_T):
        self.tg = np.memmap(file_T, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
    file_press = "%s/eosP%s" % (fdir, template)
    if os.path.isfile(file_press):
        self.pressure = np.memmap(file_press, mode="r", shape=ashape,
                                  dtype=self.dtype,
                                  order="F")
    file_rho = "%s/result_prim_0%s" % (fdir, template)
    if os.path.isfile(file_rho):
        self.rho = np.memmap(file_rho, mode="r", shape=ashape,
                             dtype=self.dtype,
                             order="F")
    file_vx = "%s/result_prim_1%s" % (fdir, template)
    if os.path.isfile(file_vx):
        self.vx = np.memmap(file_vx, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
    file_vz = "%s/result_prim_2%s" % (fdir, template)
    if os.path.isfile(file_vz):
        self.vz = np.memmap(file_vz, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
    file_vy = "%s/result_prim_3%s" % (fdir, template)
    if os.path.isfile(file_vy):
        self.vy = np.memmap(file_vy, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
    file_ei = "%s/result_prim_4%s" % (fdir, template)
    if os.path.isfile(file_ei):
        self.ei = np.memmap(file_ei, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
    file_Bx = "%s/result_prim_5%s" % (fdir, template)
    if os.path.isfile(file_Bx):
        self.bx = np.memmap(file_Bx, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
        self.bx = self.bx * bfact
    file_Bz = "%s/result_prim_6%s" % (fdir, template)
    if os.path.isfile(file_Bz):
        self.bz = np.memmap(file_Bz, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
        self.bz = self.bz * bfact
    file_By = "%s/result_prim_7%s" % (fdir, template)
    if os.path.isfile(file_By):
        self.by = np.memmap(file_By, mode="r", shape=ashape,
                            dtype=self.dtype,
                            order="F")
        self.by = self.by * bfact
    file_tau = "%s/tau%s" % (fdir, template)
    if os.path.isfile(file_tau):
        self.tau = np.memmap(file_tau, mode="r", shape=ashape,
                             dtype=self.dtype,
                             order="F")
    file_Qtot = "%s/Qtot%s" % (fdir, template)
    if os.path.isfile(file_Qtot):
        self.qtot = np.memmap(file_Qtot, mode="r", shape=ashape,
                              dtype=self.dtype,
                              order="F")

    # from moments to velocities
    #if self.prim:
    #    if hasattr(self,'rho'): 
    #        if hasattr(self,'vx'):
    #            self.vx /= self.rho
    #        if hasattr(self,'vy'):
    #            self.vy /= self.rho
    #        if hasattr(self,'vz'):
    #            self.vz /= self.rho


  def read_Iout(self):

    tmp = np.fromfile(self.fdir+'I_out.'+self.siter)

    size = tmp[1:3].astype(int)
    time = tmp[3]

    return tmp[4:].reshape([size[1],size[0]]).swapaxes(0,1),size,time


  def read_slice(self,var,depth):

    tmp = np.fromfile(self.fdir+var+'_slice_'+depth+'.'+self.siter)

    nslices = tmp[0].astype(int)
    size = tmp[1:3].astype(int)
    time = tmp[3]

    return tmp[4:].reshape([nslices,size[1],size[0]]).swapaxes(1,2),nslices,size,time


  def read_dem(self,path,max_bins=None):

    tmp = np.fromfile(path+'corona_emission_adj_dem_'+self.fdir+'.'+self.siter)

    bins   = tmp[0].astype(int)
    size   = tmp[1:3].astype(int)
    time   = tmp[3]
    lgTmin = tmp[4]
    dellgT = tmp[5]
    
    dem = tmp[6:].reshape([bins,size[1],size[0]]).transpose(2,1,0)
    
    taxis = lgTmin+dellgT*np.arange(0,bins+1)
    
    X_H = 0.7
    dem = dem*X_H*0.5*(1+X_H)*3.6e19
    
    if max_bins != None:
      if bins > max_bins :
        dem = dem[:,:,0:max_bins]
      else :
        tmp=dem
        dem=np.zeros([size[0],size[1],max_bins])
        dem[:,:,0:bins]=tmp
        
      taxis = lgTmin+dellgT*np.arange(0,max_bins+1)
    
    return dem,taxis,time


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
    For the hgcr model:
        y-axis is the vertical x and z-axes are horizontal 
    Newer runs could have x-axis the vertical. 
    
    Variable list: 
    --------------
        result_prim_0 -- Density (g/cm^3)
        result_prim_1 -- component x of the velocity (cm/s) 
        result_prim_2 -- component y of the velocity (cm/s), vertical in the hgcr
        result_prim_3 -- component z of the velocity (cm/s)
        result_prim_4 -- internal energy (erg)
        result_prim_5 -- component x of the magnetic field (G/sqrt(4*pi))
        result_prim_6 -- component y of the magnetic field (G/sqrt(4*pi))
        result_prim_7 -- component z of the magnetic field (G/sqrt(4*pi))
        eosP          -- Pressure (cgs)
        eosT          -- Temperature (K)
    '''
    
    if (not snap == None): 
      self.snap = snap 
      self.siter = '.'+inttostring(snap)
      self.read_header("%s/Header%s" % (self.fdir, self.siter))
   
    
    if var in self.varn.keys(): 
      varname=self.varn[var]
    else:
      varname=var

    try: 
    
      ashape = np.array([self.nx, self.ny, self.nz])
    
      transpose_order = self.order

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

      data = np.memmap(self.fdir+'/'+varname+ self.siter, mode="r", 
                      shape=tuple(ashape[self.order]),
                      dtype=self.dtype, order="F")
      data = data.transpose(transpose_order)
    
      if iix != None: 
        data= data[iix,:,:]
      if iiy != None: 
        data= data[:,iiy,:]
      if iiz != None: 
        data= data[:,:,iiz]

      self.data = data *cgsunits

    except:
      # Loading quantities
      if self.verbose: 
        print('Loading composite variable',end="\r",flush=True)
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

  def read_var_3d(self,var,iter=None,layout=None):

    if (not iter == None): 
      self.siter='.'+inttostring(iter)
      self.read_header("%s/Header%s" % (self.fdir, self.siter))

    tmp = np.fromfile(self.fdir+'/'+var+ self.siter)
    self.data = tmp.reshape([self.nx,self.ny,self.nz])
      
    if layout != None :
        self.data = tmp.transpose(layout)
    
    return self.data


  def read_vlos(self,path,max_bins=None):

    tmp = np.fromfile(path+'corona_emission_adj_vlos_'+self.fdir+'.'+self.siter)

    bins   = tmp[0].astype(int)
    size   = tmp[1:3].astype(int)
    time   = tmp[3]
    lgTmin = tmp[4]
    dellgT = tmp[5]
    
    vlos = tmp[6:].reshape([bins,self.ny,self.nz]).transpose(2,1,0)
    
    taxis = lgTmin+dellgT*np.arange(0,bins+1)
   
    if max_bins != None:
      if bins > max_bins :
        vlos = vlos[:,:,0:max_bins]
      else :
        tmp=vlos
        vlos=np.zeros([self.nz,self.ny,max_bins])
        vlos[:,:,0:bins]=tmp
        
      taxis = lgTmin+dellgT*np.arange(0,max_bins+1)
    
    return vlos,taxis,time


  def read_vrms(self,path,max_bins=None):

    tmp = np.fromfile(path+'corona_emission_adj_vrms_'+self.fdir+'.'+self.template)

    bins   = tmp[0].astype(int)
    size   = tmp[1:3].astype(int)
    time   = tmp[3]
    lgTmin = tmp[4]
    dellgT = tmp[5]
    
    vlos = tmp[6:].reshape([bins,self.ny,self.nz]).transpose(2,1,0)
    
    taxis = lgTmin+dellgT*np.arange(0,bins+1)
   
    if max_bins != None:
      if bins > max_bins :
        vlos = vlos[:,:,0:max_bins]
      else :
        tmp=vlos
        vlos=np.zeros([self.nz,self.ny,max_bins])
        vlos[:,:,0:bins]=tmp
        
      taxis = lgTmin+dellgT*np.arange(0,max_bins+1)
    
    return vlos,taxis,time


  def read_fil(self,path,max_bins=None):

    tmp = np.fromfile(path+'corona_emission_adj_fil_'+self.fdir+'.'+self.template)
    bins   = tmp[0].astype(int)
    size   = tmp[1:3].astype(int)
    time   = tmp[3]
    lgTmin = tmp[4]
    dellgT = tmp[5]
    
    vlos = tmp[6:].reshape([bins,size[1],size[0]]).transpose(2,1,0)
    
    taxis = lgTmin+dellgT*np.arange(0,bins+1)
   
    if max_bins != None:
      if bins > max_bins :
        vlos = vlos[:,:,0:max_bins]
      else :
        tmp=vlos
        vlos=np.zeros([size[0],size[1],max_bins])
        vlos[:,:,0:bins]=tmp
        
      taxis = lgTmin+dellgT*np.arange(0,max_bins+1)
    
    return vlos,taxis,time


  def units(self): 
    '''
    Units and constants in cgs
    '''
    self.uni={}
    self.uni['tg']     = 1.0 # K
    self.uni['l']      = 1.0e5 # to cm
    self.uni['rho']    = 1.0 # g cm^-3 
    self.uni['u']      = 1.0 # cm/s
    self.uni['b']      = np.sqrt(4.0*np.pi) # convert to Gauss
    self.uni['t']      = 1.0 # seconds
    self.uni['j']      = 1.0 # current density 

    # Units and constants in SI
    convertcsgsi(self)

    globalvars(self)


  def genvar(self, order=[0,1,2]): 
    '''
    Dictionary of original variables which will allow to convert to cgs. 
    '''
    self.varn={}
    self.varn['rho']= 'result_prim_0'
    self.varn['tg'] = 'eosT'
    self.varn['pg'] = 'eosP'
    unames = np.array(['result_prim_1','result_prim_2','result_prim_3']) 
    unames = unames[order]
    self.varn['ux'] = unames[0]
    self.varn['uy'] = unames[1]
    self.varn['uz'] = unames[2]
    self.varn['e']  = 'result_prim_4'
    unames = np.array(['result_prim_5','result_prim_6','result_prim_7'])
    unames = unames[order]
    self.varn['bx'] = unames[0]
    self.varn['by'] = unames[1]
    self.varn['bz'] = unames[2]

      
  def write_rh15d(self, outfile, desc=None, append=True, writeB=False,
                  sx=slice(None), sy=slice(None), sz=slice(None),
                  wght_per_h=1.4271):
    ''' Writes RH 1.5D NetCDF snapshot '''
    from . import rh15d
    import scipy.constants as ct
    from .bifrost import Rhoeetab
    # unit conversion to SI
    ul = 1.e-2  # to metres
    ur = 1.e3   # from g/cm^3 to kg/m^3
    ut = 1.     # to seconds
    uv = ul / ut
    ub = 1.e-4  # to Tesla
    ue = 1.      # to erg/g
    # slicing and unit conversion (default slice of None selects all)
    if sx != slice(None):
        sx = slice(sx[0], sx[1], sx[2])
    if sy != slice(None):
        sy = slice(sy[0], sy[1], sy[2])
    if sz != slice(None):
        sz = slice(sz[0], sz[1], sz[2])
    print('Slicing and unit conversion...')
    temp = self.tg[sx, sy, sz]
    rho = self.rho[sx, sy, sz]
    rho = rho * ur
    if writeB:
        Bx = self.bx[sx, sy, sz]
        By = self.by[sx, sy, sz]
        Bz = self.bz[sx, sy, sz]
        Bx = Bx * ub
        By = By * ub
        Bz = Bz * ub
    else:
        Bx, By, Bz = [None] * 3
    vx = self.vx[sx, sy, sz] * ul
    vy = self.vy[sx, sy, sz] * ul
    vz = self.vz[sx, sy, sz] * ul
    x = self.x[sx] * ul
    y = self.y[sy] * ul
    z = self.z[sz] * ul
    # convert from rho to H atoms
    nh = rho / (wght_per_h * ct.atomic_mass)  # from rho to nH
    # description
    if desc is None:
        desc = 'MURAM shapshot sequence %s, sx=%s sy=%s sz=%s.' % \
               (self.fdir, repr(sx), repr(sy), repr(sz))
    # write to file
    print('Write to file...')
    rh15d.make_xarray_atmos(outfile, temp, vz, z, nH=nh, x=x, y=y, vx=vx,
                            vy=vy, rho=rho, append=append, Bx=Bx, By=By,
                            Bz=Bz, desc=desc, snap=self.snap)


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

    return get_var(varname,snap=snap)


  def trasn2fits(self, varname, snap=None, instrument = 'MURaM', 
    name='ar098192', origin='HGCR    '): 
    '''
    converts the original data into fits files following Bifrost publicly available 
    format, i.e., SI, vertical axis, z and top corona is positive and last index. 
    '''

    var = self.trans2comm(varname,snap=snap)
    
    if varname[-1]=='y': 
      varname==varname[:-1]+'z'
    elif varname=='z': 
      varname==varname[:-1]+'y'

    varu=varname.replace('x','')
    varu=varu.replace('y','')
    varu=varu.replace('z','')
    varu=varu.replace('lg','')
    if (varname in self.varn.keys()) and (varu in self.uni.keys()): 
      siunits = self.unisi[varu]/self.uni[varu]
    else: 
      siunits = 1.0

    units_title(self)

    self.datafits=np.transpose(var,(1,2,0))
    if varname[:2] == 'lg': 
      self.datafits = self.datafits + np.log10(siunits) # cgs -> SI
    else: 
      self.datafits = self.datafits * siunits

    self.fitsunits = self.unisi_title[varu]

    self.xfits=self.x / 1e8
    self.yfits=self.z / 1e8
    self.zfits=self.y / 1e8

    self.dxfits=self.dx / 1e8
    self.dyfits=self.dz / 1e8
    self.dzfits=self.dy / 1e8

    writefits(self,varname, instrument = instrument, name=name, origin=origin)




