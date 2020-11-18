import os
import numpy as np
import scipy.constants as ct

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
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        self.read_header("%s/Header%s" % (fdir, template))
        #self.read_atmos(fdir, template)
        self.units()
        self.snap = 0
        self.siter = template
        self.genvar()
        
    def read_header(self, headerfile):
        tmp = np.loadtxt(headerfile)
        self.nx, self.ny, self.nz = tmp[:3].astype("i")
        self.dx, self.dy, self.dz, self.time= tmp[3:7] # km
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.z = np.arange(self.nz) * self.dz

    def read_atmos(self, fdir, template):
        ashape = (self.nx, self.nz, self.ny)
        file_T = "%s/eosT%s" % (fdir, template)
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
        
        Axes: 
        -----
        For the hgcr model:
            y-axis is the vertical x and z-axes are horizontal 
        Newer runs could have x-axis the vertical. 
        
        Variable list: 
        --------------
            result_prim_0 -- Density (g/cm^3)
            eosT          -- Temperature (K)
            result_prim_1 -- component x of the velocity (cm/s) 
            result_prim_2 -- component y of the velocity (cm/s), vertical in the hgcr
            result_prim_3 -- component z of the velocity (cm/s)
            result_prim_4 -- internal energy (erg)
            result_prim_5 -- component x of the magnetic field (G)
            result_prim_6 -- component y of the magnetic field (G)
            result_prim_7 -- component z of the magnetic field (G)
            eosP          -- Pressure 
        '''
        if var == '':
            print(help(self.get_var))
            print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
            for ii in self.varn: 
                print('use ', ii,' for ',self.varn[ii])
            return None
        
        ashape = (self.nx, self.ny, self.nz)
        if (not it == None): 
            self.siter='.'+inttostring(it)
            self.read_header("%s/Header%s" % (self.fdir, self.siter))
       
        if (cgs): 
            varu=var.replace('x','')
            varu=varu.replace('y','')
            varu=varu.replace('z','')
            if var in self.varn.keys(): 
                cgsunits = self.uni[varu]
            else: 
                cgsunits = 1.0
        else: 
            cgsunits = 1.0
        
        if var in self.varn.keys(): 
            varname=self.varn[var]
        else:
            varname=var
            
        data = np.memmap(self.fdir+'/'+varname+ self.siter, mode="r", shape=ashape,
                                dtype=self.dtype,
                                order="F")
        if iix != None: 
            data= data[iix,:,:]
        if iiy != None: 
            data= data[:,iiy,:]
        if iiz != None: 
            data= data[:,:,iiz]
        self.data = data
        
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

    def get_ems(self,iter=None,layout=None, wght_per_h=1.4271, unitsnorm = 1e27, axis=2): 
        '''
        Computes emission meassure in cgs and normalized to unitsnorm
        '''       
        rho = self.get_var('rho',it=iter,layout=layout)
        nh = rho / (wght_per_h * ct.atomic_mass * 1e3)  # from rho to nH and added unitsnorm
        if axis == 0:
            ds = self.dx * self.uni['l']
        elif axis == 1:
            ds = self.dy * self.uni['l']
        else:
            ds = self.dz * self.uni['l']
            
        en = nh + 2.*nh*(wght_per_h-1.) # this may need a better adjustment.             
        nh *= ds

        return en * (nh / unitsnorm )

    def units(self): 
        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.uni['proton'] = 1.67262158e-24 #gr
        self.uni['kboltz'] = 1.380658e-16 
        self.uni['c']      = 299792.458 * 1e5 #cm/s
        self.uni['gamma']  = 5./3.
        self.uni['tg']     = 1.0 # K
        self.uni['l']      = 1.0e5 # to cm
        self.uni['rho']    = 1.0 # g cm^-3 
        self.uni['u']      = 1.0 # cm/s
        self.uni['b']      = 1.0 # Gauss
        self.uni['t']      = 1.0 # seconds
        self.uni['j']      = 1.0 # current density
   
    def genvar(self): 
        '''
        Dictionary of original variables which will allow to convert to cgs. 
        '''
        self.varn={}
        self.varn['rho']= 'result_prim_0'
        self.varn['tg'] = 'eosT'
        self.varn['pg'] = 'eosP'
        self.varn['ux'] = 'result_prim_1'
        self.varn['uy'] = 'result_prim_2'
        self.varn['uz'] = 'result_prim_3'
        self.varn['e']  = 'result_prim_4'
        self.varn['bx'] = 'result_prim_5'
        self.varn['by'] = 'result_prim_6'
        self.varn['bz'] = 'result_prim_7'
        
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

def inttostring(ii,ts_size=7):

  str_num = str(ii)

  for bb in range(len(str_num),ts_size,1):
    str_num = '0'+str_num
  
  return str_num


