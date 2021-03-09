import os
import numpy as np
import scipy.constants as ct
from .load_quantities import *
from .load_noeos_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from scipy.io import FortranFile

class Mah:
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
  def __init__(self, run_name, snap, fdir='.', sel_units = 'cgs', verbose=True, 
                num_pts = 300, ngridc = 256, nzc = 615, nzc5sav = 7, nt5sav=1846):

    self.fdir     = fdir        
    self.rootname = run_name
    self.snap = snap 
    self.sel_units = sel_units
    self.verbose = verbose
    self.uni = Mah_units()
    
    self.read_ini()
    self.read_dat1()
    self.read_dat2()
    self.read_dat3()
    self.read_dat4()
    self.read_dat5()
    self.read_dat6()


    #self.x = dd.input_ini['xpos']
    #self.z = dd.input_ini['zpos']
    self.z = self.input_ini['spos']
    if self.sel_units=='cgs': 
    #    self.x *= self.uni.uni['l']
    #    self.y *= self.uni.uni['l']
        self.z *= self.uni.uni['l']
    
    self.num_pts  = num_pts
    self.nx = ngridc
    self.ny = ngridc
    self.nz = ngridc
    
    ''' 
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
    ''' 
    
    self.transunits = False

    self.cstagop = False # This will not allow to use cstagger from Bifrost in load
    self.hion = False # This will not allow to use HION from Bifrost in load

    self.genvar()

  def read_ini(self):
    f = open('%s.ini'%self.rootname,'rb')
    varnamenew=['unk1','opt',
                'unk2','unk3','kmaxc','nsizec','ngridc','nzc','ntube', 
                'unk4','unk5', 'kmaxt','nsizet','ngridt','nza','nzb',
                'unk6','unk7', 'nlev_max','max_section','max_jump', 'max_step',
                'unk8','unk9', 'ntmax', 'ndrv', 'nzc4sav','nq4sav','nm3sav',
                'unk10','unk11', 'nza2sav','nzb2sav','nzc2sav','nzc5sav','num_pts',
                'unk12','unk13', 'nt1sav','nt1del','nt2sav','nt2del',
                'unk14','unk15', 'nt5sav','nt5del','nt6sav','nt6del',
                'unk16','unk17','nmodec','kmax1dc']
    input = np.fromfile(f,dtype='int32',count=np.size(varnamenew))
    input_dic={}
    for idx,iv in enumerate(varnamenew):
        input_dic[iv] = input[idx]
    input_dic['amaxc'] = np.fromfile(f,dtype='float32',count=1)
    input_dic['nnxc'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxc'])
    input_dic['nnyc'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxc'])
    input_dic['aaxc'] = np.fromfile(f,dtype='float32',count=input_dic['kmaxc'])
    input_dic['aayc'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxc'])
    input_dic['aac'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxc'])
    varnamenew2=['unk18','unk19','nmodet','kmax1dt'] 
    input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
    for idx,iv in enumerate(varnamenew2):
        input_dic[iv] = input2[idx]
    input_dic['amaxt'] = np.fromfile(f,dtype='float32',count=1)
    input_dic['nnxt'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxt'])
    input_dic['nnyt'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxt'])
    input_dic['aaxt'] = np.fromfile(f,dtype='float32',count=input_dic['kmaxt'])
    input_dic['aayt'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxt'])
    input_dic['aat'] = np.fromfile(f,dtype='int32',count=input_dic['kmaxt'])
    unk2021 = np.fromfile(f,dtype='int32',count=2)
    input_dic['nza2arr'] = np.fromfile(f,dtype='int32',count=input_dic['nza2sav'])
    input_dic['nzb2arr'] = np.fromfile(f,dtype='int32',count=input_dic['nzb2sav'])
    input_dic['nzc2arr'] = np.fromfile(f,dtype='int32',count=input_dic['nzc2sav'])
    unk2223 = np.fromfile(f,dtype='int32',count=2)
    input_dic['nzc4arr'] = np.fromfile(f,dtype='int32',count=input_dic['nzc4sav'])
    input_dic['aa_casc'] = np.fromfile(f,dtype='float32',count=input_dic['nq4sav'])
    input_dic['km3arr'] = np.fromfile(f,dtype='int32',count=input_dic['nm3sav'])
    unk2425 = np.fromfile(f,dtype='int32',count=2)
    input_dic['nzc5arr'] = np.fromfile(f,dtype='int32',count=input_dic['nzc5sav'])
    input_dic['xpts_ini'] = (np.fromfile(f,dtype='float32',count=input_dic['num_pts']*input_dic['nzc5sav'])).reshape((input_dic['num_pts'],input_dic['nzc5sav']))
    input_dic['ypts_ini'] = (np.fromfile(f,dtype='float32',count=input_dic['num_pts']*input_dic['nzc5sav'])).reshape((input_dic['num_pts'],input_dic['nzc5sav']))

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew3 = ['dt0', 'tmin0','tmax0']
    for idx,iv in enumerate(varnamenew3):
        input_dic[iv] =  np.fromfile(f,dtype='float32',count=1)
    varnamenew4= ['imer1','imer2','itr1', 'itr2']
    for idx,iv in enumerate(varnamenew4):
        input_dic[iv] = np.fromfile(f,dtype='int32',count=1)
    varnamenew5= ['str1','str2']
    for idx,iv in enumerate(varnamenew5):
        input_dic[iv] = np.fromfile(f,dtype='float64',count=1)
    input_dic['ztr0']  = np.fromfile(f,dtype='float32',count=1)
    varnamenew6 = ['xoff1', 'yoff1', 'xoff2','yoff2']
    for idx,iv in enumerate(varnamenew6):
        input_dic[iv] = np.fromfile(f,dtype='int32',count=1)
    varnamenew7= ['xm0','zm0','hm0','bm0','len_tot','tau0','vbase0','aat1_drv',
                  'aat2_drv','bphot','width_base','temp_chrom','fact_chrom',
                  'fipfactor','nexpo','len_cor','pres_cor','temp_max','bm_cor',
                  'nu','nutim','nuconpc','nuconmc','aac1_mid','aac2_mid','nuconpt',
                  'nuconmt','at1_mid','aat2_mid']
    for idx,iv in enumerate(varnamenew7):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=1)


    input_dic['nz']=input_dic['nzc']+input_dic['nza']+input_dic['nzb']-2
    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew8 = ['trav','spos']  # trav is the total travel time
    for idx,iv in enumerate(varnamenew8):
        input_dic[iv] = np.fromfile(f,dtype='float64',count=input_dic['nz'])
    varnamenew9 = ['xpos','zpos','gamm','grav','temp','mu','pres','rho',
                   'bmag','va']
    for idx,iv in enumerate(varnamenew9):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=input_dic['nz'])
    input_dic['nlev'] = np.fromfile(f,dtype='int32',count=input_dic['nz']-1)
    input_dic['dva'] = np.fromfile(f,dtype='float32',count=input_dic['nz']-1)

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew9=['vel','qwav','qrad','fent','fcon','zzp_scale','zzm_scale']
    for idx,iv in enumerate(varnamenew9):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=input_dic['nz'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    input_dic['num_section'] = np.fromfile(f,dtype='int32',count=input_dic['nlev_max']+1)
    input_dic['section'] = np.fromfile(f,dtype='int32',
        count=2*input_dic['max_section']*(input_dic['nlev_max']+1)).reshape((2,input_dic['max_section'],input_dic['nlev_max']+1))
    input_dic['num_jump'] = np.fromfile(f,dtype='int32',count=input_dic['nlev_max'])
    input_dic['jump'] = np.fromfile(f,dtype='int32',
        count=2*input_dic['max_jump']*(input_dic['nlev_max'])).reshape((2,input_dic['max_jump'],input_dic['nlev_max']))
    input_dic['list'] = np.fromfile(f,dtype='int32',count=3*input_dic['max_step']).reshape((input_dic['max_step'],3))

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    input_dic['rat0c'] = np.fromfile(f,dtype='float32',count=input_dic['kmaxc'])
    input_dic['rat0t'] = np.fromfile(f,dtype='float32',count=input_dic['kmaxt'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew10=['width_a','rat1pa_ini','rat1ma_ini','zzpa_scale','zzma_scale']
    for idx,iv in enumerate(varnamenew10):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=input_dic['nza'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew10=['width_b','rat1pb_ini','rat1mb_ini','zzpb_scale','zzmb_scale']
    for idx,iv in enumerate(varnamenew10):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=input_dic['nza'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    varnamenew10=['width_c','rat1pc_ini','rat1mc_ini','zzpc_scale','zzmc_scale']
    for idx,iv in enumerate(varnamenew10):
        input_dic[iv] = np.fromfile(f,dtype='float32',count=input_dic['nzc'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    input_dic['kind_drv'] = np.fromfile(f,dtype='int32',count=input_dic['ndrv'])
    input_dic['omega_rms_drv']= np.fromfile(f,dtype='float32',count=input_dic['ndrv'])

    unk2627 = np.fromfile(f,dtype='int32',count=2)
    input_dic['time'] = np.fromfile(f,dtype='float32',count=input_dic['ntmax']+1)
    input_dic['omega_driver'] = np.fromfile(f,dtype='float32',
           count=input_dic['ndrv']*input_dic['ntube']*2*(input_dic['ntmax']+1)).reshape((
                 input_dic['ndrv'],input_dic['ntube'],2,(input_dic['ntmax']+1)))
    self.input_ini = input_dic

  def read_dat1(self):
    f = open('%s.dat1'%self.rootname,'rb')
    varlist=['zzp','zzm','vrms','orms','brms','arms','eep',
             'eem','eer','ekin','emag','etot','rat1p','rat1m','qperp_p',
             'qperp_m','qtot']
    '''
    zzp  -- amplitude of the outward waves (km/s) [nz,nt]
    zzm  -- amplitude of the inward waves (km/s) [nz,nt]
    brms -- rms of the |B| (cgs) [nz,nt]
    vrms -- rms of the |vel| (cgs) [nz,nt]
    orms -- rms vorticity (cgs) [nz,nt]
    etot -- total energy (cgs) [nz,nt]
    ee  -- energy denstiy (cgs) [nz,nt]
    emag -- magnetic energy (cgs) [nz,nt]
    ekin -- kinetic energy (cgs) [nz,nt]
    qperp_p -- perpendicular heating rate (cgs) [nz,nt]
    '''
    self.input_dat1 = {}
    input2 = np.fromfile(f,dtype='int32',count=1)
    for idx,iv in enumerate(varlist):
        self.input_dat1[iv] = np.zeros((self.input_ini['nz'],self.input_ini['nt1sav']))
    for it in range(self.input_ini['nt1sav']):
      try: 
        for idx,iv in enumerate(varlist):
          self.input_dat1[iv][:,it] = np.fromfile(f,dtype='float32',count=self.input_ini['nz'])
      except: 
        self.input_ini['nt1sav'] = it
      varnamenew2=['unk18','unk19'] 
      input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        
  def read_dat4(self):
    f = open('%s.ini'%self.rootname,'rb')
    varlist=['qcasc_p','qcasc_m']
    self.input_dat4 = {}
    input2 = np.fromfile(f,dtype='int32',count=1)
    for idx,iv in enumerate(varlist):
        self.input_dat4[iv] = np.zeros((self.input_ini['nq4sav'],
                                        self.input_ini['nzc4sav'],
                                        self.input_ini['nt1sav']))
    for it in range(self.input_ini['nt1sav']):
      try: 
        for idx,iv in enumerate(varlist):
          self.input_dat4[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                                          np.size(self.input_dat4[iv][...,0]))).reshape((
                                                self.input_ini['nq4sav'],self.input_ini['nzc4sav']))
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
      except: 
        self.input_ini['nt1sav'] = it
      varnamenew2=['unk18','unk19'] 
      input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))

  def read_dat2(self):
    f = open('%s.dat2'%self.rootname,'rb')
    input2 = np.fromfile(f,dtype='int32',count=1)
    varlista=['omegpa','omegma'] # voriticty w+, w- (x,y)
    self.input_dat2 = {}
    for idx,iv in enumerate(varlista):
        self.input_dat2[iv] = np.zeros((self.input_ini['kmaxt'],
                                        self.input_ini['nza2sav'],
                                        self.input_ini['ntube'],
                                        self.input_ini['nt2sav']))
    varlistb=['omegpb','omegmb']
    for idx,iv in enumerate(varlistb):
        self.input_dat2[iv] = np.zeros((self.input_ini['kmaxt'],
                                        self.input_ini['nzb2sav'],
                                        self.input_ini['ntube'],
                                        self.input_ini['nt2sav']))
    
    varlistc=['omegpc','omegmc']
    for idx,iv in enumerate(varlistc):
        self.input_dat2[iv] = np.zeros((self.input_ini['kmaxt'],
                                        self.input_ini['nzc2sav'],
                                        self.input_ini['ntube'])) 
        
    for it in range(self.input_ini['nt2sav']):
      try: 
        for idx,iv in enumerate(varlista):
          self.input_dat2[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat2[iv][...,0])))*reshape((
                            self.input_ini['ntube'],self.input_ini['nza2sav'],self.input_ini['kmaxt'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistb):
          self.input_dat2[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat2[iv][...,0])))*reshape((
                            self.input_ini['ntube'],self.input_ini['nzb2sav'],self.input_ini['kmaxt'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistc):
          self.input_dat2[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat2[iv][...,0])))*reshape((
                            self.input_ini['nzc2sav'],self.input_ini['kmaxt'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
      except: 
        self.input_ini['nt2sav'] = it        

  def read_dat3(self):
    f = open('%s.dat3'%self.rootname,'rb')
    input2 = np.fromfile(f,dtype='int32',count=1)
    varlist=['omegpc3','omegmc3']
    self.input_dat3 = {}
    for idx,iv in enumerate(varlist):
        self.input_dat3[iv] = np.zeros((self.input_ini['nm3sav'],
                                        self.input_ini['nzc'],
                                        self.input_ini['nt2sav']))
    for it in range(self.input_ini['nt2sav']):
      try: 
        for idx,iv in enumerate(varlist):
          self.input_dat3[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                                          np.size(self.input_dat3[iv][...,0]))).reshape((
                                                self.input_ini['nzc'],self.input_ini['nm3sav'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
      except: 
        self.input_ini['nt2sav'] = it

  def read_dat5(self):
    f = open('%s.dat5'%self.rootname,'rb')
    input2 = np.fromfile(f,dtype='int32',count=1)
    varlista=['vel1','vel2'] # LOS vel, Non thermal velocity (both perp to the loop). 
    self.input_dat5 = {}
    for idx,iv in enumerate(varlista):
        self.input_dat5[iv] = np.zeros((self.input_ini['ngridc'],
                                        self.input_ini['nzc'],
                                        self.input_ini['nt5sav']))
    varlistb=['qqq0'] # heating rate. 
    for idx,iv in enumerate(varlistb):
        self.input_dat5[iv] = np.zeros((self.input_ini['ngridc'],
                                        self.input_ini['ngridc'],
                                        self.input_ini['nzc5sav'],
                                        self.input_ini['nt5sav']))
    
    varlistc=['xpts','ypts','qpts','vxpts','vypts']
    for idx,iv in enumerate(varlistc):
        self.input_dat5[iv] = np.zeros((self.input_ini['num_pts'],
                                        self.input_ini['nzc5sav'],
                                        self.input_ini['nt5sav'])) 
        
    for it in range(self.input_ini['nt5sav']):
      #try: 
        for idx,iv in enumerate(varlista):
          self.input_dat5[iv][...,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat5[iv][...,0]))).reshape((
                            self.input_ini['nzc'],self.input_ini['ngridc'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistb):
          self.input_dat5[iv][...,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat5[iv][...,0]))).reshape((
                            self.input_ini['nzc5sav'],self.input_ini['ngridc'],self.input_ini['ngridc'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistc):
          self.input_dat5[iv][...,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat5[iv][...,0]))).reshape((
                            self.input_ini['nzc5sav'],self.input_ini['num_pts'])).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
      #except: 
      #  self.input_ini['nt5sav'] = it        


  def read_dat6(self):
    f = open('%s.dat6'%self.rootname,'rb')
    input2 = np.fromfile(f,dtype='int32',count=1)
    varlista=['bbxa','bbya']
    self.input_dat6 = {}
    for idx,iv in enumerate(varlista):
        self.input_dat6[iv] = np.zeros((self.input_ini['ngridt']+1,
                                        self.input_ini['ngridt']+1,
                                        self.input_ini['nza'],
                                        self.input_ini['ntube'],
                                        self.input_ini['nt6sav']))
    varlistb=['bbxb','bbyb']
    for idx,iv in enumerate(varlistb):
        self.input_dat6[iv] = np.zeros((self.input_ini['ngridt']+1,
                                        self.input_ini['ngridt']+1,
                                        self.input_ini['nzb'],
                                        self.input_ini['ntube'],
                                        self.input_ini['nt6sav']))
    
    varlistc=['bbxc','bbyc']
    for idx,iv in enumerate(varlistc):
        self.input_dat6[iv] = np.zeros((self.input_ini['ngridt']+1,
                                        self.input_ini['ngridt']+1,
                                        self.input_ini['nzc'],
                                        self.input_ini['nt6sav']))
        
    for it in range(self.input_ini['nt6sav']):
      try: 
        for idx,iv in enumerate(varlista):
          self.input_dat6[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat6[iv][...,0])))*reshape((
                            self.input_ini['ntube'],self.input_ini['nza'],
                            self.input_ini['ngridt']+1,self.input_ini['ngridt']+1)).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistb):
          self.input_dat6[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat6[iv][...,0])))*reshape((
                            self.input_ini['ntube'],self.input_ini['nzb'],
                            self.input_ini['ngridt']+1,self.input_ini['ngridt']+1)).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
        for idx,iv in enumerate(varlistc):
          self.input_dat6[iv][:,:,it] = (np.fromfile(f,dtype='float32',count=
                        np.size(self.input_dat6[iv][...,0])))*reshape((
                            self.input_ini['nzb'],self.input_ini['ngridt']+1,
                            self.input_ini['ngridt']+1)).T
          varnamenew2=['unk18','unk19'] 
          input2 = np.fromfile(f,dtype='int32',count=np.size(varnamenew2))
      except: 
        self.input_ini['nt6sav'] = it        
                                                 
                                         
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
        rho         -- Density (g/cm^3) [nz]
        energy      -- Energy (erg) [nz]
        tg          -- Temperature (K) [nz]
        modb        -- |B| (G) [nz]
        va          -- Alfven speed (km/s) [nz]
        pg          -- Pressure (cgs)  [nz]
        vx          -- component x of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        vy          -- component y of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        vz          -- component z of the velocity (multipy by self.uni['u'] to get in cm/s) [nx+1, ny+1, nz+1]
        bx          -- component x of the magnetic field (multipy by self.uni['b'] to get in G) [nx+1, ny, nz]
        by          -- component y of the magnetic field (multipy by self.uni['b'] to get in G) [nx, ny+1, nz]
        bz          -- component z of the magnetic field (multipy by self.uni['b'] to get in G) [nx, ny, nz+1]
        jx          -- component x of the current [nx+1, ny+1, nz+1]
        jy          -- component x of the current [nx+1, ny+1, nz+1]
        jz          -- component x of the current [nx+1, ny+1, nz+1]
        eta         -- eta (?) [nx, ny, nz]
    
    '''

    if var in self.varn.keys(): 
        varname=self.varn[var]
    elif var in self.varn1.keys(): 
        varname=self.varn1[var]
    elif var in self.varn2.keys(): 
        varname=self.varn2[var]
    elif var in self.varn3.keys(): 
        varname=self.varn3[var]
    elif var in self.varn4.keys(): 
        varname=self.varn4[var]
    elif var in self.varn5.keys(): 
        varname=self.varn5[var]
    elif var in self.varn6.keys(): 
        varname=self.varn6[var]
    else:
        varname=var
        
    if snap != None: 
      self.snap = snap

    #try: 
        
    varu=var.replace('x','')
    varu=varu.replace('y','')
    varu=varu.replace('z','')

    if self.sel_units == 'cgs': 
        if (var in self.varn.keys()) and (varu in self.uni.uni.keys()): 
          cgsunits = self.uni.uni[varu]
        else: 
          cgsunits = 1.0
    else: 
        cgsunits = 1.0

    if var in self.varn.keys():
        self.data = (self.input_ini[varname])
    elif var in self.varn1.keys(): 
        self.data = (self.input_dat1[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
    elif var in self.varn2.keys(): 
        self.data = (self.input_dat2[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
    elif var in self.varn3.keys(): 
        self.data = (self.input_dat3[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
    elif var in self.varn4.keys(): 
        self.data = (self.input_dat4[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
    elif var in self.varn5.keys(): 
        self.data = (self.input_dat5[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
    elif var in self.varn6.keys(): 
        self.data = (self.input_dat6[varname])
        if snap != None: 
            self.data = self.data[...,self.snap]
        '''
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
        '''
    elif var == '': 

      print(help(self.get_var))
      print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
      for ii in self.varn: 
          print('use ', ii,' for ',self.varn[ii])
      #print(self.description['ALL']) 

      return None
   
    return self.data

  def readvar(self,inputfilename,nx,ny,nz,snap,nvar):
    f = open(inputfilename,'rb')
    f.seek(8*nvar*nx*ny*nz + 64*snap*nx*ny*nz)
    print(8*nvar*nx*ny*nz,64*snap*nx*ny*nz)
    var = np.fromfile(f,dtype='float32',count=nx*ny*nz)
    var = np.reshape(var,(self.nx,self.ny,self.nz))
    f.close()
    return var

  def genvar(self): 
    '''
    Dictionary of original variables which will allow to convert to cgs. 
    '''
    self.varn={}
    self.varn['rho']= 'rho'
    self.varn['tg'] = 'temp'
    self.varn['e']  = 'e'
    self.varn['pg']  = 'pres'
    self.varn['ux'] = 'ux'
    self.varn['uy'] = 'uy'
    self.varn['uz'] = 'uz'
    self.varn['modb'] = 'bmag'
    self.varn['va'] = 'va'
    self.varn['by'] = 'by'
    self.varn['bz'] = 'bz'
    self.varn['jx'] = 'jx'
    self.varn['jy'] = 'jy'
    self.varn['jz'] = 'jz'

    varlist=['zzp','zzm','vrms','orms','brms','arms','eep',
             'eem','eer','ekin','emag','etot','rat1p','rat1m','qperp_p',
             'qperp_m','qtot']

    self.varn1={}
    for var in varlist: 
        self.varn1[var]  = var
    #self.varn1['zzp']  = 'zzp' # amplitude of the outward waves (km/s)
    #self.varn1['zzm']  = 'zzm' # amplitude of the inward waves (km/s)
    #self.varn1['vrms'] = 'vrms' # rms of the |vel| (cgs)
    #self.varn1['orms'] = 'orms' # rms vorticity (cgs)
    #self.varn1['brms'] = 'brms' # rms of the |B| (cgs)
    #self.varn1['e']    = 'etot' # total energy (cgs)
    #self.varn1['emag'] = 'emag' # magnetic energy (cgs)
    #self.varn1['ekin'] = 'ekin' # kinetic energy (cgs)
    #self.varn1['qperp_p']= 'qperp_p' # perpendicular heating rate (cgs)

    varlist=['omegpa','omegma','omegpb','omegmb','omegpc','omegmc']
    self.varn2={}
    for var in varlist: 
        self.varn2[var]  = var
        
    varlist=['omegpc3','omegmc3']
    self.varn3={}
    for var in varlist: 
        self.varn3[var]  = var

    varlist=['qcasc_p','qcasc_m']
    self.varn4={}
    for var in varlist: 
        self.varn4[var]  = var
        
    varlist=['vel1','vel2','qqq0','xpts','ypts','qpts','vxpts','vypts']
    self.varn5={}
    for var in varlist: 
        self.varn5[var]  = var
        
    varlist=['bbxa','bbya','bbxb','bbyb','bbxc','bbyc']
    self.varn6={}
    for var in varlist: 
        self.varn6[var]  = var
        
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

class Mah_units(object): 

    def __init__(self,verbose=False):

        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.verbose=verbose
        self.uni['gamma']  = 5./3.
        self.uni['tg']     = 1.0 # K
        self.uni['l']      = 1.0e5 # km -> cm
        self.uni['rho']    = 1.0 # gr cm^-3 
        self.uni['u']      = 1.0 # cm/s
        self.uni['b']      = 1.0 # Gauss
        self.uni['t']      = 1.0 # seconds

        # Units and constants in SI

        convertcsgsi(self)

        globalvars(self)

        self.uni['n']      = self.uni['rho'] / self.m_p / 2. # cm^-3