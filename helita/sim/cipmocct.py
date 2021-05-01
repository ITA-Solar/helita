import os
import numpy as np
import scipy.constants as ct
from scipy.io import readsav as rsav
from .load_quantities import *
from .load_arithmetic_quantities import *
from .tools import *
from .load_noeos_quantities import *
from scipy.ndimage import rotate
from . import document_vars
from scipy import interpolate

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
    def __init__(self, rootname, snap, fdir='./', sel_units = 'cgs', verbose=True):
        
        self.rootname = rootname
        self.fdir = fdir
        self.snap = snap 
        self.sel_units = sel_units
        self.verbose = verbose
        self.uni = Cipmocct_units()
        
        params = rsav(os.path.join(self.fdir,'params_'+rootname+'.sav'))
        if snap == None: 
            self.x = params['x1'].copy()
            self.y = params['x3'].copy()
            self.z = params['time'].copy()
            self.nx = len(params['x1'])
            self.ny = len(params['x3'])
            self.nz = len(params['time'])

            if self.sel_units=='cgs': 
                self.x *= self.uni.uni['l']
                self.y *= self.uni.uni['l']
                
            self.time =  params['time'] # No uniform (array)
            self.varfile = rsav(os.path.join(self.fdir,'variables_'+self.rootname+'.sav'))
        else: 

            self.x = params['x1'].copy()
            self.y = params['x3'].copy()
            self.z = params['x2'].copy() 

            self.nx = len(params['x1'])
            self.ny = len(params['x3'])
            self.nz = len(params['x2'])

            if self.sel_units=='cgs': 
                self.x *= self.uni.uni['l']
                self.y *= self.uni.uni['l']
                self.z *= self.uni.uni['l']

            self.time =  params['time'] # No uniform (array)

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

        self.transunits = False

        self.cstagop = False # This will not allow to use cstagger from Bifrost in load
        self.hion = False # This will not allow to use HION from Bifrost in load  
        self.genvar()
        
        document_vars.create_vardict(self)
        document_vars.set_vardocs(self)
        
        
    def get_var(self,var, *args, snap=None, iix=None, iiy=None, iiz=None, layout=None, **kargs): 
        '''
        Reads the variables from a snapshot (it).

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        snap - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot.       
        Axes: 
        -----
            z-axis is along the loop
            x and y axes are perperdicular to the loop
        
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

          if self.snap == None: 
              varfile = self.varfile
              self.data = np.transpose(varfile[varname]) * cgsunits

          else: 
              itname = '{:04d}'.format(self.snap)
              varfile = rsav(self.fdir+'vars_'+self.rootname+'_'+itname+'.sav')
              self.data = np.transpose(varfile[varname]) * cgsunits




#          varfile = rsav(os.path.join(self.fdir,self.rootname+'_'+itname+'.sav'))
            

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

        if document_vars.creating_vardict(self):
            return None
        elif  var == '': 
          print(help(self.get_var))
          print('VARIABLES USING CGS OR GENERIC NOMENCLATURE')
          for ii in self.varn: 
              print('use ', ii,' for ',self.varn[ii])
          if hasattr(self,'vardict'):
            self.vardocs()

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
        self.varn['uy'] = 'vz_cube'
        self.varn['uz'] = 'vy_cube'
        self.varn['bx'] = 'bx_cube'
        self.varn['by'] = 'bz_cube'
        self.varn['bz'] = 'by_cube'

    def trans2comm(self,varname, snap=None, angle=45, loop = 'quarter'): 
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

        INPUT: 
        varname - string
        snap - integer
        angle - real (degrees). Any number -90 to 90, default = 45
        '''

        self.sel_units = 'cgs'

        self.trans2commaxes(loop) 
        
        if angle != 0:       
            if varname[-1] in ['x']: 
                varx = self.get_var(varname,snap=snap)
                vary = self.get_var(varname[0]+'y',snap=snap)
                var = varx * np.cos(angle/90.0*np.pi/2.0) - vary * np.sin(angle/90.0*np.pi/2.0)
            elif varname[-1] in ['y']: 
                vary = self.get_var(varname,snap=snap)
                varx = self.get_var(varname[0]+'x',snap=snap)
                var = vary * np.cos(angle/90.0*np.pi/2.0) + varx * np.sin(angle/90.0*np.pi/2.0)
            else:  
                var = self.get_var(varname,snap=snap)
            var = rotate(var, angle=angle, reshape=False, mode='nearest', axes=(0,1))
           
        else: 
            var = self.get_var(varname,snap=snap)
        
        if loop == 'quarter': 
            
            if varname[-1] in ['x']: 
                var = self.make_loop(var)
                varz = self.get_var(varname[0]+'z',snap=snap)
                varz = self.make_loop(varz)
                xx, zz = np.meshgrid(self.x,self.z)
                aa=np.angle(xx+1j*zz)
                for iiy, iy in enumerate(self.y):
                    var[:,iiy,:] = varz[:,iiy,:] * np.cos(aa.T) - var[:,iiy,:] * np.sin(aa.T)
            elif varname[-1] in ['z']: 
                var = self.make_loop(var)
                varz = self.get_var(varname[0]+'z',snap=snap)
                varz = self.make_loop(varz)
                xx, zz = np.meshgrid(self.x,self.z)
                aa=np.angle(xx+1j*zz)
                for iiy, iy in enumerate(self.y):
                    var[:,iiy,:] = var[:,iiy,:] * np.cos(aa.T) + varz[:,iiy,:] * np.sin(aa.T)
            else: 
                var = self.make_loop(var)

        return var

    def make_loop(self,var): 
        R = np.max(self.z*2)/np.pi/2.
        rad=self.x_orig+np.max(self.x_loop)-np.max(self.x_orig)/2
        angl=self.z_orig / R 
        var_new=np.zeros((self.nx,self.ny,self.nz))
        iiy0=np.argmin(np.abs(self.y_orig))

        for iiy, iy in enumerate(self.y): 
            temp=var[:,iiy*2+iiy0,:]
            data = polar2cartesian(rad,angl,temp,self.z,self.x)

            var_new[:,iiy,:] = data
        return var_new
    
    def trans2commaxes(self,loop): 

        if self.transunits == False:
            self.x_orig = self.x
            self.y_orig = self.y
            self.z_orig = self.z
            if loop == 'quarter':
                R = np.max(self.z*2)/np.pi/2.
                self.x_loop=np.linspace(R*np.cos([np.pi/4]),R,
                              int((R-R*np.cos([np.pi/4]))/2/np.min(self.dx1d*2)))
                self.z_loop=np.linspace(0,R*np.sin([np.pi/4]),
                              int(R*np.sin([np.pi/4])/2/np.min(self.dx1d*2))) 
                
                self.x=self.x_loop.squeeze()
                self.z=self.z_loop.squeeze()
                self.y=self.y[np.argmin(np.abs(self.y))+1::2]
                
                self.dx1d = np.gradient(self.x)
                self.dy1d = np.gradient(self.y)
                self.dz1d = np.gradient(self.z)
                self.nx=np.size(self.x)
                self.ny=np.size(self.y)
                self.nz=np.size(self.z)

            self.transunits = True

    def trans2noncommaxes(self): 

        if self.transunits == True:
            self.x=self.x_orig
            self.y=self.y_orig
            self.z=self.z_orig
            self.dx1d = np.gradient(self.x)
            self.dy1d = np.gradient(self.y)
            self.dz1d = np.gradient(self.z)
            self.nx=np.size(self.x)
            self.ny=np.size(self.y)
            self.nz=np.size(self.z)
            # opposite to the previous function 
            self.transunits = False

class Cipmocct_units(object): 

    def __init__(self,verbose=False):
        import scipy.constants as const
        from astropy import constants as aconst


        '''
        Units and constants in cgs
        '''
        self.uni={}
        self.verbose=verbose
        self.uni['gamma']  = 5./3.
        self.uni['proton'] = 1.67262158e-24 # g
        self.uni['tg']     = 1.0e6 # K
        self.uni['fact']   = 2
        self.uni['l']      = 1000.*self.uni['fact']*1.0e5 # for having a 2000 km wide loop
        self.uni['n']      = 1.0e9 # cm^-3    
   
        # Units and constants in SI
        globalvars(self)

        self.uni['rho']    = self.uni['n'] * self.uni['proton'] /2. # gr cm^-3 
        self.uni['u']      = np.sqrt(2*self.uni['gamma']*self.k_b/self.m_p*self.uni['tg']) # cm/s

        self.uni['b']      = self.uni['u']*np.sqrt(self.uni['rho']) # Gauss
        self.uni['j']      = self.uni['b']/self.uni['l']* aconst.c.to_value('cm/s') # current density
        self.uni['t']      = self.uni['l']/self.uni['u'] # seconds

        convertcsgsi(self)
        self.unisi['ee']   = self.unisi['u']**2
        self.unisi['e']    = self.unisi['rho'] * self.unisi['ee'] 
        self.unisi['pg']   = self.unisi['rho'] * (self.unisi['l'] / self.unisi['t'])**2
        self.unisi['u']    = self.uni['u'] * const.centi # m/s



def polar2cartesian(r, t, grid, x, y, order=3):
    '''
    Converts polar grid to cartesian grid
    '''
    from scipy import ndimage

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X * X + Y * Y)
    new_t = np.arctan2(X, Y)

    ir = interpolate.interp1d(r, np.arange(len(r)), bounds_error=False, fill_value=0.0)
    it = interpolate.interp1d(t, np.arange(len(t)), bounds_error=False, fill_value=0.0)
    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r) - 1
    new_ir[new_r.ravel() < r.min()] = 0

    return ndimage.map_coordinates(grid, np.array([new_ir, new_it]),
                           order=order).reshape(new_r.shape)