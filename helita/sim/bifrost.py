"""
Set of programs to read and interact with output from Bifrost
"""

# import builtin modules
import os
import functools
import weakref
from glob import glob
import warnings
import time
import ast
import collections

# import external public modules
import numpy as np
from scipy import interpolate
from scipy.ndimage import map_coordinates

# import internal modules
from .load_quantities import *
from .load_arithmetic_quantities import *
from . import load_fromfile_quantities 
from .tools import *
from . import document_vars
from . import file_memory
from . import stagger

whsp = '  '


class BifrostData():
    """
    Reads data from Bifrost simulations in native format.

    Parameters
    ----------
    file_root - string
        Basename for all file names (without underscore!). Snapshot number
        will be added afterwards, and directory will be added before.
    snap - integer, optional
        Snapshot number. If None, will read first snapshot in sequence.
    meshfile - string, optional
        File name (including full path) for file with mesh. If set
        to None (default), a uniform mesh will be created.
    fdir - string, optional
        Directory where simulation files are. Must be a real path.
    verbose - bool, optional
        If True, will print out more diagnostic messages
    dtype - string, optional
        Data type for reading variables. Default is 32 bit float.
    big_endian - string, optional
        If True, will read variables in big endian. Default is False
        (reading in little endian).
    ghost_analyse - bool, optional
        If True, will read data from ghost zones when this is saved
        to files. Default is never to read ghost zones.
    do_stagger - bool, optional
        whether to correctly account for the stagger mesh when doing operations.
        if enabled, operations will take more time but produce more accurate results.
    stagger_kind - string, optional
        which method to use for performing stagger operations, if do_stagger.
        options are 'cstagger', 'numba' (default), 'numpy'. See stagger.py for details.
    lowbus  - bool, optional
        Use True only if data is too big to load. It will do cstagger
        operations layer by layer using threads (slower).
    numThreads - integer, optional
        number of threads for certain operations that use parallelism.
    fast - whether to read data "fast", by only reading the requested data.
        implemented as a flag, with False as default, for backwards 
        compatibility; some previous codes may have assumed non-requested 
        data was read. To avoid issues, just ensure you use get_var() 
        every time you want to have data, and don't assume things exist 
        (e.g. self.bx) unless you do get_var for that thing 
        (e.g. get_var('bx')).

    Examples
    --------
    This reads snapshot 383 from simulation "cb24bih", whose file
    root is "cb24bih", and is found at directory /data/cb24bih:

    >>> a = BifrostData("cb24bih", snap=383, fdir="/data/cb24bih")

    Scalar variables do not need de-staggering and are available as
    memory map (only loaded to memory when needed), e.g.:

    >>> a.r.shape
    (504, 504, 496)

    Composite variables need to be obtained by get_var():

    >>> vx = a.get_var("ux")
    """

    snap = None

    ## CREATION ##
    def __init__(self, file_root, snap=None, meshfile=None, fdir='.', 
                 fast=False, verbose=True, dtype='f4', big_endian=False, 
                 cstagop=None, do_stagger=True, ghost_analyse=False, lowbus=False, 
                 numThreads=1, params_only=False, sel_units=None, 
                 use_relpath=False, stagger_kind=stagger.DEFAULT_STAGGER_KIND,
                 iix=None, iiy=None, iiz=None):
        """
        Loads metadata and initialises variables.
        """
        # bookkeeping
        self.fdir = fdir if use_relpath else os.path.abspath(fdir)
        self.verbose = verbose
        self.do_stagger = do_stagger if (cstagop is None) else cstagop
        self.lowbus = lowbus
        self.numThreads = numThreads
        self.file_root = os.path.join(self.fdir, file_root)
        self.root_name = file_root
        self.meshfile = meshfile
        self.ghost_analyse = ghost_analyse
        self.stagger_kind = stagger_kind
        self.sel_units = sel_units 
        self.numThreads = numThreads
        self.fast = fast
        self._fast_skip_flag = False if fast else None  # None-> never skip

        setattr(self, document_vars.LOADING_LEVEL, -1) # tells how deep we are into loading a quantity now.

        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        self.hion = False
        self.heion = False

        try:
            tmp = find_first_match("%s*idl" % file_root, fdir)
        except IndexError:
            try:
                tmp = find_first_match("%s*idl.scr" % file_root, fdir)
            except IndexError:
                try:
                    tmp = find_first_match("mhd.in", fdir)
                except IndexError:
                    raise ValueError(("(EEE) init: no .idl or mhd.in files "
                                      "found"))
        self.uni = Bifrost_units(filename=tmp, fdir=fdir)

        self.set_snap(snap, True, params_only=params_only)

        self.set_domain_iiaxes(iix=iix, iiy=iiy, iiz=iiz, internal=False)

        self.genvar()
        self.transunits = False
        self.cross_sect = cross_sect_for_obj(self)
        if 'tabinputfile' in self.params.keys(): 
            tabfile = os.path.join(self.fdir, self.get_param('tabinputfile').strip())
            if os.access(tabfile, os.R_OK):
                self.rhoee = Rhoeetab(tabfile=tabfile, fdir=fdir, radtab=True, verbose=self.verbose)

        self.stagger = stagger.StaggerInterface(self)

        document_vars.create_vardict(self)
        document_vars.set_vardocs(self)

    ## PROPERTIES ##
    shape = property(lambda self: (self.xLength, self.yLength, self.zLength))
    size  = property(lambda self: (self.xLength * self.yLength * self.zLength))
    ndim  = property(lambda self: 3)

    stagger_kind = stagger.STAGGER_KIND_PROPERTY(internal_name='_stagger_kind')

    @property
    def cstagop(self): # cstagop is an alias to do_stagger. Maintained for backwards compatibility.
        return self.do_stagger
    @cstagop.setter
    def cstagop(self, value):
        self.do_stagger = value

    ## SET SNAPSHOT ##
    def __getitem__(self, i):
        '''sets snap to i then returns self.

        i: string, or anything which can index a list
            string --> set snap to int(i)
            else --> set snap to self.get_snaps()[i]

        Example usage:
            bb = BifrostData(...)
            bb['3']('r')
            # is equivalent to: bb.set_snap(3); bb.get_var('r')
            bb[3]('r')
            # is equivalent to: bb.set_snap(bb.get_snaps()[3]); bb.get_var('r')
            #   if the existing snaps are [0,1,2,3,...], this is equivalent to bb['3']('r')
            #   if the existing snaps are [4,5,6,7,...], this is equivalent to bb['7']('r')
        '''
        if isinstance(i, str):
            self.set_snap(int(i))
        else:
            self.set_snap(self.get_snaps()[i])
        return self
    
    def _set_snapvars(self, firstime=False):
        """
            Sets list of avaible variables
        """
        self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        self.auxvars = self.get_param('aux', error_prop=True).split()
        if self.do_mhd:
            self.snapvars += ['bx', 'by', 'bz']
        self.hionvars = []
        self.heliumvars = []
        if self.get_param('do_hion', default=0) > 0:
            self.hionvars = ['hionne', 'hiontg', 'n1',
                             'n2', 'n3', 'n4', 'n5', 'n6', 'nh2']
            self.hion = True
        if self.get_param('do_helium', default=0) > 0:
            self.heliumvars = ['nhe1', 'nhe2', 'nhe3']
            self.heion = True
        self.compvars = ['ux', 'uy', 'uz', 's', 'ee']
        self.simple_vars = self.snapvars + self.auxvars + self.hionvars + \
            self.heliumvars
        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.auxxyvars.append('ixy1')
        self.vars2d = []
        # special case for 2D variables, stored in a separate file
        for var in self.auxvars:
            if any(i in var for i in ('xy', 'yz', 'xz')):
                self.auxvars.remove(var)
                self.vars2d.append(var)

    def set_snap(self, snap, firstime=False, params_only=False):
        """
        Reads metadata and sets variable memmap links for a given snapshot
        number.
        Parameters
        ----------
        snap - integer or array
            Number of simulation snapshot to load.
        """
        if snap is None:
            try:
                tmp = sorted(glob("%s*idl" % self.file_root))[0]
                snap_string = tmp.split(
                    self.file_root + '_')[-1].split(".idl")[0]
                if snap_string.isdigit():
                    snap = int(snap_string)
                else:
                    tmp = glob("%s.idl" % self.file_root)
                    snap = 0
            except Exception:
                try:
                    tmp = sorted(glob("%s*idl.scr" % self.file_root))[0]
                    snap = -1
                except IndexError:
                    try:
                        tmp = glob("%s.idl" % self.file_root)
                        snap = 0
                    except IndexError:
                        raise ValueError(("(EEE) set_snap: snapshot not defined "
                                      "and no .idl files found"))

        def snap_str_from_snap(snap):
            if snap == 0:
                return ''
            else:
                return '_%03i' % snap

        self.snap = snap
        if np.shape(self.snap) != ():
            self.snap_str = []
            for num in snap:
                self.snap_str.append(snap_str_from_snap(num))
        else:
            self.snap_str = snap_str_from_snap(snap)
        self.snapInd = 0

        self._read_params(firstime=firstime)
        # Read mesh for all snaps because meshfiles could differ
        self.__read_mesh(self.meshfile, firstime=firstime)
        # variables: lists and initialisation
        self._set_snapvars(firstime=firstime)
        # Do not call if params_only requested
        if(not params_only):
            self._init_vars(firstime=firstime)

    def _read_params(self, firstime=False):
        """
        Reads parameter file (.idl)
        """
        if np.shape(self.snap) == ():
            snap = [self.snap]
            snap_str = [self.snap_str]
        else:
            snap = self.snap
            snap_str = self.snap_str

        filename = []
        self.paramList = []

        for i, num in enumerate(snap):
            if num < 0:
                filename.append(self.file_root + '.idl.scr')
            elif num == 0:
                filename.append(self.file_root + '.idl')
            else:
                filename.append(self.file_root + snap_str[i] + '.idl')

        for file in filename:
            self.paramList.append(read_idl_ascii(file,firstime=firstime, obj=self))

        # assign some parameters as attributes
        for params in self.paramList:
            for p in ['x', 'y', 'z', 'b']:
                try:
                    setattr(self, 'n' + p, params['m' + p])
                except KeyError:
                    raise KeyError(('read_params: could not find '
                                    'm%s in idl file!' % p))
            for p in ['dx', 'dy', 'dz', 'do_mhd']:
                try:
                    setattr(self, p, params[p])
                except KeyError:
                    raise KeyError(('read_params: could not find '
                                    '%s in idl file!' % p))
            try:
                if ((params['boundarychk'] == 1) and (params['isnap'] !=0)):
                    self.nzb = self.nz + 2 * self.nb
                else:
                    self.nzb = self.nz
                if ((params['boundarychky'] == 1) and (params['isnap'] !=0)):
                    self.nyb = self.ny + 2 * self.nb
                else:
                    self.nyb = self.ny
                if ((params['boundarychkx'] == 1) and (params['isnap'] !=0)):
                    self.nxb = self.nx + 2 * self.nb
                else:
                    self.nxb = self.nx
            except KeyError:
                self.nzb = self.nz
                self.nyb = self.ny
                self.nxb = self.nx
            # check if units are there, if not use defaults and print warning
            unit_def = {'u_l': 1.e8, 'u_t': 1.e2, 'u_r': 1.e-7,
                        'u_b': 1.121e3, 'u_ee': 1.e12}
            for unit in unit_def:
                if unit not in params:
                    default = unit_def[unit]
                    if hasattr(self, 'uni'):
                        default = getattr(self.uni, unit, default)
                    if getattr(self, 'verbose', True):
                        print("(WWW) read_params:"" %s not found, using "
                              "default of %.3e" % (unit, default), 2*whsp,
                              end="\r", flush=True)
                    params[unit] = default

        self.params = {}
        for key in self.paramList[0]:
            self.params[key] = np.array(
                [self.paramList[i][key] for i in range(0, len(self.paramList))    \
                    if key in self.paramList[i].keys()])
                    # the if statement is required in case extra params in 
                    # self.ParmList[0]
        self.time = self.params['t']
        if self.sel_units=='cgs': 
            self.time *= self.uni.uni['t']

    def get_param(self, param, default=None, warning=None, error_prop=None):
        ''' get param via self.params[param][self.snapInd].

        if param not in self.params.keys(), then the following kwargs may play a role:
            default: None (default) or any value.
                return this value (eventually) instead. (check warning and error_prop first.)
            warning: None (default) or any Warning or string.
                if not None, do warnings.warn(warning).
            error_prop: None (default), True, or any Exception object.
                None --> ignore this kwarg.
                True --> raise the original KeyError caused by trying to get self.params[param].
                else --> raise error_prop from None.
        '''
        try:
            p = self.params[param]
        except KeyError as err_triggered:
            if (warning is not None) and (self.verbose):
                warnings.warn(warning)
            if error_prop is not None:
                if isinstance(error_prop, BaseException):
                    raise error_prop from None  # "from None" --> show just this error, not also err_triggered
                elif error_prop:
                    raise err_triggered
            return default
        else:
            p = p[self.snapInd]
        return p

    def get_params(self, *params, **kw):
        '''return a dict of the values of params in self.
        Equivalent to {p: self.get_param(p, **kw) for p in params}.
        '''
        return {p: self.get_param(p, **kw) for p in params}
        
    def __read_mesh(self, meshfile, firstime=False):
        """
        Reads mesh file
        """
        if meshfile is None:
            meshfile = os.path.join(
                self.fdir, self.get_param('meshfile', error_prop=True).strip())
        if os.path.isfile(meshfile):
            f = open(meshfile, 'r')
            for p in ['x', 'y', 'z']:
                dim = int(f.readline().strip('\n').strip())
                assert dim == getattr(self, 'n' + p)
                # quantity
                setattr(self, p, np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # quantity "down"
                setattr(self, p + 'dn', np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # up derivative of quantity
                setattr(self, 'd%sid%sup' % (p, p), np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
                # down derivative of quantity
                setattr(self, 'd%sid%sdn' % (p, p), np.array(
                    [float(v) for v in f.readline().strip('\n').split()]))
            f.close()
            if self.ghost_analyse:
                # extend mesh to cover ghost zones
                self.z = np.concatenate((self.z[0] - np.linspace(
                    self.dz * self.nb, self.dz, self.nb),
                    self.z, self.z[-1] + np.linspace(
                    self.dz, self.dz * self.nb, self.nb)))
                self.zdn = np.concatenate((self.zdn[0] - np.linspace(
                    self.dz * self.nb, self.dz, self.nb),
                    self.zdn, (self.zdn[-1] + np.linspace(
                        self.dz, self.dz * self.nb, self.nb))))
                self.dzidzup = np.concatenate((
                    np.repeat(self.dzidzup[0], self.nb),
                    self.dzidzup,
                    np.repeat(self.dzidzup[-1], self.nb)))
                self.dzidzdn = np.concatenate((
                    np.repeat(self.dzidzdn[0], self.nb),
                    self.dzidzdn,
                    np.repeat(self.dzidzdn[-1], self.nb)))
                self.nx = self.nxb
                self.ny = self.nyb
                self.nz = self.nzb
        else:  # no mesh file
            if self.dx == 0.0:
                self.dx = 1.0
            if self.dy == 0.0:
                self.dy = 1.0
            if self.dz == 0.0:
                self.dz = 1.0
            if self.verbose and firstime:
                warnings.warn(('Mesh file {mf} does not exist. Creating uniform grid '
                               'with (dx,dy,dz)=({dx:.2e},{dy:.2e},{dz:.2e})').format(
                               mf=repr(meshfile), dx=self.dx, dy=self.dy, dz=self.dz))
            # x
            self.x = np.arange(self.nx) * self.dx
            self.xdn = self.x - 0.5 * self.dx
            self.dxidxup = np.zeros(self.nx) + 1. / self.dx
            self.dxidxdn = np.zeros(self.nx) + 1. / self.dx
            # y
            self.y = np.arange(self.ny) * self.dy
            self.ydn = self.y - 0.5 * self.dy
            self.dyidyup = np.zeros(self.ny) + 1. / self.dy
            self.dyidydn = np.zeros(self.ny) + 1. / self.dy
            # z
            if self.ghost_analyse:
                self.nx = self.nxb
                self.ny = self.nyb
                self.nz = self.nzb
            self.z = np.arange(self.nz) * self.dz
            self.zdn = self.z - 0.5 * self.dz
            self.dzidzup = np.zeros(self.nz) + 1. / self.dz
            self.dzidzdn = np.zeros(self.nz) + 1. / self.dz

        for x in ('x', 'y', 'z'):
            setattr(self, x, getattr(self, x)[getattr(self, 'ii'+x, slice(None))])

        for x in ('x', 'y', 'z'):
            xcoords = getattr(self, x)
            if len(xcoords) > 1:
                dx1d = np.gradient(xcoords)
            else:
                dx1d = np.zeros(len(xcoords))
            setattr(self, 'd'+x+'1d', dx1d)
        
        if self.sel_units=='cgs': 
            self.x *= self.uni.uni['l']
            self.y *= self.uni.uni['l']
            self.z *= self.uni.uni['l']
            self.zdn *= self.uni.uni['l']
            self.dx *= self.uni.uni['l']
            self.dy *= self.uni.uni['l']
            self.dz *= self.uni.uni['l']
            self.dx1d *= self.uni.uni['l']
            self.dy1d *= self.uni.uni['l']
            self.dz1d *= self.uni.uni['l']
            
            self.dxidxup /= self.uni.uni['l']
            self.dxidxdn /= self.uni.uni['l']
            self.dyidyup /= self.uni.uni['l']
            self.dyidydn /= self.uni.uni['l']
            self.dzidzup /= self.uni.uni['l']
            self.dzidzdn /= self.uni.uni['l']  
        
        self.transunits = False
        
    def _init_vars(self, firstime=False,  fast=None, *args, **kwargs):
        """
        Memmaps "simple" variables, and maps them to methods.
        Also, sets file name[s] from which to read a data
        
        fast: None, True, or False.
            whether to only read density (and not all the other variables).
            if None, use self.fast instead.
        """
        fast = fast if fast is not None else self.fast
        if self._fast_skip_flag is True:
            return
        elif self._fast_skip_flag is False:
            self._fast_skip_flag = True # swaps flag to True, then runs the rest of the code (this time around).
        # else, fast_skip_flag is None, so the code should never be skipped.
        # as long as fast is False, fast_skip_flag should be None.

        self.variables = {}
        for var in self.simple_vars:
            try:
                self.variables[var] = self._get_simple_var(
                    var, *args, **kwargs)
                setattr(self, var, self.variables[var])
            except Exception as err:
                if self.verbose:
                    if firstime:
                        print('(WWW) init_vars: could not read '
                            'variable {} due to {}'.format(var, err))
        for var in self.auxxyvars:
            try:
                self.variables[var] = self._get_simple_var_xy(var, *args,
                                                              **kwargs)
                setattr(self, var, self.variables[var])
            except Exception as err:
                if self.verbose:
                    if firstime:
                        print('(WWW) init_vars: could not read '
                            'variable {} due to {}'.format(var, err))
        rdt = self.r.dtype
        if self.stagger_kind == 'cstagger': 
            if (self.nz > 1): 
                cstagger.init_stagger(self.nz, self.dx, self.dy, self.z.astype(rdt),
                                  self.zdn.astype(rdt), self.dzidzup.astype(rdt),
                                  self.dzidzdn.astype(rdt))
                self.cstagger_exists = True   # we can use cstagger methods!
            else:
                cstagger.init_stagger_mz1d(self.nz, self.dx, self.dy, self.z.astype(rdt))
                self.cstagger_exists = True  # we must avoid using cstagger methods.
        else: 
            self.cstagger_exists = True

    ## GET VARIABLE ##
    def __call__(self, var, *args, **kwargs):
        '''equivalent to self.get_var(var, *args, **kwargs)'''
        __tracebackhide__ = True  # hide this func from error traceback stack
        return self.get_var(var, *args, **kwargs)

    def get_varTime(self, var, snap=None, iix=None, iiy=None, iiz=None, 
                    print_freq=None, 
                    *args__get_var, **kw__get_var):
        """
        Reads a given variable as a function of time.

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be a valid Bifrost variable name,
            see Bifrost.get_var().
        snap - array of integers
            Snapshot numbers to read.
        iix -- integer or array of integers, optional
            reads yz slices.
        iiy -- integer or array of integers, optional
            reads xz slices.
        iiz -- integer or array of integers, optional
            reads xy slices.
        print_freq - number, default 2
            print progress update every print_freq seconds.
            Use print_freq < 0 to never print update.
            Use print_freq ==0 to print all updates.

        additional *args and **kwargs are passed to get_var.
        """
        # set print_freq
        if print_freq is None:
            if not self.verbose:
                print_freq = -1  # never print.
            print_freq = getattr(self, 'print_freq', 2) # default 2
        else:
            setattr(self, 'print_freq', print_freq)

        # set snap
        if snap is None:
            snap = kw__get_var.pop('snaps', None) # look for 'snaps' kwarg
            if snap is None:
                snap = self.snap
        snap = np.array(snap, copy=False)
        if len(snap.shape)==0:
            raise ValueError('Expected snap to be list (in get_varTime) but got snap={}'.format(snap))
        if not np.array_equal(snap, self.snap):
            self.set_snap(snap)
            self.variables={}

        # set iix,iiy,iiz.
        self.set_domain_iiaxes(iix=iix, iiy=iiy, iiz=iiz, internal=False)
        snapLen = np.size(self.snap)

        # bookkeeping - maintain self.snap; handle self.recoverData; track timing.
        remembersnaps = self.snap                   # remember self.snap (restore later if crash)
        if hasattr(self, 'recoverData'):
            delattr(self, 'recoverData')            # smash any existing saved data
        timestart = now = time.time()               # track timing, so we can make updates.
        printed_update  = False
        def _print_clearline(N=100):        # clear N chars, and move cursor to start of line.
            print('\r'+ ' '*N +'\r',end='') # troubleshooting: ensure end='' in other prints.

        try:
            firstit = True
            for it in range(0, snapLen):
                self.snapInd = it
                # print update if it is time to print update
                if (print_freq >= 0) and (time.time() - now > print_freq):
                    _print_clearline()
                    print('Getting {:^10s}; at snap={:2d} (snap_it={:2d} out of {:2d}).'.format(
                                    var,     snap[it],         it,    snapLen        ), end='')
                    now = time.time()
                    print(' Total time elapsed = {:.1f} s'.format(now - timestart), end='')
                    printed_update=True
                    
                # actually get the values here:
                if firstit:
                    # get value at first snap
                    val0 = self.get_var(var, snap=snap[it], *args__get_var, **kw__get_var)
                    # figure out dimensions and initialize the output array.
                    value = np.empty([*np.shape(val0), snapLen], dtype=self.dtype)
                    value[..., 0] = val0
                    firstit = False
                else:
                    value[..., it] = self.get_var(var, snap=snap[it],
                                                 *args__get_var, **kw__get_var)
        except:   # here it is ok to except all errors, because we always raise.
            if it > 0:
                self.recoverData = value[..., :it]   # save data 
                if self.verbose:
                    print(('Crashed during get_varTime, but managed to get data from {} '
                           'snaps before crashing. Data was saved and can be recovered '
                           'via self.recoverData.'.format(it)))
            raise
        finally:
            self.set_snap(remembersnaps)             # restore snaps
            if printed_update:
                _print_clearline()
                print('Completed in {:.1f} s'.format(time.time() - timestart), end='\r')
        
        return value

    def set_domain_iiaxis(self, iinum=None, iiaxis='x'):
        """
        Sets iix=iinum and xLength=len(iinum). (x=iiaxis)
        if iinum is a slice, use self.nx (or self.nzb, for x='z') to determine xLength.

        Also, if we end up using a non-None slice, disable stagger.
        TODO: maybe we can leave do_stagger=True if stagger_kind != 'cstagger' ?

        Parameters
        ----------
        iinum - slice, int, list, array, or None (default)
            Slice to be taken from get_var quantity in that axis (iiaxis)
            int --> convert to slice(iinum, iinum+1) (to maintain dimensions of output)
            None --> don't change existing self.iix (or iiy or iiz).
                     if it doesn't exist, set it to slice(None).
            To set existing self.iix to slice(None), use iinum=slice(None).
        iiaxis - string
            Axis from which the slice will be taken ('x', 'y', or 'z')

        Returns True if any changes were made, else None.
        """
        iix = 'ii' + iiaxis
        if hasattr(self, iix):
            # if iinum is None or self.iix == iinum, do nothing and return nothing.
            if (iinum is None):
                return None
            elif np.all(iinum == getattr(self, iix)):
                return None

        if iinum is None:
            iinum = slice(None)

        if not np.array_equal(iinum, slice(None)):
            # smash self.variables. Necessary, since we will change the domain size.
            self.variables={}

        if isinstance(iinum, (int, np.integer)): # we convert to slice, to maintain dimensions of output.
            iinum = slice(iinum, iinum+1)  # E.g. [0,1,2][slice(1,2)] --> [1]; [0,1,2][1] --> 1

        # set self.iix
        setattr(self, iix, iinum)
        if self.verbose:
            # convert iinum to string that wont be super long (in case iinum is a long list)
            try:
                assert len(iinum)>20
            except (TypeError, AssertionError):
                iinumprint = iinum
            else:
                iinumprint = 'list with length={:4d}, min={:4d}, max={:4d}, x[1]={:2d}'
                iinumprint = iinumprint.format(len(iinum), min(iinum), max(iinum), iinum[1])
            # print info.
            print('(set_domain) {}: {}'.format(iix, iinumprint),
                  whsp*4, end="\r",flush=True)

        #set self.xLength
        if isinstance(iinum, slice):
            nx = getattr(self, 'n'+iiaxis+'b')
            indSize = len(range(*iinum.indices(nx)))
        else:
            iinum = np.asarray(iinum)
            if iinum.dtype == 'bool':
                indSize = np.sum(iinum)
            else:
                indSize = np.size(iinum)
        setattr(self, iiaxis + 'Length', indSize)

        return True

    def set_domain_iiaxes(self, iix=None, iiy=None, iiz=None, internal=False):
        '''sets iix, iiy, iiz, xLength, yLength, zLength.
        iix: slice, int, list, array, or None (default)
            Slice to be taken from get_var quantity in x axis
            None --> don't change existing self.iix.
                     if self.iix doesn't exist, set it to slice(None).
            To set existing self.iix to slice(None), use iix=slice(None).
        iiy, iiz: similar to iix.
        internal: bool (default: False)
            if internal and self.do_stagger, don't change slices.
            internal=True inside get_var.

        updates x, y, z, dx1d, dy1d, dz1d afterwards, if any domains were changed.
        '''
        AXES = ('x', 'y', 'z')
        if internal and self.do_stagger:
            # we slice at the end, only. For now, set all to slice(None)
            slices = (slice(None), slice(None), slice(None))
        else:
            slices = (iix, iiy, iiz)

        any_domain_changes = False
        for x, iix in zip(AXES, slices):
            domain_changed     = self.set_domain_iiaxis(iix, x)
            any_domain_changes = any_domain_changes or domain_changed

        # update x, y, z, dx1d, dy1d, dz1d appropriately.
        if any_domain_changes:
            self.__read_mesh(self.meshfile, firstime=False)

    def genvar(self): 
        '''
        Dictionary of original variables which will allow to convert to cgs. 
        '''
        self.varn={}
        self.varn['rho']= 'r'
        self.varn['tg'] = 'tg'
        self.varn['pg'] = 'p'
        self.varn['ux'] = 'ux'
        self.varn['uy'] = 'uy'
        self.varn['uz'] = 'uz'
        self.varn['e']  = 'e'
        self.varn['bx'] = 'bx'
        self.varn['by'] = 'by'
        self.varn['bz'] = 'bz'
    
    @document_vars.quant_tracking_top_level
    def _load_quantity(self, var, cgsunits=1.0, **kwargs):
        '''helper function for get_var; actually calls load_quantities for var.'''
        __tracebackhide__ = True  # hide this func from error traceback stack
        # look for var in self.variables
        if cgsunits==1.0:
            if var in self.variables:                 # if var is still in memory,
                return self.variables[var]  # load from memory instead of re-reading.
        # Try to load simple quantities.
        val = load_fromfile_quantities.load_fromfile_quantities(self, var,
                                                save_if_composite=True, cgsunits=cgsunits, **kwargs)

        # Try to load "regular" quantities
        if val is None:
            val = load_quantities(self, var, **kwargs)
        # Try to load "arithmetic" quantities.
        if val is None:
            val = load_arithmetic_quantities(self, var, **kwargs) 

        return val

    def get_var(self, var, snap=None, *args, iix=None, iiy=None, iiz=None, **kwargs):
        """
        Reads a variable from the relevant files.

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        snap - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot
            by running self.set_snap(snap).

        **kwargs go to load_..._quantities functions.
        """
        if self.verbose:
            print('(get_var): reading ', var, whsp*6, end="\r", flush=True)

        if var in ['x', 'y', 'z']:
            return getattr(self, var)

        if (snap is not None) and np.any(snap != self.snap):
            if self.verbose:
                print('(get_var): setsnap ', snap, self.snap, whsp*6,
                    end="\r",flush=True)
            self.set_snap(snap)
            self.variables={}

        # set iix, iiy, iiz appropriately
        slices_names_and_vals = (('iix', iix), ('iiy', iiy), ('iiz', iiz))
        original_slice = [iix if iix is not None else getattr(self, slicename, slice(None))
                           for slicename, iix in slices_names_and_vals]
        self.set_domain_iiaxes(iix=iix, iiy=iiy, iiz=iiz, internal=True)
        
        if var in self.varn.keys(): 
            var=self.varn[var]

        if (self.sel_units=='cgs'): 
            varu=var.replace('x','')
            varu=varu.replace('y','')
            varu=varu.replace('z','')
            if varu == 'r': 
                varu = 'rho'
            if (varu in self.uni.uni.keys()): 
                cgsunits = self.uni.uni[varu]
            else: 
                cgsunits = 1.0

        else: 
            cgsunits = 1.0
                
        # get value of variable.

        val = self._load_quantity(var, cgsunits=cgsunits, **kwargs)

        # do post-processing
        val = self._get_var_postprocess(val, var=var, original_slice=original_slice)
        return val

    def _get_var_postprocess(self, val, var='', original_slice=[slice(None) for x in ('x', 'y', 'z')]):
        '''does post-processing for get_var.
        This includes:
            - handle "creating documentation" or "var==''" case
            - handle "don't know how to get this var" case
            - reshape result as appropriate (based on iix,iiy,iiz)
        returns val after the processing is complete.
        '''
        # handle documentation case
        if document_vars.creating_vardict(self):
            return None
        elif var == '':
            print('Variables from snap or aux files:')
            print(self.simple_vars)
            print('Variables from xy aux files:')
            print(self.auxxyvars)
            if hasattr(self,'vardict'):
                self.vardocs()
            return None

        # handle "don't know how to get this var" case
        if val is None:
            errmsg = ('get_var: do not know (yet) how to calculate quantity {}. '
                '(Got None while trying to calculate it.) '
                'Note that simple_var available variables are: {}. '
                '\nIn addition, get_quantity can read others computed variables; '
                "see e.g. help(self.get_var) or get_var('')) for guidance.")
            raise ValueError(errmsg.format(repr(var), repr(self.simple_vars)))

        # set original_slice if do_stagger and we are at the outermost layer.
        if self.do_stagger and not self._getting_internal_var():
            self.set_domain_iiaxes(*original_slice, internal=False)
        
        # reshape if necessary... E.g. if var is a simple var, and iix tells to slice array.
        if (np.ndim(val) == self.ndim) and (np.shape(val) != self.shape):
            def isslice(x): return isinstance(x, slice)
            if isslice(self.iix) and isslice(self.iiy) and isslice(self.iiz):
                val = val[self.iix, self.iiy, self.iiz]  # we can index all together
            else:  # we need to index separately due to numpy multidimensional index array rules.
                val = val[self.iix,:,:]
                val = val[:,self.iiy,:]
                val = val[:,:,self.iiz]

        return val

    def _getting_internal_var(self):
        '''returns whether we are currently inside of an internal call to _load_quantity.
        (_load_quantity is called inside of get_var.)

        Here is an example, with the comments telling self._getting_internal_var() at that line:
            # False
            get_var('ux') -->
                # False
                px = get_var('px') -->
                    # True
                    returns the value of px
                # False
                rxdn = get_var('rxdn') -->
                    # True
                    r = get_var('r') -->
                        # True
                        returns the value of r
                    # True
                    returns apply_xdn_to(r)
                # False
                return px / rxdn
            # False
        (Of course, this example assumes get_var('ux') was called externally.)
        '''
        return getattr(self, document_vars.LOADING_LEVEL) >= 0

    def trans2comm(self, varname, snap=None, *args, **kwargs): 
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

        
        self.trans2commaxes() 

        self.sel_units = 'cgs'

        sign = 1.0
        if varname[-1] in ['x','y','z']: 
            varname = varname+'c'
            if varname[-2] in ['y','z']: 
                sign = -1.0 
        
        var = self.get_var(varname,snap=snap, *args, **kwargs)
        var = sign * var

        var = var[...,::-1].copy()

        return var

    def trans2commaxes(self): 
        if self.transunits == False:
          self.transunits = True
          if self.sel_units == 'cgs':
            cte=1.0
          else: 
            cte=1.0e8
          self.x = self.x*cte
          self.dx = self.dx*cte
          self.y = self.y*cte
          self.dy = self.dy*cte
          self.z = - self.z[::-1].copy()*cte
          self.dz = - self.dz1d[::-1].copy()*cte 

    def trans2noncommaxes(self): 

        if self.transunits == True:
          self.transunits = False
          if self.sel_units == 'cgs':
            cte=1.0
          else: 
            cte=1.0e8
          self.x =  self.x/cte
          self.dx =  self.dx/cte
          self.y =  self.y/cte
          self.dy =  self.dy/cte
          self.z = - self.z[::-1].copy()/cte
          self.dz = - self.dz1d[::-1].copy()/cte

    @document_vars.quant_tracking_simple('SIMPLE_VARS')
    def _get_simple_var(self, var, order='F', mode='r', 
                        panic=False, *args, **kwargs):
        """
        Gets "simple" variable (ie, only memmap, not load into memory).
        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        order - string, optional
            Must be either 'C' (C order) or 'F' (Fortran order, default).
        mode - string, optional
            numpy.memmap read mode. By default is read only ('r'), but
            you can use 'r+' to read and write. DO NOT USE 'w+'.
        Returns
        -------
        result - numpy.memmap array
            Requested variable.
        """
        if var == '':
            _simple_vars_msg = ('Quantities which are stored by the simulation. These are '
                                'loaded as numpy memmaps by reading data files directly.')
            docvar = document_vars.vars_documenter(self, 'SIMPLE_VARS', None, _simple_vars_msg)
            # TODO: << add documentation for bifrost simple vars, here.
            return None

        if var not in self.simple_vars:
            return None

        if self.verbose:
            print('(get_var): reading simple ', var, whsp*5,  # TODO: show np.shape(val) info somehow?
                end="\r",flush=True)

        if np.shape(self.snap) != ():
            currSnap = self.snap[self.snapInd]
            currStr = self.snap_str[self.snapInd]
        else:
            currSnap = self.snap
            currStr = self.snap_str
        if currSnap < 0:
            filename = self.file_root
            if panic: 
                fsuffix_b = ''
            else: 
                fsuffix_b = '.scr'
        elif currSnap == 0:
            filename = self.file_root
            fsuffix_b = ''
        else:
            filename = self.file_root + currStr
            fsuffix_b = ''

        if var in self.snapvars:
            if panic: 
                fsuffix_a = '.panic'
            else: 
                fsuffix_a = '.snap'
            idx = (self.snapvars).index(var)
            filename += fsuffix_a + fsuffix_b
        elif var in self.auxvars:
            fsuffix_a = '.aux'
            idx = self.auxvars.index(var)
            filename += fsuffix_a + fsuffix_b
        elif var in self.hionvars:
            idx = self.hionvars.index(var)
            isnap = self.get_param('isnap', error_prop=True)
            if panic: 
                filename = filename + '.hion.panic'
            else: 
                if isnap <= -1:
                    filename = filename + '.hion.snap.scr'
                elif isnap == 0:
                    filename = filename + '.hion.snap'
                elif isnap > 0:
                    filename = '%s.hion_%03d.snap' % (self.file_root, isnap)
                    if not os.path.isfile(filename):
                        filename = '%s_.hion%s.snap' % (self.file_root, isnap)
        elif var in self.heliumvars:
            idx = self.heliumvars.index(var)
            isnap = self.get_param('isnap', error_prop=True)
            if panic: 
                filename = filename + '.helium.panic'
            else:
                if isnap <= -1:
                    filename = filename + '.helium.snap.scr'
                elif isnap == 0:
                    filename = filename + '.helium.snap'
                elif isnap > 0:
                    filename = '%s.helium_%s.snap' % (self.file_root, isnap)
        else:
            raise ValueError(('_get_simple_var: could not find variable '
                              '%s. Available variables:' % (var) +
                              '\n' + repr(self.simple_vars)))
        dsize = np.dtype(self.dtype).itemsize
        if self.ghost_analyse:
            offset = self.nxb * self.nyb * self.nzb * idx * dsize
            ss = (self.nxb, self.nyb, self.nzb)
        else:
            offset = ((self.nxb + (self.nxb - self.nx)) * 
                      (self.nyb + (self.nyb - self.ny)) *
                      (self.nzb + (self.nzb - self.nz) // 2) * idx * dsize)
            ss = (self.nx, self.ny, self.nz)

        if var in self.heliumvars:
            return np.exp(np.memmap(filename, dtype=self.dtype, order=order,
                                    mode=mode, offset=offset, shape=ss))
        else:
            return np.memmap(filename, dtype=self.dtype, order=order,
                             mode=mode, offset=offset, shape=ss)

    def _get_simple_var_xy(self, *args, **kwargs):
        '''returns load_fromfile_quantities._get_simple_var_xy(self, *args, **kwargs).
        raises ValueError if result is None (to match historical behavior of this function).

        included for backwards compatibility purposes, only.
        new code should instead use the function from load_fromfile_quantitites.
        '''
        val = load_fromfile_quantities._get_simple_var_xy(self, *args, **kwargs)
        if val is None:
            raise ValueError(('_get_simple_var_xy: variable'
                              ' %s not available. Available vars:'
                              % (var) + '\n' + repr(self.auxxyvars)))

    def _get_composite_var(self, *args, **kwargs):
        '''returns load_fromfile_quantities._get_composite_var(self, *args, **kwargs).

        included for backwards compatibility purposes, only.
        new code should instead use the function from load_fromfile_quantitites.
        '''
        return load_fromfile_quantities._get_composite_var(self, *args, **kwargs)

    def get_electron_density(self, sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Gets electron density

        Parameters
        ----------
        self : BifrostData instance
            A BifrostData object loaded for a given snapshot.
        sx, sy, sz : slice objecs
            Slice objects for x, y, and z dimensions, when not all points
            are needed.
        """
        from astropy.units import Quantity
        if self.hion:
            ne = self.get_var('hionne')[sx, sy, sz]
        else:
            ee = self.get_var('ee')[sx, sy, sz]
            ee = ee * self.uni.u_ee
            eostab = Rhoeetab(fdir=self.fdir)
            rho = self.r[sx, sy, sz] * self.uni.u_r   # to cm^-3
            ne = eostab.tab_interp(rho, ee, order=1)
        return Quantity(ne, unit='1/cm3')


    def get_hydrogen_pops(self, sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Gets hydrogen populations, or total number of hydrogen atoms,
        if hydrogen populations not available.

        Parameters
        ----------
        sx, sy, sz : slice objecs
            Slice objects for x, y, and z dimensions, when not all points
            are needed.
        """
        from astropy.units import Quantity
        if self.hion:
            shape = [6, ]
            # calculate size of slices to determine array shape
            for item, n in zip([sx, sy, sz], [self.nx, self.ny, self.nz]):
                slice_size = np.mgrid[item].size
                if slice_size == 0:
                    slice_size = n
                shape.append(slice_size)
            nh = np.empty(shape, dtype='Float32')
            for k in range(6):
                nv = self.get_var('n%i' % (k + 1))
                nh[k] = nv[sx, sy, sz]
        else:
            rho = self.r[sx, sy, sz] * self.uni.u_r
            subsfile = os.path.join(self.fdir, 'subs.dat')
            tabfile = os.path.join(self.fdir, self.get_param('tabinputfile', error_prop=True).strip())
            tabparams = []
            if os.access(tabfile, os.R_OK):
                tabparams = read_idl_ascii(tabfile, obj=self)
            if 'abund' in tabparams and 'aweight' in tabparams:
                abund = np.array(tabparams['abund']).astype('f')
                aweight = np.array(tabparams['aweight']).astype('f')
                grph = calc_grph(abund, aweight)
            elif os.access(subsfile, os.R_OK):
                grph = subs2grph(subsfile)
            else:
                grph = 2.380491e-24
            nh = rho / grph
            nh = nh[None]  # add extra empty dimension when nhydr = 1
        return Quantity(nh, unit='1/cm3')

    def write_rh15d(self, outfile, desc=None, append=True, sx=slice(None),
                    sy=slice(None), sz=slice(None), write_all_v=False):
        """
        Writes snapshot in RH 1.5D format.
        Parameters
        ----------
        outfile - string
            File name to write
        append - bool, optional
            If True (default) will append output as a new snapshot in file.
            Otherwise, creates new file (will fail if file exists).
        desc - string, optional
            Description string
        sx, sy, sz - slice object
            Slice objects for x, y, and z dimensions, when not all points
            are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
            for every second point up to 100.
        write_all_v - bool, optional
            If true, will write also the vx and vy components.
        Returns
        -------
        None.
        """
        from . import rh15d
        verbose = self.verbose
        if verbose:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=4)
            except ModuleNotFoundError:
                verbose = False
        # unit conversion to SI
        # Comment to Tiago, params['units'] is hard coded. I
        # strongly recoment to use Bifrost_units.
        ul = self.params['u_l'][self.snapInd] / 1.e2  # to metres
        # to g/cm^3  (for ne_rt_table)
        ur = self.params['u_r'][self.snapInd]
        ut = self.params['u_t'][self.snapInd]         # to seconds
        uv = ul / ut
        ub = self.params['u_b'][self.snapInd] * 1e-4  # to Tesla
        ue = self.params['u_ee'][self.snapInd]        # to erg/g
        if verbose:
            pbar.set_description("Slicing and unit conversion")
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]

        if self.do_mhd:
            Bx = do_cstagger(self.bx, 'xup', obj=self)[sx, sy, sz]
            By = do_cstagger(self.by, 'yup', obj=self)[sx, sy, sz]
            Bz = do_cstagger(self.bz, 'zup', obj=self)[sx, sy, sz]
            # Bx = cstagger.xup(self.bx)[sx, sy, sz]
            # By = cstagger.yup(self.by)[sx, sy, sz]
            # Bz = cstagger.zup(self.bz)[sx, sy, sz]
            # Change sign of Bz (because of height scale) and By
            # (to make right-handed system)
            Bx = Bx * ub
            By = -By * ub
            Bz = -Bz * ub
        else:
            Bx = By = Bz = None

        vz = do_cstagger(self.pz, 'zup', obj=self)[sx, sy, sz] / rho
        # vz = cstagger.zup(self.pz)[sx, sy, sz] / rho
        vz *= -uv
        if write_all_v:
            vx = cstagger.xup(self.px)[sx, sy, sz] / rho
            vx *= uv
            vy = cstagger.yup(self.py)[sx, sy, sz] / rho
            vy *= -uv
        else:
            vx = None
            vy = None
        x = self.x[sx] * ul
        y = self.y[sy] * (-ul)
        z = self.z[sz] * (-ul)
        if verbose:
            pbar.update()
            pbar.set_description("Getting hydrogen populations")
        nh = self.get_hydrogen_pops(sx, sy, sz).to_value('1/m3')
        if verbose:
            pbar.update()
            pbar.set_description("Getting electron density")
        ne = self.get_electron_density(sx, sy, sz).to_value('1/m3')
        if desc is None:
            desc = 'BIFROST snapshot from sequence %s, sx=%s sy=%s sz=%s.' % \
                   (self.file_root, repr(sx), repr(sy), repr(sz))
            if self.hion:
                desc = 'hion ' + desc
        # write to file
        if verbose:
            pbar.update()
            pbar.set_description("Writing to file")
        rh15d.make_xarray_atmos(outfile, temp, vz, z, nH=nh, ne=ne, x=x, y=y,
                                vx=vx, vy=vy, Bx=Bx, By=By, Bz=Bz, desc=desc,
                                append=append, snap=self.snap)
        if verbose:
            pbar.update()

    def write_multi3d(self, outfile, mesh='mesh.dat', desc=None,
                      sx=slice(None), sy=slice(None), sz=slice(None)):
        """
        Writes snapshot in Multi3D format.
        Parameters
        ----------
        outfile - string
            File name to write
        mesh - string, optional
            File name of the mesh file to write.
        desc - string, optional
            Description string
        sx, sy, sz - slice object
            Slice objects for x, y, and z dimensions, when not all points
            are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
            for every second point up to 100.
        Returns
        -------
        None.
        """
        from .multi3d import Multi3dAtmos
        # unit conversion to cgs and km/s
        ul = self.params['u_l'][self.snapInd]   # to cm
        ur = self.params['u_r'][self.snapInd]   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t'][self.snapInd]   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ue = self.params['u_ee'][self.snapInd]  # to erg/g
        nh = None
        if self.verbose:
            print('Slicing and unit conversion...', whsp*4, end="\r",
                flush=True)
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]
        # Change sign of vz (because of height scale) and vy (to make
        # right-handed system)
        # vx = cstagger.xup(self.px)[sx, sy, sz] / rho
        vx = do_cstagger(self.px, 'xup', obj=self)[sx, sy, sz] / rho
        vx *= uv
        vy = do_cstagger(self.py, 'yup', obj=self)[sx, sy, sz] / rho
        vy *= -uv
        vz = do_cstagger(self.pz, 'zup', obj=self)[sx, sy, sz] / rho
        vz *= -uv
        rho = rho * ur  # to cgs
        x = self.x[sx] * ul
        y = self.y[sy] * ul
        z = self.z[sz] * (-ul)
        ne = self.get_electron_density(sx, sy, sz).to_value('1/cm3')
        # write to file
        if self.verbose:
            print('Write to file...', whsp*8, end="\r", flush=True)
        nx, ny, nz = temp.shape
        fout = Multi3dAtmos(outfile, nx, ny, nz, mode="w+", read_nh=self.hion)
        fout.ne[:] = ne
        fout.temp[:] = temp
        fout.vx[:] = vx
        fout.vy[:] = vy
        fout.vz[:] = vz
        fout.rho[:] = rho
        if self.hion:
            nh = self.get_hydrogen_pops(sx, sy, sz).to_value('1/cm3')
            fout.nh[:] = np.transpose(nh, axes=(1, 2, 3, 0))
        # write mesh?
        if mesh:
            fout2 = open(mesh, "w")
            fout2.write("%i\n" % nx)
            x.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % ny)
            y.tofile(fout2, sep="  ", format="%11.5e")
            fout2.write("\n%i\n" % nz)
            z.tofile(fout2, sep="  ", format="%11.5e")
            fout2.close()

    ## MISC. CONVENIENCE METHODS ##

    def get_varm(self, *args__get_var, **kwargs__get_var):
        '''get_var but returns np.mean() of result.
        provided for convenience for quicker debugging.
        '''
        return np.mean(self.get_var(*args__get_var, **kwargs__get_var))

    def get_varu(self, *args__get_var, mode='si', **kwargs__get_var):
        '''get_var() then get_units() and return (result * units factor, units name).
        e.g. r = self.get_var('r'); units = self.get_units('si'); return (r*units.factor, units.name).
        e.g. self.get_varu('r') --> (r * units.factor, 'kg / m^{3}')
        '''
        x = self.get_var(*args__get_var, **kwargs__get_var)
        u = self.get_units(mode=mode)
        return (x * u.factor, u.name)

    def get_varU(self, *args__get_var, mode='si', **kwargs__get_var):
        '''get_varm() then get_units and return (result * units factor, units name).
        equivalent to: x=self.get_varu(...); return (np.mean(x[0]), x[1]).
        '''
        x = self.get_varm(*args__get_var, **kwargs__get_var)
        u = self.get_units(mode=mode)
        return (x * u.factor, u.name)

    get_varmu = get_varum = get_varU  # aliases for get_varU

    def get_varV(self, var, *args__get_var, mode='si', vmode='modhat', **kwargs__get_var):
        '''returns get_varU info but for a vector.
        Output format depends on vmode:
            'modhat' ---> ((|var|,units), get_unit_vector(var, mean=True))
            'modangle' -> ((|var|,units), (angle between +x and var, units of angle))
            'xyz' ------> ([varx, vary, varz], units of var)
        '''
        VALIDMODES = ('modhat', 'modangle', 'xyz')
        vmode = vmode.lower()
        assert vmode in VALIDMODES, 'vmode {} invalid! Expected vmode in {}.'.format(repr(vmode), VALIDMODES)
        if vmode in ('modhat', 'modangle'):
            mod = self.get_varU('mod'+var, *args__get_var, mode=mode, **kwargs__get_var)
            if vmode == 'modhat':
                hat = self.get_unit_vector(var, mean=True, **kwargs__get_var)
                return (mod, hat)
            elif vmode == 'modangle':
                angle = self.get_varU(var+'_anglexxy', *args__get_var, mode=mode, **kwargs__get_var)
                return (mod, angle)
        elif vmode == 'xyz':
            varxyz = [self.get_varm(var + x, *args__get_var, **kwargs__get_var) for x in ('x', 'y', 'z')]
            units  = self.get_units(mode=mode)
            return (np.array(varxyz) * units.factor, units.name)
        assert False  # if we made it to this line it means something is wrong with the code here.

    def _varV_formatter(self, vmode, fmt_values='{: .2e}', fmt_units='{:^7s}'):
        '''returns a format function for pretty formatting of the result of get_varV.'''
        VALIDMODES = ('modhat', 'modangle', 'xyz')
        vmode = vmode.lower()
        assert vmode in VALIDMODES, 'vmode {} invalid! Expected vmode in {}.'.format(repr(vmode), VALIDMODES)
        if vmode == 'modhat':
            def fmt(x):
                mag   = fmt_values.format(x[0][0])
                units = fmt_units.format(x[0][1])
                hat   = ('[ '+fmt_values+', '+fmt_values+', '+fmt_values+' ]').format(*x[1])
                return 'magnitude = {} [{}];  unit vector = {}'.format(mag, units, hat)
        elif vmode == 'modangle':
            def fmt(x):
                mag   = fmt_values.format(x[0][0])
                units = fmt_units.format(x[0][1])
                angle = fmt_values.format(x[1][0])
                angle_units = fmt_units.format(x[1][1])
                return 'magnitude = {} [{}];  angle (from +x) = {} [{}]'.format(mag, units, angle, angle_units)
        elif vmode == 'xyz':
            def fmt(x):
                vec   = ('[ '+fmt_values+', '+fmt_values+', '+fmt_values+' ]').format(*x[0])
                units = fmt_units.format(x[1])
                return '{}  [{}]'.format(vec, units)
        fmt.__doc__ = 'formats result of get_varV. I was made by helita.sim.bifrost._varV_formatter.'
        return fmt

    def zero(self, **kw__np_zeros):
        '''return np.zeros() with shape equal to shape of result of get_var()'''
        return np.zeros(self.shape, **kw__np_zeros)

    def get_snap_at_time(self, t, units='simu'):
        '''get snap number which is closest to time t.
        
        units: 's', 'si', 'cgs', or 'simu' (default).
            's', 'si', 'cgs' --> enter t in seconds; return time at snap in seconds.
            'simu' (default) --> enter t in simulation units; return time at snap in simulation units.

        Return (snap number, time at this snap).
        '''
        snaps = self.snap
        try:
            snaps[0]
        except TypeError:
            raise TypeError('expected self.snap (={}) to be a list. You can set it via self.set_snap()'.format(snaps))
        units = units.lower()
        VALIDUNITS = ('s', 'si', 'cgs', 'simu') 
        assert units in VALIDUNITS, 'expected units (={}) to be one of {}'.format(repr(units), VALIDUNITS)
        if units in ('s', 'si', 'cgs'):
            u_t = self.uni.u_t   # == self.uni.usi_t.   time [simu units] * u_t = time [seconds].
        else:
            u_t = 1
        t_get = t / u_t   # time [simu units]
        idxmin = np.argmin(np.abs(self.time - t_get))
        return snaps[idxmin], self.time[idxmin] * u_t

    def set_snap_time(self, t, units='simu', snaps=None, snap=None):
        '''set self.snap to the snap which is closest to time t.
        
        units: 's', 'si', 'cgs', or 'simu' (default).
            's', 'si', 'cgs' --> enter t in seconds; return time at snap in seconds.
            'simu' (default) --> enter t in simulation units; return time at snap in simulation units.
        snaps: None (default) or list of snaps.
            None --> use self.snap for list of snaps to choose from.
            list --> use snaps for list of snaps to choose from.
                self.set_snap_time(t, ..., snaps=SNAPLIST) is equivalent to:
                self.set_snap(SNAPLIST); self.set_snap_time(t, ...)
        snap: alias for snaps kwarg. (Ignore snap if snaps is also entered, though.)

        Return (snap number, time at this snap).
        '''

        snaps = snaps if (snaps is not None) else snap
        if snaps is not None:
            self.set_snap(snaps)
        try:
            result_snap, result_time = self.get_snap_at_time(t, units=units)
        except TypeError:
            raise TypeError('expected self.snap to be a list, or snaps=list_of_snaps input to function.')
        self.set_snap(result_snap)
        return (result_snap, result_time)

    def get_lmin(self):
        '''return smallest length resolvable for each direction ['x', 'y', 'z'].
        result is in [simu. length units]. Multiply by self.uni.usi_l to convert to SI.

        return 1 (instead of 0) for any direction with number of points < 2.
        '''
        def _dxmin(x):
            dx1d = getattr(self, 'd'+x+'1d')
            if len(dx1d)==1:
                return 1
            else:
                return dx1d.min()
        return np.array([_dxmin(x) for x in AXES])

    def get_kmax(self):
        '''return largest value of each component of wavevector resolvable by self.
        I.e. returns [max kx, max ky, max kz].
        result is in [1/ simu. length units]. Divide by self.uni.usi_l to convert to SI.
        '''
        return 2 * np.pi / self.get_lmin()

    def get_unit_vector(self, var, mean=False, **kw__get_var):
        '''return unit vector of var. [varx, vary, varz]/|var|.'''
        varx = self.get_var(var+'x', **kw__get_var)
        vary = self.get_var(var+'y', **kw__get_var)
        varz = self.get_var(var+'z', **kw__get_var)
        varmag = self.get_var('mod'+var, **kw__get_var)
        if mean:
            varx, vary, varz, varmag = varx.mean(), vary.mean(), varz.mean(), varmag.mean()
        return np.array([varx, vary, varz]) / varmag

    def write_mesh_file(self, meshfile='untitled_mesh.mesh', u_l=None):
        '''writes mesh to meshfilename.
        mesh will be the mesh implied by self,
        using values for x, y, z, dx1d, dy1d, dz1d, indexed by iix, iiy, iiz.

        u_l: None, or a number
            cgs length units (length [simulation units] * u_l = length [cm]),
                for whoever will be reading the meshfile.
            None -> use length units of self.

        Returns abspath to generated meshfile.
        '''
        if not meshfile.endswith('.mesh'):
            meshfile += '.mesh'
        if u_l is None:
            scaling = 1.0
        else:
            scaling = self.uni.u_l / u_l
        AXES = ('x', 'y', 'z')
        kw_x    = {  x   : getattr(self,    x      ) * scaling for x in AXES}
        kw_dx   = {'d'+x : getattr(self, 'd'+x+'1d') / scaling for x in AXES}
        kw_nx   = {'n'+x : getattr(self, x+'Length')           for x in AXES}
        kw_mesh = {**kw_x, **kw_nx, **kw_dx}
        Create_new_br_files().write_mesh(**kw_mesh, meshfile=meshfile)
        return os.path.abspath(meshfile)

    write_meshfile = write_mesh_file  # alias

    def get_coords(self, mode='si', axes=None):
        '''returns dict of coords, with keys ['x', 'y', 'z', 't'].
        coords units are based on mode.
            'si' (default) -> [meters] for x,y,z; [seconds] for t.
            'cgs'     ->      [cm] for x,y,z;  [seconds] for t.
            'simu'    ->      [simulation units] for all coords.
        if axes is not None:
            instead of returning a dict, return coords for the axes provided, in the order listed.
            axes can be provided in either of these formats:
                strings: 'x', 'y', 'z', 't'.
                ints:     0 ,  1 ,  2 ,  3 .
            For example:
                c = self.get_coords()
                c['y'], c['t'] == self.get_coords(axes=('y', 'z'))
        '''
        mode = mode.lower()
        VALIDMODES = ('si', 'cgs', 'simu')
        assert mode in VALIDMODES, "Invalid mode ({})! Expected one of {}".format(repr(mode), VALIDMODES)
        if mode=='si':
            u_l = self.uni.usi_l
            u_t = self.uni.usi_t
        elif mode == 'cgs':
            u_l = self.uni.u_l
            u_t = self.uni.u_t
        else: # mode == 'simu'
            u_l = 1
            u_t = 1
        x, y, z = (self_x * u_l for self_x in (self.x, self.y, self.z))
        t       = self.time * u_t
        result = dict(x=x, y=y, z=z, t=t)
        if axes is not None:
            AXES_LOOKUP = {'x':'x', 0:'x', 'y':'y', 1:'y', 'z':'z', 2:'z', 't':'t', 3:'t'}
            result = tuple(result[AXES_LOOKUP[axis]] for axis in axes)
        return result

    if file_memory.DEBUG_MEMORY_LEAK:
        def __del__(self):
            print('deleted {}'.format(self), flush=True)


####################
#  LOCATING SNAPS  #
####################

SnapStuff = collections.namedtuple('SnapStuff', ('snapname', 'snaps'))

def get_snapstuff(dd=None):
    '''return (get_snapname(), available_snaps()).
    dd: None or BifrostData object.
        None -> do operations locally.
        else -> cd to dd.fdir, first.
    '''
    snapname = get_snapname(dd=dd)
    snaps    = get_snaps(snapname=snapname, dd=dd)
    return SnapStuff(snapname=snapname, snaps=snaps)

snapstuff = get_snapstuff   # alias

def get_snapname(dd=None):
    '''gets snapname by reading it from mhd.in'''
    with EnterDirectory(_get_dd_fdir(dd)):
        mhdin_ascii = read_idl_ascii('mhd.in')
        return mhdin_ascii['snapname']

snapname = get_snapname   # alias

def available_snaps(dd=None, snapname=None):
    '''list available snap numbers.
    Does look for: snapname_*.idl, snapname.idl (i.e. snap 0)
    Doesn't look for: .pan, .scr, .aux files.
    snapname: None (default) or str
        snapname parameter from mhd.in. If None, get snapname.
    if dd is not None, look in dd.fdir.
    '''
    with EnterDirectory(_get_dd_fdir(dd)):
        snapname = snapname if snapname is not None else get_snapname()
        snaps = [_snap_to_N(f, snapname) for f in os.listdir()]
        snaps = [s for s in snaps if s is not None]
        snaps = sorted(snaps)
        return snaps

snaps      = available_snaps   # alias
get_snaps  = available_snaps   # alias
list_snaps = available_snaps   # alias

def snaps_info(dd=None, snapname=None):
    '''returns string with length of snaps, as well as min and max.'''
    snaps = get_snaps(dd=dd, snapname=snapname)
    return 'There are {} snaps, from {} (min) to {} (max)'.format(len(snaps), min(snaps), max(snaps))

class EnterDir:
    '''context manager for remembering directory.
    upon enter, cd to directory. upon exit, restore original working directory.
    '''
    def __init__(self, directory=os.curdir):
        self.cwd       = os.path.abspath(os.getcwd())
        self.directory = directory

    def __enter__ (self):
        os.chdir(self.directory)

    def __exit__ (self, exc_type, exc_value, traceback):
        os.chdir(self.cwd)

EnterDirectory = EnterDir  #alias

def _get_dd_fdir(dd=None):
    '''return dd.fdir if dd is not None, else os.curdir.'''
    if dd is not None:
        fdir = dd.fdir
    else:
        fdir = os.curdir
    return fdir

def _snap_to_N(name, base, sep='_', ext='.idl'):
    '''returns N as number given snapname (and basename) if possible, else None.
    for all strings in exclude, if name contains string, return None.
    E.g. _snap_to_N('s_075.idl', 's') == 75
    E.g. _snap_to_N('s.idl', 's')     == 0
    E.g. _snap_to_N('notasnap', 's')  == None
    '''
    if not name.startswith(base):
        return None
    namext = os.path.splitext(name)
    if   namext[1] != ext :
        return None
    elif namext[0] == base:
        return 0
    else:
        try:
            snapN = int(namext[0][len(base+sep):])
        except ValueError:
            return None
        else:
            return snapN

# include methods (and some aliases) for getting snaps in BifrostData object
BifrostData.get_snapstuff   = get_snapstuff
BifrostData.get_snapname    = get_snapname
BifrostData.available_snaps = available_snaps
BifrostData.get_snaps       = available_snaps
BifrostData.snaps_info      = snaps_info


####################
#  WRITING SNAPS   #
####################

def write_br_snap(rootname,r,px,py,pz,e,bx,by,bz):
    nx, ny, nz = r.shape
    data = np.memmap(rootname, dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,8))
    data[...,0] = r
    data[...,1] = px
    data[...,2] = py
    data[...,3] = pz
    data[...,4] = e
    data[...,5] = bx
    data[...,6] = by
    data[...,7] = bz
    data.flush()


def paramfile_br_update(infile, outfile, new_values):
    ''' Updates a given number of fields with values on a bifrost.idl file.
        These are given in a dictionary: fvalues = {field: value}.
        Reads from infile and writes into outfile.'''
    out = open(outfile, 'w')
    with open(infile) as fin:
        for line in fin:
            if line[0] == ';':
                out.write(line)
            elif line.find('=') < 0:
                out.write(line)
            else:
                ss = line.split('=')[0]
                ssv = ss.strip().upper()
                if ssv in list(new_values.keys()):
                    out.write('%s= %s\n' % (ss, str(new_values[ssv])))
                else:
                    out.write(line)
    return


class Create_new_br_files:
    def write_mesh(self, x=None, y=None, z=None, nx=None, ny=None, nz=None,
                   dx=None, dy=None, dz=None, meshfile="newmesh.mesh"):
        """
        Writes mesh to ascii file.

        The meshfile units are simulation units for length (or 1/length, for derivatives).
        """
        def __xxdn(f):
            '''
            f is centered on (i-.5,j,k)
            '''
            nx = len(f)
            d = -5. / 2048
            c = 49. / 2048
            b = -245. / 2048
            a = .5 - b - c - d
            x = (a * (f + np.roll(f, 1)) +
                 b * (np.roll(f, -1) + np.roll(f, 2)) +
                 c * (np.roll(f, -2) + np.roll(f, 3)) +
                 d * (np.roll(f, -3) + np.roll(f, 4)))
            for i in range(0, 4):
                x[i] = x[4] - (4 - i) * (x[5] - x[4])
            for i in range(1, 4):
                x[nx - i] = x[nx - 4] + i * (x[nx - 4] - x[nx - 5])

            x[nx-3:] = x[nx-3:][::-1] # fixes order in the tail of x
            return x

        def __ddxxup(f, dx=None):
            '''
            X partial up derivative
            '''
            if dx is None:
                dx = 1.
            nx = len(f)
            d = -75. / 107520. / dx
            c = 1029 / 107520. / dx
            b = -8575 / 107520. / dx
            a = 1. / dx - 3 * b - 5 * c - 7 * d
            x = (a * (np.roll(f, -1) - f) +
                 b * (np.roll(f, -2) - np.roll(f, 1)) +
                 c * (np.roll(f, -3) - np.roll(f, 2)) +
                 d * (np.roll(f, -4) - np.roll(f, 3)))
            x[:3] = x[3]
            for i in range(1, 5):
                x[nx - i] = x[nx - 5]
            return x

        def __ddxxdn(f, dx=None):
            '''
            X partial down derivative
            '''
            if dx is None:
                dx = 1.
            nx = len(f)
            d = -75. / 107520. / dx
            c = 1029 / 107520. / dx
            b = -8575 / 107520. / dx
            a = 1. / dx - 3 * b - 5 * c - 7 * d
            x = (a * (f - np.roll(f, 1)) +
                 b * (np.roll(f, -1) - np.roll(f, 2)) +
                 c * (np.roll(f, -2) - np.roll(f, 3)) +
                 d * (np.roll(f, -3) - np.roll(f, 4)))
            x[:4] = x[4]
            for i in range(1, 4):
                x[nx - i] = x[nx - 4]
            return x

        f = open(meshfile, 'w')

        for p in ['x', 'y', 'z']:
            setattr(self, p, locals()[p])
            if (getattr(self, p) is None):
                setattr(self, 'n' + p, locals()['n' + p])
                setattr(self, 'd' + p, locals()['d' + p])
                setattr(self, p, np.linspace(0,
                                             getattr(self, 'n' + p) *
                                             getattr(self, 'd' + p),
                                             getattr(self, 'n' + p)))
            else:
                if (len(locals()[p]) < 1):
                    raise ValueError("(EEE): "+p+" axis has length zero")
                setattr(self, 'n' + p, len(locals()[p]))
            if getattr(self, 'n' + p) > 1:
                xmdn = __xxdn(getattr(self, p))
                dxidxup = __ddxxup(getattr(self, p))
                dxidxdn = __ddxxdn(getattr(self, p))
            else:
                xmdn = getattr(self, p)
                dxidxup = np.array([1.0])
                dxidxdn = np.array([1.0])
            f.write(str(getattr(self, 'n' + p)) + "\n")
            f.write(" ".join(map("{:.5f}".format, getattr(self, p))) + "\n")
            f.write(" ".join(map("{:.5f}".format, xmdn)) + "\n")
            f.write(" ".join(map("{:.5f}".format, 1.0/dxidxup)) + "\n")
            f.write(" ".join(map("{:.5f}".format, 1.0/dxidxdn)) + "\n")
        f.close()


############
#  UNITS   #
############

class Bifrost_units(object):
    '''stores units as attributes.

    units starting with 'u_' are in cgs. starting with 'usi_' are in SI.
    Convert to these units by multiplying data by this factor.
    Example:
        r    = obj.get_var('r')  # r    = mass density / (simulation units)
        rcgs = r * obj.uni.u_r   # rcgs = mass density / (cgs units, i.e. (g * cm^-3))
        rsi  = r * obj.uni.usi_r # rsi  = mass density / (si units, i.e. (kg * m^-3))

    all units are uniquely determined by the following minimal set of units:
        (length, time, mass density, gamma)

    you can access documentation on the units themselves via:
        self.help().    (for BifrostData object obj, do obj.uni.help())
        this documentation is not very detailed, but at least tells you
        which physical quantity the units are for.
    '''

    def __init__(self,filename='mhd.in',fdir='./',verbose=True,base_units=None):
        '''get units from file (by reading values of u_l, u_t, u_r, gamma).

        filename: str; name of file. Default 'mhd.in'
        fdir: str; directory of file. Default './'
        verbose: True (default) or False
            True -> if we use default value for a base unit because
                    we can't find its value otherwise, print warning.
        base_units: None (default), dict, or list
            None -> ignore this keyword.
            dict -> if contains any of the keys: u_l, u_t, u_r, gamma,
                    initialize the corresponding unit to the value found.
                    if base_units contains ALL of those keys, IGNORE file.
            list -> provides value for u_l, u_t, u_r, gamma; in that order.
        '''
        import scipy.constants as const
        from astropy import constants as aconst

        self.doc_units = dict()
        self.BASE_UNITS = ['u_l', 'u_t', 'u_r', 'gamma']
        DEFAULT_UNITS = dict(u_l=1.0e8, u_t=1.0e2, u_r=1.0e-7, gamma=1.667)
        _n_base_set = 0  # number of base units which have been set.

        # set units from base_units, if applicable
        if base_units is not None:
            try:
                items = base_units.items()
            except AttributeError: # base_units is a list
                for i, val in enumerate(base_units):
                    setattr(self, self.BASE_UNITS[i], val)
            else:
                for key, val in base_units.items():
                    if key in DEFAULT_UNITS.keys():
                        setattr(self, key, val)
                        _n_base_set += 1
                    elif verbose:
                        print(('(WWW) the key {} is not a base unit',
                              ' so it was ignored').format(key))
        # set units from file (or defaults), if still necessary.
        if _n_base_set != len(DEFAULT_UNITS):
            if filename is None:
                file_exists = False
            else:
                file = os.path.join(fdir, filename)
                file_exists = os.path.isfile(file)
            if file_exists:
                # file exists -> set units using file.
                self.params = read_idl_ascii(file,firstime=True)
                
                def set_unit(key):
                    if getattr(self, key, None) is not None:
                        return
                    try:
                        value = self.params[key]
                    except Exception:
                        value = DEFAULT_UNITS[key]
                        if verbose:
                            printstr = ("(WWW) the file '{file}' does not contain '{unit}'. "
                                        "Default Solar Bifrost {unit}={value} has been selected.")
                            print(printstr.format(file=file, unit=key, value=value))
                    setattr(self, key, value)

                for unit in DEFAULT_UNITS.keys():
                    set_unit(unit)
            else:
                # file does not exist -> set default units.
                units_to_set = {unit: DEFAULT_UNITS[unit] for unit in DEFAULT_UNITS.keys()
                                        if getattr(self, unit, None) is None}
                if verbose:
                    print("(WWW) selected file '{file}' is not available.".format(file=filename),
                          "Setting the following Default Solar Bifrost units: ", units_to_set)
                for unit, value in units_to_set.items():
                    setattr(self, unit, value)

        # I think we shouldn't keep "params" in Bifrost_units anymore. - SE Apr 28 2021
        ## it obfuscates the contents of Bifrost_units, especially when checking self.__dict__.
        ## Here I am going to set self.params to an object which, if someone tries to access a key of it,
        ## will raise an exception with a clear message
        ## of how to fix it.
        class params_are_empty():
            def __init__(self):
                pass
                self.errmsg = ('We are testing to remove self.params from Bifrost_units object. '
                      'If you are seeing this Exception, please consider if your code '
                      'can be written without doing self.params[key] (e.g. obj.uni.params[key]). '
                      'A good alternative is probably to use obj.params[key][obj.snapInd]. '
                      'If you decide you really need to access self.uni.params, then you can '
                      'remove the line of code which deletes self.params, in helita.sim.bifrost.py.'
                      '(the line looks like: "self.params = params_are_empty()"')

            def __contains__(self, i): raise Exception(self.errmsg)
            def __getitem__(self, i):  raise Exception(self.errmsg)
            def keys(self):            raise Exception(self.errmsg)
            def values(self):          raise Exception(self.errmsg)
            def items(self):           raise Exception(self.errmsg)

        self.params = params_are_empty()  # "delete" self.params

        # set cgs units
        self.verbose=verbose
        self.u_u  = self.u_l / self.u_t
        self.u_p  = self.u_r * (self.u_u)**2           # Pressure [dyne/cm2]
        self.u_kr = 1 / (self.u_r * self.u_l)         # Rosseland opacity [cm2/g]
        self.u_ee = self.u_u**2
        self.u_e  = self.u_p      # energy density units are the same as pressure units.
        self.u_te = self.u_e / self.u_t * self.u_l  # Box therm. em. [erg/(s ster cm2)]
        self.u_n  = 3.00e+10                      # Density number n_0 * 1/cm^3
        self.pi   = const.pi
        self.u_b  = self.u_u * np.sqrt(4. * self.pi * self.u_r)

        # self.uni tells how to convert from simu. units to cgs units, for simple vars.
        self.uni={}
        self.uni['l'] = self.u_l
        self.uni['t'] = self.u_t
        self.uni['rho'] = self.u_r
        self.uni['p'] = self.u_r * self.u_u # self.u_p
        self.uni['u'] = self.u_u
        self.uni['e'] = self.u_e
        self.uni['ee'] = self.u_ee
        self.uni['n'] = self.u_n
        self.uni['tg'] = 1.0
        self.uni['b'] = self.u_b

        convertcsgsi(self)
        globalvars(self)
  
        self.u_tg = (self.m_h / self.k_b) * self.u_ee
        self.u_tge = (self.m_e / self.k_b) * self.u_ee

        # set si units
        self.usi_t = self.u_t
        self.usi_l = self.u_l * const.centi                   # 1e-2
        self.usi_r = self.u_r * const.gram / const.centi**3   # 1e-4
        self.usi_u = self.usi_l / self.u_t
        self.usi_p = self.usi_r * (self.usi_u)**2             # Pressure [N/m2]
        self.usi_kr = 1 / (self.usi_r * self.usi_l)           # Rosseland opacity [m2/kg]
        self.usi_ee = self.usi_u**2
        self.usi_e = self.usi_p    # energy density units are the same as pressure units.
        self.usi_te = self.usi_e / self.u_t * self.usi_l      # Box therm. em. [J/(s ster m2)]
        self.ksi_b = aconst.k_B.to_value('J/K')               # Boltzman's cst. [J/K]
        self.msi_h = const.m_n                                # 1.674927471e-27
        self.msi_he = 6.65e-27
        self.msi_p = self.mu * self.msi_h                     # Mass per particle
        self.usi_tg = (self.msi_h / self.ksi_b) * self.usi_ee
        self.msi_e = const.m_e  # 9.1093897e-31
        self.usi_b = self.u_b * 1e-4

        # documentation for units above:
        self.docu('t', 'time')
        self.docu('l', 'length')
        self.docu('r', 'mass density')
        self.docu('u', 'velocity')
        self.docu('p', 'pressure')
        self.docu('kr', 'Rosseland opacity')
        self.docu('ee', 'energy (total; i.e. not energy density)')
        self.docu('e', 'energy density')
        self.docu('te', 'Box therm. em. [J/(s ster m2)]')
        self.docu('b', 'magnetic field')

        # additional units (added for convenience) - started by SE, Apr 26 2021
        self.docu('m', 'mass')
        self.u_m    = self.u_r   * self.u_l**3   # rho = mass / length^3
        self.usi_m  = self.usi_r * self.usi_l**3 # rho = mass / length^3
        self.docu('ef', 'electric field')
        self.u_ef   = self.u_b                   # in cgs: F = q(E + (u/c) x B)
        self.usi_ef = self.usi_b * self.usi_u    # in SI:  F = q(E + u x B)
        self.docu('f', 'force')
        self.u_f    = self.u_p   * self.u_l**2   # pressure = force / area
        self.usi_f  = self.usi_p * self.usi_l**2 # pressure = force / area
        self.docu('q', 'charge')
        self.u_q    = self.u_f   / self.u_ef     # F = q E
        self.usi_q  = self.usi_f / self.usi_ef   # F = q E
        self.docu('nr', 'number density')
        self.u_nr   = self.u_r   / self.u_m      # nr = r / m
        self.usi_nr = self.usi_r / self.usi_m    # nr = r / m
        self.docu('nq', 'charge density')
        self.u_nq   = self.u_q   * self.u_nr
        self.usi_nq = self.usi_q * self.usi_nr
        self.docu('pm', 'momentum density')
        self.u_pm   = self.u_u   * self.u_r      # mom. dens. = mom * nr = u * r
        self.usi_pm = self.usi_u * self.usi_r
        self.docu('hz', 'frequency')
        self.u_hz   = 1./self.u_t
        self.usi_hz = 1./self.usi_t
        self.docu('phz', 'momentum density frequency (see e.g. momentum density exchange terms)')
        self.u_phz  = self.u_pm   * self.u_hz
        self.usi_phz= self.usi_pm * self.usi_hz
        self.docu('i', 'current per unit area')
        self.u_i    = self.u_nq   * self.u_u     # ue = ... + J / (ne qe)
        self.usi_i  = self.usi_nq * self.usi_u

        # additional constants (added for convenience)
        ## masses
        self.simu_amu = self.amu / self.u_m         # 1 amu
        self.simu_m_e = self.m_electron / self.u_m  # 1 electron mass
        ## charge (1 elementary charge)
        self.simu_q_e   = self.q_electron   / self.u_q   # [derived from cgs]
        self.simu_qsi_e = self.qsi_electron / self.usi_q # [derived from si]
        ### note simu_q_e != simu_qsi_e because charge is defined
        ### by different equations, for cgs and si. 

        # update the dict doc_units with the values of units
        self._update_doc_units_with_values()


    def __repr__(self):
        '''show self in a pretty way (i.e. including info about base units)'''
        return "<{} with base_units={}>".format(object.__repr__(self),
                            self.prettyprint_base_units(printout=False))

    def base_units(self):
        '''returns dict of u_l, u_t, u_r, gamma, for self.'''
        return {unit: getattr(self, unit) for unit in self.BASE_UNITS}

    def prettyprint_base_units(self, printout=True):
        '''print (or return, if not printout) prettystring for base_units for self.'''
        fmt = '{:.2e}'  # formatting for keys (except gamma)
        fmtgam = '{}'   # formatting for key gamma
        s = []
        for unit in self.BASE_UNITS:
            val = getattr(self,unit)
            if unit=='gamma':
                valstr = fmtgam.format(val)
            else:
                valstr = fmt.format(val)
            s += [unit+'='+valstr]
        result = 'dict({})'.format(', '.join(s))
        if printout:
            print(result)
        else:
            return(result)

    def _unit_name(self, u):
        '''returns name of unit u. e.g. u_r -> 'r'; usi_hz -> 'hz', 'nq' -> 'nq'.'''
        for prefix in ['u_', 'usi_']:
            if u.startswith(prefix):
                u = u[len(prefix):]
                break
        return u

    def _unit_values(self, u):
        '''return values of u, as a dict'''
        u = self._unit_name(u)
        result = {}
        u_u   = 'u_'+u
        usi_u = 'usi_'+u
        if hasattr(self, u_u):
            result[u_u] = getattr(self, u_u)
        if hasattr(self, usi_u):
            result[usi_u] = getattr(self, usi_u)
        return result

    def prettyprint_unit_values(self, x, printout=True, fmtname='{:<3s}', fmtval='{:.2e}', sep=' ;  '):
        '''pretty string for unit values. print if printout, else return string.'''
        if isinstance(x, str):
            x = self._unit_values(x)
        result = []
        for key, value in x.items():
            u_, p, name = key.partition('_')
            result += [u_ + p + fmtname.format(name) + ' = ' + fmtval.format(value)]
        result = sep.join(result)
        if printout:
            print(result)
        else:
            return result

    def _update_doc_units_with_values(self, sep=' |  ', fmtdoc='{:20s}'):
        '''for u in self.doc_units, update self.doc_units[u] with values of u.'''
        for u, doc in self.doc_units.items():
            valstr = self.prettyprint_unit_values(u, printout=False)
            if len(valstr)>0:
                doc = sep.join([fmtdoc.format(doc), valstr])
                self.doc_units[u] = doc

    def docu(self, u, doc):
        '''documents u by adding u=doc to dict self.doc_units'''
        self.doc_units[u]=doc

    def help(self, u=None, printout=True, fmt='{:3s}: {}'):
        '''prints documentation for u, or all units if u is None.
        printout=False --> return dict, instead of printing.
        '''
        if u is None:
            result = self.doc_units
        else:
            if isinstance(u, str):
                u = [u]
            result = dict()
            for unit in u:
                unit = self._unit_name(unit)
                doc  = self.doc_units.get(unit, "u='{}' is not yet documented!".format(unit))
                result[unit] = doc
        if not printout:
            return result
        else:
            for key, doc in result.items():
                print(fmt.format(key, doc))

    def closest(self, value, sign_sensitive=True, reltol=1e-8):
        '''returns [(attr, value)] for attr(s) in self whose value is closest to value.
        sign_sensitive: True (default) or False
            whether to care about signs (plus or minus) when comparing values
        reltol: number (default 1e-8)
            if multiple attrs are closest, and all match (to within reltol)
            return a list of (attr, value) pairs for all such attrs.
        closeness is determined by doing ratios.
        '''
        result = []
        best = np.inf
        for key, val in self.__dict__.items():
            if val == 0:
                if value != 0:
                    continue
                else:
                    result += [(key, val)]
            try:
                rat = value / val
            except TypeError:
                continue
            if sign_sensitive: 
                rat = abs(rat)
            compare_val = abs(rat - 1)
            if best == 0:  # we handle this separately to prevent division by 0 error.
                if compare_val < reltol:
                    result += [(key, val)]
            elif abs(1 - compare_val / best) < reltol:
                result += [(key, val)]
            elif compare_val < best:
                result = [(key, val)]
                best = compare_val
        return result


#####################
#  CROSS SECTIONS   #
#####################

class Rhoeetab:
    def __init__(self, tabfile=None, fdir='.', big_endian=False, dtype='f4',
                 verbose=True, radtab=False):
        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.eosload = False
        self.radload = False
        self.entload = False
        # read table file and calculate parameters
        if tabfile is None:
            tabfile = '%s/tabparam.in' % (fdir)
        self.param = self.read_tab_file(tabfile)
        try:
            tmp = find_first_match("*idl", fdir)
        except IndexError:
            try:
                tmp = find_first_match("*idl.scr", fdir)
            except IndexError:
                try:
                    tmp = find_first_match("mhd.in", fdir)
                except IndexError:
                    tmp = None
                    print("(WWW) init: no .idl or mhd.in files found." +
                          "Units set to 'standard' Bifrost units.")
        self.uni = Bifrost_units(filename=tmp,fdir=fdir)
        # load table(s)
        self.load_eos_table()
        if radtab:
            self.load_rad_table()

    def read_tab_file(self, tabfile):
        ''' Reads tabparam.in file, populates parameters. '''
        self.params = read_idl_ascii(tabfile, obj=self)
        if self.verbose:
            print(('*** Read parameters from ' + tabfile), whsp*4, end="\r",
                    flush=True)
        p = self.params
        # construct lnrho array
        self.lnrho = np.linspace(
            np.log(p['rhomin']), np.log(p['rhomax']), p['nrhobin'])
        self.dlnrho = self.lnrho[1] - self.lnrho[0]
        # construct ei array
        self.lnei = np.linspace(
            np.log(p['eimin']), np.log(p['eimax']), p['neibin'])
        self.dlnei = self.lnei[1] - self.lnei[0]

    def load_eos_table(self, eostabfile=None):
        ''' Loads EOS table. '''
        if eostabfile is None:
            eostabfile = '%s/%s' % (self.fdir, self.params['eostablefile'])
        nei = self.params['neibin']
        nrho = self.params['nrhobin']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(eostabfile, mode='r', shape=(nei, nrho, 4),
                          dtype=dtype, order='F')
        self.lnpg = table[:, :, 0]
        self.tgt = table[:, :, 1]
        self.lnne = table[:, :, 2]
        self.lnkr = table[:, :, 3]
        self.eosload = True
        if self.verbose:
            print('*** Read EOS table from ' + eostabfile, whsp*4, end="\r",
                flush=True)

    def load_ent_table(self, eostabfile=None):
        '''
        Generates Entropy table from Bifrost EOS table
        '''
        self.enttab = np.zeros((self.params['neibin'], self.params['nrhobin']))
        for irho in range(1, self.params['nrhobin']):
            dinvrho = (1.0 / np.exp(self.lnrho[irho]) - 1.0 / np.exp(
                self.lnrho[irho - 1]))

            self.enttab[0, irho] = self.enttab[0, irho - 1] + 1.0 / \
                self.tgt[0, irho] * np.exp(self.lnpg[0, irho]) * dinvrho

            for iei in range(1, self.params['neibin']):
                dei = np.exp(self.lnei[iei]) - np.exp(self.lnei[iei - 1])
                self.enttab[iei, irho] = self.enttab[iei - 1, irho] + 1.0 / \
                    self.tgt[iei, irho] * dei
        for iei in range(1, self.params['neibin']):
            dei = np.exp(self.lnei[iei]) - np.exp(self.lnei[iei - 1])
            self.enttab[iei, 0] = self.enttab[iei - 1, 0] + \
                1.0 / self.tgt[iei, 0] * dei

        self.enttab = np.log(self.enttab - np.min(self.enttab) - 5.0e8)

    def load_rad_table(self, radtabfile=None):
        ''' Loads rhoei_radtab table. '''
        if radtabfile is None:
            radtabfile = '%s/%s' % (self.fdir,
                                    self.params['rhoeiradtablefile'])
        nei = self.params['neibin']
        nrho = self.params['nrhobin']
        nbins = self.params['nradbins']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(radtabfile, mode='r', shape=(nei, nrho, nbins, 3),
                          dtype=dtype, order='F')
        self.epstab = table[:, :, :, 0]
        self.temtab = table[:, :, :, 1]
        self.opatab = table[:, :, :, 2]
        self.radload = True
        if self.verbose:
            print('*** Read rad table from ' + radtabfile, whsp*4, end="\r",
                flush=True)

    def get_table(self, out='ne', bine=None, order=1):

        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab',
                 'ent': 'enttab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
        if out in ['ent'] and not self.entload:
            if not self.eosload:
                raise ValueError("(EEE) tab_interp: EOS table not loaded!")
            if not self.entload:
                self.load_ent_table()
                self.entload = True
        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")
        quant = getattr(self, qdict[out])
        if out in ['opa eps temp'.split()]:
            if bin is None:
                print("(WWW) tab_interp: radiation bin not set,"
                       " using first bin.")
                bin = 0
            quant = quant[..., bin]
        return quant

    def tab_interp(self, rho, ei, out='ne', bin=None, order=1):
        '''
        Interpolates the EOS/rad table for the required quantity in out.

        Parameters
        ----------
        rho  : ndarray
            Density in g/cm^3
        ei   : ndarray
            Internal energy in erg/g
        bin  : int, optional
            Radiation bin number for bin parameters
        order: int, optional
            Interpolation order (1: linear, 3: cubic)

        Returns
        -------
        output : array
            Same dimensions as input. Depeding on the selected option,
            could be:
            'nel'  : electron density [cm^-3]
            'tg'   : temperature [K]
            'pg'   : gas pressure [dyn/cm^2]
            'kr'   : Rosseland opacity [cm^2/g]
            'eps'  : scattering probability
            'opa'  : opacity
            'temt' : thermal emission
        '''

        qdict = {'ne': 'lnne', 'tg': 'tgt', 'pg': 'lnpg', 'kr': 'lnkr',
                 'eps': 'epstab', 'opa': 'opatab', 'temp': 'temtab',
                 'ent': 'enttab'}
        if out in ['ne tg pg kr'.split()] and not self.eosload:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")
        if out in ['ent'] and not self.entload:
            if not self.eosload:
                raise ValueError("(EEE) tab_interp: EOS table not loaded!")
            if not self.entload:
                self.load_ent_table()
                self.entload = True
        if out in ['opa eps temp'.split()] and not self.radload:
            raise ValueError("(EEE) tab_interp: rad table not loaded!")
        quant = getattr(self, qdict[out])
        if out in ['opa', 'eps', 'temp']:
            if bin is None:
                print("(WWW) tab_interp: radiation bin not set, using first.")
                bin = 0
            quant = quant[:, :, bin]
        # warnings for values outside of table
        rhomin = np.min(rho)
        rhomax = np.max(rho)
        eimin = np.min(ei)
        eimax = np.max(ei)
        if rhomin < self.params['rhomin']:
            print('(WWW) tab_interp: density outside table bounds.' +
                  'Table rho min=%.3e, requested rho min=%.3e' %
                  (self.params['rhomin'], rhomin))
        if rhomax > self.params['rhomax']:
            print('(WWW) tab_interp: density outside table bounds. ' +
                  'Table rho max=%.1f, requested rho max=%.1f' %
                  (self.params['rhomax'], rhomax))
        if eimin < self.params['eimin']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei min=%.2f, requested Ei min=%.2f' %
                  (self.params['eimin'], eimin))
        if eimax > self.params['eimax']:
            print('(WWW) tab_interp: Ei outside of table bounds. ' +
                  'Table Ei max=%.2f, requested Ei max=%.2f' %
                  (self.params['eimax'], eimax))
        # translate to table coordinates
        x = (np.log(ei) - self.lnei[0]) / self.dlnei
        y = (np.log(rho) - self.lnrho[0]) / self.dlnrho
        # interpolate quantity

        result = map_coordinates(
            quant, [x, y], order=order, mode='nearest')
        return (np.exp(result) if out != 'tg' else result)


class Opatab:
    """
    Class to loads opacity table and calculate the photoionization cross
    sections given by anzer & heinzel apj 622: 714-721, 2005, march 20
    they have big typos in their reported c's.... correct values to
    be found in rumph et al 1994 aj, 107: 2108, june 1994

    gaunt factors are set to 0.99 for h and 0.85 for heii,
    which should be good enough for the purposes of this code
    """
    
    def __init__(self, tabname=None, fdir='.',  dtype='f4',
                 verbose=True, lambd=100.0, big_endian=False):
        self.fdir = fdir
        self.dtype = dtype
        self.verbose = verbose
        self.big_endian = big_endian
        self.lambd = lambd
        self.radload = False
        self.teinit = 3.0
        self.dte = 0.05
        self.nte = 100
        self.ch_tabname = "chianti" # alternatives are e.g. 'mazzotta' and others found in Chianti
        # read table file and calculate parameters
        if tabname is None:
            tabname = os.path.join(fdir, 'ionization.dat')
        self.tabname = tabname

    def hopac(self):
        ghi = 0.99
        o0 = 7.91e-18  # cm^2
        ohi = 0
        if self.lambd <= 912:
            ohi = o0 * ghi * (self.lambd / 912.0)**3
        return ohi

    def heiopac(self):
        c = [-2.953607e1, 7.083061e0, 8.678646e-1,
             -1.221932e0, 4.052997e-2, 1.317109e-1,
             -3.265795e-2, 2.500933e-3]
        ohei = 0
        if self.lambd <= 504:
            for i, cf in enumerate(c):
                ohei += cf * (np.log10(self.lambd))**i
            ohei = 10.0**ohei
        return ohei

    def heiiopac(self):
        gheii = 0.85
        o0 = 7.91e-18  # cm^2
        oheii = 0
        if self.lambd <= 228:
            oheii = 16 * o0 * gheii * (self.lambd / 912.0)**3
        return oheii

    def load_opa_table(self, tabname=None):
        ''' Loads ionizationstate table. '''
        if tabname is None:
            tabname = '%s/%s' % (self.fdir, 'ionization.dat')
        eostab = Rhoeetab(fdir=self.fdir)
        nei = eostab.params['neibin']
        nrho = eostab.params['nrhobin']
        dtype = ('>' if self.big_endian else '<') + self.dtype
        table = np.memmap(tabname, mode='r', shape=(nei, nrho, 3), dtype=dtype,
                          order='F')
        self.ionh = table[:, :, 0]
        self.ionhe = table[:, :, 1]
        self.ionhei = table[:, :, 2]
        self.opaload = True
        if self.verbose:
            print('*** Read EOS table from ' + tabname, whsp*4, end="\r",
                flush=True)

    def tg_tab_interp(self, order=1):
        '''
        Interpolates the opa table to same format as tg table.
        '''
        self.load_opa1d_table()
        #rhoeetab = Rhoeetab(fdir=self.fdir)
        #tgTable = rhoeetab.get_table('tg')
        tgTable = np.linspace(self.teinit,self.teinit + self.dte*self.nte,self.nte)
        # translate to table coordinates
        x = ((tgTable) - self.teinit) / self.dte
        # interpolate quantity
        self.ionh = map_coordinates(self.ionh1d, [x], order=order)
        self.ionhe = map_coordinates(self.ionhe1d, [x], order=order)
        self.ionhei = map_coordinates(self.ionhei1d, [x], order=order)

    def h_he_absorb(self, lambd=None):
        '''
        Gets the opacities for a particular wavelength of light.
        If lambd is None, then looks at the current level for wavelength
        '''
        rhe = 0.1
        if lambd is not None:
            self.lambd = lambd
        self.tg_tab_interp()
        arr = (self.ionh) * self.hopac() + rhe * ((1 - self.ionhei - (1-self.ionhei-self.ionhe)) * 
                           self.heiopac() + (self.ionhei) * self.heiiopac())
        #ion_h = self.ionh
        #ion_he = self.ionhe
        #ion_hei = self.ionhei
        #ohi = self.hopac()
        #ohei = self.heiopac()
        #oheii = self.heiiopac()
        #arr = (1 - ion_h) * ohi + rhe * ((1 - ion_he - ion_hei) *
        #                                 ohei + ion_he * oheii)
        arr[arr < 0] = 0
        return arr

    def load_opa1d_table(self, tabname='chianti', tgmin=3.0, tgmax=9.0, ntg=121):
        ''' Loads ionizationstate table. '''
        import ChiantiPy.core as ch        
        if tabname is None:
            tabname = '%s/%s' % (self.fdir, 'ionization1d.dat')
        if tabname == '%s/%s' % (self.fdir, 'ionization1d.dat'):
            dtype = ('>' if self.big_endian else '<') + self.dtype
            table = np.memmap(tabname, mode='r', shape=(41, 3), dtype=dtype,
                          order='F')
            self.ionh1d = table[:, 0]
            self.ionhe1d = table[:, 1]
            self.ionhei1d = table[:, 2]
            self.opaload = True
        else: # Chianti table
            import ChiantiPy.core as ch
            if self.verbose:
                print('*** Reading Chianti table', whsp*4, end="\r",
                          flush=True)
            h = ch.Ioneq.ioneq(1)
            h.load(tabname)
            temp=np.linspace(tgmin,tgmax,ntg)
            h.calculate(10**temp)
            logte = np.log10(h.Temperature)
            self.dte = logte[1]-logte[0]
            self.teinit = logte[0]
            self.nte = np.size(logte)
            self.ionh1d = h.Ioneq[0,:]
            he = ch.Ioneq.ioneq(2)
            he.load(tabname)
            self.ionhe1d = he.Ioneq[0,:]
            self.ionhei1d = he.Ioneq[1,:]
        if self.verbose:
            print('*** Read OPA table from ' + tabname, whsp*4, end="\r",
                flush=True)


class Cross_sect:
    """
    Reads data from Bifrost collisional cross section tables.

    Parameters
    ----------
    cross_tab - string or array of strings
        File names of the ascii cross table files.
    fdir - string, optional
        Directory where simulation files are. Must be a real path.
    verbose - bool, optional
        If True, will print out more diagnostic messages
    dtype - string, optional
        Data type for reading variables. Default is 32 bit float.
    kelvin - bool (default True)
        Whether to load data in Kelvin. (uses eV otherwise)

    Examples
    --------
    >>> a = cross_sect(['h-h-data2.txt','h-h2-data.txt'], fdir="/data/cb24bih")

    """
    def __init__(self, cross_tab=None, fdir=os.curdir, dtype='f4', verbose=None, kelvin=True, obj=None):
        '''
        Loads cross section tables and calculates collision frequencies and
        ambipolar diffusion.

        parameters:
        cross_tab: None or list of strings
            None -> use default cross tab list of strings.
            else -> treat each string as the name of a cross tab file.
        fdir: str (default '.')
            directory of files (prepend to each filename in cross_tab).
        dtype: default 'f4'
            sets self.dtype. aside from that, internally does NOTHING.
        verbose: None (default) or bool.
            controls verbosity. presently, internally does NOTHING.
            if None, use obj.verbose if possible, else use False (default)
        kelvin - bool (default True)
            Whether to load data in Kelvin. (uses eV otherwise)
        obj: None (default) or an object
            None -> does nothing; ignore this parameter.
            else -> improve time-efficiency by saving data from cross_tab files
                    into memory of obj (save in obj._memory_read_cross_txt).
        '''
        self.fdir = fdir
        self.dtype = dtype
        if verbose is None: 
            verbose = False if obj is None else getattr(obj, 'verbose', False)
        self.verbose = verbose
        self.kelvin = kelvin
        self.units = {True: 'K', False: 'eV'}[self.kelvin]
        # save pointer to obj. Use weakref to help ensure we don't create a circular reference.
        self.obj = (lambda: None) if (obj is None) else weakref.ref(obj)  # self.obj() returns obj.
        # read table file and calculate parameters
        if cross_tab is None:
            cross_tab = ['h-h-data2.txt', 'h-h2-data.txt', 'he-he.txt',
                          'e-h.txt', 'e-he.txt', 'h2_molecule_bc.txt',
                          'h2_molecule_pj.txt', 'p-h-elast.txt', 'p-he.txt',
                          'proton-h2-data.txt']
        self._cross_tab_strs = cross_tab
        self.cross_tab_list = {}
        for i, cross_txt in enumerate(cross_tab):
            self.cross_tab_list[i] = os.path.join(fdir, cross_txt)

        # load table(s)
        self.load_cross_tables(firstime=True)

    def load_cross_tables(self,firstime=False):
        '''
        Collects the information in the cross table files.
        '''
        self.cross_tab = dict()
        for itab in range(len(self.cross_tab_list)):
            self.cross_tab[itab] = read_cross_txt(self.cross_tab_list[itab],firstime=firstime,
                                                  obj=self.obj(), kelvin=self.kelvin)
            

    def tab_interp(self, tg, itab=0, out='el', order=1):
        ''' Interpolates the cross section tables in the simulated domain.
            IN:
                tg  : Temperature [K]
                order: interpolation order (1: linear, 3: cubic)
            OUT:
                'se'  : Spin exchange cross section [a.u.]
                'el'  : Integral Elastic cross section [a.u.]
                'mt'  : momentum transfer cross section [a.u.]
                'vi'  : viscosity cross section [a.u.]
        '''

        if out in ['se el vi mt'.split()] and not self.load_cross_tables:
            raise ValueError("(EEE) tab_interp: EOS table not loaded!")

        finterp = interpolate.interp1d(self.cross_tab[itab]['tg'],
                                          self.cross_tab[itab][out])
        tgreg = np.array(tg, copy=True)
        max_temp = np.max(self.cross_tab[itab]['tg'])
        tgreg[tg > max_temp] = max_temp
        min_temp = np.min(self.cross_tab[itab]['tg'])
        tgreg[tg < min_temp] = min_temp

        return finterp(tgreg)

    def __call__(self, tg, *args, **kwargs):
        '''alias for self.tab_interp.'''
        return self.tab_interp(tg, *args, **kwargs)

    def __repr__(self):
        return '{} == {}'.format(object.__repr__(self), str(self))

    def __str__(self):
        return "Cross_sect(cross_tab={}, fdir='{}')".format(self._cross_tab_strs, self.fdir)


def cross_sect_for_obj(obj=None):
    '''return function which returns Cross_sect with self.obj=obj.
    obj: None (default) or an object
        None -> does nothing; ignore this parameter.
        else -> improve time-efficiency by saving data from cross_tab files
                into memory of obj (save in obj._memory_read_cross_txt).
                Also, use fdir=obj.fdir, unless fdir is entered explicitly.
    '''
    @functools.wraps(Cross_sect)
    def _init_cross_sect(cross_tab=None, fdir=None, *args__Cross_sect, **kw__Cross_sect):
        if fdir is None: fdir = getattr(obj, 'fdir', '.')
        return Cross_sect(cross_tab, fdir, *args__Cross_sect, **kw__Cross_sect, obj=obj)
    return _init_cross_sect

## Tools for making cross section table such that colfreq is independent of temperature ##
def constant_colfreq_cross(tg0, Q0, tg=range(1000, 400000, 100), T_to_eV = lambda T: T / 11604):
    '''makes values for constant collision frequency vs temperature cross section table.
    tg0, Q0:
        enforce Q(tg0) = Q0.
    tg: array of values for temperature.
        (recommend: 1000 to 400000, with intervals of 100.)
    T_to_eV: function
        T_to_eV(T) --> value in eV.
    
    colfreq = consts * Q(tg) * sqrt(tg).
        For constant colfreq:
        Q(tg1) sqrt(tg1) = Q(tg0) sqrt(tg0)
        
    returns dict of arrays. keys: 'E' (for energy in eV), 'T' (for temperature), 'Q' (for cross)
    '''
    tg = np.asarray(tg)
    E  = T_to_eV(tg)
    Q  = Q0 * np.sqrt(tg0) / np.sqrt(tg)
    return dict(E=E, T=tg, Q=Q)

def cross_table_str(E, T, Q, comment=''):
    '''make a string for the table for cross sections.
    put comment at top of file if provided.
    '''
    header = ''
    if len(comment) > 0:
        if not comment.startswith(';'):
            comment = ';' + comment
        header += comment + '\n'
    header += '\n'.join(["",
                         "; 1 atomic unit of square distance = 2.80e-17 cm^2",
                         "; 1eV = 11604K",
                         "",
                         "2.80e-17",
                         "",
                         "",
                         ";   E            T          Q11  ",
                         ";  (eV)         (K)        (a.u.)",
                         "",
                         "",
                        ])
    lines = []
    for e, t, q in zip(E, T, Q):
        lines.append('{:.6f}       {:d}       {:.3f}'.format(e, t, q))
    return header + '\n'.join(lines)

def constant_colfreq_cross_table_str(tg0, Q0, **kw):
    '''make a string for a cross section table which will give constant collision frequency (vs tg).'''
    if 'comment' in kw:
        comment = kw.pop('comment')
    else:
        comment = '\n'.join(['; This table provides cross sections such that',
                             '; the collision frequency will be independent of temperature,',
                             '; assuming the functional form colfreq proportional to sqrt(T).',
                            ])
    ccc = constant_colfreq_cross(tg0, Q0, **kw)
    result = cross_table_str(**ccc, comment=comment)
    return result


###########
#  TOOLS  #
###########

def bifrost2d_to_rh15d(snaps, outfile, file_root, meshfile, fdir, writeB=False,
                       sx=slice(None), sz=slice(None), desc=None):
    """
    Reads a Bifrost 2D atmosphere are writes into RH 1.5D format,
    with the time dimension written in the y dimension (to maximise
    parallelism).
    Parameters
    ----------
    snaps : list or 1D array
        Numbers of snapshots to write.
    outfile: str
        Name of RH 1.5D atmosphere file to write.
    file_root: str
        Basename for bifrost files.
    meshfile : str
        Filename of mesh file (including full path).
    writeB : bool, optional
        If True, will also write magnetic field. Default is False.
    sx, sz : slice object, optional
        Slice objects for x and z dimensions, when not all points
        are needed. E.g. use slice(None) for all points, slice(0, 100, 2)
        for every second point up to 100.
    desc : str
        Description.
    """
    from . import rh15d
    data = BifrostData(file_root, snap=snaps[0], meshfile=meshfile, fdir=fdir,
                       ghost_analyse=False)
    nz = len(data.z[sz])
    nx = max(len(data.x[sx]), len(data.y[sx]))
    ny = len(snaps)
    tgas = np.empty((nx, ny, nz), dtype='f')
    vz = np.empty_like(tgas)
    if writeB:
        Bz = np.empty_like(tgas)
        Bx = np.empty_like(tgas)
        By = np.empty_like(tgas)
    else:
        Bz = None
        Bx = None
        By = None
    ne = np.empty_like(tgas, dtype='d')
    if data.hion:
        nH = np.empty((6, ) + tgas.shape, dtype='f')
    else:
        nH = np.empty((1, ) + tgas.shape, dtype='f')

    # unit conversion to SI
    ul = data.params['u_l'] / 1.e2 # to metres
    ur = data.params['u_r']        # to g/cm^3  (for ne_rt_table)
    ut = data.params['u_t']        # to seconds
    uv = ul / ut
    ub = data.params['u_b'] * 1e-4 # to tgasesl
    ue = data.params['u_ee']       # to erg/g

    if not desc:
        desc = 'BIFROST snapshot from 2D sequence %s, sx=%s sy=1 sz=%s.' % \
                    (file_root, repr(sx), repr(sz))
        if data.hion:
            desc = 'hion ' + desc
    x = data.x[sx] * ul
    y = snaps
    z = data.z[sz] * (-ul)

    rdt = data.r.dtype
    # cstagger.init_stagger(data.nz, data.dx, data.dy, data.z.astype(rdt),
    #                      data.zdn.astype(rdt), data.dzidzup.astype(rdt),
    #                      data.dzidzdn.astype(rdt))

    for i, s in enumerate(snaps):
        data.set_snap(s)
        tgas[:, i] = np.squeeze(data.tg)[sx, sz]
        rho = np.squeeze(data.r)[sx, sz]
        vz[:, i] = np.squeeze(do_cstagger(data.pz,'zup',obj=data))[sx, sz] / rho * (-uv)
        if writeB:
            Bx[:, i] = np.squeeze(data.bx)[sx, sz] * ub
            By[:, i] = np.squeeze(-data.by)[sx, sz] * ub
            Bz[:, i] = np.squeeze(-data.bz)[sx, sz] * ub
        ne[:, i] = np.squeeze(data.get_electron_density(sx=sx, sz=sz)).to_value('1/m3')
        nH[:, :, i] = np.squeeze(data.get_hydrogen_pops(sx=sx, sz=sz)).to_value('1/m3')

    rh15d.make_xarray_atmos(outfile, tgas, vz, z, nH=nH, ne=ne, x=x, y=y,
                            append=False, Bx=Bx, By=By, Bz=Bz, desc=desc,
                            snap=snaps[0])

@file_memory.remember_and_recall('_memory_read_idl_ascii')
def read_idl_ascii(filename,firstime=False):
    ''' Reads IDL-formatted (command style) ascii file into dictionary.
    if obj is not None, remember the result and restore it if ever reading the same exact file again.
    '''
    li = -1
    params = {}
    # go through the file, add stuff to dictionary

    with open(filename) as fp:
        for line in fp:
            li += 1
            # ignore empty lines and comments
            line, _, comment = line.partition(';')
            key, _, value    = line.partition('=')
            key   = key.strip().lower()
            value = value.strip()
            if len(key) == 0:
                continue    # this was a blank line.
            elif len(value) == 0:
                if firstime:
                    print('(WWW) read_params: line %i is invalid, skipping' % li)
                continue
            # --- evaluate value --- #
            ## allow '.false.' or '.true.' for bools
            if (value.lower() in ['.false.', '.true.']):
                value = False if value.lower() == '.false.' else True
            else:
                ## safely evaluate any other type of value
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    ## failed to evaluate. Might be string, or might be int with leading 0's.
                    try:
                        value = int(value)
                    except ValueError:
                        ## failed to convert to int; interpret value as string.
                        pass  # leave value as string without evaluating it.

            params[key] = value

    return params

@file_memory.remember_and_recall('_memory_read_cross_txt', kw_mem=['kelvin'])
def read_cross_txt(filename,firstime=False, kelvin=True):
    ''' Reads IDL-formatted (command style) ascii file into dictionary.
    tg will be converted to Kelvin, unless kelvin==False.
    '''
    li = 0
    params = {}
    count = 0
    # go through the file, add stuff to dictionary
    with open(filename) as fp:
        for line in fp:
            # ignore empty lines and comments
            line = line.strip()
            if len(line) < 1:
                li += 1
                continue
            if line[0] == ';':
                li += 1
                continue
            line = line.split(';')[0].split()
            if (len(line) == 1):
                params['crossunits'] = float(line[0].strip())
                li += 1
                continue
            elif not('crossunits' in params.keys()):
                print('(WWW) read_cross: line %i is invalid, missing crossunits, file %s' % (li,filename))

            if (len(line) < 2):
                if (firstime):
                    print('(WWW) read_cross: line %i is invalid, skipping, file %s' % (li,filename))
                li += 1
                continue
            # force lowercase because IDL is case-insensitive
            temp = line[0].strip()
            cross = line[2].strip()

            # instead of the insecure 'exec', find out the datatypes
            if ((temp.upper().find('E') >= 0) or (temp.find('.') >= 0)):
                # float type
                temp = float(temp)
            else:
                # int type
                try:
                    temp = int(temp)
                except Exception:
                    if (firstime):
                        print('(WWW) read_cross: could not find datatype in '
                            'line %i, skipping' % li)
                    li += 1
                    continue
            if not('tg' in params.keys()):
                params['tg'] = temp
            else:
                params['tg'] = np.append(params['tg'], temp)

            if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                # float type
                cross = float(cross)
            else:
                # int type
                try:
                    cross = int(cross)
                except Exception:
                    if (firstime):
                        print('(WWW) read_cross: could not find datatype in '
                            'line %i, skipping' % li)
                    li += 1
                    continue
            if not('el' in params.keys()):
                params['el'] = cross
            else:
                params['el'] = np.append(params['el'], cross)

            if len(line) > 2:
                cross = line[2].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
                                'in line %i, skipping' % li)
                        li += 1
                        continue
                if not('mt' in params.keys()):
                    params['mt'] = cross
                else:
                    params['mt'] = np.append(params['mt'], cross)

            if len(line) > 3:
                cross = line[3].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
                                'in line %i, skipping' % li)
                        li += 1
                        continue
                if not hasattr(params, 'vi'):
                    params['vi'] = cross
                else:
                    params['vi'] = np.append(params['vi'], cross)

            if len(line) > 4:
                cross = line[4].strip()

                if ((cross.upper().find('E') >= 0) or (cross.find('.') >= 0)):
                    # float type
                    cross = float(cross)
                else:
                    # int type
                    try:
                        cross = int(cross)
                    except Exception:
                        if (firstime):
                            print('(WWW) read_cross: could not find datatype'
                                'in line %i, skipping' % li)
                        li += 1
                        continue
                if not hasattr(params, 'se'):
                    params['se'] = cross
                else:
                    params['se'] = np.append(params['se'], cross)
            li += 1

    # convert to kelvin
    if kelvin:
        params['tg'] *= Bifrost_units(verbose=False).ev_to_k

    return params


def calc_grph(abundances, atomic_weights):
    """
    Calculate grams per hydrogen atom, given a mix of abundances
    and respective atomic weights.

    Parameters
    ----------
    abundances : 1D array
        Element abundances relative to hydrogen in log scale,
        where hydrogen is defined as 12.
    atomic_weights : 1D array
        Atomic weights for each element in atomic mass units.

    Returns
    -------
    grph : float
        Grams per hydrogen atom.
    """
    from astropy.constants import u as amu
    linear_abundances = 10.**(abundances - 12.)
    masses = atomic_weights * amu.to_value('g')
    return np.sum(linear_abundances * masses)


def subs2grph(subsfile):
    """
    Extract abundances and atomic masses from subs.dat, and calculate
    the number of grams per hydrogen atom.

    Parameters
    ----------
    subsfile : str
        File name of subs.dat.

    Returns
    -------
    grph : float
        Grams per hydrogen atom.
    """
    f = open(subsfile, 'r')
    nspecies = np.fromfile(f, count=1, sep=' ', dtype='i')[0]
    f.readline()  # second line not important
    ab = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    am = np.fromfile(f, count=nspecies, sep=' ', dtype='f')
    f.close()
    return calc_grph(ab, am)


def find_first_match(name, path,incl_path=False):
    '''
    This will find the first match,
    name : string, e.g., 'patern*'
    incl_root: boolean, if true will add full path, otherwise, the name.
    path : sring, e.g., '.'
    '''
    originalpath=os.getcwd()
    os.chdir(path)
    for file in glob(name):
        if incl_path:
            os.chdir(originalpath)
            return os.path.join(path, file)
        else:
            os.chdir(originalpath)
            return file
    os.chdir(originalpath)
