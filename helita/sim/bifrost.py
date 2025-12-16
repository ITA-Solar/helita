"""
Set of programs to read and interact with output from Bifrost
"""

# import builtin modules
import os
import ast
import time
import weakref
import warnings
import functools
import collections
from glob import glob

# import external public modules
import numpy as np
from scipy import interpolate
from scipy.ndimage import map_coordinates

from . import document_vars, file_memory, load_fromfile_quantities, stagger, tools, units
from .load_arithmetic_quantities import *
# import internal modules
from .load_quantities import *
from .tools import *

# defaults
whsp = '  '
AXES = ('x', 'y', 'z')

# BifrostData class


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
        More options may be defined later. Set stagger_kind='' to see all options.
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
    units_output - string, optional
        unit system for output. default 'simu' for simulation output.
        options are 'simu', 'si', 'cgs'.
        Only affects final values from (external calls to) get_var.
        if not 'simu', self.got_units_name will store units string from latest get_var.
        Do not use at the same time as non-default sel_units.
    squeeze_output - bool, optional. default False
        whether to apply np.squeeze() before returning the result of get_var.
    print_freq - value, default 2.
        number of seconds between print statements during get_varTime.
        == 0 --> print update at every snapshot during get_varTime.
        < 0  --> never print updates during get_varTime.
    printing_stats - bool or dict, optional. default False
        whether to print stats about values of var upon completing a(n external) call to get_var.
        False --> don't print stats.
        True  --> do print stats.
        dict  --> do print stats, passing this dictionary as kwargs.

    Examples
    --------
    This reads snapshot 383 from simulation "cb24bih", whose file
    root is "cb24bih", and is found at directory /data/cb24bih:

        a = BifrostData("cb24bih", snap=383, fdir="/data/cb24bih")

    Scalar variables do not need de-staggering and are available as
    memory map (only loaded to memory when needed), e.g.:

        a.r.shape
        (504, 504, 496)

    Composite variables need to be obtained by get_var():

        vx = a.get_var("ux")
    """

    ## CREATION ##
    def __init__(self, file_root, snap=None, meshfile=None, fdir='.',
                 fast=False, verbose=True, dtype='f4', big_endian=False,
                 cstagop=None, do_stagger=True, ghost_analyse=False, lowbus=False,
                 numThreads=1, params_only=False, sel_units=None,
                 use_relpath=False, stagger_kind=stagger.DEFAULT_STAGGER_KIND,
                 units_output='simu', squeeze_output=False,
                 print_freq=2, printing_stats=False,
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
        self.numThreads = numThreads
        self.fast = fast
        self._fast_skip_flag = False if fast else None  # None-> never skip
        self.squeeze_output = squeeze_output
        self.print_freq = print_freq
        self.printing_stats = printing_stats

        # units. Two options for management. Should only use one at a time; leave the other at default value.
        self.units_output = units_output    # < units.py system of managing units.
        self.sel_units = sel_units          # < other system of managing units.

        setattr(self, document_vars.LOADING_LEVEL, -1)  # tells how deep we are into loading a quantity now.

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
        self.uni = Bifrost_units(filename=tmp, fdir=fdir, parent=self)

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
    help = property(lambda self: self.vardoc)

    shape = property(lambda self: (self.xLength, self.yLength, self.zLength))
    size = property(lambda self: (self.xLength * self.yLength * self.zLength))
    ndim = property(lambda self: 3)

    units_output = units.UNITS_OUTPUT_PROPERTY(internal_name='_units_output')

    @property
    def internal_means(self):
        '''whether to take means of get_var internally, immediately (for simple vars).
        DISABLED by default.

        E.g. if enabled, self.get_var('r') will be single-valued, not an array.
        Note this will have many consequences. E.g. derivatives will all be 0.
        Original intent: analyzing simulations with just a small perturbation around the mean.
        '''
        return getattr(self, '_internal_means', False)

    @internal_means.setter
    def internal_means(self, value):
        self._internal_means = value

    @property
    def printing_stats(self):
        '''whether to print stats about values of var upon completing a(n external) call to get_var.

        Options:
        False (default) --> don't print stats.
        True --> do print stats.
        dict --> call print stats with these kwargs
                e.g. printing_stats=dict(fmt='{:.3e}') --> self.print_stats(fmt='{:.3e}')

        This is useful especially while investigating just the approximate values for each quantity.
        '''
        return getattr(self, '_printing_stats', False)

    @printing_stats.setter
    def printing_stats(self, value):
        self._printing_stats = value

    stagger_kind = stagger.STAGGER_KIND_PROPERTY(internal_name='_stagger_kind')

    @property
    def cstagop(self):  # cstagop is an alias to do_stagger. Maintained for backwards compatibility.
        return self.do_stagger

    @cstagop.setter
    def cstagop(self, value):
        self.do_stagger = value

    @property
    def snap(self):
        '''snapshot number, or list of snapshot numbers.'''
        return getattr(self, '_snap', None)

    @snap.setter
    def snap(self, value):
        self.set_snap(value)

    @property
    def snaps(self):
        '''equivalent to self.snap when it is a list (or other iterable). Otherwise, raise TypeError.'''
        snaps = self.snap
        try:
            iter(snaps)
        except TypeError:
            raise TypeError(f'self.snap (={self.snap}) is not a list!') from None
        return snaps

    @property
    def snapname(self):
        '''alias for self.root_name. Set by 'snapname' in mhd.in / .idl files.'''
        return self.root_name

    kx = property(lambda self: 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.xLength, self.dx)),
                  doc='kx coordinates [simulation units] (fftshifted such that 0 is in the middle).')
    ky = property(lambda self: 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.yLength, self.dy)),
                  doc='ky coordinates [simulation units] (fftshifted such that 0 is in the middle).')
    kz = property(lambda self: 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.zLength, self.dz)),
                  doc='kz coordinates [simulation units] (fftshifted such that 0 is in the middle).')
    # ^ convert k to physical units by dividing by self.uni.usi_l  (or u_l for cgs)

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
        def _param_to_int(param_name, default=0):
            """Convert parameter to int, handling Fortran logicals"""
            val = self.get_param(param_name, default=default)
            if isinstance(val, (str, np.str_)):
                val_str = str(val).strip().upper()
                if val_str in ['T', 'TRUE', '.TRUE.']:
                    return 1
                elif val_str in ['F', 'FALSE', '.FALSE.']:
                    return 0
                else:
                    return int(val_str)
            return int(val)

        if _param_to_int('do_hion') > 0:
            self.hionvars = ['hionne', 'hiontg', 'n1',
                             'n2', 'n3', 'n4', 'n5', 'n6', 'nh2']
            self.hion = True
        if _param_to_int('do_helium') > 0:
            self.heliumvars = ['nhe1', 'nhe2', 'nhe3']
            self.heion = True
        self.compvars = ['ux', 'uy', 'uz', 's', 'ee']
        self.simple_vars = self.snapvars + self.auxvars + self.hionvars + \
            self.heliumvars
        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.simple_vars.remove('ixy1')
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

        self._snap = snap
        if np.shape(self.snap) != ():
            self.snap_str = []
            for num in snap:
                self.snap_str.append(_N_to_snapstr(num))
        else:
            self.snap_str = _N_to_snapstr(snap)
        self.snapInd = 0

        self._read_params(firstime=firstime)
        # Read mesh for all snaps because meshfiles could differ
        self.__read_mesh(self.meshfile, firstime=firstime)
        # variables: lists and initialisation
        self._set_snapvars(firstime=firstime)
        # Do not call if params_only requested
        if (not params_only):
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
            self.paramList.append(read_idl_ascii(file, firstime=firstime, obj=self))

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
                if ((params['boundarychk'] == 1) and (params['isnap'] != 0)):
                    self.nzb = self.nz + 2 * self.nb
                else:
                    self.nzb = self.nz
                if ((params['boundarychky'] == 1) and (params['isnap'] != 0)):
                    self.nyb = self.ny + 2 * self.nb
                else:
                    self.nyb = self.ny
                if ((params['boundarychkx'] == 1) and (params['isnap'] != 0)):
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
                [self.paramList[i][key] for i in range(0, len(self.paramList))
                    if key in self.paramList[i].keys()])
            # the if statement is required in case extra params in
            # self.ParmList[0]
        self.time = self.params['t']
        if self.sel_units == 'cgs':
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
                # Skip assertion check if we auto-corrected ghost zones for Z dimension
                if p == 'z' and hasattr(self, '_autocorrected_ghost') and self._autocorrected_ghost:
                    # Mesh file has original dimension, but we've expanded to include ghost zones
                    if self.verbose and dim != getattr(self, 'n' + p):
                        print(f"(WWW) Mesh file has nz={dim}, but using auto-corrected nz={getattr(self, 'n' + p)} for ghost zones")
                else:
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

            if self.dx < 0.0:
                self.dx = -self.dx / self.nx
            if self.dy < 0.0:
                self.dy = -self.dy / self.ny
            if self.dz < 0.0:
                self.dz = -self.dz / self.nz 

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

        if self.sel_units == 'cgs':
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
            self._fast_skip_flag = True  # swaps flag to True, then runs the rest of the code (this time around).
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
        if not hasattr(self, 'r'):
            raise ValueError("Failed to load any snap variables. Check ghost_analyse setting and file availability.")
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
            self.variables = {}

        if isinstance(iinum, (int, np.integer)):  # we convert to slice, to maintain dimensions of output.
            iinum = slice(iinum, iinum+1)  # E.g. [0,1,2][slice(1,2)] --> [1]; [0,1,2][1] --> 1

        # set self.iix
        setattr(self, iix, iinum)
        if self.verbose:
            # convert iinum to string that wont be super long (in case iinum is a long list)
            try:
                assert len(iinum) > 20
            except (TypeError, AssertionError):
                iinumprint = iinum
            else:
                iinumprint = 'list with length={:4d}, min={:4d}, max={:4d}, x[1]={:2d}'
                iinumprint = iinumprint.format(len(iinum), min(iinum), max(iinum), iinum[1])
            # print info.
            print('(set_domain) {}: {}'.format(iix, iinumprint),
                  whsp*4, end="\r", flush=True)

        # set self.xLength
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
        if internal and self.do_stagger:
            # we slice at the end, only. For now, set all to slice(None)
            slices = (slice(None), slice(None), slice(None))
        else:
            slices = (iix, iiy, iiz)

        any_domain_changes = False
        for x, iix in zip(AXES, slices):
            domain_changed = self.set_domain_iiaxis(iix, x)
            any_domain_changes = any_domain_changes or domain_changed

        # update x, y, z, dx1d, dy1d, dz1d appropriately.
        if any_domain_changes:
            self.__read_mesh(self.meshfile, firstime=False)

    def genvar(self):
        '''
        Dictionary of original variables which will allow to convert to cgs.
        '''
        self.varn = {}
        self.varn['rho'] = 'r'
        self.varn['totr'] = 'r'
        self.varn['tg'] = 'tg'
        self.varn['pg'] = 'p'
        self.varn['ux'] = 'ux'
        self.varn['uy'] = 'uy'
        self.varn['uz'] = 'uz'
        self.varn['e'] = 'e'
        self.varn['bx'] = 'bx'
        self.varn['by'] = 'by'
        self.varn['bz'] = 'bz'

    @document_vars.quant_tracking_top_level
    def _load_quantity(self, var, cgsunits=1.0, **kwargs):
        '''helper function for get_var; actually calls load_quantities for var.'''
        __tracebackhide__ = True  # hide this func from error traceback stack
        # look for var in self.variables
        if cgsunits == 1.0:
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

    def get_var(self, var, snap=None, *args, iix=None, iiy=None, iiz=None, printing_stats=None, **kwargs):
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
                      end="\r", flush=True)
            self.set_snap(snap)
            self.variables = {}

        # set iix, iiy, iiz appropriately
        slices_names_and_vals = (('iix', iix), ('iiy', iiy), ('iiz', iiz))
        original_slice = [iix if iix is not None else getattr(self, slicename, slice(None))
                          for slicename, iix in slices_names_and_vals]
        self.set_domain_iiaxes(iix=iix, iiy=iiy, iiz=iiz, internal=True)

        if var in self.varn.keys():
            var = self.varn[var]

        if (self.sel_units == 'cgs'):
            varu = var.replace('x', '')
            varu = varu.replace('y', '')
            varu = varu.replace('z', '')
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
        val = self._get_var_postprocess(val, var=var, original_slice=original_slice, printing_stats=printing_stats)
        return val

    def _get_var_postprocess(self, val, var='', printing_stats=None, original_slice=[slice(None) for x in ('x', 'y', 'z')]):
        '''does post-processing for get_var.
        This includes:
            - handle "creating documentation" or "var==''" case
            - handle "don't know how to get this var" case
            - reshape result as appropriate (based on iix,iiy,iiz)
            - take mean if self.internal_means (disabled by default).
            - squeeze if self.squeeze_output (disabled by default).
            - convert units as appropriate (based on self.units_output.)
                - default is to keep result in simulation units, doing no conversions.
                - if converting, note that any caching would happen in _load_quantity,
                  outside this function. The cache will always be in simulation units.
            - print stats if printing_stats or ((printing_stats is None) and self.printing_stats).
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
            if hasattr(self, 'vardict'):
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
        if (np.ndim(val) >= self.ndim) and (np.shape(val) != self.shape):
            if all(isinstance(s, slice) for s in (self.iix, self.iiy, self.iiz)):
                val = val[self.iix, self.iiy, self.iiz]  # we can index all together
            else:  # we need to index separately due to numpy multidimensional index array rules.
                val = val[self.iix, :, :]
                val = val[:, self.iiy, :]
                val = val[:, :, self.iiz]

        # take mean if self.internal_means (disabled by default)
        if self.internal_means:
            val = val.mean()

        # handle post-processing steps which we only do for top-level calls to get_var:
        if not self._getting_internal_var():

            # squeeze if self.squeeze_output (disabled by default)
            if self.squeeze_output and (np.ndim(val) > 0):
                val = val.squeeze()

            # convert units if we are using units_output != 'simu'.
            if self.units_output != 'simu':
                units_f, units_name = self.get_units(mode=self.units_output, _force_from_simu=True)
                self.got_units_name = units_name   # << this line is just for reference. Not used internally.
                val = val * units_f   # can't do *= in case val is a read-only memmap.

            # print stats if self.printing_stats
            self.print_stats(val, printing_stats=printing_stats)

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
        if varname[-1] in ['x', 'y', 'z']:
            varname = varname+'c'
            if varname[-2] in ['y', 'z']:
                sign = -1.0

        var = self.get_var(varname, snap=snap, *args, **kwargs)
        var = sign * var

        var = var[..., ::-1].copy()

        return var

    def trans2commaxes(self):
        if self.transunits == False:
            self.transunits = True
            if self.sel_units == 'cgs':
                cte = 1.0
            else:
                cte = self.uni.u_l  # not sure if this works, u_l seems to be 1.e8
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
                cte = 1.0
            else:
                cte = self.uni.u_l
            self.x = self.x/cte
            self.dx = self.dx/cte
            self.y = self.y/cte
            self.dy = self.dy/cte
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
            document_vars.vars_documenter(self, 'SIMPLE_VARS', None, _simple_vars_msg)
            # TODO: << add documentation for bifrost simple vars, here.
            return None

        if var not in self.simple_vars:
            return None

        if self.verbose:
            print('(get_var): reading simple ', var, whsp*5,  # TODO: show np.shape(val) info somehow?
                  end="\r", flush=True)

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
            # Check if we have auto-corrected dimensions
            if hasattr(self, '_autocorrected_ghost') and self._autocorrected_ghost:
                nzb_to_use = self._autocorrected_nzb
            else:
                nzb_to_use = self.nzb
            # Ghost zones only in Z dimension: (nx, ny, nzb)
            # Use np.int64 to avoid integer overflow for large offsets
            offset = np.int64(self.nx) * np.int64(self.ny) * np.int64(nzb_to_use) * np.int64(idx) * np.int64(dsize)
            ss = (self.nx, self.ny, nzb_to_use)
        else:
            offset = np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nz) * np.int64(idx) * np.int64(dsize)
            ss = (self.nx, self.ny, self.nz)
            
            # Check if user set ghost_analyse=False but file actually contains ghost zones
            if var in self.snapvars:  # Only check for snap variables
                import os
                nvars = len(self.snapvars)
                expected_size_noghost = np.int64(nvars) * np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nz) * np.int64(dsize)
                actual_size = os.path.getsize(filename)
                
                # Calculate what the file actually contains
                total_values = actual_size // dsize
                values_per_var = total_values // nvars
                actual_nz = values_per_var // (self.nx * self.ny)
                
                if actual_size != expected_size_noghost:
                    # Check if it matches ghost zone pattern (actual_nz = nz + 2*nb)
                    if actual_nz == (self.nz + 2 * self.nb):
                        raise ValueError(f"File '{filename}' contains ghost zone data but ghost_analyse=False. "
                                       f"File size: {actual_size} bytes contains {actual_nz} Z-points "
                                       f"(expected {self.nz} without ghost zones, got {actual_nz} = {self.nz}+2×{self.nb}). "
                                       f"Set ghost_analyse=True to read this data.")
                    else:
                        # File has unexpected size - provide debugging info
                        expected_size_autodetect = np.int64(nvars) * np.int64(self.nx) * np.int64(self.ny) * np.int64(actual_nz) * np.int64(dsize)
                        if actual_size == expected_size_autodetect:
                            raise ValueError(f"File '{filename}' contains {actual_nz} Z-points but expected {self.nz}. "
                                           f"File appears to contain ghost zone data. Set ghost_analyse=True to read this data.")
                        else:
                            raise ValueError(f"File '{filename}' has unexpected size: {actual_size} bytes. "
                                           f"Expected {expected_size_noghost} for {self.nx}×{self.ny}×{self.nz} dimensions, "
                                           f"but file contains {actual_nz} Z-points.")

        # Check if file size matches expected size for ghost zones
        if self.ghost_analyse and var in self.snapvars:
            import os
            # Ghost zones only exist in Z dimension: (mx, my, mz+2*mb)
            # Use np.int64 to avoid integer overflow for large files
            expected_size_ghost = np.int64(len(self.snapvars)) * np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nzb) * np.int64(dsize)
            # Non-ghost size: (mx, my, mz)  
            expected_size_noghost = np.int64(len(self.snapvars)) * np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nzb - 2 * self.nb) * np.int64(dsize)
            actual_size = os.path.getsize(filename)
            
            if actual_size == expected_size_noghost:
                raise ValueError(f"ghost_analyse=True but snap file '{filename}' contains no ghost zone data. "
                               f"File size is {actual_size} bytes (expected {expected_size_ghost} for ghost zones). "
                               f"Set ghost_analyse=False to read this data.")
            elif actual_size != expected_size_ghost:
                # Calculate what the file actually contains
                nvars = len(self.snapvars)
                total_values = actual_size // dsize
                values_per_var = total_values // nvars
                actual_nz = values_per_var // (self.nx * self.ny)
                
                # Check if file contains ghost zones by auto-detecting from file size
                expected_nz_with_ghost = actual_nz
                expected_size_autodetect = np.int64(nvars) * np.int64(self.nx) * np.int64(self.ny) * np.int64(actual_nz) * np.int64(dsize)
                
                if actual_size == expected_size_autodetect:
                    # Store the auto-corrected value and mark that auto-correction happened
                    if not hasattr(self, '_autocorrected_ghost') or not self._autocorrected_ghost:
                        # First time auto-correction - print message and expand mesh
                        if self.verbose:
                            print(f"(WWW) Auto-detected ghost zones: correcting nzb from {self.nzb} to {actual_nz}")
                        self._autocorrected_nzb = actual_nz
                        self._autocorrected_ghost = True
                        
                        # Expand mesh to include ghost zones (same logic as in __read_mesh)
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
                        # Update dimensions to match expanded mesh
                        self.nz = actual_nz
                        if self.verbose:
                            print(f"(WWW) Expanded mesh to include ghost zones: z.shape={self.z.shape}")
                    
                    # Use the corrected dimensions for reading this variable
                    offset = np.int64(self.nx) * np.int64(self.ny) * np.int64(actual_nz) * np.int64(idx) * np.int64(dsize)
                    ss = (self.nx, self.ny, actual_nz)
                else:
                    # Still doesn't match - report the issue
                    var_size_ghost = np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nzb) * np.int64(dsize)
                    var_size_noghost = np.int64(self.nx) * np.int64(self.ny) * np.int64(self.nzb - 2 * self.nb) * np.int64(dsize)
                    nz_orig = self.nzb - 2 * self.nb
                    raise ValueError(f"Snap file '{filename}' has unexpected size: {actual_size} bytes.\n"
                                   f"Current dimensions: nx={self.nx}, ny={self.ny}, nz={self.nz} (after ghost_analyse setting)\n"
                                   f"Original dimensions: nx={self.nx}, ny={self.ny}, nz={nz_orig}, with ghost: nzb={self.nzb}, nb={self.nb}\n"
                                   f"File actually contains: nx={self.nx}, ny={self.ny}, nz={actual_nz} (calculated from file size)\n"
                                   f"Data type: {self.dtype} ({dsize} bytes per value)\n"
                                   f"Variables expected: {nvars} ({self.snapvars})\n"
                                   f"Expected for ghost zones: {expected_size_ghost} bytes ({nvars} vars × {var_size_ghost} bytes/var)\n"
                                   f"Expected for no ghost zones: {expected_size_noghost} bytes ({nvars} vars × {var_size_noghost} bytes/var)\n"
                                   f"BUG: nzb should be {actual_nz} but is {self.nzb}. Check boundary parameter calculation.")

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
            nh = np.empty(shape, dtype=np.float32)
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
                abund = np.array(tabparams['abund'].split()).astype('f')
                aweight = np.array(tabparams['aweight'].split()).astype('f')
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
        self.params['u_r'][self.snapInd]
        ut = self.params['u_t'][self.snapInd]         # to seconds
        uv = ul / ut
        ub = self.params['u_b'][self.snapInd] * 1e-4  # to Tesla
        ue = self.params['u_ee'][self.snapInd]        # to erg/g
        if verbose:
            pbar.set_description("Slicing and unit conversion")
        temp = self.tg[sx, sy, sz]
        rho = self.r[sx, sy, sz]

        if self.do_mhd:
            Bx = do_stagger(self.bx, 'xup', obj=self)[sx, sy, sz]
            By = do_stagger(self.by, 'yup', obj=self)[sx, sy, sz]
            Bz = do_stagger(self.bz, 'zup', obj=self)[sx, sy, sz]
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

        vz = do_stagger(self.pz, 'zup', obj=self)[sx, sy, sz] / rho
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
                      sx=slice(None), sy=slice(None), sz=slice(None),
                      write_magnetic=False):
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
        write_magnetic - bool, optional
            Whether to write a magnetic field file. Default is False.
        Returns
        -------
        None.
        """
        from .multi3d import Multi3dAtmos
        from .multi3d import Multi3dMagnetic
        # unit conversion to cgs and km/s
        ul = self.params['u_l'][self.snapInd]   # to cm
        ur = self.params['u_r'][self.snapInd]   # to g/cm^3  (for ne_rt_table)
        ut = self.params['u_t'][self.snapInd]   # to seconds
        uv = ul / ut / 1e5        # to km/s
        ub = self.params['u_b'][self.snapInd] # to G
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
        vx = do_stagger(self.px, 'xup', obj=self)[sx, sy, sz] / rho
        vx *= uv
        vy = do_stagger(self.py, 'yup', obj=self)[sx, sy, sz] / rho
        vy *= -uv
        vz = do_stagger(self.pz, 'zup', obj=self)[sx, sy, sz] / rho
        vz *= -uv
        rho = rho * ur  # to cgs
        x = self.x[sx] * ul
        y = self.y[sy] * (-ul)
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
        if write_magnetic:
            Bx = do_stagger(self.bx, 'xup', obj=self)[sx, sy, sz]
            By = do_stagger(self.by, 'yup', obj=self)[sx, sy, sz]
            Bz = do_stagger(self.bz, 'zup', obj=self)[sx, sy, sz]
            # Change sign of Bz (because of height scale) and By
            # (to make right-handed system)
            Bx = Bx * ub
            By = -By * ub # [M.Sz] Should By be inverted too  (Bz points downwards)? [TEM] Yes, to preserve right-handedness the y-axis should change sign too
            Bz = -Bz * ub
            fout3 = Multi3dMagnetic('magnetic.dat', nx, ny, nz, mode='w+')
            fout3.Bx[:] = Bx
            fout3.By[:] = By
            fout3.Bz[:] = Bz

    ## VALUES OVER TIME, and TIME DERIVATIVES ##

    def get_varTime(self, var, snap=None, iix=None, iiy=None, iiz=None,
                    print_freq=None, printing_stats=None,
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
        printing_stats - None, bool, or dict
            whether to print stats of result (via self.print_stats).
            None  --> use value of self.printing_stats.
            False --> don't print stats. (This is the default value for self.printing_stats.)
            True  --> do print stats.
            dict  --> do print stats, passing this dictionary as kwargs.

        additional *args and **kwargs are passed to get_var.
        """
        # set print_freq
        if print_freq is None:
            print_freq = getattr(self, 'print_freq', 2)  # default 2
        else:
            setattr(self, 'print_freq', print_freq)

        # set snap
        if snap is None:
            snap = kw__get_var.pop('snaps', None)  # look for 'snaps' kwarg
            if snap is None:
                snap = self.snap
        snap = np.array(snap, copy=False)
        if len(snap.shape) == 0:
            raise ValueError('Expected snap to be list (in get_varTime) but got snap={}'.format(snap))
        if not np.array_equal(snap, self.snap):
            self.set_snap(snap)
            self.variables = {}

        # set iix,iiy,iiz.
        self.set_domain_iiaxes(iix=iix, iiy=iiy, iiz=iiz, internal=False)
        snapLen = np.size(self.snap)

        # bookkeeping - maintain self.snap; handle self.recoverData; don't print stats in the middle; track timing.
        remembersnaps = self.snap                   # remember self.snap (restore later if crash)
        if hasattr(self, 'recoverData'):
            delattr(self, 'recoverData')            # smash any existing saved data
        kw__get_var.update(printing_stats=False)   # never print_stats in the middle of get_varTime.
        timestart = now = time.time()           # track timing, so we can make updates.
        printed_update = False

        def _print_clearline(N=100):        # clear N chars, and move cursor to start of line.
            print('\r' + ' '*N + '\r', end='')  # troubleshooting: ensure end='' in other prints.

        try:
            firstit = True
            for it in range(0, snapLen):
                self.snapInd = it
                # print update if it is time to print update
                if (print_freq >= 0) and (time.time() - now > print_freq):
                    _print_clearline()
                    print('Getting {:^10s}; at snap={:2d} (snap_it={:2d} out of {:2d}).'.format(
                        var,     snap[it],         it,    snapLen), end='')
                    now = time.time()
                    print(' Total time elapsed = {:.1f} s'.format(now - timestart), end='')
                    printed_update = True

                # actually get the values here:
                if firstit:
                    # get value at first snap
                    val0 = self.get_var(var, snap=snap[it], *args__get_var, **kw__get_var)
                    # figure out dimensions and initialize the output array.
                    value = np.empty_like(val0, shape=[*np.shape(val0), snapLen])
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

        self.print_stats(value, printing_stats=printing_stats)
        return value

    @tools.maintain_attrs('snap')
    def ddt(self, var, snap=None, *args__get_var, method='centered', printing_stats=None, **kw__get_var):
        '''time derivative of var, at current snapshot.
        Units are determined by self.units_output (default: [simulation units]).

        snap: None or value
            if provided (not None), first self.set_snap(snap).
        method: ('forward', 'backward', 'centered')
            tells how to take the time derivative.
            forward  --> (var[snap+1] - var[snap]) / (t[snap+1] - t[snap])
            backward --> (var[snap] - var[snap-1]) / (t[snap] - t[snap-1])
            centered --> (var[snap+1] - var[snap-1]) / (t[snap+1] - t[snap-1])
        '''
        if snap is not None:
            self.set_snap(snap)
        method = method.lower()
        if method == 'forward':
            snaps = [self.get_snap_here(), self.get_snap_next()]
        elif method == 'backward':
            snaps = [self.get_snap_prev(), self.get_snap_here()]
        elif method == 'centered':
            snaps = [self.get_snap_prev(), self.get_snap_next()]
        else:
            raise ValueError(f'Unrecognized method in ddt: {repr(method)}')
        kw__get_var.update(printing_stats=False)   # never print_stats in the middle of ddt.
        self.set_snap(snaps[0])
        value0 = self(var, *args__get_var, **kw__get_var)
        time0 = self.get_coord('t')[0]
        self.set_snap(snaps[1])
        value1 = self(var, *args__get_var, **kw__get_var)
        time1 = self.get_coord('t')[0]
        result = (value1 - value0) / (time1 - time0)
        self.print_stats(result, printing_stats=printing_stats)   # print stats iff self.printing_stats.
        return result

    def get_dvarTime(self, var, method='numpy', kw__gradient=dict(), printing_stats=None, **kw__get_varTime):
        '''time derivative of var, across time.
        Units are determined by self.units_output (default: [simulation units]).

        method: ('numpy', 'simple', 'centered')
            tells how to take the time derivative:
                numpy    --> np.gradient(v, axis=-1) / np.gradient(tt, axis=-1)
                            result will be shape (..., M),
                            corresponding to times (tt).
                simple   --> (v[..., 1:] - v[..., :-1]) / (tt[..., 1:] - tt[..., :-1])
                            result will be shape (..., M-1),
                            corresponding to times (tt[..., 1:] + tt[..., :-1]) / 2.
                centered --> (v[..., 2:] - v[..., :-2]) / (tt[..., 2:] - tt[..., :-2])
                            result will be shape (..., M-2),
                            corresponding to times (tt[..., 1:-1])
            where, above, v = self.get_varTime(var);
            tt=self.get_coord('t'), with dims expanded (np.expand_dims) appropriately.
        kw__gradient: dict
            if method=='numpy', kw__gradient are passed to np.gradient.
            (do not include 'axis' in kw__gradient.)
        additional **kwargs are passed to self.get_varTime.

        returns: array of shape (..., M),
        where M=len(self.snap) if method=='numpy', or len(self.snap)-1 if method=='simple'.
        '''
        KNOWN_METHODS = ('numpy', 'simple', 'centered')
        method = method.lower()
        assert method in KNOWN_METHODS, f"Unrecognized method for get_dvarTime: {repr(method)}"
        v = self.get_varTime(var, printing_stats=False, **kw__get_varTime)
        tt = self.get_coord('t')
        tt = np.expand_dims(tt, axis=tuple(range(0, v.ndim - tt.ndim)))  # e.g. shape (1,1,1,len(self.snaps))
        method = method.lower()
        if method == 'numpy':
            result = np.gradient(v, **kw__gradient, axis=-1) / np.gradient(tt, axis=-1)
        elif method == 'simple':
            result = (v[..., 1:] - v[..., :-1]) / (tt[..., 1:] - tt[..., :-1])
        else:  # method == 'centered'
            result = (v[..., 2:] - v[..., :-2]) / (tt[..., 2:] - tt[..., :-2])
        self.print_stats(result, printing_stats=printing_stats)
        return result

    def get_atime(self):
        '''get average time, corresponding to times of derivative from get_dvarTime(..., method='simple').'''
        tt = self.get_coord('t')
        return (tt[..., 1:] + tt[..., :-1]) / 2

    ## MISC. CONVENIENCE METHODS ##
    def print_stats(self, value, *args, printing_stats=True, **kwargs):
        '''print stats of value, via tools.print_stats.
        printing_stats: None, bool, or dict.
            None  --> use value of self.printing_stats.
            False --> don't print stats.
            True  --> do print stats.
            dict  --> do print stats, passing this dictionary as kwargs.
        '''
        if printing_stats is None:
            printing_stats = self.printing_stats
        if printing_stats:
            kw__print_stats = printing_stats if isinstance(printing_stats, dict) else dict()
            tools.print_stats(value, **kw__print_stats)

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
            units = self.get_units(mode=mode)
            return (np.array(varxyz) * units.factor, units.name)
        assert False  # if we made it to this line it means something is wrong with the code here.

    def _varV_formatter(self, vmode, fmt_values='{: .2e}', fmt_units='{:^7s}'):
        '''returns a format function for pretty formatting of the result of get_varV.'''
        VALIDMODES = ('modhat', 'modangle', 'xyz')
        vmode = vmode.lower()
        assert vmode in VALIDMODES, 'vmode {} invalid! Expected vmode in {}.'.format(repr(vmode), VALIDMODES)
        if vmode == 'modhat':
            def fmt(x):
                mag = fmt_values.format(x[0][0])
                units = fmt_units.format(x[0][1])
                hat = ('[ '+fmt_values+', '+fmt_values+', '+fmt_values+' ]').format(*x[1])
                return 'magnitude = {} [{}];  unit vector = {}'.format(mag, units, hat)
        elif vmode == 'modangle':
            def fmt(x):
                mag = fmt_values.format(x[0][0])
                units = fmt_units.format(x[0][1])
                angle = fmt_values.format(x[1][0])
                angle_units = fmt_units.format(x[1][1])
                return 'magnitude = {} [{}];  angle (from +x) = {} [{}]'.format(mag, units, angle, angle_units)
        elif vmode == 'xyz':
            def fmt(x):
                vec = ('[ '+fmt_values+', '+fmt_values+', '+fmt_values+' ]').format(*x[0])
                units = fmt_units.format(x[1])
                return '{}  [{}]'.format(vec, units)
        fmt.__doc__ = 'formats result of get_varV. I was made by helita.sim.bifrost._varV_formatter.'
        return fmt

    def zero(self, **kw__np_zeros):
        '''return np.zeros() with shape equal to shape of result of get_var()'''
        return np.zeros(self.shape, **kw__np_zeros)

    def get_snap_here(self):
        '''return self.snap, or self.snap[0] if self.snap is a list.
        This is the snap which get_var() will work at, for the given self.snap value.
        '''
        try:
            iter(self.snap)
        except TypeError:
            return self.snap
        else:
            return self.snap[0]

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
            if len(dx1d) == 1:
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
        kw_x = {x: getattr(self,    x) * scaling for x in AXES}
        kw_dx = {'d'+x: getattr(self, 'd'+x+'1d') / scaling for x in AXES}
        kw_nx = {'n'+x: getattr(self, x+'Length') for x in AXES}
        kw_mesh = {**kw_x, **kw_nx, **kw_dx}
        Create_new_br_files().write_mesh(**kw_mesh, meshfile=meshfile)
        return os.path.abspath(meshfile)

    write_meshfile = write_mesh_file  # alias

    def get_coords(self, units='si', axes=None, mode=None):
        '''returns dict of coords, with keys ['x', 'y', 'z', 't'].
        units:
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
                c['z'], c['x'], c['y'] == self.get_coords(axes='zxy')
        mode: alias for units. (for backwards compatibility)
            if entered, ignore units kwarg; use mode instead.
        '''
        if mode is None:
            mode = units
        mode = mode.lower()
        VALIDMODES = ('si', 'cgs', 'simu')
        assert mode in VALIDMODES, "Invalid mode ({})! Expected one of {}".format(repr(mode), VALIDMODES)
        if mode == 'si':
            u_l = self.uni.usi_l
            u_t = self.uni.usi_t
        elif mode == 'cgs':
            u_l = self.uni.u_l
            u_t = self.uni.u_t
        else:  # mode == 'simu'
            u_l = 1
            u_t = 1
        x, y, z = (self_x * u_l for self_x in (self.x, self.y, self.z))
        t = self.time * u_t
        result = dict(x=x, y=y, z=z, t=t)
        if axes is not None:
            AXES_LOOKUP = {'x': 'x', 0: 'x', 'y': 'y', 1: 'y', 'z': 'z', 2: 'z', 't': 't', 3: 't'}
            result = tuple(result[AXES_LOOKUP[axis]] for axis in axes)
        return result

    def get_coord(self, axis, units=None):
        '''gets coord for the given axis, in the given unit system.
        axis: string ('x', 'y', 'z', 't') or int (0, 1, 2, 3)
        units: None (default) or string ('si', 'cgs', 'simu')   ('simu' for 'simulation units')
            None --> use self.units_output.

        The result will be an array (possibly with only 1 element).
        '''
        if units is None:
            units = self.units_output
        return self.get_coords(units=units, axes=[axis])[0]

    def coord_grid(self, axes='xyz', units='si', sparse=True, **kw__meshgrid):
        '''returns grid of coords for self along the given axes.

        axes: list of strings ('x', 'y', 'z', 't'), or ints (0, 1, 2, 3)
        units: string ('si', 'cgs', 'simu')   ('simu' for 'simulation units')
        sparse: bool. Example:
            coord_grid('xyz', sparse=True)[0].shape == (Nx, 1, 1)
            coord_grid('xyz', sparse=False)[0].shape == (Nx, Ny, Nz)

        This function basically just calls np.meshgrid, using coords from self.get_coords.

        Example:
            xx, yy, zz = self.coord_grid('xyz', sparse=True)
            # yy.shape == (1, self.yLength, 1)
            # yy[0, i, 0] == self.get_coord('x')[i]

            xx, tt = self.coord_grid('xt', sparse=False)
            # xx.shape == (self.xLength, len(self.time))
            # tt.shape == (self.XLength, len(self.time))
        '''
        coords = self.get_coords(axes=axes, units=units)
        indexing = kw__meshgrid.pop('indexing', 'ij')  # default 'ij' indexing
        return np.meshgrid(*coords, sparse=sparse, indexing=indexing, **kw__meshgrid)

    def get_kcoords(self, units='si', axes=None):
        '''returns dict of k-space coords, with keys ['kx', 'ky', 'kz']
        coords units are based on mode.
            'si' (default) -> [ 1 / m]
            'cgs'     ->      [ 1 / cm]
            'simu'    ->      [ 1 / simulation unit length]
        if axes is not None:
            instead of returning a dict, return coords for the axes provided, in the order listed.
            axes can be provided in either of these formats:
                strings: 'x', 'y', 'z'
                ints:     0 ,  1 ,  2
        '''
        # units
        units = units.lower()
        assert units in ('si', 'cgs', 'simu')
        u_l = {'si': self.uni.usi_l, 'cgs': self.uni.u_l, 'simu': 1}[units]
        # axes bookkeeping
        if axes is None:
            axes = AXES
            return_dict = True
        else:
            AXES_LOOKUP = {'x': 'x', 0: 'x', 'y': 'y', 1: 'y', 'z': 'z', 2: 'z'}
            axes = [AXES_LOOKUP[x] for x in axes]
            return_dict = False
        result = {f'k{x}': getattr(self, f'k{x}') for x in axes}   # get k
        result = {key: val / u_l for key, val in result.items()}  # convert units
        # return
        if return_dict:
            return result
        else:
            return [result[f'k{x}'] for x in axes]

    def get_extent(self, axes, units='si'):
        '''use plt.imshow(extent=get_extent()) to make a 2D plot in x,y,z,t coords.
        (Be careful if coords are unevenly spaced; imshow assumes even spacing.)
        units: 'si' (default), 'cgs', or 'simu'
            unit system for result
        axes: None, strings (e.g. ('x', 'z') or 'xz'), or list of indices (e.g. (0, 2))
            which axes to get the extent for.
            first axis will be the plot's x axis; second will be the plot's y axis.
            E.g. axes='yz' means 'y' as the horizontal axis, 'z' as the vertical axis.
        '''
        assert len(axes) == 2, f"require exactly 2 axes for get_extent, but got {len(axes)}"
        x, y = self.get_coords(units=units, axes=axes)
        return tools.extent(x, y)

    def get_kextent(self, axes=None, units='si'):
        '''use plt.imshow(extent=get_kextent()) to make a plot in k-space.
        units: 'si' (default), 'cgs', or 'simu'
            unit system for result
        axes: None, strings (e.g. ('x', 'z') or 'xz'), or list of indices (e.g. (0, 2))
            which axes to get the extent for.
            if None, use obj._latest_fft_axes (see helita.sim.load_arithmetic_quantities.get_fft_quant)
            first axis will be the plot's x axis; second will be the plot's y axis.
            E.g. axes='yz' means 'y' as the horizontal axis, 'z' as the vertical axis.
        '''
        if axes is None:
            try:
                axes = self._latest_fft_axes
            except AttributeError:
                errmsg = "self._latest_fft_axes not set; maybe you meant to get a quant from " +\
                         "FFT_QUANT first? Use self.vardoc('FFT_QUANT') to see list of options."
                raise AttributeError(errmsg) from None
        assert len(axes) == 2, f"require exactly 2 axes for get_kextent, but got {len(axes)}"
        kx, ky = self.get_kcoords(units=units, axes=axes)
        return tools.extent(kx, ky)

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
    snaps = get_snaps(snapname=snapname, dd=dd)
    return SnapStuff(snapname=snapname, snaps=snaps)


snapstuff = get_snapstuff   # alias


def get_snapname(dd=None):
    '''gets snapname by reading it from local mhd.in, or dd.snapname if dd is provided.'''
    if dd is None:
        mhdin_ascii = read_idl_ascii('mhd.in')
        return mhdin_ascii['snapname']
    else:
        return dd.snapname


snapname = get_snapname   # alias


def get_snaps(dd=None, snapname=None):
    '''list available snap numbers.
    Does look for: snapname_*.idl, snapname.idl (i.e. snap 0)
    Doesn't look for: .pan, .scr, .aux files.
    snapname: None (default) or str
        snapname parameter from mhd.in. If None, get snapname.
    if dd is not None, look in dd.fdir.
    '''
    with tools.EnterDirectory(_get_dd_fdir(dd)):
        snapname = snapname if snapname is not None else get_snapname()
        snaps = [_snap_to_N(f, snapname) for f in os.listdir()]
        snaps = [s for s in snaps if s is not None]
        snaps = sorted(snaps)
        return snaps


snaps = get_snaps   # alias
available_snaps = get_snaps   # alias
list_snaps = get_snaps   # alias


def snaps_info(dd=None, snapname=None, snaps=None):
    '''returns string with length of snaps, as well as min and max.
    if snaps is None, lookup all available snaps.
    '''
    if snaps is None:
        snaps = get_snaps(dd=dd, snapname=snapname)
    return 'There are {} snaps, from {} (min) to {} (max)'.format(len(snaps), min(snaps), max(snaps))


def get_snap_shifted(dd=None, shift=0, snapname=None, snap=None):
    '''returns snap's number for snap at index (current_snap_index + shift).
    Must provide dd or snap, so we can figure out current_snap_index.
    '''
    snaps = list(get_snaps(dd=dd, snapname=snapname))
    snap_here = snap if snap is not None else dd.get_snap_here()
    i_here = snaps.index(snap_here)
    i_result = i_here + shift
    if i_result < 0:
        if shift == -1:
            raise ValueError(f'No snap found prior to snap={snap_here}')
        else:
            raise ValueError(f'No snap found {abs(shift)} prior to snap={snap_here}')
    elif i_result >= len(snaps):
        if shift == 1:
            raise ValueError(f'No snap found after snap={snap_here}')
        else:
            raise ValueError(f'No snap found {abs(shift)} after snap={snap_here}')
    else:
        return snaps[i_result]


def get_snap_prev(dd=None, snapname=None, snap=None):
    '''returns previous available snap's number. TODO: implement more efficiently.
    Must provide dd or snap, so we can figure out the snap here, first.
    '''
    return get_snap_shifted(dd=dd, shift=-1, snapname=snapname, snap=snap)


def get_snap_next(dd=None, snapname=None, snap=None):
    '''returns next available snap's number. TODO: implement more efficiently.
    Must provide dd or snap, so we can figure out the snap here, first.
    '''
    return get_snap_shifted(dd=dd, shift=+1, snapname=snapname, snap=snap)


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
    if namext[1] != ext:
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


def _N_to_snapstr(N):
    '''return string representing snap number N.'''
    if N == 0:
        return ''
    else:
        assert tools.is_integer(N), f"snap values must be integers! (snap={N})"
        return '_%03i' % N


# include methods (and some aliases) for getting snaps in BifrostData object
BifrostData.get_snapstuff = get_snapstuff
BifrostData.get_snapname = get_snapname
BifrostData.available_snaps = available_snaps
BifrostData.get_snaps = get_snaps
BifrostData.get_snap_prev = get_snap_prev
BifrostData.get_snap_next = get_snap_next
BifrostData.snaps_info = snaps_info


####################
#  WRITING SNAPS   #
####################

def write_br_snap(rootname, r, px, py, pz, e, bx, by, bz):
    nx, ny, nz = r.shape
    data = np.memmap(rootname, dtype=np.float32, mode='w+', order='f', shape=(nx, ny, nz, 8))
    data[..., 0] = r
    data[..., 1] = px
    data[..., 2] = py
    data[..., 3] = pz
    data[..., 4] = e
    data[..., 5] = bx
    data[..., 6] = by
    data[..., 7] = bz
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

            x[nx-3:] = x[nx-3:][::-1]  # fixes order in the tail of x
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

class BifrostUnits(units.HelitaUnits):
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

    def __init__(self, filename='mhd.in', fdir='./', verbose=True, base_units=None, **kw__super_init):
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
        DEFAULT_UNITS = dict(u_l=1.0e8, u_t=1.0e2, u_r=1.0e-7, gamma=1.667)
        base_to_use = dict()  # << here we will put the u_l, u_t, u_r, gamma to actually use.
        _n_base_set = 0  # number of base units set (i.e. assigned in base_to_use)

        # setup units from base_units, if applicable
        if base_units is not None:
            try:
                base_units.items()
            except AttributeError:  # base_units is a list
                for i, val in enumerate(base_units):
                    base_to_use[self.BASE_UNITS[i]] = val
                    _n_base_set += 1
            else:
                for key, val in base_units.items():
                    if key in DEFAULT_UNITS.keys():
                        base_to_use[key] = val
                        _n_base_set += 1
                    elif verbose:
                        print(('(WWW) the key {} is not a base unit',
                              ' so it was ignored').format(key))

        # setup units from file (or defaults), if still necessary.
        if _n_base_set != len(DEFAULT_UNITS):
            if filename is None:
                file_exists = False
            else:
                file = os.path.join(fdir, filename)
                file_exists = os.path.isfile(file)
            if file_exists:
                # file exists -> set units using file.
                self.params = read_idl_ascii(file, firstime=True)

                def setup_unit(key):
                    if base_to_use.get(key, None) is not None:
                        return
                    # else:
                    try:
                        value = self.params[key]
                    except Exception:
                        value = DEFAULT_UNITS[key]
                        if verbose:
                            printstr = ("(WWW) the file '{file}' does not contain '{unit}'. "
                                        "Default Solar Bifrost {unit}={value} has been selected.")
                            print(printstr.format(file=file, unit=key, value=value))
                    base_to_use[key] = value

                for unit in DEFAULT_UNITS.keys():
                    setup_unit(unit)
            else:
                # file does not exist -> setup default units.
                units_to_set = {unit: DEFAULT_UNITS[unit] for unit in DEFAULT_UNITS.keys()
                                if getattr(self, unit, None) is None}
                if verbose:
                    print("(WWW) selected file '{file}' is not available.".format(file=filename),
                          "Setting the following Default Solar Bifrost units: ", units_to_set)
                for key, value in units_to_set.items():
                    base_to_use[key] = value

        # initialize using instructions from HelitaUnits (see helita.sim.units.py)
        super().__init__(**base_to_use, verbose=verbose, **kw__super_init)


Bifrost_units = BifrostUnits  # alias (required for historical compatibility)

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
        self.uni = Bifrost_units(filename=tmp, fdir=fdir)
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

    def get_table(self, out='ne', bin=None, order=1):

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
        self.ch_tabname = "chianti"  # alternatives are e.g. 'mazzotta' and others found in Chianti
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
        tgTable = np.linspace(self.teinit, self.teinit + self.dte*self.nte, self.nte)
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
        # arr = (1 - ion_h) * ohi + rhe * ((1 - ion_he - ion_hei) *
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
        else:  # Chianti table
            import ChiantiPy.core as ch
            if self.verbose:
                print('*** Reading Chianti table', whsp*4, end="\r",
                      flush=True)
            h = ch.Ioneq.ioneq(1)
            h.load(tabname)
            temp = np.linspace(tgmin, tgmax, ntg)
            h.calculate(10**temp)
            logte = np.log10(h.Temperature)
            self.dte = logte[1]-logte[0]
            self.teinit = logte[0]
            self.nte = np.size(logte)
            self.ionh1d = h.Ioneq[0, :]
            he = ch.Ioneq.ioneq(2)
            he.load(tabname)
            self.ionhe1d = he.Ioneq[0, :]
            self.ionhei1d = he.Ioneq[1, :]
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
        a = cross_sect(['h-h-data2.txt','h-h2-data.txt'], fdir="/data/cb24bih")

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

    def load_cross_tables(self, firstime=False):
        '''
        Collects the information in the cross table files.
        '''
        self.cross_tab = dict()
        for itab in range(len(self.cross_tab_list)):
            self.cross_tab[itab] = read_cross_txt(self.cross_tab_list[itab], firstime=firstime,
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

        finterp = interpolate.interp1d(np.log(self.cross_tab[itab]['tg']),
                                       self.cross_tab[itab][out])
        tgreg = np.array(tg, copy=True)
        max_temp = np.max(self.cross_tab[itab]['tg'])
        tgreg[tg > max_temp] = max_temp
        min_temp = np.min(self.cross_tab[itab]['tg'])
        tgreg[tg < min_temp] = min_temp

        return finterp(np.log(tgreg))

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
        if fdir is None:
            fdir = getattr(obj, 'fdir', '.')
        return Cross_sect(cross_tab, fdir, *args__Cross_sect, **kw__Cross_sect, obj=obj)
    return _init_cross_sect

## Tools for making cross section table such that colfreq is independent of temperature ##


def constant_colfreq_cross(tg0, Q0, tg=range(1000, 400000, 100), T_to_eV=lambda T: T / 11604):
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
    E = T_to_eV(tg)
    Q = Q0 * np.sqrt(tg0) / np.sqrt(tg)
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
    ul = data.params['u_l'] / 1.e2  # to metres
    ur = data.params['u_r']        # to g/cm^3  (for ne_rt_table)
    ut = data.params['u_t']        # to seconds
    uv = ul / ut
    ub = data.params['u_b'] * 1e-4  # to tgasesl
    ue = data.params['u_ee']       # to erg/g

    if not desc:
        desc = 'BIFROST snapshot from 2D sequence %s, sx=%s sy=1 sz=%s.' % \
            (file_root, repr(sx), repr(sz))
        if data.hion:
            desc = 'hion ' + desc
    x = data.x[sx] * ul
    y = snaps
    z = data.z[sz] * (-ul)

    data.r.dtype
    # cstagger.init_stagger(data.nz, data.dx, data.dy, data.z.astype(rdt),
    #                      data.zdn.astype(rdt), data.dzidzup.astype(rdt),
    #                      data.dzidzdn.astype(rdt))

    for i, s in enumerate(snaps):
        data.set_snap(s)
        tgas[:, i] = np.squeeze(data.tg)[sx, sz]
        rho = np.squeeze(data.r)[sx, sz]
        vz[:, i] = np.squeeze(do_stagger(data.pz, 'zup', obj=data))[sx, sz] / rho * (-uv)
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
def read_idl_ascii(filename, firstime=False):
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
            key, _, value = line.partition('=')
            key = key.strip().lower()
            value = value.strip()
            if len(key) == 0:
                continue    # this was a blank line.
            elif len(value) == 0:
                if firstime:
                    print('(WWW) read_params: line %i is invalid, skipping' % li)
                continue
            # --- evaluate value --- #
            # allow '.false.' or '.true.' for bools
            if (value.lower() in ['.false.', '.true.']):
                value = False if value.lower() == '.false.' else True
            else:
                # safely evaluate any other type of value
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    # failed to evaluate. Might be string, or might be int with leading 0's.
                    try:
                        value = int(value)
                    except ValueError:
                        # failed to convert to int; interpret value as string.
                        pass  # leave value as string without evaluating it.

            params[key] = value

    return params


@file_memory.remember_and_recall('_memory_read_cross_txt', kw_mem=['kelvin'])
def read_cross_txt(filename, firstime=False, kelvin=True):
    ''' Reads IDL-formatted (command style) ascii file into dictionary.
    tg will be converted to Kelvin, unless kelvin==False.
    '''
    li = 0
    params = {}
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
            elif not ('crossunits' in params.keys()):
                print('(WWW) read_cross: line %i is invalid, missing crossunits, file %s' % (li, filename))

            if (len(line) < 2):
                if (firstime):
                    print('(WWW) read_cross: line %i is invalid, skipping, file %s' % (li, filename))
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
            if not ('tg' in params.keys()):
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
            if not ('el' in params.keys()):
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
                if not ('mt' in params.keys()):
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


def find_first_match(name, path, incl_path=False):
    '''
    This will find the first match,
    name : string, e.g., 'patern*'
    incl_root: boolean, if true will add full path, otherwise, the name.
    path : sring, e.g., '.'
    '''
    originalpath = os.getcwd()
    os.chdir(path)
    for file in glob(name):
        if incl_path:
            os.chdir(originalpath)
            return os.path.join(path, file)
        else:
            os.chdir(originalpath)
            return file
    os.chdir(originalpath)
