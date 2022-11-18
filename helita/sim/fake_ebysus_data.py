"""
File purpose:
    Access the load_..._quantities calculations without needing to write a full snapshot.
    
    Examples where this module is particularly useful:
        - quickly testing values of quantities from helita postprocessing, for small arrays of data.
        - check what would happen if a small number of changes are made to existing data.

[TODO]
    - allow FakeEbysusData.set_var to use units other than 'simu'
      (probably by looking up var in vardict to determine units).
      Note - this is already allowed in set_var_fundamanetal
      (which, by default, gets called from set_var for var in FUNDAMENTAL_SETTABLES).
"""

# import built-ins
import shutil
import os
import collections
import warnings

# import internal modules
from . import ebysus
from . import tools
from . import document_vars
from . import file_memory
from . import units

# import external public modules
import numpy as np


AXES = ('x', 'y', 'z')


class FakeEbysusData(ebysus.EbysusData):
    '''Behaves like EbysusData but allows to choose artificial values for any quant.
    
    If a quant is requested (via self(quant) or self.get_var(quant)),
    first check if that quant has been set artificially, using the artificial value if found.
    If the quant has not been set artificially, try to read it from snapshot as normal.

    No snapshot data are required to utilize FakeEbysusData.
    All "supporting materials" are required as normal, though:
        - snapname.idl  ^(see footnote)
        - mf_params.in (filename determined by mf_param_file in snapname.idl)
        - any relevant .atom files
        - collision tables if collision module(s) are enabled:
            - mf_coltab.in, mf_ecoltab.in
            - any collision files referenced by those files ^ 
        ^(footnote) if mhd.in is provided:
            if snapname is not entered at __init__, use snapname from mhd.in.
            if snapname.idl file does exist, copy mhd.in to a new file named snapname.idl.

    units_input: None, 'simu', 'si', or 'cgs'
        units of value input using set_var.
        only 'simu' system is implemented right now.
        None --> use same mode as units_output.
    units_input_fundamental: None, 'simu', 'si', or 'cgs'
        units of value input using set_fundamental_var
        None --> use same mode as units_input.
    '''

    def __init__(self, *args, verbose=False, units_input=None, units_input_fundamental=None, **kw):
        '''initialize self using method from parent. But first:
            - if there is no .idl file with appropriate snapname (from mhd.in or args[0]),
              make one (by copying mhd.in)

        TODO: non-default options for units_input
        '''
        # setup memory for fake data
        self.setvars = collections.defaultdict(tools.GenericDict_with_equals(self._metadata_equals))
        self.nset = 0   # nset tracks the number of times set_var has been called.

        # units
        self.units_input = units_input
        self.units_input_fundamental = units_input_fundamental

        # make snapname.idl if necessary.
        snapname = args[0] if len(args)>0 else ebysus.get_snapname()
        idlfilename = f'{snapname}.idl'
        if not os.path.isfile(idlfilename):
            shutil.copyfile('mhd.in', idlfilename)
            if verbose:
                print(f"copied 'mhd.in' to '{idlfilename}'")
        # initialize self using method from parent.
        super(FakeEbysusData, self).__init__(*args, verbose=verbose, **kw)

    @property
    def units_input_fundamental(self):
        '''units of value input using set_fundamental_var
        None --> use same mode as units_input.
        '''
        result = getattr(self, '_units_input_fundamental', None)
        if result is None: 
            result = getattr(self, 'units_input', 'simu')
        return result
    @units_input_fundamental.setter
    def units_input_fundamental(self, value):
        if value is not None:
            value = value.lower()
            units.ASSERT_UNIT_SYSTEM(value)
        self._units_input_fundamental = value

    @property
    def units_input(self):
        '''units of value input using set_var.
        only 'simu' system is implemented right now.
        None --> use same mode as units_output.
        '''
        result = getattr(self, '_units_input', 'simu')
        if result is None:
            result = getattr(self, 'units_output', 'simu')
        return result
    @units_input.setter
    def units_input(self, value):
        if value is not None:
            value = value.lower()
            units.ASSERT_UNIT_SYSTEM(value)
            if value != 'simu':
                raise NotImplementedError(f'units_input = {repr(value)}')
        self._units_input = value

    def _init_vars_get(self, *args__None, **kw__None):
        '''do nothing and return None.
        (overriding the initial data grabbing from EbysusData.)
        '''
        pass

    ## SET_VAR ##
    def set_var(self, var, value, *args, nfluid=None, units=None, fundamental=None,
                _skip_preprocess=False, fundamental_only=False, **kwargs):
        '''set var in memory of self.
        Use this to set the value for some fake data.
        Any time we get var, we will check memory first;
            if the value is in memory (with the appropriate metadata, e.g. ifluid,)
            use the value from memory. Otherwise try to get it a different way.

        NOTE: set_var also clears self.cache (which otherwise has no way to know the data has changed).

        *args, and **kwargs go to self._get_var_preprocess.
            E.g. using set_var(..., ifluid=(1,2)) will first set self.ifluid=(1,2).

        nfluid: None (default), 0, 1, or 2
            number of fluids which this var depends on.
            None - try to guess, using documentation about the vars in self.vardict.
                   This option is good enough for most cases.
                   But fails for constructed vars which don't appear in vardict directly, e.g. 'b_mod'.
            0; 1; or 2 - depends on neither; just ifluid; or both ifluid and jfluid.
        units: None, 'simu', 'si', or 'cgs'
            units associated with value input to this function.
            (Note that all values will be internally stored in the same units as helita would output,
             given units_output='simu'. This usually means values are stored in simulation units.)
            None --> use self.units_input.
            else --> use the value of this kwarg.
        fundamental: None (default), True, or False
            None --> check first if var is in self.FUNDAMENTAL_SETTABLES.
                     if it is, use set_fundamental_var instead.
            True --> immediately call set_fundamental_var instead.
            False --> do not even consider using set_fundamental_var.
        fundamental_only: True (default), or False
            if calling set_fundamental_var...
            True  --> only set value of fundamental quantity corresponding to var.
            False --> also set value of var.
            Example of why it matters...:
                If running the following lines:
                    (1)  obj.set_var('tg', 4321)
                    (2)  obj.set_var('e', obj('e') * 100)
                    (3)  obj.get_var('tg')
                with fundamental_only==True:
                    (1)  sets 'e' to the appropriate value such that 'tg' will be 4321
                    (2)  adjusts the value of 'e' (only)
                    (3)  gives the answer 432100
                with fundamental_only==False:
                    (1)  sets 'tg' to 4321. (AND 'e' appropriately, if fundamental is True or None.)
                    (2)  adjusts the value of 'e' (only)
                    (3)  gives the answer 4321, because it reads the value of 'tg' directly, instead of checking 'e'.

        '''
        if fundamental is None:
            if var in self.FUNDAMENTAL_SETTABLES:
                fundamental = True
        if fundamental:
            return self.set_fundamental_var(var, value, *args, units=units,
                                            fundamental_only=fundamental_only, **kwargs)

        self._warning_during_setvar_if_slicing_and_stagger()

        if not _skip_preprocess:
            self._get_var_preprocess(var, *args, **kwargs)

        # bookkeeping - nfluid
        if nfluid is None:
            nfluid = self.get_var_nfluid(var)
            if nfluid is None:  # if still None, raise instructive error (instead of confusing crash later).
                raise ValueError(f"nfluid=None for var='{var}'. Workaround: manually enter nfluid in set_var.")
        # bookkeeping - units
        units_input = units if units is not None else self.units_input
        if units_input != 'simu':
            raise NotImplementedError(f'set_var(..., units={repr(units_input)})')

        # save to memory.
        meta = self._metadata(with_nfluid=nfluid)
        self.setvars[var][meta] = value

        # do any updates that we should do whenever a var is set.
        self._signal_set_var(var=var)

    def _signal_set_var(self, var=None):
        '''update anything that needs to be updated whenever a var is set.
        This code should probably be run any time self.setvars is altered in any way.
        '''
        # bookkeeping - increment nset
        self.nset += 1
        # clear the cache.
        if hasattr(self, 'cache'):
            self.cache.clear()

    # tell quant lookup to search vardict for var if metaquant == 'setvars'
    VDSEARCH_IF_META = getattr(ebysus.EbysusData, 'VDSEARCH_IF_META', []) + ['setvars']

    @tools.maintain_attrs('match_type', 'ifluid', 'jfluid')
    @file_memory.with_caching(cache=False, check_cache=True, cache_with_nfluid=None)
    @document_vars.quant_tracking_top_level
    def _load_quantity(self, var, *args, **kwargs):
        '''load quantity, but first check if the value is in memory with the appropriate metadata.'''
        if var in self.setvars:
            meta = self._metadata()
            try:
                result = self.setvars[var][meta]
                document_vars.setattr_quant_selected(self, var, 'SET_VAR', metaquant='setvars',
                                                     varname=var, level='(FROM SETVARS)', delay=False)
                return result
            except KeyError:  # var is in memory, but not with appropriate metadata.
                pass          #    e.g. we know some 'nr', but not for the currently-set ifluid.
        # else
        return self._raw_load_quantity(var, *args, **kwargs)

    FUNDAMENTAL_SETTABLES = ('r', 'nr', 'e', 'tg', *(f'{v}{x}' for x in AXES for v in ('p', 'u', 'ui', 'b')))

    def set_fundamental_var(self, var, value, *args, fundamental_only=True, units=None, **kwargs):
        '''sets fundamental quantity corresponding to var; also sets var (unless fundamental_only).
        fundamental quantities, and alternate options for vars will allow to set them, are:
            r - nr
            e - tg, p
            p{x} - u{x}, ui{x}   (for {x} in 'x', 'y', 'z')
            b{x} – (no alternates.)
        
        fundamental_only: True (default) or False
            True  --> only set value of fundamental quantity corresponding to var.
            False --> also set value of var.
        units: None, 'simu', 'si', or 'cgs'
            units associated with value input to this function.
            (Note that all values will be internally stored in the same units as helita would output,
             given units_output='simu'. This usually means values are stored in simulation units.)
            None --> use self.units_input_fundamental.
            else --> use the value of this kwarg.

        returns (name of fundamental var which was set, value to which it was set)
        '''
        assert var in self.FUNDAMENTAL_SETTABLES, f"I don't know how this var relates to a fundamental var: '{var}'"
        self._warning_during_setvar_if_slicing_and_stagger()

        # preprocess
        self._get_var_preprocess(var, *args, **kwargs)  # sets snap, ifluid, etc.
        also_set_var = (not fundamental_only)
        # units
        units_input = units if units is not None else self.units_input_fundamental
        def ulookup(key):
            '''return self.uni(key, units_input, 'simu').
            if key is for variable (e.g. 'u', 'r'), value [simu] * ulookup(key) == value [units_input]
            if key is for constant (e.g. 'kB'), ulookup(key) is value of constant in [units_input] system.'''
            return self.uni(key, units_input, 'simu')
        ## more units: we will set the following values below:
        ###  u_res = divide result by this value to convert to units for internal storage.
        ###  u_var = divide   var  by this value to convert to units for internal storage.
        # set fundamental var
        ## 'r' - mass density
        if var in ['r', 'nr']:
            setting = 'r'
            u_res   = ulookup('r')
            if var == 'r':
                also_set_var = False
                result = value
            elif var == 'nr':   # nr = r / m
                u_var  = ulookup('nr')
                result = value * self.get_mass(units=units_input)
        ## 'e' - energy density
        elif var in ['e', 'p', 'tg']:
            setting = 'e'
            u_res   = ulookup('e')
            if var == 'e':
                also_set_var = False
                result = value
            elif var == 'p':    # p = e * (gamma - 1)
                u_var  = u_res
                result = value / (self.uni.gamma - 1)
            elif var == 'tg':   # T = p / (nr * kB) = (e * (gamma - 1)) / (nr * kB)
                u_var  = 1   # temperature units always K.
                e_to_tg = self('e_to_tg')
                result  = value / e_to_tg
        ## 'p{x}' - momentum density ({x}-component)
        elif var in tuple(f'{v}{x}' for x in AXES for v in ('p', 'u', 'ui')):
            base, x = var[:-1], var[-1]
            setting = f'p{x}'
            u_res   = ulookup('pm')
            if base == 'p':
                also_set_var = False
                result = value
            elif base in ['u', 'ui']: # u = p / rxdn
                u_var  = ulookup('u')
                r      = self('r'+f'{x}dn') * ulookup('r')
                result = value * r
        ## 'b{x}' - magnetic field ({x}-component)
        elif var in tuple(f'b{x}' for x in AXES):
            base, x = var[:-1], var[-1]
            setting = f'b{x}'
            u_res   = ulookup('b')
            if base == 'b':
                also_set_var = False
                result = value
        else:
            raise NotImplementedError(f'{var} in set_fundamental_var')

        # set fundamental var
        self.set_var(setting, result / u_res, *args, **kwargs,
                     units='simu',          # we already handled the units; set_var shouldn't mess with them.
                     fundamental=False,     # we already handled the 'fundamental' possibility.
                     _skip_preprocess=True, # we already handled preprocessing.
                     )
        # set var (the one that was entered to this function)
        if also_set_var:
            self.set_var(var, value / u_var, *args, **kwargs,
                         units='simu',          # we already handled the units; set_var shouldn't mess with them.
                         fundamental=False,     # we already handled the 'fundamental' possibility.
                         _skip_preprocess=True, # we already handled preprocessing.
                         )
        return (setting, result)

    def _warn_if_slicing_and_stagger(self, message):
        '''if any slice is not slice(None), and do_stagger=True, warnings.warn(message)'''
        if self.do_stagger and any(iiax!=slice(None) for iiax in (self.iix, self.iiy, self.iiz)):
            warnings.warn(message)

    def _warning_during_setvar_if_slicing_and_stagger(self):
        self._warn_if_slicing_and_stagger((
          'setting var with iix, iiy, or iiz != slice(None) and do_stagger=True'
          ' may lead to unexpectedly not using values from setvars. \n\n(Internally,'
          ' when do_stagger=True, slices are set to slice(None) while getting vars, and'
          ' the original slices are only applied after completing all other calculations.)'
          f'\n\nGot slices: iix={self.iix}, iiy={self.iiy}, iiz={self.iiz}'
        ))

    ## UNSET_VAR ##
    def unset_var(self, var):
        '''unset the value of var if it has been set in setvars.
        if var hasn't been set, do nothing.
        returns whether var was previously set.
        '''
        try:
            del self.setvars[var]
        except KeyError:
            return False   # var wasn't in self.setvars.
        else:
            self._signal_set_var(var=var)
            return True

    FUNDAMENTAL_VARS = (*(f'b{x}' for x in AXES), 'r', *(f'p{x}' for x in AXES), 'e')

    def unset_extras(self):
        '''unsets the values of all non-fundamental vars.
        returns list of vars which were previously set but are no longer set.
        '''
        desetted = []
        for var in iter(set(self.setvars.keys()) - set(self.FUNDAMENTAL_VARS)):
            u = self.unset_var(var)
            if u:
                desetted.append(var)
        return desetted

    unset_nonfundamentals = unset_non_fundamentals = keep_only_fundamentals = unset_extras   # aliases

    ## ITER FUNDAMENTALS ##
    def iter_fundamentals(self, b=True, r=True, p=True, e=True, AXES=AXES):
        '''iterate through fundamental vars:
            b (per axis)
            r (per non-electron fluid)
            p (per axis, per non-electron fluid)
            e (per fluid (including electrons))
        during iteration through fluids, set self.ifluid appropriately.

        b, r, p, e: bool
            whether to iterate through this fundamental var.
        AXES: string or list of strings from ('x', 'y', 'z')
            axes to use when iterating through fundamental vars.
        
        yields var name. (sets ifluid immediately before yielding each value.)
        '''
        if b:
            for x in AXES:
                yield f'b{x}'
        if r:
            for fluid in self.fluid_SLs(with_electrons=False):
                self.ifluid = fluid
                yield 'r'
        if p:
            for fluid in self.fluid_SLs(with_electrons=False):
                for x in AXES:
                    self.ifluid = fluid   # in inner loop, to set immediately before yielding
                    yield f'p{x}'
        if e:
            for fluid in self.fluid_SLs(with_electrons=True):
                self.ifluid = fluid
                yield 'e'

    @tools.maintain_attrs('ifluid')
    def set_fundamental_means(self):
        '''sets all fundamental vars to their mean values. (Method provided for convenience.)'''
        for var in self.iter_fundamentals():
            self.set_fundamental_var(var, np.mean(self(var)))

    @tools.maintain_attrs('ifluid')
    def set_fundamental_full(self):
        '''sets all fundamental vars to their fully-shaped values.
        (If their shape is not self.shape, add self.zero())
        '''
        for var in self.iter_fundamentals():
            self.set_fundamental_var(var, self.reshape_if_necessary(self(var)))

    ## WRITE SNAPSHOT ##
    def write_snap0(self, warning=True):
        '''write data from self to snapshot 0.
        if warning, first warn user that snapshot 0 will be overwritten, and request confirmation.
        '''
        if not self._confirm_write('Snapshot 0', warning):
            return   # skip writing unless confirmed.
        self.write_mfr(warning=False)
        self.write_mfp(warning=False)
        self.write_mfe(warning=False)
        self.write_mf_common(warning=False)

    def _confirm_write(self, name, warning=True):
        '''returns whether user truly wants to write name at self.file_root.
        if warning==False, return True (i.e. "yes, overwrite") without asking user.
        '''
        if warning:
            confirm = input(f'Write {name} at {self.file_root}? (y/n)')
            if confirm.lower() not in ('y', 'yes'):
                print('Aborted. Nothing was written.')
                return False
        return True

    @tools.with_attrs(units_output='simu')
    def write_mfr(self, warning=True):
        '''write mass densities from self to snapshot 0.'''
        if not self._confirm_write('Snapshot 0 mass densities', warning):
            return   # skip writing unless confirmed.
        for ifluid in self.iter_fluid_SLs(with_electrons=False):
            r_i = self.reshape_if_necessary( self('r', ifluid=ifluid) )
            ebysus.write_mfr(self.root_name, r_i, ifluid=ifluid)

    @tools.with_attrs(units_output='simu')
    def write_mfp(self, warning=True):
        '''write momentum densitites from self to snapshot 0.'''
        if not self._confirm_write('Snapshot 0 momentum densities', warning):
            return   # skip writing unless confirmed.
        for ifluid in self.iter_fluid_SLs(with_electrons=False):
            self.ifluid = ifluid
            p_xyz_i = [self.reshape_if_necessary( self(f'p{x}') ) for x in AXES]
            ebysus.write_mfp(self.root_name, *p_xyz_i, ifluid=ifluid)

    @tools.with_attrs(units_output='simu')
    def write_mfe(self, warning=True):
        '''write energy densitites from self to snapshot 0.
        Note: if there is only 1 non-electron fluid, this function does nothing and returns None
            (because ebysus treats e as 'common' in single fluid case. See also: write_common()).
        '''
        non_e_fluids = self.fluid_SLs(with_electrons=False)
        if len(non_e_fluids) == 1:
            return
        if not self._confirm_write('Snapshot 0 energy densities', warning):
            return   # skip writing unless confirmed.
        for ifluid in non_e_fluids:
            e_i = self.reshape_if_necessary( self('e', ifluid=ifluid) )
            ebysus.write_mfe(self.root_name, e_i, ifluid=ifluid)
        e_e = self.reshape_if_necessary( self('e', ifluid=(-1,0)) )
        ebysus.write_mf_e(self.root_name, e_e)

    @tools.with_attrs(units_output='simu')
    def write_mf_common(self, warning=True):
        '''write magnetic field from self to snapshot 0. (Also writes energy density if singlefluid.)'''
        b_xyz = [self.reshape_if_necessary( self(f'b{x}') ) for x in AXES]
        non_e_fluids = self.fluid_SLs(with_electrons=False)
        if len(non_e_fluids) == 1:
            if not self._confirm_write('Snapshot 0 magnetic field and single fluid energy density', warning):
                return   # skip writing unless confirmed.
            self.ifluid = non_e_fluids[0]
            e_singlefluid = self.reshape_if_necessary( self('e') )
            ebysus.write_mf_common(self.root_name, *b_xyz, e_singlefluid)
        else:
            if not self._confirm_write('Snapshot 0 magnetic field', warning):
                return   # skip writing unless confirmed.
            ebysus.write_mf_common(self.root_name, *b_xyz)

    ## CONVENIENCE ##
    def reshape_if_necessary(self, val):
        '''returns val + self.zero() if shape(val) != self.shape, else val (unchanged)'''
        if np.shape(val) != self.shape:
            val = val + self.zero()
        return val
