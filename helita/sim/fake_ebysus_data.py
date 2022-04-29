"""
File purpose:
    Access the load_..._quantities calculations without needing to write a full snapshot.
    
    Examples where this module is particularly useful:
        - quickly testing values of quantities from helita postprocessing, for small arrays of data.
        - check what would happen if a small number of changes are made to existing data.

[TODO]
    - properly hook FakeEbysusData._load_quantity into units, caching, etc.
        (until doing that^, this module is only compatible with 'simu' units,
        and the self.got_vars_tree() results may be confusing / misleading.)
    - 'units_input' attribute, which tells the units for 'value' in set_var.
"""

# import built-ins
import shutil
import os
import collections

# import internal modules
from . import ebysus
from . import tools
from .units import (
    UNIT_SYSTEMS,
)


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
        super(type(self), self).__init__(*args, verbose=verbose, **kw)

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
            assert value in UNIT_SYSTEMS, f"invalid units: {repr(value)}. Expected one of {UNIT_SYSTEMS} or None."
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
            assert value in UNIT_SYSTEMS, f"invalid units: {repr(value)}. Expected one of {UNIT_SYSTEMS} or None."
            if value != 'simu':
                raise NotImplementedError(f'units_input = {repr(value)}')
        self._units_input = value

    def _init_vars_get(self, *args__None, **kw__None):
        '''do nothing and return None.
        (overriding the initial data grabbing from EbysusData.)
        '''
        pass

    def set_var(self, var, value, *args, nfluid=None, units=None, fundamental=None,
                _skip_preprocess=False, **kwargs):
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
        '''
        if fundamental is None:
            if var in self.FUNDAMENTAL_SETTABLES:
                fundamental = True
        if fundamental:
            return self.set_fundamental_var(var, value, *args, units=units, **kwargs)

        if not _skip_preprocess:
            self._get_var_preprocess(var, *args, **kwargs)

        # bookkeeping - nset
        self.nset += 1
        # bookkeeping - nfluid
        if nfluid is None:
            nfluid = self.get_var_nfluid(var)
            if nfluid is None:  # if still None, raise instructive error (instead of confusing crash later).
                raise ValueError(f"nfluid=None for var='{var}'. Workaround: manually enter nfluid in set_var.")
        # bookkeeping - units
        units_input = units if units is not None else self.units_input
        if units_input != 'simu':
            raise NotImplementedError(f'set_var(..., units={repr(value)})')

        # save to memory.
        meta = self._metadata(with_nfluid=nfluid)
        self.setvars[var][meta] = value

        # clear the cache.
        if hasattr(self, 'cache'):
            self.cache.clear()

    def _load_quantity(self, var, *args, **kwargs):
        '''load quantity, but first check if the value is in memory with the appropriate metadata.
        [TODO] hook this properly into caching, units, etc.
        '''
        if var in self.setvars:
            meta = self._metadata()
            try:
                return self.setvars[var][meta]
            except KeyError:  # var is in memory, but not with appropriate metadata.
                pass          #    e.g. we know some 'nr', but not for the currently-set ifluid.
        # else
        return super(type(self), self)._load_quantity(var, *args, **kwargs)

    FUNDAMENTAL_SETTABLES = ('r', 'nr', 'e', 'tg', *(f'{v}{x}' for x in AXES for v in ('p', 'u', 'ui', 'b')))

    def set_fundamental_var(self, var, value, *args, fundamental_only=False, units=None, **kwargs):
        '''sets fundamental quantity corresponding to var; also sets var (unless fundamental_only).
        fundamental quantities, and alternate options for vars will allow to set them, are:
            r - nr
            e - tg, p
            p{x} - u{x}, ui{x}   (for {x} in 'x', 'y', 'z')
            b{x} – (no alternates.)
        
        fundamental_only: False (default) or True
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
        # preprocess
        self._get_var_preprocess(var, *args, **kwargs)  # sets snap, ifluid, etc.
        also_set_var = (not fundamental_only)
        # units
        units_input = units if units is not None else self.units_input_fundamental
        def ulookup(key):
            return {'simu': 1,
                    'si': getattr(self.uni, f'usi_{key}'),
                    'cgs': getattr(self.uni, f'u_{key}')}[units_input]
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
                nr     = self('nr') * ulookup('nr')
                kB     = {'simu':self.uni.simu_kB, 'si':self.uni.ksi_b, 'cgs':self.uni.k_b}[units_input]
                result = value * nr * kB / (self.uni.gamma - 1)
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
