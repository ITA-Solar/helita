"""
File purpose:
    Access the load_..._quantities calculations without needing to write a full snapshot.
    
    Examples where this module is particularly useful:
        - quickly testing values of quantities from helita postprocessing, for small arrays of data.
        - [TODO] check what would happen if a small number of changes are made to existing data.

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


AXES = ('x', 'y', 'z')


class FakeEbysusData(ebysus.EbysusData):
    def __init__(self, *args, verbose=False, units_input='simu', **kw):
        '''initialize self using method from parent. But first:
            - if there is no .idl file with appropriate snapname (from mhd.in or args[0]),
              make one (by copying mhd.in)

        TODO: non-default options for units_input
        '''
        # setup memory for fake data
        self.setvars = collections.defaultdict(tools.GenericDict_with_equals(self._metadata_equals))

        # units
        self.units_input = units_input

        # make snapname.idl if necessary.
        snapname = args[0] if len(args)>0 else ebysus.get_snapname()
        idlfilename = f'{snapname}.idl'
        if not os.path.isfile(idlfilename):
            shutil.copyfile('mhd.in', idlfilename)
            if verbose:
                print(f"copied 'mhd.in' to '{idlfilename}'")
        # initialize self using method from parent.
        super(type(self), self).__init__(*args, verbose=verbose, **kw)

    def _init_vars_get(self, *args__None, **kw__None):
        '''do nothing and return None.
        (overriding the initial data grabbing from EbysusData.)
        '''
        pass

    def set_var(self, var, value, *args, nfluid=None, _skip_preprocess=False, **kwargs):
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
        '''
        if not _skip_preprocess:
            self._get_var_preprocess(var, *args, **kwargs)

        if nfluid is None:
            nfluid = self.get_var_nfluid(var)
            if nfluid is None:  # if still None, raise instructive error (instead of confusing crash later).
                raise ValueError(f"nfluid=None for var='{var}'. Workaround: manually enter nfluid in set_var.")

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

    def set_fundamental_var(self, var, value, *args, fundamental_only=False, **kwargs):
        '''sets fundamental quantity corresponding to var; also sets var (unless fundamental_only).
        fundamental quantities, and options for which other vars will allow to set them, are:
            r - nr
            e - tg, p
            p{x} - u{x}, ui{x}   (for {x} in 'x', 'y', 'z')

        returns (name of fundamental var which was set, value to which it was set)
        '''
        assert var in self.FUNDAMENTAL_SETTABLES, f"I don't know how this var relates to a fundamental var: '{var}'"
        # preprocess
        self._get_var_preprocess(var, *args, **kwargs)  # sets snap, ifluid, etc.
        # set var (as entered)
        if not fundamental_only:
            self.set_var(var, value, *args, **kwargs, _skip_preprocess=True)
        # set fundamental var
        ## 'r' - mass density
        if var in ['r', 'nr']:
            setting = 'r'
            if var == 'r':
                result = value
            elif var == 'nr':   # nr = r / m
                result = value * self.get_mass(units=self.units_input)
        ## 'e' - energy density
        elif var in ['e', 'p', 'tg']:
            setting = 'e'
            if var == 'e':
                result = value
            elif var == 'p':    # p = e * (gamma - 1)
                result = value / (self.uni.gamma - 1)
            elif var == 'tg':   # T = p / (nr * kB) = (e * (gamma - 1)) / (nr * kB)
                nr = self('nr')
                kB = {'simu':self.uni.simu_kB, 'si':self.uni.ksi_b, 'cgs':self.uni.k_b}[self.units_input]
                result = value * nr * kB / (self.uni.gamma - 1)
        ## 'p{x}' - momentum density ({x}-component)
        elif var in tuple(f'{v}{x}' for x in AXES for v in ('p', 'u', 'ui')):
            base, x = var[:-1], var[-1]
            setting = f'p{x}'
            if base == 'p':
                result = value
            elif base in ['u', 'ui']: # u = p / rxdn
                ux = self(var)
                r  = self('r'+f'{x}dn')
                result = ux * r
        ## 'b{x}' - magnetic field ({x}-component)
        elif var in tuple(f'b{x}' for x in AXES):
            base, x = var[:-1], var[-1]
            setting = f'b{x}'
            if base == 'b':
                result = value
        else:
            raise NotImplementedError(f'{var} in set_fundamental_var')

        self.set_var(setting, result, *args, **kwargs, _skip_preprocess=True)
        return (setting, result)
