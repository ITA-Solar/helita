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
    - set_fundamental_var() which will allow to set fundamentals using common alternatives,
        e.g. set_fundamental_var('nr', value) actually sets 'r'=value * mass;
        and set_fundamental_var('tg', value) actually sets 'e' after converting from temperature.
"""

# import built-ins
import shutil
import os
import collections

# import internal modules
from . import ebysus
from . import tools


class FakeEbysusData(ebysus.EbysusData):
    def __init__(self, *args, verbose=False, **kw):
        '''initialize self using method from parent. But first:
            - if there is no .idl file with appropriate snapname (from mhd.in or args[0]),
              make one (by copying mhd.in)
        '''
        # setup memory for fake data
        self.setvars = collections.defaultdict(tools.GenericDict_with_equals(self._metadata_equals))

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

    def set_var(self, var, value, *args, nfluid=None, **kwargs):
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