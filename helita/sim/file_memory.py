"""
created by Sam Evans on Apr 12 2021 

purpose:

    - don't re-read files multiple times. (see remember_and_recall())
    - limit number of open memmaps; avoid crash via "too many files open". (see manage_memmaps())

TODO:
    try to manage_memmaps a bit more intelligently..
    current implementation will delete the oldest-created memmap first.
    This leads to non-useful looping behavior esp. if using get_varTime.
    whereas we could instead do something intelligent. Options include:
        - dedicate the first N memmaps to the first N that we read.
        - maintain separate list of just the memmap file names
            count how many times we read each file;
            keep in memory the memmaps for the files we are reading more often.
        - some combination of the above ideas.

    allow for check_cache to propagate downwards throughout all calls to get_var.
        E.g. right now get_var(x, check_cache=False) will not check cache for x,
            however if it requires to get_var(y) it will still check cache for y.

"""

# import builtins
import resource
import warnings
import functools
import os
from collections import OrderedDict, namedtuple
import sys  # for debugging 'too many files' crash; will be removed in the future
import time  # for time profiling for caching
import weakref  # for refering to parent in cache without making circular reference.

# import local modules
from . import document_vars

# import external public modules
try:
    import numpy as np
except ImportError:
    warnings.warn('failed to import numpy; some functions in helita.sim.file_memory may crash')

# import internal modules
# from .fluid_tools import fluid_equals   # can't import this here, due to dependency loop:
                                        # bifrost imports file_memory
                                        # fluid_tools imports at_tools
                                        # at_tools import Bifrost_units from bifrost

# set defaults
## apparently there is good efficiency improvement even if we only remember the last few memmaps.
## NMLIM_ATTR is the name of the attr which will tell us max number of memmaps to remember.
NMLIM_ATTR     = 'N_memmap'  
MEMORY_MEMMAP  = '_memory_memmap'
MM_PERSNAP     = 'mm_persnap'
## hard limit on number of open files = limit set by system; cannot be changed.
_, HARD = resource.getrlimit(resource.RLIMIT_NOFILE)
## soft limit on number of open files = limit observed by programs; can be changed, must always be less than HARD.
SOFT_INCREASE = 1.2     # amount to increase soft limit, when increasing.     int -> add; float -> multiply.
MAX_SOFT      = int(min(1e6, 0.1 * HARD))  # we will never set the soft limit to a value larger than this.
SOFT_WARNING  = 8192    # if soft limit exceeds this value we will warn user every time we increase it.
SOFT_PER_OBJ  = 0.1     # limit number of open memmaps in one object to SOFT_PER_OBJ * soft limit.

HIDE_DECORATOR_TRACEBACKS = True  # whether to hide decorators from this file when showing error traceback.


DEBUG_MEMORY_LEAK = False  # whether to turn on debug messages to tell when Cache and/or EbysusData are deleted.
                           # There is currently a memory leak which seems unrelated to file_memory.py,
                           # because even with _force_disable_memory=True, the EbysusData objects
                           # are not actually being deleted when del is called.  - SE June 10, 2021
                           # E.g. dd = eb.EbysusData(...); del dd    --> dd.__del__() is not being called.
                           # This could be caused by some attribute of dd containing a pointer to dd.
                           # Those pointers should be replaced by weakrefs; see e.g. Cache class in this file.


''' --------------------- remember_and_recall() --------------------- '''

def remember_and_recall(MEMORYATTR, ORDERED=False, kw_mem=[]):
    '''wrapper which returns function but with optional args obj, MEMORYATTR.
    default obj=None, MEMORYATTR=MEMORYATTR.
    if obj is None, behavior is unchanged;
    else, remembers the values from reading files (by saving to the dict obj.MEMORYATTR),
          and returns those values instead of rereading files. (This improves efficiency.)

    track modification timestamp for file in memory lookup dict.
        this ensures if file is modified we will read the new file data.
    '''
    def decorator(f):
        @functools.wraps(f)
        def f_but_remember_and_recall(filename, *args, obj=None, MEMORYATTR=MEMORYATTR, kw_mem=kw_mem, **kwargs):
            '''if obj is None, simply does f(filename, *args, **kwargs).
            Else, recall or remember result, as appropriate.
                memory location is obj.MEMORYATTR[filename.lower()].
            kw_mem: list of strs (default [])
                for key in kw_mem, associate key kwargs[key] with uniqueness of result.
            '''
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            if getattr(obj, '_force_disable_memory', False):
                obj = None
            if obj is not None:
                if not hasattr(obj, '_recalled'):
                    obj._recalled = dict()
                if MEMORYATTR not in obj._recalled:
                    obj._recalled[MEMORYATTR] = 0
                if not hasattr(obj, MEMORYATTR):
                    setattr(obj, MEMORYATTR, dict())
                    memory = getattr(obj, MEMORYATTR)
                    memory['data'] = dict()
                    memory['len'] = 0
                    memory['recalled'] = 0
                    if ORDERED:
                        memory['order'] = OrderedDict()
                else:
                    memory = getattr(obj, MEMORYATTR)
                memdata = memory['data']
                if os.path.exists(filename):
                    timestamp = os.stat(filename).st_mtime   # timestamp of when file was last modified
                else:
                    timestamp = '???'
                # set filekey (key string for memory dict with filename keys)
                filekey   = filename.lower()   # this would be enough, but we remove common prefix for readability.
                if hasattr(obj, '_memory_filekey_fdir'):
                    _memory_filekey_fdir = obj._memory_filekey_fdir
                elif hasattr(obj, 'fdir'):
                    _memory_filekey_fdir = os.path.abspath(obj.fdir).lower()
                    obj._memory_filekey_fdir = _memory_filekey_fdir
                else:
                    _memory_filekey_fdir = os.path.abspath(os.sep)  # 'root directory' (plays nicely with relpath, below)
                filekey = os.path.relpath(filekey, _memory_filekey_fdir)
                # determine whether we have this (filename, timestamp, kwargs) in memory already.
                need_to_read = True
                existing_mid = None
                if filekey in memdata.keys():
                    memfile = memdata[filekey]
                    for mid, memdict in memfile['memdicts'].items():
                        # determine if the values of kwargs (which appear in kw_mem) match those in memdict.
                        kws_match = True
                        for key in kw_mem:
                            # if key (in kw_mem) appears in kwargs, it must appear in memdict and have matching value.
                            if key in kwargs.keys():
                                if key not in memdict.keys():
                                    kws_match = False
                                    break
                                elif kwargs[key] != memdict[key]:
                                    kws_match = False
                                    break
                        # if kwargs and timestamp match, we don't need to read; instead use value from memdict.
                        if kws_match and memdict['file_timestamp']==timestamp:
                            need_to_read = False
                            break
                    # if we found a memdict matching (filename, timestamp, kwargs),
                    ## we need to read if and only if memdict['value'] has been smashed.
                    if not need_to_read:
                        if 'value' not in memdict.keys():
                            need_to_read = True
                            existing_mid = mid      # mid is the unique index for this (timestamp, kwargs)
                                                    # combination, for this filekey. This allows to have
                                                    # a unique dict key which is (filekey, mid); and use
                                                    # (filekey, mid) to uniquely specify (file, timestamp, kwargs).
                else:
                    memdata[filekey] = dict(memdicts=dict(), mid_next=1) # mid_next is the next available mid.
                    memfile = memdata[filekey]
                # read file if necessary (and store result to memory)
                if need_to_read:
                    result  = f(filename, *args, **kwargs) # here is where we call f, if obj is not None.
                    if not existing_mid:
                        mid                 = memfile['mid_next']
                        memfile['mid_next'] = mid + 1
                        memdict = dict(value=result,                                   # value is in memdict
                                       file_timestamp=timestamp, mid=mid, recalled=0)  # << metadata about file, kwargs, etc
                        memdict.update(kwargs)                                         # << metadata about file, kwargs, etc
                        memfile['memdicts'][mid] = memdict    # store memdict in memory.
                    else:
                        mid                 = existing_mid
                        memdict             = memfile['memdicts'][mid]
                        memdict['value']    = result
                    memory['len']   += 1    # total number of 'value's stored in memory
                    if ORDERED:
                        memory['order'][(filekey, mid)] = None
                        # this is faster than a list due to the re-ordering of memory['order']
                        ## which occurs if we ever access the elements again.
                        ## Really, a dict is unnecessary, we just need a "hashed" list,
                        ## but we can abuse OrderedDict to get that.
                else:
                    memory['recalled'] += 1
                    memdict['recalled'] += 1
                    obj._recalled[MEMORYATTR] += 1
                    if ORDERED:
                        # move this memdict to end of order list; order is order of access.
                        memory['order'].move_to_end((filekey, memdict['mid']))
                # return value from memory
                return memdict['value']
            else:  # obj is None, so there is no memory, so we just call f and return the result.
                return f(filename, *args, **kwargs)           # here is where we call f, if obj is None.
        return f_but_remember_and_recall
    return decorator

''' --------------------- manage_memmaps() --------------------- '''

def get_nfiles_soft_limit():
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    return soft

def increase_soft_limit(soft=None, increase=SOFT_INCREASE):
    '''increase soft by increase (int -> add; float -> multiply).'''
    if soft is None: soft = get_nfiles_soft_limit()
    soft0 = soft
    if isinstance(increase, int):
        soft += increase
    elif isinstance(increase, float):
        soft = int(soft * increase)
    else:
        raise TypeError('invalid increase type! expected int or float but got {}'.format(type(increase)))
    increase_str = ' limit on number of simultaneous open files from {} to {}'.format(soft0, soft)
    if soft > MAX_SOFT:
        raise ValueError('refusing to increase'+increase_str+' because this exceeds MAX_SOFT={}'.format(MAX_SOFT))
    if soft > SOFT_WARNING:
        warnings.warn('increasing'+increase_str+'.')
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, HARD))

def manage_memmaps(MEMORYATTR, kw_mem=['dtype', 'order', 'offset', 'shape']):
    '''decorator which manages number of memmaps. f should at most add one memmap to memory.'''
    def decorator(f):
        @functools.wraps(f)
        def f_but_forget_memmaps_if_needed(*args, **kwargs):
            '''forget one memmap if there are too many in MEMORYATTR.
            determine what is "too many" via NMLIM_ATTR.

            Then return f(*args, **kwargs).
            '''
            # check if we need to forget a memmap; forget one if needd.
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            try:
                obj = kwargs['obj']
            except KeyError:
                obj = None
            if getattr(obj, '_force_disable_memory', False):
                obj = None
            if obj is not None:
                memory = getattr(obj, MEMORYATTR, None)
                if memory is not None:
                    # memory is a dict of {filekey: memdictlist}; each memdict in memdictlist stores one memmap.
                    soft = get_nfiles_soft_limit()
                    forget_one = False
                    val = getattr(obj, NMLIM_ATTR, -1)
                    if val == -1:   # we limit number of open memmaps based on limit for simultaneous open files.
                        if memory['len'] >= SOFT_PER_OBJ * soft:
                            try:
                                increase_soft_limit(soft)
                            except ValueError: # we are not allowed to increase soft limit any more.
                                warnings.warn('refusing to increase soft Nfile limit further than {}!'.format(soft))
                                forget_one = True
                    elif val < 0:
                        raise ValueError('obj.'+NMLIM_ATTR+'must be -1 or 0 or >0 but got {}'.format(val))
                    else:           # we limit number of open memmaps based on NMLIM_ATTR.
                        if memory['len'] >= val:
                            forget_one = True
                    if forget_one:
                        # forget oldest memmap.
                        ## ... TODO: possibly add a warning? It may be okay to be silent though.
                        filekey, mid = next(iter(memory['order'].keys()))
                        # commented lines for debugging 'too many files' crash; will be removed in the future:
                        #x = memory[filekey]['memdicts'][mid]['value']  # this is the memmap
                        #print('there are {} references to the map.'.format(sys.getrefcount(x)))
                        memdata = memory['data']
                        memdict = memdata[filekey]['memdicts'][mid]
                        del memdict['value']         # this is the memmap
                        #print('there are {} references to the map (after deleting dict)'.format(sys.getrefcount(x)))
                        #print('referrers are: ', referrers(x))
                        del memory['order'][(filekey, mid)]
                        memory['len'] -= 1
            # return f(*args, **kwargs)
            return f(*args, kw_mem=kw_mem, **kwargs)

        return f_but_forget_memmaps_if_needed
    return decorator

# for debugging 'too many files' crash; will be removed in the future:
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# for debugging 'too many files' crash; will be removed in the future:
def referrers(obj):
    return [namestr(refe, globals()) for refe in gc.get_referrers(obj)]


''' --------------------- cache --------------------- '''

CacheEntry = namedtuple('CacheEntry', ['value', 'metadata', 'id', 'nbytes', 'calctime', 'qtracking_state'],
                        defaults = [None, None, None, None, None, dict()])
#        value: value.
#     metadata: additional params which are associated with this value of var.
#           id: unique id associated to this var and cache_params for this cache.
#       nbytes: number of bytes in value
#     calctime: amount of time taken to calculate value.

def _fmt_SL(SL, sizing=2):
    '''pretty formatting for species,level'''
    if SL is None:
        totlen = len('(') + sizing + len(', ') + sizing + len(')')
        totlen = str(totlen)
        fmtstr = '{:^'+totlen+'s}'  #e.g. '{:8s}'
        return fmtstr.format(str(None))
    else:
        sizing = str(sizing)
        fmtnum = '{:'+sizing+'d}'  # e.g. '{:2d}'
        fmtstr = '('+fmtnum+', '+fmtnum+')'
        return fmtstr.format(SL[0], SL[1])

def _new_cache_entry_str_(x):
    '''new __str__ method for CacheEntry, which shows a much more readable format.
    To get the original (namedtuple-style) representation of CacheEntry object x, use repr(x).
    '''
    FMT_SNAP = '{:3d}'
    FMT_DATA = '{: .3e}'
    FMT_META = '{: .2e}'
    snap     = FMT_SNAP.format(x.metadata.get('snap', None))
    ifluid   = _fmt_SL(x.metadata.get('ifluid', None))
    jfluid   = _fmt_SL(x.metadata.get('jfluid', None))
    value = x.value
    valmin   = None if value is None else FMT_DATA.format(np.min(value))
    valmean  = None if value is None else FMT_DATA.format(np.mean(value))
    valmax   = None if value is None else FMT_DATA.format(np.max(value))
    nbytes   = FMT_META.format(x.nbytes)
    calctime = FMT_META.format(x.calctime)
    result = ('CacheEntryView(snap={snap:}, ifluid={ifluid:}, jfluid={jfluid:}, '
              'valmin={valmin:}, valmean={valmean:}, valmax={valmax:}, '
              'nbytes={nbytes:}, calctime={calctime:})'
             )
    result = result.format(
              snap=snap, ifluid=ifluid, jfluid=jfluid,
              valmin=valmin, valmean=valmean, valmax=valmax,
              nbytes=nbytes, calctime=calctime)
    return result
# actually overwrite the __str__ method for CacheEntry:
CacheEntry.__str__ = _new_cache_entry_str_

class Cache:
    '''cache results of get_var.
    can contain up to self.max_MB MB of data, and up to self.max_Narr entries.
    Deletes oldest entries first when needing to free up space.

    self.performance tells total number of times arrays have been recalled,
    and total amount of time saved (estimate based on time it took to read the first time.)
    (Note the time saved is usually an overestimate unless you have N_memmap=0.)

    self.contents() shows a human-readable view of cache contents.
    '''
    def __init__(self, obj=None, max_MB=10, max_Narr=20):
        '''initialize Cache.

        obj: None or object with _metadata() and _metadata_matches() methods.
            Cache remembers this obj and uses these methods, if possible.
            obj._metadata() must accept kwarg with_nfluid, and must return a dict.
            obj._metadata_matches() must take a single dict as input, and must return a bool.
        max_MB: 10 (default) or number
            maximum number of MB of data which cache is allowed to store at once.
        max_Narr: 20 (default) or number
            maximum number of arrays which cache is allowed to store at once.
        '''
        # set attrs which dictate max size of cache
        self.max_MB   = max_MB
        self.max_Narr = max_Narr
        # set parent, using weakref, to ensure we don't keep parent alive just because Cache points to it.
        self.parent   = (lambda: None) if (obj is None) else weakref.ref(obj)
        # initialize self.performance, which will track the performance of Cache.
        self.performance = dict(time_saved_estimate=0, N_recalled=0, N_recalled_unknown_time_savings=0)
        # initialize attrs for internal use.
        self._content = dict()
        self._next_cacheid = 0   # unique id associated to each cache entry (increases by 1 each time)
        self._order   = []  # list of (var, id)
        self._nbytes  = 0   # number of bytes of data stored in self.
        self.debugging = False   # if true, print some helpful debugging statements.

    def get_parent_attr(self, attr, default=None):
        '''return getattr(self.parent(), attr, default)
        Caution: to ensure weakref is useful and no circular reference is created,
            make sure to not save the result of get_parent_attr as an attribute of self.
        '''
        return getattr(self.parent(), attr, None)

    def _metadata(self, *args__parent_metadata, **kw__parent_metadata):
        '''returns self.parent()._metadata() if it exists; else None.'''
        get_metadata_func = self.get_parent_attr('_metadata')
        if get_metadata_func is not None:
            return get_metadata_func(*args__parent_metadata, **kw__parent_metadata)
        else:
            return None

    def get_metadata(self, metadata=None, obj=None, with_nfluid=2):
        '''returns metadata, given args.

        metadata: None or dict
            if not None, return this value immediately.
        obj: None or object with _metadata() method which returns dict
            if not None, return obj._metadata(with_nfluid=with_nfluid)
        with_nfluid: 2, 1, or 0
            if obj is not None, with_nfluid is passed to obj._metadata.
            else, with_nfluid is passed to self.parent_get_metadata.

        This method's default behavior (i.e. behavior when called with no arguments)
        is to return self.parent_get_metadata(with_nfluid=2).
        '''
        if metadata is not None:
            return metadata
        if obj is not None:
            return obj._metadata(with_nfluid=with_nfluid)
        parent_metadata = self._metadata(with_nfluid=with_nfluid)
        if parent_metadata is not None:
            return parent_metadata
        raise ValueError('Expected non-None metadata, obj, or self.parent_get_metadata, but all were None.')

    def _metadata_matches(self, cached_metadata, metadata=None, obj=None):
        '''return whether metadata matches cached_metadata.
        
        if self has parent and self.parent() has _metadata_matches method:
            return self.parent()._metadata_matches(cached_metadata)
        else:
            return _dict_equals(cached_metadata, self.get_metadata(metadata, obj))
        '''
        metadata_matches_func = self.get_parent_attr('_metadata_matches')
        if metadata_matches_func is not None:
            return metadata_matches_func(cached_metadata)
        else:
            return _dict_equals(cached_metadata, self.get_metadata(metadata=metadata, obj=obj))

    def get(self, var, metadata=None, obj=None):
        '''return entry associated with var and metadata in self,
        if such an entry exists. Else, return empty CacheEntry.

        if Cache was initialized with obj, use obj._metadata() to 
        var: string
        metadata: None (default) or dict
            check that this agrees with cached metadata before returning result.
        obj: None (default) or EbysusData object
            if not None, use obj to determine params and fluids.

        if metadata and obj are None, tries to use metadata from self.parent().
        '''
        try:
            var_cache_entries = self._content[var]
        except KeyError:
            if self.debugging >= 2: print(' > Getting {:15s}; var not found in cache.'.format(var))
            return CacheEntry(None)   # var is not in self.
        # else (var is in self):
        for entry in var_cache_entries:
            if self._metadata_matches(entry.metadata, metadata=metadata, obj=obj):
                # we found a match! So, return this entry (after doing some bookkeeping).
                if self.debugging >= 1: print(' -> Loaded   {:^15s} -> {}'.format(var, entry))
                ## update performance tracker.
                self._update_performance_tracker(entry)
                ## update QUANT_SELECTED in self.parent()
                parent = self.parent()
                if parent is not None:
                    document_vars.restore_quant_tracking_state(parent, entry.qtracking_state)
                return entry
        # else (var is in self but not associated with this metadata):
        if self.debugging >= 2: print(' > Getting {:15s}, var in cache but not with this metadata.'.format(var))
        return CacheEntry(None)

    def cache(self, var, val, metadata=None, obj=None, with_nfluid=2, calctime=None, from_internal=False):
        '''add var with value val (and associated with cache_params) to self.'''
        if self.debugging >= 2: print(' < Caching {:15s}; with_nfluid={}'.format(var, with_nfluid))
        val = np.array(val, copy=True, subok=True)  # copy ensures value in cache isn't altered even if val array changes.
        nbytes = val.nbytes
        self._nbytes += nbytes
        metadata = self.get_metadata(metadata=metadata, obj=obj, with_nfluid=with_nfluid)
        quant_tracking_state = document_vars.get_quant_tracking_state(self.parent(), from_internal=from_internal)
        entry  = CacheEntry(value=val, metadata=metadata,
                            id=self._take_next_cacheid(), nbytes=nbytes, calctime=calctime,
                            qtracking_state=quant_tracking_state)
        if self.debugging >= 1: print(' <- Caching {:^15s} <- {}'.format(var, entry))
        if var in self._content.keys():
            self._content[var] += [entry]
        else:
            self._content[var] = [entry]
        self._order += [(var, entry.id)]
        self._shrink_cache_as_needed()

    def remove_one_entry(self, id=None):
        '''removes the oldest entry in self. returns id of entry removed.
        if id is not None, instead removes the entry with id==id.
        '''
        if id is None:
            oidx = 0
            var, eid = self._order[oidx]
        else:
            try:
                oidx, (var, eid) = next(  ( (i, x) for i, x in enumerate(self._order) if x[1]==id)  )
            except StopIteration:
                raise KeyError('id={} not found in cache {}'.format(id, self))
        var_entries = self._content[var]
        i = next((i for i, entry in enumerate(var_entries) if entry.id == eid))
        self._nbytes -= var_entries[i].nbytes
        del var_entries[i]
        del self._order[oidx]
        return eid

    def __repr__(self):
        '''pretty print of self'''
        s = '<{self:} totaling {MB:0.3f} MB, containing {N:} cached values from {k:} vars: {vars:}>'
        vars = list(self._content.keys())
        if len(vars) > 20:  # then we will show only the first 20.
            svars = '[' + ', '.join(vars[:20]) + ', ...]'
        else:
            svars = '[' + ', '.join(vars) + ']'
        return s.format(self=object.__repr__(self), MB=self._nMB(), N=len(self._order), k=len(vars), vars=svars)

    def contents(self):
        '''pretty display of contents (as CacheEntryView tuples).
        To access the content data directly, use self._content.
        '''
        result = dict()
        for var, content in self._content.items():
            result[var] = []
            for entry in content:
                result[var] += [str(entry)]
        return result

    def _update_performance_tracker(self, entry):
        '''update self.performance as if we just got entry from cache once.'''
        self.performance['N_recalled'] += 1
        savedtime = entry.calctime
        if savedtime is None:
            self.performance['N_recalled_unknown_time_savings'] += 1
        else:
            self.performance['time_saved_estimate'] += savedtime

    def _take_next_cacheid(self):
        result = self._next_cacheid
        self._next_cacheid += 1
        return result

    def _max_nbytes(self):
        return self.max_MB * 1024 * 1024

    def _nMB(self):
        return self._nbytes / (1024 * 1024)

    def _shrink_cache_as_needed(self):
        '''shrink cache to stay within limits of number of entries and amount of data.'''
        while len(self._order) > self.max_Narr:
            self.remove_one_entry()
        max_nbytes = self._max_nbytes()
        while self._nbytes > max_nbytes:
            self.remove_one_entry()

    def is_NoneCache(self):
        '''return if self.max_MB <= 0 or self.max_Narr <= 0'''
        return (self.max_MB <= 0 or self.max_Narr <= 0)

    if DEBUG_MEMORY_LEAK:
        def __del__(self):
            print('deleted {}'.format(self))


def with_caching(check_cache=True, cache=False, cache_with_nfluid=None):
    '''decorate function so that it does caching things.

    cache, check_cache, and nfluid values passed to with_caching()
    will become the _default_ values of these kwargs for the function
    which is being decorated (but other values for these kwargs
    can still be passed to that function, later).

    cache: whether to store result in obj.cache
    check_cache: whether to try to get result from obj.cache if it exists there.
    cache_with_nfluid - None (default), 0, 1, or 2
        if not None, cache result and associate it with this many fluids.
        0 -> neither; 1 -> just ifluid; 2 -> both ifluid and jfluid.
    '''
    def decorator(f):
        @functools.wraps(f)
        def f_but_caching(obj, var, *args_f,
                          check_cache=check_cache, cache=cache, cache_with_nfluid=cache_with_nfluid,
                          **kwargs_f):
            '''do f(obj, *args_f, **kwargs_f) but do caching things as appropriate,
            i.e. check cache first (if check_cache) and store result (if cache).
            '''
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            if getattr(obj, '_force_disable_memory', False):
                cache = check_cache = False

            val = None
            if (not getattr(obj, 'do_caching', True)) or (not hasattr(obj, 'cache')) or (obj.cache.is_NoneCache()):
                cache = check_cache = False
            elif cache_with_nfluid is not None:
                cache = True
            if (cache or check_cache):
                track_timing = True
                # check cache for result (if check_cache==True)
                if check_cache:
                    entry = obj.cache.get(var)
                    val = entry.value
                    if cache and (val is not None):
                        # remove entry from cache to prevent duplicates (because we will re-add entry soon)
                        obj.cache.remove_one_entry(id=entry.id)
                        # use timing from original entry
                        track_timing = False
                        calctime = entry.calctime
                if cache and track_timing:
                    now = time.time()   # track timing, so we can estimate how much time cache is saving.
            # calculate result (if necessary)
            if val is None:
                val = f(obj, var, *args_f, **kwargs_f)
            # save result to obj.cache (if cache==True)
            if cache:
                if track_timing:
                    calctime = time.time() - now
                obj.cache.cache(var, val, with_nfluid=cache_with_nfluid, calctime=calctime)
            # return result
            return val
        return f_but_caching
    return decorator


class Caching():
    '''context manager which lets you do caching by setting self.result to a value.'''
    def __init__(self, obj, nfluid=None):
        self.obj    = obj
        self.nfluid = nfluid

    def __enter__(self):
        self.caching = (getattr(self.obj, 'do_caching', True)) \
                        and (hasattr(self.obj, 'cache'))       \
                        and (not self.obj.cache.is_NoneCache())
        if self.caching:
            self.start = time.time()
            self.metadata = self.obj._metadata(with_nfluid=self.nfluid)

        def _cache_it(var, value, restart_timer=True):
            '''save this result to cache.'''
            if not self.caching:
                return
            #else
            calctime = time.time() - self.start
            self.obj.cache.cache(var, value, metadata=self.metadata, calctime=calctime, from_internal=True)
            if restart_timer:
                self.start = time.time() # restart the clock.

        return _cache_it

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def _dict_matches(A, B, subset_ok=True, ignore_keys=[]):
    '''returns whether A matches B for dicts A, B.

    A "matches" B if for all keys in A, A[key] == B[key].

    subset_ok: True (default) or False
        if False, additionally "A matches B" requires A.keys() == B.keys()
    ignore_keys: list (default [])
        these keys are never checked; A[key] need not equal B[key] for key in ignore_keys.

    This function is especially useful when checking dicts which may contain numpy arrays,
    because numpy arrays override __equals__ to return an array instead of True or False.
    '''
    ignore_keys = set(ignore_keys)
    keysA = set(A.keys()) - ignore_keys
    keysB = set(B.keys()) - ignore_keys
    if not subset_ok:
        if not keysA == keysB:
            return False
    for key in keysA:
        eq = (A[key] == B[key])
        if isinstance(eq, np.ndarray):
            if not np.all(eq):
                return False  
        elif eq == False:
            return False
        elif eq == True:
            pass #continue on to next key.
        else:
            raise ValueError("Object equality was not boolean nor np.ndarray. Don't know what to do. " + \
                             "Objects = {:}, {:}; (x == y) = {:}; type((x==y)) = {:}".format(            \
                                     A[key], B[key],         eq,              type(eq)      )     )
    return True

def _dict_equals(A, B, ignore_keys=[]):
    '''returns whether A==B for dicts A, B.
    Even works if some contents are numpy arrays.
    '''
    return _dict_matches(A, B, subset_ok=False, ignore_keys=ignore_keys)

def _dict_is_subset(A, B, ignore_keys=[]):
    '''returns whether A is a subset of B, i.e. whether for all keys in A, A[key]==B[key].
    Even works if some contents are numpy arrays.
    '''
    return _dict_matches(A, B, subset_ok=True, ignore_keys=ignore_keys)