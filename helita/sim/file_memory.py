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
"""

# import builtins
import resource
import warnings
import functools
import os

# set defaults
## apparently there is good efficiency improvement even if we only remember the last few memmaps.
## NMLIM_ATTR is the name of the attr which will tell us max number of memmaps to remember.
NMLIM_ATTR     = 'N_memmap'   
## hard limit on number of open files = limit set by system; cannot be changed.
_, HARD = resource.getrlimit(resource.RLIMIT_NOFILE)
## soft limit on number of open files = limit observed by programs; can be changed, must always be less than HARD.
SOFT_INCREASE = 2.0     # amount to increase soft limit, when increasing.     int -> add; float -> multiply.
MAX_SOFT      = int(min(1e5, 0.1 * HARD))  # we will never set the soft limit to a value larger than this.
SOFT_WARNING  = 8192    # if soft limit exceeds this value we will warn user every time we increase it.
SOFT_PER_OBJ  = 0.1     # limit number of open memmaps in one object to SOFT_PER_OBJ * soft limit.


''' --------------------- remember_and_recall() --------------------- '''

def remember_and_recall(MEMORYATTR, ORDERED=False):
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
        def f_but_remember_and_recall(filename, *args, obj=None, MEMORYATTR=MEMORYATTR, kw_mem=[], **kwargs):
            '''if obj is None, simply does f(filename, *args, **kwargs).
            Else, recall or remember result, as appropriate.
                memory location is obj.MEMORYATTR[filename.lower()].
            kw_mem: list of strs (default [])
                for key in kw_mem, associate key kwargs[key] with uniqueness of result.
            '''
            if obj is not None:
                if not hasattr(obj, MEMORYATTR):
                    setattr(obj, MEMORYATTR, dict())
                    memory = getattr(obj, MEMORYATTR)
                    memory['len'] = 0
                    memory['recalled'] = 0
                    if ORDERED:
                        memory['order'] = []
                else:
                    memory = getattr(obj, MEMORYATTR)
                if os.path.exists(filename):
                    timestamp = os.stat(filename).st_mtime   # timestamp of when file was last modified
                else:
                    timestamp = '???'
                filekey   = filename.lower()
                # determine whether we have this (filename, timestamp, kwargs) in memory already.
                need_to_read = True
                if filekey in memory.keys():
                    mem = memory[filekey]
                    for mid, memdict in mem['memdicts'].items():
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
                else:
                    memory[filekey] = dict(memdicts=dict(), read_n=0) #read_n is number of times this file has been read.
                    mem = memory[filekey]
                # read file if necessary (and store result to memory)
                if need_to_read:
                    result  = f(filename, *args, **kwargs)    # here is where we call f, if obj is not None.
                    read_n  = mem['read_n'] + 1
                    mem['read_n'] = read_n
                    memdict = dict(value=result, file_timestamp=timestamp, read_n=read_n)
                    memdict.update(kwargs)
                    mem['memdicts'][read_n] = memdict
                    memory['len']   += 1    # total number of 'value's in memory
                    if ORDERED:
                        memory['order'] += [(filekey, read_n)]
                else:
                    memory['recalled'] += 1
                # return value from memory
                return memdict['value']
            else:
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
            try:
                obj = kwargs['obj']
            except KeyError:
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
                                print('cannot increase soft limit further!', soft)
                                forget_one = True
                    elif val < 0:
                        raise ValueError('obj.'+NMLIM_ATTR+'must be -1 or 0 or >0 but got {}'.format(val))
                    else:           # we limit number of open memmaps based on NMLIM_ATTR.
                        if memory['len'] >= val:
                            forget_one = True
                    if forget_one:
                        # forget oldest memmap.
                        ## ... TODO: possibly add a warning? It may be okay to be silent though.
                        filekey, mid = memory['order'][0]
                        del memory[filekey]['memdicts'][mid]['value']  # this is the memmap
                        del memory[filekey]['memdicts'][mid]           # this is the dict that had the memmap
                        del memory['order'][0]
                        memory['len'] -= 1
            # return f(*args, **kwargs)
            return f(*args, kw_mem=kw_mem, **kwargs)

        return f_but_forget_memmaps_if_needed
    return decorator
