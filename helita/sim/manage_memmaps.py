"""
created by Sam Evans on Apr 12 2021 

purpose:
limit number of open memmaps, to avoid crashing due to "too many files open"
"""

# import builtins
import resource
import warnings
import functools

# OrderedDict class which keeps track of order keys were added
from collections import OrderedDict
class LastUpdatedOrderedDict(OrderedDict):
    '''Store items in the order the keys were last added. Newest -> end of list.'''
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)

# set defaults
## apparently there is good efficiency improvement even if we only remember the last few memmaps.
## NMLIM_ATTR is the name of the attr which will tell us max number of memmaps to remember.
NMLIM_ATTR     = 'N_memmap'   
## hard limit on number of open files = limit set by system; cannot be changed.
_, HARD = resource.getrlimit(resource.RLIMIT_NOFILE)
## soft limit on number of open files = limit observed by programs; can be changed, must always be less than HARD.
SOFT_INCREASE = 2.0     # amount to increase soft limit, when increasing.     int -> add; float -> multiply.
MAX_SOFT      = int(min(1e3, 0.1 * HARD))  # we will never set the soft limit to a value larger than this.
SOFT_WARNING  = 8192    # if soft limit exceeds this value we will warn user every time we increase it.
SOFT_PER_OBJ  = 0.1     # limit number of open memmaps in one object to SOFT_PER_OBJ * soft limit.


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

def get_n_memmap(obj, soft=None):
    if soft is not None: soft = get_nfiles_soft_limit()
    val = getattr(obj, NMLIM_ATTR, -1)
    if val == -1:
        return SOFT_PER_OBJ * soft  # limit based on the max number of files we are allowed to have open at once.
    elif val < 0:
        raise ValueError('obj.'+NMLIM_ATTR+'must be -1 or 0 or >0 but got {}'.format(val))
    else:
        return val   # limit is whatever value was entered for NMLIM_ATTR.

def manage_memmaps(MEMORYATTR):
    '''decorator which manages number of memmaps. f should at most add one memmap to memory.'''
    def decorator(f):
        @functools.wraps(f)
        def f_but_forget_memmaps_if_needed(*args, **kwargs):
            '''forget one memmap if there are too many in MEMORYATTR.
            determine what is "too many" via NMLIM_ATTR.

            Then return f(*args, **kwargs).
            '''
            # forget a memmap if necessary.
            try:
                obj = kwargs['obj']
            except KeyError:
                obj = None
            if obj is not None:
                memory = getattr(obj, MEMORYATTR, None)
                if memory is not None:
                    soft   = get_nfiles_soft_limit()
                    forget_one = False
                    val = getattr(obj, NMLIM_ATTR, -1)
                    if val == -1:   # we limit number of open memmaps based on limit for simultaneous open files.
                        if len(memory) >= SOFT_PER_OBJ * soft:
                            try:
                                increase_soft_limit(soft)
                            except ValueError: # we are not allowed to increase soft limit any more.
                                forget_one = True
                    elif val < 0:
                        raise ValueError('obj.'+NMLIM_ATTR+'must be -1 or 0 or >0 but got {}'.format(val))
                    else:           # we limit number of open memmaps based on NMLIM_ATTR.
                        if len(memory) >= val:
                            forget_one = True
                    if forget_one:
                        # ... possibly add a warning to this line? It may be okay to be silent though.
                        key = next(iter(memory.keys()))
                        del memory[key]  # forget oldest memmap.
            # return f(*args, **kwargs)
            return f(*args, **kwargs)

        return f_but_forget_memmaps_if_needed
    return decorator
