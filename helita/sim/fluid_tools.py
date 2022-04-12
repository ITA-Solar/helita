"""
created by Sam Evans on Apr 19 2021 

purpose:
    - tools for fluids in ebysus.py
"""

# import built-ins
import functools
import warnings
import itertools

# import external private modules
try:
  from at_tools import fluids as fl
except ImportError:
  fl = tools.ImportFailed('at_tools.fluids')

# set defaults
HIDE_DECORATOR_TRACEBACKS = True  # whether to hide decorators from this file when showing error traceback.

## list of functions from fluid_tools which will be set as methods of the Multifluid class.
## for example, EbysusData inherits from Multifluid, so if Multifluid gets get_mass, then:
##   for dd=EbysusData(...), dd.get_mass(*args, **kw) == fluid_tools.get_mass(dd, *args, **kw).
MULTIFLUID_FUNCS = \
    ['set_mf_fluid', 'set_mfi', 'set_mfj', 'set_fluids',
    'get_species_name', 'get_fluid_name', 'get_mass', 'get_charge',
    'get_cross_tab', 'get_cross_sect', 'get_coll_type',
    'i_j_same_fluid', 'iter_fluid_SLs']

''' --------------------- setting fluids --------------------- '''

# NOTE: these functions are largely obsolete, now.
## Thanks to the "magic" of property(), doing something like obj.ifluid=(1,2)
## will effectively set mf_ispecies and mf_ilevel appropriately.
## And, reading something like obj.ifluid will give values (obj.mf_ispecies, obj.mf_ilevel)
# However, we cannot delete these functions, for historical reasons.
# And, maybe they are still useful thanks to the kwarg interpretation in set_fluids.

def set_mf_fluid(obj, species=None, level=None, i='i'):
    '''sets obj.mf_{i}species and obj.mf_{i}level.
    species, level: None or int
        None -> don't change obj.mf_{i}species, mf_{i}level.
        ints -> set mf_{i}species=species, mf_{i}level=level.
    '''
    setattr(obj, 'mf_'+i+'species', species)
    setattr(obj, 'mf_'+i+'level', level)

def set_mfi(obj, mf_ispecies=None, mf_ilevel=None):
    return obj.set_mf_fluid(mf_ispecies, mf_ilevel, 'i')
set_mfi.__doc__ = set_mf_fluid.__doc__.format(i='i')

def set_mfj(obj, mf_jspecies=None, mf_jlevel=None):
    return obj.set_mf_fluid(mf_jspecies, mf_jlevel, 'j')
set_mfj.__doc__ = set_mf_fluid.__doc__.format(i='j')

def set_fluids(obj, **kw__fluids):
    '''interprets kw__fluids then sets them using set_mfi and set_mfj.
    returns (ifluid, jfluid).
    '''
    (si, li, sj, lj) = _interpret_kw_fluids(**kw__fluids)
    obj.set_mfi(si, li)
    obj.set_mfj(sj, lj)
    return (obj.ifluid, obj.jfluid)

''' --------------------- fluid kwargs --------------------- '''

def _interpret_kw_fluids(mf_ispecies=None, mf_ilevel=None, mf_jspecies=None, mf_jlevel=None,
                         ifluid=None, jfluid=None, iSL=None, jSL=None,
                         iS=None, iL=None, jS=None, jL=None,
                         **kw__None):
    '''interpret kwargs entered for fluids. Returns (mf_ispecies, mf_ilevel, mf_jspecies, mf_jlevel).
    kwargs are meant to be shorthand notation. If conflicting kwargs are entered, raise ValueError.
    **kw__None are ignored; it is part of the function def'n so that it will not break if extra kwargs are entered.
    Meanings for non-None kwargs (similar for j, only writing for i here):
        mf_ispecies, mf_ilevel = ifluid
        mf_ispecies, mf_ilevel = iSL
        mf_ispecies, mf_ilevel = iS, iL
    Examples:
        These all return (1,2,3,4) (they are equivalent):
            _interpret_kw_fluids(mf_ispecies=1, mf_ilevel=2, mf_jspecies=3, mf_jlevel=4)
            _interpret_kw_fluids(ifluid=(1,2), jfluid=(3,4))
            _interpret_kw_fluids(iSL=(1,2), jSL=(3,4))
            _interpret_kw_fluids(iS=1, iL=2, jS=3, jL=4)
        Un-entered fluids will be returned as None:
            _interpret_kw_fluids(ifluid=(1,2))
            >> (1,2,None,None)
        Conflicting non-None kwargs will cause ValueError:
            _interpret_kw_fluids(mf_ispecies=3, ifluid=(1,2))
            >> ValueError('mf_ispecies (==3) was incompatible with ifluid[0] (==1)')
            _interpret_kw_fluids(mf_ispecies=1, ifluid=(1,2))
            >> (1,2,None,None)
    '''
    si, li = _interpret_kw_fluid(mf_ispecies, mf_ilevel, ifluid, iSL, iS, iL, i='i')
    sj, lj = _interpret_kw_fluid(mf_jspecies, mf_jlevel, jfluid, jSL, jS, jL, i='j')
    return (si, li, sj, lj)

def _interpret_kw_ifluid(mf_ispecies=None, mf_ilevel=None, ifluid=None, iSL=None, iS=None, iL=None, None_ok=True):
    '''interpret kwargs entered for ifluid. See _interpret_kw_fluids for more documentation.'''
    return _interpret_kw_fluid(mf_ispecies, mf_ilevel, ifluid, iSL, iS, iL, None_ok=None_ok, i='i')

def _interpret_kw_jfluid(mf_jspecies=None, mf_jlevel=None, jfluid=None, jSL=None, jS=None, jL=None, None_ok=True):
    '''interpret kwargs entered for jfluid. See _interpret_kw_fluids for more documentation.'''
    return _interpret_kw_fluid(mf_jspecies, mf_jlevel, jfluid, jSL, jS, jL, None_ok=None_ok, i='j')

def _interpret_kw_fluid(mf_species=None, mf_level=None, fluid=None, SL=None, S=None, L=None, i='', None_ok=True):
    '''interpret kwargs entered for fluid. Returns (mf_ispecies, mf_ilevel).
    See _interpret_kw_fluids for more documentation.
    i      : 'i', or 'j'; Used to make clearer error messages, if entered.
    None_ok: True (default) or False;
        whether to allow answer of None or species and/or level.
        if False and species and/or level is None, raise TypeError.
    '''
    s  , l   = None, None
    kws, kwl = '', ''
    errmsg = 'Two incompatible fluid kwargs entered! {oldkw:} and {newkw:} must be equal ' + \
                 '(unless one is None), but got {oldkw:}={oldval:} and {newkw:}={newval:}'
    def set_sl(news, newl, newkws, newkwl, olds, oldl, oldkws, oldkwl, i):
        newkws, newkwl = newkws.format(i), newkwl.format(i)
        if (olds is not None):
            if (news is not None):
                if (news != olds):
                    raise ValueError(errmsg.format(newkw=newkws, newval=news, oldkw=oldkws, oldval=olds))
            else:
                news = olds
        if (oldl is not None):
            if (newl is not None):
                if (newl != oldl):
                    raise ValueError(errmsg.format(newkw=newkwl, newval=newl, oldkw=oldkwl, oldval=oldl))
            else:
                newl = oldl
        return news, newl, newkws, newkwl

    if fluid is None: fluid = (None, None)
    if SL    is None: SL    = (None, None)
    s, l, kws, kwl = set_sl(mf_species, mf_level, 'mf_{:}species', 'mf_{:}level', s, l, kws, kwl, i)
    s, l, kws, kwl = set_sl(fluid[0]  , fluid[1], '{:}fluid[0]'  , '{:}fluid[1]', s, l, kws, kwl, i)
    s, l, kws, kwl = set_sl(SL[0]     , SL[1]   , '{:}SL[0]'     , '{:}SL[1]'   , s, l, kws, kwl, i)
    s, l, kws, kwl = set_sl(S         , L       , '{:}S'         , '{:}L'       , s, l, kws, kwl, i)
    if not None_ok:
        if s is None or l is None:
            raise TypeError('{0:}species and {0:}level cannot be None, but got: '.format(i) +
                            'mf_{0:}species={1:}; mf_{0:}level={2:}.'.format(i, s, l))
    return s, l


''' --------------------- fluid SL context managers --------------------- '''

class _MaintainingFluids():
    '''context manager which restores ifluid and jfluid to original values, upon exit.

    Example:
    dd = EbysusData(...)
    dd.set_mfi(2,3)
    print(dd.ifluid)  #>> (2,3)
    with _MaintainingFluids(dd):
        print(dd.ifluid)  #>> (2,3)
        dd.set_mfi(4,5)
        print(dd.ifluid)  #>> (4,5)
    print(dd.ifluid)  #>> (2,3)
    '''
    def __init__(self, obj):
        self.obj = obj
        self.orig_ifluid = obj.ifluid
        self.orig_jfluid = obj.jfluid

    def __enter__ (self):
        pass

    def __exit__ (self, exc_type, exc_value, traceback):
        self.obj.set_mfi(*self.orig_ifluid)
        self.obj.set_mfj(*self.orig_jfluid)

_MaintainFluids = _MaintainingFluids   # alias

class _UsingFluids(_MaintainingFluids):
    '''context manager for using fluids, but ending up with the same ifluid & jfluid at the end.
    upon enter, set fluids, based on kw__fluids.
    upon exit, restore original fluids.

    Example:
    dd = EbysusData(...)
    dd.set_mfi(1,1)
    print(dd.ifluid)  #>> (1,1)
    with _UsingFluids(dd, ifluid=(2,3)):
        print(dd.ifluid)  #>> (2,3)
        dd.set_mfi(4,5)
        print(dd.ifluid)  #>> (4,5)
    print(dd.ifluid)  #>> (1,1)
    '''
    def __init__(self, obj, **kw__fluids):
        _MaintainingFluids.__init__(self, obj)
        (si, li, sj, lj) = _interpret_kw_fluids(**kw__fluids)
        self.ifluid = (si, li)
        self.jfluid = (sj, lj)

    def __enter__ (self):
        self.obj.set_mfi(*self.ifluid)
        self.obj.set_mfj(*self.jfluid)

    # __exit__ is inheritted from MaintainingFluids.

_UseFluids = _UsingFluids  # alias

def maintain_fluids(f):
    '''decorator version of _MaintainFluids. first arg of f must be an EbysusData object.'''
    @functools.wraps(f)
    def f_but_maintain_fluids(obj, *args, **kwargs):
        __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
        with _MaintainingFluids(obj):
            return f(obj, *args, **kwargs)
    return f_but_maintain_fluids

def use_fluids(**kw__fluids):
    '''returns decorator version of _UseFluids. first arg of f must be an EbysusData object.'''
    def decorator(f):
        @functools.wraps(f)
        def f_but_use_fluids(obj, *args, **kwargs):
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            with _UsingFluids(obj, **kw__fluids):
                return f(obj, *args, **kwargs)
        return f_but_use_fluids
    return decorator


''' --------------------- iterators over fluids --------------------- '''

def fluid_pairs(fluids, ordered=False, allow_same=False):
    '''returns an iterator over fluids of obj.

    ordered: False (default) or True
        False -> (A,B) and (B,A) will be yielded separately.
        True  -> (A,B) will be yielded, but (B,A) will not.
    allow_same: False (default) or True
        False -> (A,A) will never be yielded.
        True  -> (A,A) will be yielded.

    This function just returns a combinatoric iterators from itertools.
    defaults lead to calling itertools.permutations(fluids)

    Example:
    for (ifluid, jfluid) in fluid_pairs([(1,2),(3,4),(5,6)], ordered=True, allow_same=False):
        print(ifluid, jfluid, end=' | ')
    # >> (1, 2) (3, 4) | (1, 2) (5, 6) | (3, 4) (5, 6) | 
    '''
    if       ordered and     allow_same: return itertools.combinations_with_replacement(fluids, 2)
    elif     ordered and not allow_same: return itertools.combinations(fluids, 2)
    elif not ordered and not allow_same: return itertools.permutations(fluids, 2)
    elif not ordered and     allow_same: return itertools.product(fluids, repeat=2)
    assert False #we should never reach this line...

def iter_fluid_SLs(dd, with_electrons=True):
    '''returns an iterator over the fluids of dd, and electrons.
    yields SL pairs; NOT at_tools.fluids.Fluid objects!
    example: list(iter_fluids(dd)) = [(-1,0), (1,1), (1,2)].
    '''
    if with_electrons:
        yield (-1,0)
    for fluid in fl.Fluids(dd=dd):
        yield fluid.SL

''' --------------------- compare fluids --------------------- '''

def i_j_same_fluid(obj):
    '''returns whether obj.ifluid and obj.jfluid represent the same fluid.'''
    return fluid_equals(obj.ifluid, obj.jfluid)

def fluid_equals(iSL, jSL):
    '''returns whether iSL and jSL represent the same fluid.'''
    if iSL[0] < 0 and jSL[0] < 0:
        return True
    else:
        return (iSL == jSL)

''' --------------------- small helper functions --------------------- '''
# for each of these functions, obj should be an EbysusData object.

def get_species_name(obj, specie):
    '''return specie's name: 'e' for electrons; element (atomic symbol) for other fluids.'''
    if specie < 0:
        return 'e'
    else:
        return obj.att[specie].params.element

def get_fluid_name(obj, fluid):
    '''return fluid's name: 'e-' for electrons; element & ionization for other fluids (e.g. 'H II').
    fluid can be at_tools.fluids.Fluid object, (species, level) pair, or -1 (for electrons).
    '''
    try:
        return fluid.name
    except AttributeError:
        try:
            specie = fluid[0]
            electrons_or_bust = False
        except TypeError:
            specie = fluid
            if not (specie < 0):
                errmsg_badfluid = ('Expected at_tools.fluids.Fluid object or (species, level) for fluid, '
                                   'but got fluid = {}'.format(fluid))
                raise TypeError(errmsg_badfluid)
        if specie < 0:
            return 'e-'
        else:
            return fl.Fluids(dd=obj)[fluid].name

def get_mass(obj, specie, units='amu'):
    '''return specie's mass [units]. default units is amu.
    units: one of: ['amu', 'g', 'kg', 'cgs', 'si', 'simu']. Default 'amu'
        'amu'        -> mass in amu.    For these units, mH ~= 1
        'g' or 'cgs' -> mass in grams.  For these units, mH ~= 1.66E-24
        'kg' or 'si' -> mass in kg.     For these units, mH ~= 1.66E-27
        'simu'       -> mass in simulation units.

    '''
    # if specie is actually (spec, level) return get_mass(obj, spec) instead.
    try:
        specie = next(iter(specie))
    except TypeError:
        pass
    else:
        return get_mass(obj, specie, units=units)
    units = units.lower()
    VALID_UNITS = ['amu', 'g', 'kg', 'cgs', 'si', 'simu']
    assert units in VALID_UNITS, "Units invalid; got units={}".format(units)
    if specie < 0:
        # electron
        if units == 'amu':
            return obj.uni.m_electron / obj.uni.amu
        elif units in ['g', 'cgs']:
            return obj.uni.m_electron
        elif units in ['kg', 'si']:
            return obj.uni.msi_e
        else: # units == 'simu'
            return obj.uni.simu_m_e
    else:
        # not electron
        m_amu = obj.att[specie].params.atomic_weight
        if units == 'amu':
            return m_amu
        elif units in ['g', 'cgs']:
            return m_amu * obj.uni.amu
        elif units in ['kg', 'si']:
            return m_amu * obj.uni.amusi
        else: # units == 'simu'
            return m_amu * obj.uni.simu_amu

def get_charge(obj, SL, units='e'):
    '''return the charge fluid SL in [units]. default is elementary charge units.
    units: one of ['e', 'elementary', 'esu', 'c', 'cgs', 'si', 'simu']. Default 'elementary'.
        'e' or 'elementary' -> charge in elementary charge units. For these units, qH+ ~= 1.
        'c' or 'si'         -> charge in SI units (Coulombs).     For these units, qH+ ~= 1.6E-19
        'esu' or 'cgs'      -> charge in cgs units (esu).         For these units, qH+ ~= 4.8E-10
        'simu'              -> charge in simulation units.
    '''
    units = units.lower()
    VALID_UNITS = ['e', 'elementary', 'esu', 'c', 'cgs', 'si', 'simu']
    assert units in VALID_UNITS, "Units invalid; got units={}".format(units)
    # get charge, in 'elementary charge' units:
    if (SL==-1) or (SL[0] < 0):
        # electron
        charge = -1.
    else:
        # not electron
        charge = fl.Fluids(dd=obj)[SL].ionization
    # convert to proper units and return:
    if units in ['e', 'elementary']:
        return charge
    elif units in ['esu', 'cgs']:
        return charge * obj.uni.q_electron
    elif units in ['c', 'si']:
        return charge * obj.uni.qsi_electron
    else: #units=='simu'
        return charge * obj.uni.simu_qsi_e

def get_cross_tab(obj, iSL=None, jSL=None, **kw__fluids):
    '''return (filename of) cross section table for obj.ifluid, obj.jfluid.
    use S=-1 for electrons. (e.g. iSL=(-1,1) represents electrons.)
    either ifluid or jfluid must be neutral. (charged -> Coulomb collisions.)
    iSL, jSL, kw__fluids behavior is the same as in get_var.
    '''
    iSL, jSL = obj.set_fluids(iSL=iSL, jSL=jSL, **kw__fluids)
    if iSL==jSL:
        warnings.warn('Tried to get cross_tab when ifluid==jfluid. (Both equal {})'.format(iSL))
    icharge, jcharge = (get_charge(obj, SL) for SL in (iSL, jSL))
    assert icharge==0 or jcharge==0, "cannot get cross_tab for charge-charge interaction."
    # force ispecies to be neutral (swap i & j if necessary; cross tab is symmetric).
    if icharge != 0:
        return get_cross_tab(obj, jSL, iSL)
    # now, ispecies is the neutral one.
    # now we will actually get the filename.
    CTK = 'CROSS_SECTIONS_TABLES'
    if (jSL==-1) or (jSL[0] < 0): 
        # electrons
        cross_tab_table = obj.mf_etabparam[CTK]
        for row in cross_tab_table:
            # example row looks like: ['01', 'e-h-bruno-fits.txt']
            ## contents are: [mf_species, filename]
            if int(row[0])==iSL[0]:
                return row[1]
    else:
        # not electrons
        cross_tab_table = obj.mf_tabparam[CTK]
        for row in cross_tab_table:
            # example row looks like: ['01', '02', '01', 'he-h-bruno-fits.txt']
            ## contents are: [mf_ispecies, mf_jspecies, mf_jlevel, filename]
            if int(row[0])==iSL[0]:
                if int(row[1])==jSL[0]:
                    if int(row[2])==jSL[1]:
                        return row[3]
    # if we reach this line, we couldn't find cross section file, so make the code crash.
    errmsg = "Couldn't find cross section file for ifluid={}, jfluid={}. ".format(iSL, jSL) + \
             "(We looked in obj.mf_{}tabparam['{}'].)".format(('e' if jSL[0] < 0 else ''), CTK)
    raise ValueError(errmsg)

def get_cross_sect(obj, **kw__fluids):
    '''returns Cross_sect object containing cross section data for obj.ifluid & obj.jfluid.
    equivalent to obj.cross_sect(cross_tab=[get_cross_tab(obj, **kw__fluids)])

    common use-case:
    obj.get_cross_sect().tab_interp(tg_array)
    '''
    return obj.cross_sect([obj.get_cross_tab(**kw__fluids)])

def get_coll_type(obj, iSL=None, jSL=None, **kw__fluids):
    '''return type of collisions between obj.ifluid, obj.jfluid.
    use S=-1 for electrons. (e.g. iSL=(-1,1) represents electrons.)
    iSL, jSL, kw__fluids behavior is the same as in get_var.

    result is 'EL' for elastic collisions, 'MX' for maxwell, 'CL' for coulomb, or None
    In the following cases, return None:
        - ifluid and jfluid are not both charged and 'EL' and 'MX' are not in their coll_keys.
        - ifluid and jfluid   are   both charged and 'CL' is not in their coll_keys.
    if ifluid or jfluid is electrons:
        if both are charged: return ('EE', 'CL')
        if one is neutral:   return ('EE', 'EL')
    '''
    iSL, jSL = obj.set_fluids(iSL=iSL, jSL=jSL, **kw__fluids)
    icharge = obj.get_charge(iSL)
    jcharge = obj.get_charge(jSL)
    if icharge < 0 or jcharge < 0:
        implied_coll_key = 'CL' if (icharge != 0 and jcharge != 0) else 'EL'
        return ('EE', implied_coll_key)
    try:
        coll_keys = obj.coll_keys[(iSL[0], jSL[0])]   # obj.coll_keys only knows about species.
    except KeyError:
        errmsg_collkey_missing = 'coll key not found for (iS, jS) = ({}, {})! '.format(iSL[0], jSL[0]) +\
                                 'Common cause: mistakes / missing keys in COLL KEYS in mf_param_file.'
        raise KeyError(errmsg_collkey_missing)
    if icharge != 0 and jcharge != 0:    # two charged fluids --> return CL or None
        if 'CL' in coll_keys:
            return 'CL'
        else:
            return None
    else:                                # at least one neutral --> return EL or MX or None.
        EL = 'EL' in coll_keys
        MX = 'MX' in coll_keys
        if EL and MX:
            errmsg = 'got EL and MX in coll_keys for ifluid={}, jfluid={}.' +\
                     'But EL and MX are mutually exclusive. Crashing...'
            raise ValueError(errmsg.format(ifluid=iSL, jfluid=jSL))
        elif EL:
            return 'EL'
        elif MX:
            return 'MX'
        else:
            return None


''' --------------------- MultiFluid class --------------------- '''

def simple_property(internal_name, doc=None, name=None, **kw):
    '''return a property with a setter and getter method for internal_name.
    if 'default' in kw:
        - getter will have a default of kw['default'], if attr has not been set.
        - setter will do nothing if value is kw['default'].
    '''
    if 'default' in kw:
        default = kw['default']
        # define getter method
        def getter(self):
            return getattr(self, internal_name, default)
        # define setter method
        def setter(self, value):
            if value is not default:
                setattr(self, internal_name, value)
    else:
        # define getter method
        def getter(self):
            return getattr(self, internal_name)
        # define setter method
        def setter(self, value):
            setattr(self, internal_name, value)
    # define deleter method
    def deleter(self):
        delattr(self, internal_name)
    # bookkeeping
    if name is not None:
        getter.__name__ = 'get_'+name
        setter.__name__ = 'set_'+name
        deleter.__name__ = 'del_'+name
    # collect and return result.
    return property(getter, setter, deleter, doc=doc)

def simple_tuple_property(*internal_names, doc=None, name=None, **kw):
    '''return a property which refers to a tuple of internal names.
    if 'default' in kw:
        - getter will have a default of kw['default'], if attr has not been set.
        - setter will do nothing if value is kw['default'].
        This applies to each name in internal_names, individually.
    '''
    if 'default' in kw:
        default = kw['default']
        # define getter method
        def getter(self):
            return tuple(getattr(self, name, default) for name in internal_names)
        # define setter method
        def setter(self, value):
            for name, val in zip(internal_names, value):
                if val is not default:
                    setattr(self, name, val)
    else:
        # define getter method
        def getter(self):
            return tuple(getattr(self, name) for name in internal_names)
        # define setter method
        def setter(self, value):
            for name, val in zip(internal_names, value):
                setattr(self, name, val)
    # define deleter method
    def deleter(self):
        for name in internal_names:
            delattr(self, name)
    # bookkeeping
    if name is not None:
        getter.__name__ = 'get_'+name
        setter.__name__ = 'set_'+name
        deleter.__name__ = 'del_'+name
    # collect and return result.
    return property(getter, setter, deleter, doc=doc)

# internal names for properties:
_IS = '_mf_ispecies'
_JS = '_mf_jspecies'
_IL = '_mf_ilevel'
_JL = '_mf_jlevel'

class Multifluid():
    '''class which tracks fluids, and contains methods related to fluids.'''
    def __init__(self, **kw):
        self.set_fluids(**kw)

    ## PROPERTIES (FLUIDS) ##
    ### "ORIGINAL PROPERTIES" ###
    mf_ispecies = simple_property(_IS, default=None, name='mf_ispecies')
    mf_jspecies = simple_property(_JS, default=None, name='mf_jspecies')
    mf_ilevel   = simple_property(_IL, default=None, name='mf_ilevel')
    mf_jlevel   = simple_property(_JL, default=None, name='mf_jlevel')
    ### ALIASES - single ###
    iS          = simple_property(_IS, default=None, name='iS')
    jS          = simple_property(_JS, default=None, name='jS')
    iL          = simple_property(_IL, default=None, name='iL')
    jL          = simple_property(_JL, default=None, name='jL')
    ### ALIASES - multiple ###
    ifluid      = simple_tuple_property(_IS, _IL, default=None, name='ifluid')
    iSL         = simple_tuple_property(_IS, _IL, default=None, name='iSL')
    jfluid      = simple_tuple_property(_JS, _JL, default=None, name='jfluid')
    jSL         = simple_tuple_property(_JS, _JL, default=None, name='jSL')

    ### FLUIDS OBJECT (from at_tools.fluids) ###
    @property
    def fluids(self):
        '''at_tools.fluids.Fluids object describing the fluids in self.'''
        if hasattr(self, '_fluids'):
            return self._fluids
        else:
            return fl.Fluids(dd=self)

    ## METHODS ##
    def fluids_equal(self, ifluid, jfluid):
        '''returns whether ifluid and jfluid represent the same fluid.'''
        return fluid_equals(ifluid, jfluid)

    def MaintainingFluids(self):
        return _MaintainingFluids(self)
    MaintainingFluids.__doc__ = _MaintainingFluids.__doc__.replace(
                                '_MaintainingFluids(dd', 'dd.MaintainingFluids(')  # set docstring
    MaintainFluids = MaintainingFluids  # alias

    def UsingFluids(self, **kw__fluids):
        return _UsingFluids(self, **kw__fluids)

    UsingFluids.__doc__ = _UsingFluids.__doc__.replace(
                                '_UsingFluids(dd, ', 'dd.UsingFluids(') # set docstring
    UseFluids = UsingFluids  # alias

# include bound versions of methods from this module into the Multifluid class.
for func in MULTIFLUID_FUNCS:
    setattr(Multifluid, func, globals().get(func, NotImplementedError))

del func   # (we don't want func to remain in the fluid_tools.py namespace beyond this point.)