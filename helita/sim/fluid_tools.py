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
  warnings.warn('failed to import at_tools.fluids; some functions in helita.sim.fluid_tools may crash')

''' --------------------- setting fluids --------------------- '''

def set_mf_fluid(obj, species=None, level=None, i='i'):
    '''sets obj.mf_{i}species and obj.mf_{i}level. Also sets obj.{i}fluid
    species, level: None or int
        None -> if obj.mf_{i}attr already exists, don't change it.
                else, set it to 1.
        ints -> set mf_{i}species=species, mf_{i}level=level.
    '''
    DEFAULT_S, DEFAULT_L = (1, 1)
    mf_species_attr = 'mf_'+i+'species'
    mf_level_attr = 'mf_'+i+'level'
    fluid_attr = i+'fluid'
    # set species
    if species is None:
        if not hasattr(obj, mf_species_attr):
            species = DEFAULT_S
        else:
            species = getattr(obj, mf_species_attr)
    setattr(obj, mf_species_attr, species)
    # set level
    if level is None:
        if not hasattr(obj, mf_level_attr):
            level   = DEFAULT_L
        else:
            level   = getattr(obj, mf_level_attr)
    setattr(obj, mf_level_attr, level)
    # set fluid
    setattr(obj, fluid_attr, (species, level) )

def set_mfi(obj, mf_ispecies=None, mf_ilevel=None):
    '''set obj.mf_ispecies, obj.mf_ilevel, and obj.ifluid.
    mf_ispecies, mf_ilevel: None or int
        None -> if attr already exists, don't change it.
                else, set it to 1.
        int  -> set attr to this value.
    '''
    return obj.set_mf_fluid(mf_ispecies, mf_ilevel, 'i')

def set_mfj(obj, mf_jspecies=None, mf_jlevel=None):
    '''set obj.mf_jspecies, obj.mf_jlevel, and obj.jfluid.
    mf_jspecies, mf_jlevel: None or int
        None -> if attr already exists, don't change it.
                else, set it to 1.
        int  -> set attr to this value.
    '''
    return obj.set_mf_fluid(mf_jspecies, mf_jlevel, 'j')


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

class _MaintainingFluids:
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
        with _MaintainingFluids(obj):
            return f(obj, *args, **kwargs)
    return f_but_maintain_fluids

def use_fluids(**kw__fluids):
    '''returns decorator version of _UseFluids. first arg of f must be an EbysusData object.'''
    def decorator(f):
        @functools.wraps(f)
        def f_but_use_fluids(obj, *args, **kwargs):
            with _UsingFluids(obj, **kw__fluids):
                return f(obj, *args, **kwargs)
        return f_but_use_fluids
    return decorator


''' --------------------- iterators over fluid pairs --------------------- '''

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
    elif not ordered and     allow_same: return itertools.product(fluids, 2)
    assert False #we should never reach this line...

''' --------------------- small helper functions --------------------- '''
# for each of these functions, obj should be an EbysusData object.

def get_species_name(obj, specie):
    '''return specie's name: 'e' for electrons; element (atomic symbol) for other fluids.'''
    if specie < 0:
        return 'e'
    else:
        return obj.att[specie].params.element

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
        specie = iter(specie)
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
    if SL[0] < 0:
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
    iS, iL, jS, jL = _interpret_kw_fluids(iSL=iSL, jSL=jSL, **kw__fluids)
    obj.set_mfi(iS, iL)
    obj.set_mfj(jS, jL)
    iSL = obj.ifluid
    jSL = obj.jfluid
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
    if jSL[0] < 0: 
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