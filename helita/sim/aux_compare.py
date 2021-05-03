"""
Created by Sam Evans on Apr 24 2021

purpose: easily compare values between helita and aux vars from a simulation.

Highest-level use-case: compare all the aux vars with their helita counterparts!
    #<< input:
    from helita.sim import aux_compare as axc
    from helita.sim import ebysus as eb
    dd = eb.EbysusData(...)        # you must fill in the ... as appropriate.
    c = axc.compare_all(dd)

    #>> output:
             >->->->->->->-> initiate comparison for auxvar = etg <-<-<-<-<-<-<-<

    auxvar etg            min= 4.000e+03, mean= 4.000e+03,  max= 4.000e+03
    helvar  tg   -1       min= 4.000e+03, mean= 4.000e+03,  max= 4.000e+03;   mean ratio (aux / helita):  1.000e+00
    ----------------------------------------------------------------------------------------------------------------------

    comparison_result(N_differ=0, N_total=1, runtime=0.0020618438720703125) 


         >->->->->->->-> initiate comparison for auxvar = mm_cnu <-<-<-<-<-<-<-<

    auxvar mm_cnu  ( 1, 1) ( 1, 2)   min= 8.280e+05, mean= 8.280e+05,  max= 8.280e+05
    helvar  nu_ij  ( 1, 1) ( 1, 2)   min= 8.280e+05, mean= 8.280e+05,  max= 8.280e+05;   mean ratio (aux / helita):  1.000e+00
    ---------------------------------------------------------------------------------------------------------------------------------

    ... << (more lines of output, which we are not showing you in this file, to save space.)

    #<< more input:
    print(c)

    #>> more output:
    {'N_compare': 30, 'N_var': 8, 'N_differ': 4, 'N_diffvar': 1, 'N_error': 1,
    'errors': [FileNotFoundError(2, 'No such file or directory')], 'runtime': 1.581925868988037}

High-level use-case: compare a single aux var with its helita counterpart!
    #<< input:
    from helita.sim import aux_compare as axc
    from helita.sim import ebysus as eb
    dd = eb.EbysusData(...)        # you must fill in the ... as appropriate.
    axc.compare(dd, 'mfr_nu_es')

    #>> output:
    auxvar mfr_nu_es  ( 1, 1)           min= 3.393e+04, mean= 3.393e+04,  max= 3.393e+04
    helvar     nu_ij   -1     ( 1, 1)   min= 1.715e+04, mean= 1.715e+04,  max= 1.715e+04;   mean ratio (aux / helita):  1.978e+00
                                                                                         >>> WARNING: RATIO DIFFERS FROM 1.000 <<<<
    ------------------------------------------------------------------------------------------------------------------------------------
    auxvar mfr_nu_es  ( 1, 2)           min= 1.621e+05, mean= 1.621e+05,  max= 1.621e+05
    helvar     nu_ij   -1     ( 1, 2)   min= 1.622e+05, mean= 1.622e+05,  max= 1.622e+05;   mean ratio (aux / helita):  9.993e-01
    ------------------------------------------------------------------------------------------------------------------------------------

    #<< more input:
    axc.compare(dd, 'mm_cnu')

    #>> more output:
    auxvar mm_cnu  ( 1, 1) ( 1, 2)   min= 8.280e+05, mean= 8.280e+05,  max= 8.280e+05
    helvar  nu_ij  ( 1, 1) ( 1, 2)   min= 8.280e+05, mean= 8.280e+05,  max= 8.280e+05;   mean ratio (aux / helita):  1.000e+00
    ---------------------------------------------------------------------------------------------------------------------------------
    auxvar mm_cnu  ( 1, 2) ( 1, 1)   min= 8.280e+06, mean= 8.280e+06,  max= 8.280e+06
    helvar  nu_ij  ( 1, 2) ( 1, 1)   min= 8.280e+06, mean= 8.280e+06,  max= 8.280e+06;   mean ratio (aux / helita):  1.000e+00
    ---------------------------------------------------------------------------------------------------------------------------------

# output format notes:
#   vartype varname (ispecie, ilevel) (jspecie, jlevel) min mean max
# when ispecies < 0 or jspecie < 0 (i.e. for electrons), they may be shown as "specie" instead of "(ispecie, ilevel)".


TODO (maybe):
    - allow to put kwargs in auxvar lookup.
        - for example, ebysus defines mm_cross = 0 when ispecies is ion, to save space.
          meanwhile get_var('cross') in helita will tell same values even if fluids are swapped.
          e.g. get_var('mm_cross', ifluid=(1,2), jfluid=(1,1)) == 0
          get_var('cross', ifluid=(1,2), jfluid=(1,1)) == get_var('cross', ifluid=(1,1), jfluid=(1,2))
"""

# import built-in
from collections import namedtuple
import time

# import internal modules
from . import fluid_tools

# import external private modules
try:
    from at_tools import fluids as fl
except ImportError:
    fl = None
    warnings.warn('failed to import at_tools.fluids; some functions in helita.sim.aux_compare may crash')

# set defaults
DEFAULT_TOLERANCE = 0.05    # the max for (1-abs(X/Y)) before we think X != Y


''' ----------------------------- lookup helita counterpart to aux var ----------------------------- '''

# dict of defaults for converting from auxvar to helita var (aka "helvar").
AUXVARS = {
    # aux var   : helita var. if tuple, v[1] is required ifluid or mf_ispecies.
                                     #  v[2] (if it exists) jfluid or mf_jspecies.
    'etg'       : ('tg', -1),    # electron temperature
    'mfe_tg'    : 'tg',          #  fluid   temperature
    'mfr_nu_es' : ('nu_ij', -1), # electron-fluid collision frequency
    'mm_cnu'    : 'nu_ij',       #  fluid - fluid collision frequency
    'mm_cross'  : 'cross',       # cross section
    'mfr_p'     : 'p',           # pressure
    
}
# add each of these plus an axis to AUXVARS.
# e.g. {'e': 'ef'} --> {'ex': 'efx', 'ey': 'efy', 'ez': 'efz'}.
AUX_AXIAL_VARS = {
    'e'         : 'ef',
    'eu'        : 'ue',
    'i'         : 'j',
}
AXES = ['x', 'y', 'z']
# add the axial vars to auxvars.
for (aux, hel) in AUX_AXIAL_VARS.items():
    AUXVARS.update({aux+x: hel+x for x in AXES})

def get_helita_var(auxvar):
    return AUXVARS[auxvar]


''' ----------------------------- get_var for helita & aux ----------------------------- '''

def _callsig(helvar):
    '''returns dict with keys for getvar for helvar'''
    if isinstance(helvar, str):
        return dict(var=helvar)
    #else: helvar has len 2 or longer
    result = dict(var=helvar[0])
    try:
        next(iter(helvar[1]))
    except TypeError:  # helvar[1] is not a list
        result.update(dict(mf_ispecies=helvar[1]))
    else:              # helvar[1] is a list
        result.update(dict(ifluid=helvar[1]))
    if len(helvar)>2: # we have info for jfluid as well.
        try:
            next(iter(helvar[2]))
        except TypeError:
            result.update(dict(mf_jspecies=helvar[2]))
        else:
            result.update(dict(jfluid=helvar[2]))
    return result

def _loop_fluids(obj, callsig):
    '''return the fluid kws which need to be looped through.
    obj should be EbysusData object.
    callsig should be _callsig(helvar).
    returns a tuple telling whether to loop through (ifluid, jfluid) for helvar.
    '''
    var = callsig['var']
    search = obj.search_vardict(var)
    nfluid = search.result['nfluid']
    if nfluid is None:  # we do not need to loop through any fluids.
        return (False, False)
    elif nfluid == 0:   # we do not need to loop through any fluids.
        assert list(callsig.keys()) == ['var'], "invalid var tuple in AUXVARS for nfluid=0 var '{}'".format(var)
        return (False, False)
    elif nfluid == 1:   # we might need to loop through ifluid.
        result = [True, False]
        for kw in ['mf_ispecies', 'ifluid']:
            if kw in callsig.keys():
                result[0] = False  # we do not need to loop through ifluid.
                break
        return tuple(result)
    elif nfluid == 2:   # we might need to loop through ifluid and/or jfluid.
        result = [True, True]
        for kw in ['mf_jspecies', 'jfluid']:
            if kw in callsig.keys():
                result[1] = False  # we do not need to loop through jfluid.
                break
        for kw in ['mf_ispecies', 'ifluid']:
            if kw in callsig.keys():
                result[0] = False  # we do not need to loop through ifluid.
                break
        return tuple(result)
    else:
        raise NotImplementedError  # we don't know what to do when nfluid is not 0, 1, 2, or None.

def _iter_fluids(fluids, loopfluids, **kw__fluid_pairs):
    '''returns an iterator which yields pairs of dicts: (daux, dhel)
    daux are the fluid kws to call with aux var
    dhel are the fluid kws to call with helita var.

    loopfluids ==
        (False, False) -> yields (dict(), dict()) then stops iteration.
        (True,  False) -> yields (dict(ifluid=fluid), dict(ifluid=fluid)) for fluid in fluids.
        (False, True ) -> yields (dict(ifluid=fluid), dict(jfluid=fluid)) for fluid in fluids.
        (True, True) -> yields (x, x) where x is a dict with keys ifluid, jfluid,
                        and we iterate over pairs of ifluid, jfluid.
    **kw__fluid_pairs
        only matters if loopfluids == (True, True);
        these kwargs go to fluid_tools.fluid_pairs.
    '''
    loopi, loopj = loopfluids
    if   not loopi and not loopj:
        x = dict()
        yield (x, x)
    elif     loopi and not loopj:
        for fluid in fluids:
            x = dict(ifluid=fluid)
            yield (x, x)
    elif not loopi and     loopj:
        for fluid in fluids:
            yield (dict(ifluid=fluid), dict(jfluid=fluid))
    elif     loopi and     loopj:
        for ifluid, jfluid in fluid_tools.fluid_pairs(fluids, **kw__fluid_pairs):
            x = dict(ifluid=ifluid, jfluid=jfluid)
            yield (x, x)

def _SL_fluids(fluids_dict, f = lambda fluid: fluid):
    '''update values in fluids_dict by applying f'''
    return {key: f(val) for key, val in fluids_dict.items()}

def _setup_fluid_kw(auxvar, callsig, auxfluids, helfluids, f=lambda fluid: fluid):
    '''returns ((args, kwargs) to use with auxvar, (args, kwargs) to use with helitavar)
    args with be the list [var]
    kwargs will be the dict of auxfluids (or helfluids). (species, levels) only.

    f is applied to all values in auxfluids and helfluids.
        use f = (lambda fluid: fluid.SL) when fluids are at_tools.fluids.Fluids,
                to convert them to (species, level) tuples.
    '''
    # convert fluids to SLs via f
    auxfluids = _SL_fluids(auxfluids, f=f)
    helfluids = _SL_fluids(helfluids, f=f)
    # pop var from callsig (we pass var as arg rather than kwarg).
    callsigcopy = callsig.copy()  # copy to ensure callsig is not altered
    helvar = callsigcopy.pop('var')
    helfluids.update(callsigcopy)
    # make & return output
    callaux = ([auxvar], auxfluids)
    callhel = ([helvar], helfluids)
    return (callaux, callhel)

def _get_fluids_and_f(obj, fluids=None, f=lambda fluid: fluid):
    '''returns fluids, f.
    if fluids is None:
        fluids = fl.Fluids(dd=obj)
        f = lambda fluid: fluid.SL
    if we failed to import at_tools.fluids, try fluids=obj.fluids, before giving up.
    '''
    if fluids is None:
        f = lambda fluid: fluid.SL
        if fl is None:
            if not obj.hasattr('fluids'):
                errmsg = ("{} has no attribute 'fluids', we failed to import at_tools.fluids "
                          "and you didn't input fluids, so we don't know which fluids to use!")
                errmsg = errmsg.format(obj)
                raise NameError(errmsg)  # choosing NameError type because "fluids" is "not defined".
            else:
                fluids = obj.fluids
        else:
            fluids = fl.Fluids(dd=obj)
    return (fluids, f)

def iter_get_var(obj, auxvar, helvar=None, fluids=None, f=lambda fluid: fluid,
                 ordered=False, allow_same=False, **kw__get_var):
    '''gets values for auxvar and helita var.
    
        yields dict(vars   = dict(aux=auxvar,          hel=helita var name),
                    vals   = dict(aux=get_var(auxvar), hel=get_var(helvar)),
                    fluids = dict(aux=auxfluids_dict,  hel=helfluids_dict)),
                    SLs    = dict(aux=auxfluidsSL,     hel=helfluidsSL))   ,
                    )

        obj: EbysusData object
            we will do obj.get_var(...) to get the values.
        auxvar: str
            name of var in aux. e.g. 'mfe_tg' for temperature.
        helvar: None (default), or str, or tuple
            None -> lookup helvar using helita.sim.aux_compare.AUXVARS.
            str  -> use this as helvar. Impose no required fluids on helvar.
            tuple -> use helvar[0] as helvar. Impose required fluids:
                        helvar[1] imposes ifluid or mf_ispecies.
                        helvar[2] imposes jfluid or mf_jspecies (if helvar[2] exists).
        fluids: None (default) or list of fluids
            None -> use fluids = fl.Fluids(dd=obj).
        f: function which converts fluid to (species, level) tuple
            if fluids is None, f is ignored, we will instead use f = lambda fluid: fluid.SL
            otherwise, we apply f to each fluid in fluids, before putting it into get_var.
            Note: auxfluids_dict and helfluids_dict contain fluids before f is applied.
        if iterating over fluid pairs, the following kwargs also matter:
            ordered: False (default) or True
                whether to only yield ordered combinations of fluid pairs (AB but not BA)
            allow_same: False (default) or True
                whether to also yield pairs of fluids which are the same (AA, BB, etc.)
        **kw__get_var goes to obj.get_var().
    '''
    if helvar is None: helvar = get_helita_var(auxvar)
    callsig    = _callsig(helvar)
    loopfluids = _loop_fluids(obj, callsig)
    # set fluids if necessary
    if loopfluids[0] or loopfluids[1]:
        fluids, f = _get_fluids_and_f(obj, fluids, f)
    iterfluids = _iter_fluids(fluids, loopfluids, ordered=ordered, allow_same=allow_same)
    for auxfluids_dict, helfluids_dict in iterfluids:
        auxcall, helcall = _setup_fluid_kw(auxvar, callsig, auxfluids_dict, helfluids_dict, f=f)
        auxfluidsSL = auxcall[1].copy()
        helfluidsSL = helcall[1].copy()
        auxcall[1].update(**kw__get_var)
        helcall[1].update(**kw__get_var)
        # actually get values by reading data and/or doing calculations
        auxval = obj.get_var(*auxcall[0], **auxcall[1])
        helval = obj.get_var(*helcall[0], **helcall[1])
        # format output & yield it
        vardict = dict(aux=auxvar,         hel=callsig['var'])
        valdict = dict(aux=auxval,         hel=helval)
        fludict = dict(aux=auxfluids_dict, hel=helfluids_dict)
        SLsdict = dict(aux=auxfluidsSL,    hel=helfluidsSL)
        yield dict(vars=vardict, vals=valdict, fluids=fludict, SLs=SLsdict)


''' ----------------------------- prettyprint comparison ----------------------------- '''

def _stats(arr):
    '''return stats for arr. dict with min, mean, max.'''
    return dict(min=arr.min(), mean=arr.mean(), max=arr.max())

def _strstats(arr_or_stats, fmt='{: 0.3e}', fmtkey='{:>4s}'):
    '''return pretty string for stats. min=__, mean=__, max=__.'''
    keys = ['min', 'mean', 'max']
    if isinstance(arr_or_stats, dict): # arr_or_stats is stats
        x = arr_or_stats  
    else:                              # arr_or_stats is arr
        x = _stats(arr_or_stats) 
    return ', '.join([fmtkey.format(key) + '='+fmt.format(x[key]) for key in keys])

def _strvals(valdict):
    '''return dict of pretty str for vals from valdict. keys 'hel', 'aux', 'stats'.
    'stats' contains dict of stats for hel & aux.
    '''
    result = dict(stats=dict())
    for aux in valdict.keys():  # for aux in ['aux', 'hel']:
        stats = _stats(valdict[aux])
        strstats = _strstats(stats)
        result[aux] = strstats
        result['stats'][aux] = stats
    return result

def _strSL(SL, fmtSL='({:2d},{:2d})', fmtS=' {:2d}    ', fmtNone=' '*(1+2+1+2+1)):
    '''pretty string for (specie, level) SL. (or just specie SL, or None SL)'''
    if SL is None:
        return fmtNone
    try:
        next(iter(SL))  # error if SL is not a list.
    except TypeError:
        return fmtS.format(SL)    # SL is just S
    else:
        return fmtSL.format(*SL)  # SL is (S, L)

def _strfluids(fludict):
    '''return dict of pretty str for fluids from fludict. keys 'hel', 'aux'.'''
    N = max(len(fludict['aux']), len(fludict['hel']))
    result = dict()
    for aux in fludict.keys():  # for aux in ['aux', 'hel']:
        s = ''
        if N>0:
            iSL = fludict[aux].get('ifluid', fludict[aux].get('mf_ispecies', None))
            s += _strSL(iSL) + ' '
        if N>1:
            jSL = fludict[aux].get('jfluid', fludict[aux].get('mf_jspecies', None))
            s += _strSL(jSL) + ' '
        result[aux] = s
    return result

def _strvars(vardict, prefix=True):
    '''return dict of pretty str for vars from vardict. keys 'hel', 'aux'.
    prefix==True -> include prefix 'helita' or 'auxvar'.
    '''
    L = max(len(vardict['aux']), len(vardict['hel']))
    fmt = '{:>'+str(L)+'s}'
    result = dict()
    for aux in vardict.keys():  # for aux in ['aux', 'hel']:
        s = ''
        if prefix:
            s += dict(aux='auxvar', hel='helvar')[aux] + ' '
        s += fmt.format(vardict[aux]) + ' '
        result[aux] = s
    return result

def prettyprint_comparison(x, printout=True, prefix=True, underline=True,
                           rattol=DEFAULT_TOLERANCE, return_warned=False, **kw__None):
    '''pretty printing of info in x. x is one output of iter_get_var.
    e.g.: for x in iter_get_var(...): prettyprint_comparison(x)

    printout: if False, return string instead of printing.
    prefix: whether to include prefix of 'helita' or 'auxvar' at start of each line.
    underline: whether to include a line of '------'... at the end.
    rattol: if abs(1 - (mean aux / mean helita)) > rattol, print extra warning line.
    return_warned: whether to also return whether we made a warning.
    **kw__None goes to nowhere.
    '''
    # get strings / values:
    svars   = _strvars(  x['vars'])
    sfluids = _strfluids(x['SLs'] )
    svals   = _strvals(  x['vals'])
    meanaux = svals['stats']['aux']['mean']
    meanhel = svals['stats']['hel']['mean']
    if meanaux==0.0 and meanhel==0.0:
        ratio = 1.0
    else:
        ratio = meanaux / meanhel
    ratstr  = 'mean ratio (aux / helita): {: 0.3e}'.format(ratio)
    # combine strings
    key = 'aux'
    s = ' '.join([svars[key], sfluids[key], svals[key]]) + '\n'
    lline = len(s)
    key = 'hel'
    s += ' '.join([svars[key], sfluids[key], svals[key]]) + ';   '
    s += ratstr
    if abs(1 - ratio) > rattol:  # then, add warning!
        s += '\n' + ' '*(lline) + '>>> WARNING: RATIO DIFFERS FROM 1.000 <<<<'
        warned = True
    else:
        warned = False
    if underline:
        s += '\n' + '-' * (lline + len(ratstr) + 10)
    # print (or return)
    result = None
    if printout:
        print(s)
    else:
        result = s
    if return_warned:
        result = (result, warned)
    return result


''' ----------------------------- high-level comparison interface ----------------------------- '''

comparison_result = namedtuple('comparison_result', ('N_differ', 'N_total', 'runtime')) 

@fluid_tools.maintain_fluids # restore dd.ifluid and dd.jfluid after finishing compare().
def compare(obj, auxvar, helvar=None, fluids=None, **kwargs):
    '''compare values of auxvar with appropriate helita var, for obj.
    **kwargs propagate to iter_get_var, obj.get_var, and prettyprint_comparison.

    involves looping through fluids:
        none (nfluid=0, e.g. 'ex'),
        one  (nfluid=1, e.g. 'tg'), or
        two  (nfluid=2, e.g. 'nu_ij').

    Parameters
    ----------
    helvar: None (default)or str or tuple
        helita var corresponding to auxvar. E.g. 'efx' for auxvar='ex'.
        it is assumed that helvar and auxvar use the same number of fluids.
        For example, 'mfe_tg' and 'tg' each use one fluid.
        For some vars, there is a slight hiccup. Example: 'etg' and 'tg'.
            'etg' is equivalent to 'tg' only when mf_ispecies=-1.
        To accomodate such cases, we allow a tuple such as ('tg', -1) for helvar.
        type of helvar, and explanations below:
            None -> use default. (but not all auxvars have an existing default.)
                    all defaults which exist are hard-coded in helita.sim.aux_compare.
                    when default fails, use non-None helvar,
                    or edit helita.sim.aux_compare.AUXVARS to add default.
            str  -> use var = helvar. This one should do exactly what you expect.
                    Note that for this case, auxvar and helvar must depend on the
                    exact same fluids (e.g. both depend on just ifluid).
            tuple-> use var = helvar[0]. The remaining items in the tuple
                    will force fluid kwargs for helvar. ints for mf_species;
                    tuples force fluid. Example: ('nu_ij', -1) forces mf_ispecies=-1,
                    and because nu_ij depends on 2 fluids (according to obj.vardict),
                    we still need to enter one fluid. So we will loop through
                    fluids, passing each fluid to helvar as jfluid, and auxvar as ifluid.
    
    fluids: None (default) or iterable (e.g. list)
        None     -> get fluids using obj. fluids = at_tools.fluids.Fluids(dd=obj)
        iterable -> use these fluids. Should be tuples of (specie, level),
                    example: fluids = [(1,2),(2,2),(2,3)]
                    See aux_compare.iter_get_var for more documentation.

    Returns
    -------
    returns namedtuple (N_differ, N_total, runtime), where:
        N_differ = number of times helita and auxvar gave
                   different mean results (differing by more than rattol).
                   0 is good, it means helita & auxvar agreed on everything! :)
        N_total  = total number of values compared. example:
                   if we compared 'mfe_tg' and 'tg' for ifluid in 
                   [(1,1),(1,2),(2,3)], we will have N_total==3.
        runtime  = time it took to run, in seconds.
    '''
    now = time.time()
    N_warnings = 0
    N_total    = 0
    for x in iter_get_var(obj, auxvar, helvar=helvar, fluids=fluids, **kwargs):
        N_total += 1
        _, warned = prettyprint_comparison(x, return_warned=True, **kwargs)
        if warned:
            N_warnings += 1
    runtime = round(time.time() - now, 3) # round because sub-ms times are untrustworthy and ugly.
    return comparison_result(N_warnings, N_total, runtime)

def _get_aux_vars(obj):
    '''returns list of vars in aux based on obj.'''
    return obj.params['aux'][obj.snapInd].split()

def compare_all(obj, aux=None, verbose=2, **kwargs):
    '''compare all aux vars with their corresponding values in helita.

    each comparison involves looping through fluids:
        none (nfluid=0, e.g. 'ex'),
        one  (nfluid=1, e.g. 'tg'), or
        two  (nfluid=2, e.g. 'nu_ij').
    
    Parameters
    ----------
    obj: an EbysusData object.
        (or any object with get_var method.)
    aux: None (default) or a list of strs
        the list of aux vars to compare.
        None -> get the list from obj (via obj.params['aux']).
    verbose: 2 (default), 1, or 0
        2 -> full print info
        1 -> print some details but set printout=False (unless printout is in kwargs)
        0 -> don't print anything.
    **kwargs:
        extra kwargs are passed to compare(). (i.e.: helita.sim.aux_compare.compare)

    Returns
    -------
    returns dict with contents:
        N_compare = number of actual values we compared.
                    (one for each set of fluids used for each (auxvar, helvar) pair)
        N_var     = number of vars we tried to run comparisons for.
                    (one for each (auxvar, helvar) pair)
        N_differ  = number of times helita and auxvar gave
                    different mean results (by more than rattol)
        N_diffvar = number of vars for which helita and auxvar gave
                    different mean results at least once.
        N_error   = number of times compare() crashed due to error.
        errors    = list of errors raised.
        runtime   = time it took to run, in seconds.

    A 100% passing test looks like N_differ == N_error == 0.
    '''
    now = time.time()
    printout = kwargs.pop('printout', (verbose >= 2) ) # default = (verbose>=2)
    x = dict(N_compare=0, N_var=0, N_differ=0, N_diffvar=0, N_error=0, errors=[])
    auxvars = _get_aux_vars(obj) if aux is None else aux
    for aux in auxvars:
        if verbose:
            banner = '     >->->->->->->-> {} <-<-<-<-<-<-<-<'
            if printout: banner += '\n'
            print(banner.format('initiate comparison for auxvar = {}'.format(aux)))
        x['N_var'] +=1
        try:
            comp = compare(obj, aux, printout=printout, **kwargs)
        except Exception as exc:
            x['N_error'] += 1
            x['errors']  += [exc]
            if verbose >= 1:
                print('>>>', repr(exc), '\n')
        else:
            x['N_compare'] += comp[1]
            x['N_differ']  += comp[0]
            x['N_diffvar'] += (comp[0] > 0)
            if printout: print() # print a single new line
            if verbose >= 1:
                print(comp, '\n')
        if printout: print() # print a single new line
    x['runtime'] = round(time.time() - now, 3) # round because sub-ms times are untrustworthy and ugly.
    return x


