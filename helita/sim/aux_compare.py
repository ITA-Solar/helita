"""
Created by Sam Evans on Apr 24 2021

purpose: easily compare values between helita and aux vars from a simulation.
"""

# import internal modules
from . import fluid_tools

# import external private modules
try:
    from at_tools import fluids as fl
except ImportError:
    fl = None
    warnings.warn('failed to import at_tools.fluids; some functions in helita.sim.aux_compare may crash')

AUXVARS = {
    # aux var   : helita var. if tuple, v[1] is required ifluid or mf_ispecies.
                                     #  v[2] (if it exists) jfluid or mf_jspecies.
    'etg'       : ('tg', -1),
    'mfe_tg'    : 'tg',
    'ex'        : 'efx',
    'ey'        : 'efy',
    'ez'        : 'efz',
    'mm_cnu'    : 'nu_ij',
    'mfr_nu_es' : ('nu_ij', -1),
} 

def get_helita_var(auxvar):
    return AUXVARS[auxvar]

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

def _construct_calls(auxvar, callsig, auxfluids, helfluids, f=lambda fluid: fluid, **kw__get_var):
    '''returns ((args, kwargs) to use with auxvar, (args, kwargs) to use with helitavar)
    args with be the list [var]
    kwargs will be the dict of auxfluids (or helfluids), updated by kw__get_var.

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
    # update dicts with callsig kws and kw__get_var
    helfluids.update(callsigcopy)
    auxfluids.update(**kw__get_var)
    helfluids.update(**kw__get_var)
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
        auxcall, helcall = _construct_calls(auxvar, callsig, auxfluids_dict, helfluids_dict, f=f, **kw__get_var)
        #print(auxcall, helcall)
        auxval = obj.get_var(*auxcall[0], **auxcall[1])
        helval = obj.get_var(*helcall[0], **helcall[1])
        # format output & yield it
        vardict = dict(aux=auxvar,         hel=callsig['var'])
        valdict = dict(aux=auxval,         hel=helval)
        fludict = dict(aux=auxfluids_dict, hel=helfluids_dict)
        yield dict(vars=vardict, vals=valdict, fluids=fludict)

# TODO:
#   generic prettyprint of results, requiring only obj and auxvar.
#   also should include some stats such as ratio, and {min, max & mean} for hel & aux.
# example simple non-generic prettyprint:
'''
for d in axc.iter_get_var(dd, 'mm_cnu'):
    fluids = d['fluids']
    auxfluids = fluids['aux']['ifluid'].SL, fluids['aux']['jfluid'].SL
    helfluids = fluids['hel']['ifluid'].SL, fluids['hel']['jfluid'].SL
    print('{:6s}'.format(d['vars']['aux']), *auxfluids, '{: .3e}'.format(d['vals']['aux'].mean()))
    print('{:6s}'.format(d['vars']['hel']), *helfluids, '{: .3e}'.format(d['vals']['hel'].mean()))
    print('-'*40)

# the first few lines of output look like:

mm_cnu (1, 1) (1, 2)  8.280e+05
nu_ij  (1, 1) (1, 2)  8.280e+05
----------------------------------------
mm_cnu (1, 1) (2, 1)  7.225e+02
nu_ij  (1, 1) (2, 1)  7.225e+02
----------------------------------------
mm_cnu (1, 1) (2, 2)  1.252e+02
nu_ij  (1, 1) (2, 2)  1.252e+02
----------------------------------------
'''