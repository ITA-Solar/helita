"""
Created by Sam Evans on Apr 3 2021

Purpose: helper functions for documentation of variables.

create vardict which looks like:
vardict = {
    meta_quant_1 :                                # example: "mf_quantities"
    {
        QUANTDOC     : 'meta_quant_1 description',
        TYPE_QUANT_1 :                            # example: "GLOBAL_QUANT"
            {     
            QUANTDOC     : 'TYPE_QUANT_1 description',
            # '_DOC_QUANT' : 'global variables; calculated by looping through species',   # example
            mq1tq1_var_1 : 'mq1tq1_var_1 description',
            # 'nel'      : 'electron number density [cm^-3]',   # example
            mq1tq1_var_2 : 'mq1tq1_var_2 description',
            ...
            },
        TYPE_QUANT_2 :                            # example: "PLASMA_QUANT"
            {     
            QUANTDOC     : 'TYPE_QUANT_2 description',
            mq1tq2_var_1 : 'mq1tq2_var_1 description',
            ...
            },
        ...
    },
    meta_quant_2 :                                # example "arithmetic_quantities"
    {      
        QUANTDOC     : 'meta_quant_2 description',
        TYPE_QUANT_1 :
            {
            QUANTDOC     : 'TYPE_QUANT_2 description',
            mq2tq1_var_1 : 'mq2tq1_var_1 description',
            ...
            },
        ...
    },
    ...
}


TODO:
    documentation for simple vars. (And quant_tracking for simple vars)
    Add units to vars.
"""

# import built-ins
import math #for pretty strings
import collections
import functools
import types  # for MethodType

# import internal modules
from . import file_memory

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = '(not yet documented)'        #default documentation if none is provided.
QUANTDOC = '_DOC_QUANT'                 #key for dd.vardict[TYPE_QUANT] containing doc for what TYPE_QUANT means.
NFLUID  = 'nfluid'    #key which stores number of fluids. (e.g. 0 for none; 1 for "uses ifluid but not jfluid". 
CREATING_VARDICT = '_creating_vardict'  #attribute of obj which tells if we are running get_var('') to create vardict.

# defaults for quant tracking
## attributes of obj
QUANT_SELECTED  = '_quant_selected'     #stores vardict lookup info for the latest quant selected.
QUANTS_SELECTED = '_quants_selected'    #stores quant_selected for QUANT_NTRACKING recent quants.
QUANT_SELECTION = '_quant_selection'    #stores info for latest quant selected; use for hesitant setting.
QUANT_NTRACKING = '_quant_ntracking'    #if it exists, sets maxlen for _quants_selected deque.
## misc
QUANT_TRACKING_N = 1000                 #default for number of quant selections to remember.

# defaults for loading level tracking
## attribute of obj which tells how deep we are into loading a quantity right now. 0 = top level.
LOADING_LEVEL    = '_loading_level'     
## defaults for "top level" quants (passed to get_var externally, e.g. outside of load_..._quantities files)
TYPEQUANT_TOP_LEVEL = 'TOP_LEVEL'
METAQUANT_TOP_LEVEL = 'top_level'
## defaults for "mid level" quants (passed to get_var internally, e.g. inside of load_..._quantities files)
## these will be hit by .format(level=getattr(obj, LOADING_LEVEL, 0)).
TYPEQUANT_LEVEL_N   = 'LEVEL_{level:}'   #NOTE: gotten_vars() assumes these very specific forms
METAQUANT_LEVEL_N   = 'level_{level:}'   #      for TYPEQUANT_LEVEL_N and METAQUANT_LEVEL_N.
                                         # If you change these values you need to update _get_level_from_q().


HIDE_DECORATOR_TRACEBACKS = True  # whether to hide decorators from this file when showing error traceback.

# global variable which tells which quantity you are setting now.
METAQUANT = None

''' ----------------------------- create vardict ----------------------------- '''

def set_meta_quant(obj, name, QUANT_DOC=NONEDOC):
    '''sets the current "meta_quant". You must use this before starting documentation.
    see load_mf_quantities.load_mf_quantities for an example.

    QUANT_DOC is the documentation to put about this metaquant.
    for example, in load_mf_quantities.load_mf_quantities,
        set_meta_quant('MULTIFLUID_QUANTITIES', 'These are the multiple-fluid quantities.')

    The idea is that the meta_quant will be the same throughout a given load_*_quantities.py file.
    '''
    if not hasattr(obj, VARDICT):
        setattr(obj, VARDICT, dict())
    vardict = getattr(obj, VARDICT)

    global METAQUANT   # allows to edit the value of document_vars.METAQUANT
    METAQUANT = name

    if METAQUANT not in vardict.keys():
        vardict[METAQUANT] = dict()
    vardict[METAQUANT][QUANTDOC] = QUANT_DOC

def vars_documenter(obj, TYPE_QUANT, QUANT_VARS=None, QUANT_DOC=NONEDOC, nfluid=None, rewrite=False):
    '''function factory; returns function(varname, vardoc, nfluid=None) which writes documentation of var.
    The documentation goes to vd['doc'] where vd = obj.vardict[METAQUANT][TYPE_QUANT][varname].

    Also store vd['nfluid'] = nfluid.
        vars_documenter(...,nfluid) -> store as default
        f = vars_documenter(); f(var, doc, nfluid) -> store for this var, only.
        nfluid =
            None -> does not even understand what a "fluid" is. (Use this value outside of load_mf_quantities.py)
                    Or, if in mf_quantities, None indicates nfluid has not been documented for this var.
            2    -> uses obj.ifluid and obj.jfluid to calculate results. (e.g. 'nu_ij')
            1    -> uses obj.ifluid (and not jfluid) to calculate results. (e.g. 'ux', 'tg')
            0    -> does not use ifluid nor jfluid to calculate results. (e.g. 'bx', 'nel', 'tot_e')

    METAQUANT (i.e. document_vars.METAQUANT) must be set before using vars_documenter;
        use document_vars.set_meta_quant() to accomplish this.
        Raises ValueError if METAQUANT has not been set.

    if QUANT_VARS is not None:
        initialize documentation of all the vars in varnames with vardoc=NONEDOC.
        enforce that only vars in QUANT_VARS can be documented (ignore documentation for all vars not in QUANT_DOC).

    if not rewrite, and TYPE_QUANT already in obj.vardict[METAQUANT].keys() (when vars_documenter is called),
        instead do nothing and return a function which does nothing.

    also sets obj.vardict[METAQUANT][TYPE_QUANT][document_vars.QUANTDOC] = QUANT_DOC.
    '''
    if METAQUANT is None:
        raise ValueError('METAQUANT cannot be None when calling vars_documenter. ' + \
                         'Use document_vars.set_meta_quant() to set METAQUANT.')
    vardict = getattr(obj, VARDICT)[METAQUANT]   #sets vardict = obj.vardict[METAQUANT]
    write = rewrite
    if not TYPE_QUANT in vardict.keys():
        vardict[TYPE_QUANT] = dict()
        vardict[TYPE_QUANT][QUANTDOC] = QUANT_DOC
        write = True
    if write:
        # define function (which will be returned)
        def document_var(varname, vardoc, nfluid=nfluid, **kw__more_info_about_var):
            '''puts documentation about var named varname into obj.vardict[TYPE_QUANT].'''
            if (QUANT_VARS is not None) and (varname not in QUANT_VARS):
                return
            tqd = vardict[TYPE_QUANT]
            var_info_dict = {'doc': vardoc, 'nfluid': nfluid, **kw__more_info_about_var}
            try:
                vd = tqd[varname]   # vd = vardict[TYPE_QUANT][varname] (if it already exists)
            except KeyError:        # else, initialize tqd[varname]:
                tqd[varname] = var_info_dict
            else:                   # if vd assignment was successful, set info.
                vd.update(var_info_dict)

        # initialize documentation to NONEDOC for var in QUANT_VARS
        if QUANT_VARS is not None:
            for varname in QUANT_VARS:
                document_var(varname, vardoc=NONEDOC, nfluid=nfluid)

        # return document_var function which we defined.
        return document_var
    else:
        # do nothing and return a function which does nothing.
        def dont_document_var(varname, vardoc, nfluid=None):
            '''does nothing.
            (because obj.vardict[TYPE_QUANT] already existed when vars_documenter was called).
            '''
            return
        return dont_document_var

def create_vardict(obj):
    '''call obj.get_var('') but with prints turned off.
    Afterwards, obj.vardict will be full of documentation.

    Also, set obj.gotten_vars() to a function which returns obj._quants_selected.
        (conceptually this belongs elsewhere but it is convenient to do it here.)
    '''
    # creat vardict
    setattr(obj, CREATING_VARDICT, True)
    obj.get_var('')
    setattr(obj, CREATING_VARDICT, False)
    # set gotten_vars
    obj.gotten_vars = types.MethodType(gotten_vars, obj)

def creating_vardict(obj, default=False):
    '''return whether obj is currently creating vardict. If unsure, return <default>.'''
    return getattr(obj, CREATING_VARDICT, default)


''' ----------------------------- search vardict ----------------------------- '''

def _apply_keys(d, keys):
    '''result result of successive application of (key for key in in keys) to dict of dicts, d.'''
    for key in keys:
        d = d[key]
    return d

search_result = collections.namedtuple('vardict_search_result', ('result', 'type', 'keys'))

def search_vardict(vardict, x):
    '''search vardict for x. x is the key we are looking for.

    return search_result named tuple. its attributes give (in order):
        result: the dict which x is pointing to.
        type: None or a string:
            None (vardict itself)
            'metaquant' (top-level)    # a dict of typequants
            'typequant' (middle-level) # a dict of vars
            'var'       (bottom-level) # a dict with keys 'doc' (documentation) and 'nfluid'
        keys: the list of keys to apply to vardict to get to result.
            when type is None, keys is [];
            when type is metaquant, keys is [x]
            when type is typequant, keys is [metaquantkey, x]
            when type is 'var', keys is [metaquantkey, typequantkey, x]

    return False if failed to find x in vardict.
    '''
    v = vardict
    if x is None:
        return search_result(result=v, type=None, keys=[])
    if x in v.keys():
        return search_result(result=v[x], type='metaquant', keys=[x])
    for metaquant in vardict.keys():
        v = vardict[metaquant]
        if not isinstance(v, dict): continue   # skip QUANTDOC
        if x in v.keys():
            return search_result(result=v[x], type='typequant', keys=[metaquant, x])
    for metaquant in vardict.keys():
        for typequant in vardict[metaquant].keys():
            v = vardict[metaquant][typequant]
            if not isinstance(v, dict): continue   # skip QUANTDOC
            if x in v.keys():
                return search_result(result=v[x], type='var', keys=[metaquant, typequant, x])
    return False


''' ----------------------------- prettyprint vardict ----------------------------- '''

TW = 3   # tabwidth
WS = ' ' # whitespace

def _underline(s, underline='-', minlength=0):
    '''return underlined s'''
    if len(underline.strip())==0:
        return s
    line = underline * math.ceil(max(len(s), minlength)/len(underline))
    return s + '\n' + line

def _intro_line(text, length=80):
    '''return fancy formatting of text as "intro line".'''
    left, right = '(<< ', ' >>)'
    length  = max(0, (length - len(left) - len(right)))
    fmtline = '{:^' + str(length) + '}'   # {:^N} makes for line which is N long, and centered.
    return (left + fmtline + right).format(text)

def _vardocs_var(varname, vd, q=WS*TW*2):
    '''docs for vd (var_dict). returns list containing one string, or None (if undocumented)'''
    vardoc = vd['doc']
    if vardoc is NONEDOC:
        return None
    else:
        nfluid = vd['nfluid']
        rstr = q + '{:10s}'.format(varname) + ' : '
        if nfluid is not None:
            rstr += '(nfluid = {}) '.format(nfluid)
        rstr += str(vardoc)
        return [rstr]

def _vardocs_typequant(typequant_dict, tqd=WS*TW, q=WS*TW*2, ud=WS*TW*3):
    '''docs for typequant_dict. returns list of strings, each string is one line.'''
    result = []
    if QUANTDOC in typequant_dict.keys():
        s = str(typequant_dict[QUANTDOC]).lstrip().replace('\n', tqd+'\n')
        s = s.rstrip() + '\n'   # make end have exactly 1 newline.
        result += [tqd + s]
    undocumented = []
    for varname in (key for key in sorted(typequant_dict.keys()) if key!=QUANTDOC):
        vd = typequant_dict[varname]
        vdv = _vardocs_var(varname, vd, q=q)
        if vdv is None:
            undocumented += [varname]
        else:
            result += vdv
    if undocumented!=[]:
        result += ['\n' + q + 'existing but undocumented vars:\n' + ud + ', '.join(undocumented)]
    return result

def _vardocs_metaquant(metaquant_dict, underline='-',
                       mqd=''*TW, tq=WS*TW, tqd=WS*TW, q=WS*TW*2, ud=WS*TW*3):
    '''docs for metaquant_dict. returns list of strings, each string is one line.'''
    result = []
    if QUANTDOC in metaquant_dict.keys():
        result += [mqd + str(metaquant_dict[QUANTDOC]).lstrip().replace('\n', mqd+'\n')]
    for typequant in (key for key in sorted(metaquant_dict.keys()) if key!=QUANTDOC):
        result += ['', _underline(tq + typequant, underline)]
        typequant_dict = metaquant_dict[typequant]
        result += _vardocs_typequant(typequant_dict, tqd=tqd, q=q, ud=ud)
    return result

def _vardocs_print(result, printout=True):
    '''x = '\n'.join(result). if printout, print x. Else, return x.'''
    stresult = '\n'.join(result)
    if printout:
        print(stresult)
    else:
        return stresult

def set_vardocs(obj, printout=True, underline='-', min_mq_underline=80,
                mqd=''*TW, tq=WS*TW, tqd=WS*TW, q=WS*TW*2, ud=WS*TW*3):
    '''make obj.vardocs be a function which prints vardict in pretty format.
    (return string instead if printout is False.)
    mqd, tq, tqd are indents for metaquant_doc, typequant, typequant_doc,
    q, ud are indents for varname, undocumented vars

    also make obj.vardoc(x) print doc for x, only.
    x can be a var, typequant, metaquant, or None (equivalent to vardocs if None).
    '''
    def vardocs(printout=True):
        '''prettyprint docs. If printout is False, return string instead of printing.'''
        result = [
            'Following is documentation for vars compatible with self.get_var(var).',
            _intro_line('Documentation contents available in dictionary form via self.{}'.format(VARDICT)),
            _intro_line('Documentation string available via self.vardocs(printout=False)'),
            ]
        vardict = getattr(obj, VARDICT)
        for metaquant in sorted(vardict.keys()):
            result += ['', '', _underline(metaquant, underline, minlength=min_mq_underline)]
            metaquant_dict = vardict[metaquant]
            result += _vardocs_metaquant(metaquant_dict, underline=underline,
                                         mqd=mqd, tq=tq, tqd=tqd, q=q, ud=ud)
        return _vardocs_print(result, printout=printout)

    obj.vardocs = vardocs

    def vardoc(x=None, printout=True):
        '''prettyprint docs for x. x can be a var, typequant, metaquant, or None.

        default x is None; when x is None, this function is equivalent to vardocs().

        If printout is False, return string instead of printing.
        '''
        search = search_vardict(obj.vardict, x)
        if search == False:
            result = ["key '{}' does not exist in obj.vardict!".format(x)]
            return _vardocs_print(result, printout)
        # else: search was successful.
        if search.type is None:
            return vardocs(printout=printout)
        # else: search actually did something nontrivial.
        keystr = ''.join(["['{}']".format(key) for key in search.keys])
        result = ['vardoc for {}, accessible via obj.vardict{}'.format(x, keystr)]
        if search.type == 'metaquant':
            result += _vardocs_metaquant(search.result, underline=underline,
                                         mqd=mqd, tq=tq, tqd=tqd, q=q, ud=ud)
        elif search.type == 'typequant':
            result += _vardocs_typequant(search.result, tqd=tqd, q=q, ud=ud)
        elif search.type == 'var':
            result += _vardocs_var(x, search.result, q=q)
        return _vardocs_print(result, printout)

    obj.vardoc = vardoc

    def _search_vardict(x):
        '''searches self.vardict for x. x can be a var, typequant, metaquant, or None.'''
        vardict = getattr(obj, VARDICT)
        return search_vardict(vardict, x)

    obj.search_vardict = _search_vardict


''' ----------------------------- quant tracking ----------------------------- '''

QuantInfo = collections.namedtuple('QuantInfo', ('quant', 'typequant', 'metaquant'))

def _track_quants_selected(obj, info, maxlen=QUANT_TRACKING_N):
    '''updates obj._quants_selected with info.
    if _quants_selected attr doesn't exist, make a deque.

    maxlen for deque will be obj._quant_ntracking if it exists; else value of maxlen kwarg.
    '''
    if hasattr(obj, QUANTS_SELECTED):
        getattr(obj, QUANTS_SELECTED).appendleft(info)
    else:
        maxlen = getattr(obj, QUANT_NTRACKING, maxlen)  # maxlen kwarg is default value.
        setattr(obj, QUANTS_SELECTED, collections.deque([info], maxlen))
    return getattr(obj, QUANTS_SELECTED)

def setattr_quant_selected(obj, quant, typequant, metaquant=None, delay=False):
    '''sets obj._quant_selected = QuantInfo(quant, typequant, metaquant)
    if metaquant is None, use helita.sim.document_vars.METAQUANT as default.
    returns the value in obj._quant_selected.

    if delay, set QUANT_SELECTION instead of QUANT_SELECTED.
    (if delay, it is recommended to later call quant_select_selection to update.)
    QUANT_SELECTION is maintained by document_vars.quant_tracking_top_level() wrapper
    '''
    if metaquant is None:
        metaquant = METAQUANT
    info = QuantInfo(quant=quant, typequant=typequant, metaquant=metaquant)
    if delay:
        setattr(obj, QUANT_SELECTION, info)
    else:
        setattr(obj, QUANT_SELECTED, info)
        _track_quants_selected(obj, info)
    return info

def select_quant_selection(obj, info_default=None):
    '''puts data from QUANT_SELECTION into QUANT_SELECTED.
    Also, updates QUANTS_SELECTED with the info.

    Recommended to only use after doing setattr_quant_selected(..., delay=True).
    '''
    info = getattr(obj, QUANT_SELECTION, info_default)
    setattr(obj, QUANT_SELECTED, info)
    _track_quants_selected(obj, info)
    return info

def quant_tracking_simple(typequant, metaquant=None):
    '''returns a function dectorator which turns f(obj, quant, *args, **kwargs) into:
        result = f(...)
        if result is not None:
            obj._quant_selected = QuantInfo(quant, typequant, metaquant)
        return result
    if metaquant is None, use helita.sim.document_vars.METAQUANT as default.

    Suggested use:
        use this wrapper for any get_..._quant functions whose results correspond to
        the entire quant. For examples see helita.sim.load_mf_quantities.py (or load_quantities.py).

        do NOT use this wrapper for get_..._quant functions whose results correspond to
        only a PART of the quant entered. For example, don't use this for get_square() in
        load_arithmetic_quantities, since that function pulls a '2' off the end of quant,
        E.g. get_var('b2') --> bx**2 + by**2 + bz**2. For this case, see quant_tracker() instead.
    '''
    def decorator(f):
        @functools.wraps(f)
        def f_but_quant_tracking(obj, quant, *args, **kwargs):
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            result = f(obj, quant, *args, **kwargs)
            if result is not None:
                setattr_quant_selected(obj, quant, typequant, metaquant)
            return result
        return f_but_quant_tracking
    return decorator

def quant_tracking_top_level(f):
    '''function decorator which is like quant_tracking_simple(...), with contents
    depending on whether we are LOADING_QUANTITY right now or not.
    if we ARE loading_quantity right now, wrap with:
        quant_tracking_simple(TYPEQUANT_LEVEL_N, METAQUANT_LEVEL_N)
    if we are NOT loading_quantity, wrap with:
        quant_tracking_simple(TYPEQUANT_TOP_LEVEL, METAQUANT_TOP_LEVEL)
    '''
    @file_memory.maintain_attrs(QUANT_SELECTION)
    @functools.wraps(f)
    def f_but_quant_tracking_level(obj, quant, *args, **kwargs):
        __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
        # figure out names for typequant and metaquant for quant_tracking_simple wrapper
        loading_level = getattr(obj, LOADING_LEVEL, 0)
        if loading_level == 0: # we are NOT loading; use TOP level.
            typequant = TYPEQUANT_TOP_LEVEL
            metaquant = METAQUANT_TOP_LEVEL
        else:                  # we are MID loading; use mid level and format with the level info.
            typequant = TYPEQUANT_LEVEL_N.format(level=loading_level)
            metaquant = METAQUANT_LEVEL_N.format(level=loading_level)
        # define f but wrapped by quant_tracking_simple.
        @quant_tracking_simple(typequant, metaquant)
        def wrapped_f(obj, quant, *args, **kwargs):
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            return f(obj, quant, *args, **kwargs)
        # call the f wrapped by quant_tracking simple; return the result.
        return wrapped_f(obj, quant, *args, **kwargs)
    return f_but_quant_tracking_level


''' ----------------------------- quant tracking - lookup ----------------------------- '''

def gotten_vars(obj, hide_level=3, hide_interp=True, hide=[], hidef=lambda info: False,
                hide_quants=[], hide_typequants=[], hide_metaquants=[]):
    '''returns obj._quants_selected, which shows the most recent quants which get_var got.

    It is possible to hide quants from the list using the kwargs of this function.

    hide_level: integer (default 2)
        hide all QuantInfo tuples with typequant or metaquant like 'level_n' with n >= hide_level.
        case insensitive. Also, 'top_level' is treated like 'level_0'.
        (This can only hide quants with otherwise unnamed typequant/metaquant info.)
    hide_interp: True (default) or False
        whether hide all quants which are one of the following types:
        'INTERP_QUANT', 'CENTER_QUANT'
    hide: list. (default [])
        hide all QuantInfo tuples with quant, typequant, or metaquant in this list.
    hidef: function(info) --> bool. Default: (lambda info: False)
        if evaluates to True for a QuantInfo tuple, hide this info.
        Such objects are namedtuples, with contents (quant, typequant, metaquant).
    hide_quants:     list. (default [])
        hide all QuantInfo tuples with   quant   in this list.
    hide_typequants: list. (default [])
        hide all QuantInfo tuples with typequant in this list.
    hide_metaquants: list. (default [])
        hide all QuantInfo tuples with metaquant in this list.
    '''
    quants_selected = getattr(obj, QUANTS_SELECTED, collections.deque([]))
    result = collections.deque(maxlen=quants_selected.maxlen)
    for info in quants_selected:
        quant, typequant, metaquant = info
        # if we should hide this one, continue.
        level = _get_level_from_quant_info(info)
        if level is not None:
            if level >= hide_level:
                continue
        if hide_interp:
            if typequant in ['INTERP_QUANT', 'CENTER_QUANT']:
                continue
        if len(hide) > 0:
            if (quant in hide) or (typequant in hide) or (metaquant in hide):
                continue
        if (quant in hide_quants) or (typequant in hide_typequants) or (metaquant in hide_metaquants):
            continue
        if hidef(info):
            continue
        # else, add this one to result.
        result.append(info)
    return result

def _get_level_from_q(q):
    '''gets level from typequant or metaquant q.
    Returns None if q is not a level quant.
    '''
    q = q.lower()
    if not 'level' in q:
        return None
    levelstr, _, leveln = q.rpartition('_')
    if levelstr == 'level':   # q.lower() looks like level_N
        return int(leveln)
    if levelstr == 'top':     # q.lower() looks like top_level
        return 0

def _get_level_from_quant_info(quant_info):
    '''gets level from QuantInfo tuple.
    Returns None if q is not a level quant.
    '''
    typelevel = _get_level_from_q(quant_info.typequant)
    metalevel = _get_level_from_q(quant_info.metaquant)
    if typelevel is None:
        if metalevel is None:
            return None
        else:
            return metalevel
    elif metalevel is None:
        return typelevel
    else:
        return max(typelevel, metalevel)