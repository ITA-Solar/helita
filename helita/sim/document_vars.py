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

#import built-ins
import math #for pretty strings
import collections
import functools
import types  # for MethodType

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = '(not yet documented)'        #default documentation if none is provided.
QUANTDOC = '_DOC_QUANT'                 #key for dd.vardict[TYPE_QUANT] containing doc for what TYPE_QUANT means.
NFLUID  = 'nfluid'    #key which stores number of fluids. (e.g. 0 for none; 1 for "uses ifluid but not jfluid". 
CREATING_VARDICT = '_creating_vardict'  #attribute of obj which tells if we are running get_var('') to create vardict.

QUANT_SELECTED  = '_quant_selected'     #attribute of obj which stores vardict lookup info for the latest quant selected.
QUANTS_SELECTED = '_quants_selected'    #attribute of obj which stores quant_selected for QUANT_NTRACKING recent quants.
QUANT_NTRACKING = '_quant_ntracking'    #attribute of obj which, if it exists, sets maxlen for _quants_selected deque.
QUANT_TRACKING_N = 50                   #default for number of quant selections to remember.

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

def gotten_vars(obj):
    '''returns obj._quants_selected, which shows the most recent quants which get_var got.'''
    return getattr(obj, QUANTS_SELECTED, [])


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

QuantVardictInfo = collections.namedtuple('QuantVardictInfo', ('quant', 'typequant', 'metaquant'))

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

def setattr_quant_selected(obj, quant, typequant, metaquant=None):
    '''sets obj._quant_selected = QuantVardictInfo(quant, typequant, metaquant)
    if metaquant is None, use helita.sim.document_vars.METAQUANT as default.
    returns the value in obj._quant_selected.
    '''
    if metaquant is None:
        metaquant = METAQUANT
    info = QuantVardictInfo(quant=quant, typequant=typequant, metaquant=metaquant)
    setattr(obj, QUANT_SELECTED, info)
    _track_quants_selected(obj, info)
    return info

def quant_tracking_simple(typequant, metaquant=None):
    '''returns a function dectorator which turns f(obj, quant, *args, **kwargs) into:
        result = f(...)
        if result is not None:
            obj._quant_selected = QuantVardictInfo(quant, typequant, metaquant)
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

def quant_tracker(obj, typequant, metaquant=None):
    '''returns a function f(quant) which sets obj._quant_selected appropriately.
    Use this in cases where quant_tracking_simple is not sufficienct. 
    For examples see the file helita.sim.load_arithmetic_quantities.py.
    '''
    def set_quant_selected(quant):
        '''sets obj._quant_selected = QuantVardictInfo(quant, typequant, metaquant).
        This function was generated by the function helita.sim.document_vars.quant_tracker().
        '''
        return setattr_quant_selected(obj, quant, typequant, metaquant)
    return set_quant_selected