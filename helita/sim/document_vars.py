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


[TODO]:
    - [FIX] Solve sad interaction between self.variables checking and quant_tracking.
        - E.g. get_var('r') after initialization of EbysusData confuses quant_tracking,
            because it returns the data in self.r immediately, instead of loading a quantity.

"""

# import built-ins
import math #for pretty strings
import collections
import functools
import copy   # for deepcopy for QuantTree

# import internal modules
from . import units       # not used heavily; just here for setting defaults, and setting obj.get_units

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = '(not yet documented)'        #default documentation if none is provided.
QUANTDOC = '_DOC_QUANT'                 #key for dd.vardict[TYPE_QUANT] containing doc for what TYPE_QUANT means.
NFLUID  = 'nfluid'    #key which stores number of fluids. (e.g. 0 for none; 1 for "uses ifluid but not jfluid". 
CREATING_VARDICT = '_creating_vardict'  #attribute of obj which tells if we are running get_var('') to create vardict.

# defaults for quant tracking
## attributes of obj
VARNAME_INPUT   = '_varname_input'      #stores name of most recent variable which was input to get_var.
QUANT_SELECTED  = '_quant_selected'     #stores vardict lookup info for the latest quant selected.
QUANTS_SELECTED = '_quants_selected'    #stores quant_selected for QUANT_NTRACKING recent quants.
QUANTS_BY_LEVEL = '_quants_by_level'    #stores quant_selected by level.
QUANTS_TREE     = '_quants_tree'        #stores quant_selected as a tree. 
QUANT_SELECTION = '_quant_selection'    #stores info for latest quant selected; use for hesitant setting.
QUANT_NTRACKING = '_quant_ntracking'    #if it exists, sets maxlen for _quants_selected deque.

## misc
QUANT_TRACKING_N = 1000       #default for number of quant selections to remember.
QUANT_BY_LEVEL_N = 50         #default for number of quant selections to remember at each level.
QUANT_NOT_FOUND  = '???'      #default for typequant and metaquant when quant is not found. Do not use None.

# defaults for loading level tracking
## attribute of obj which tells how deep we are into loading a quantity right now. 0 = top level.
LOADING_LEVEL    = '_loading_level'


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

def vars_documenter(obj, TYPE_QUANT, QUANT_VARS=None, QUANT_DOC=NONEDOC, nfluid=None, rewrite=False, **kw__defaults):
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

    kw__defaults become default values for all quants in QUANT_VARS.
        Example:
            docvar = vars_documenter(obj, typequant, ['var1', 'var2', 'var3'], foo_kwarg='bar')
            docvar('var1', 'info about var 1')
            docvar('var2', 'info about var 2', foo_kwarg='overwritten')
            docvar('var3', 'info about var 3')
        Leads to obj.vardict[METAQUANT][typequant] like:
            {'var1': {'doc':'info about var 1', 'foo_kwarg':'bar'},
             'var2': {'doc':'info about var 2', 'foo_kwarg':'overwritten'},
             'var3': {'doc':'info about var 3', 'foo_kwarg':'bar'}}

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
            var_info_dict = {'doc': vardoc, 'nfluid': nfluid, **kw__defaults, **kw__more_info_about_var}
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

    Also, set a few more things (conceptually these belong elsewhere,
    but it is convenient to do them here because create_vardict is called in __init__ for all the DataClass objects) :
        set obj.gotten_vars() to a function which returns obj._quants_selected.
        set obj.got_vars_tree() to a function which returns obj._quants_tree.
        set obj.quant_lookup() to a function which returns dict of info about quant, as found in obj.vardict.
    '''
    # creat vardict
    setattr(obj, CREATING_VARDICT, True)
    obj.get_var('')
    setattr(obj, CREATING_VARDICT, False)
    # set some other useful functions in obj.
    def _make_weak_bound_method(f):
        @functools.wraps(f)
        def _weak_bound_method(*args, **kwargs):
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            return f(obj, *args, **kwargs)   # << obj which was passed to create_vardict
        return _weak_bound_method
    obj.gotten_vars    = _make_weak_bound_method(gotten_vars)
    obj.got_vars_tree  = _make_weak_bound_method(got_vars_tree)
    obj.get_quant_info = _make_weak_bound_method(get_quant_info)
    obj.get_var_info   = obj.get_quant_info   # alias
    obj.quant_lookup   = _make_weak_bound_method(quant_lookup)
    obj.get_units      = _make_weak_bound_method(units.get_units)

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


''' --------------------- restore attrs --------------------- '''

# this helper function probably should go in another file, but this is the best place for it for now.

def maintain_attrs(*attrs):
    '''return decorator which restores attrs of obj after running function.
    It is assumed that obj is the first arg of function.
    '''
    def attr_restorer(f):
        @functools.wraps(f)
        def f_but_maintain_attrs(obj, *args, **kwargs):
            '''f but attrs are maintained.'''
            __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
            memory = dict()  # dict of attrs to maintain
            for attr in attrs:
                if hasattr(obj, attr):
                    memory[attr] = getattr(obj, attr)
            try:
                return f(obj, *args, **kwargs)
            finally:
                # restore attrs
                for attr, val in memory.items(): 
                    setattr(obj, attr, val)
        return f_but_maintain_attrs
    return attr_restorer


''' ----------------------------- quant tracking ----------------------------- '''

QuantInfo = collections.namedtuple('QuantInfo', ('varname', 'quant', 'typequant', 'metaquant', 'level'),
                                   defaults = [None, None, None, None, None])

def setattr_quant_selected(obj, quant, typequant, metaquant=None, varname=None, level=None, delay=False):
    '''sets QUANT_SELECTED to QuantInfo(varname, quant, typequant, metaquant, level).

    varname   = name of var which was input.
        default (if None): getattr(obj, VARNAME_INPUT, None)
    quant     = name of quant which matches var.
        (e.g. '2' maches 'b2'; see get_square from load_arithmetic_quantities.)
    typequant = type associated with quant
        e.g. 'SQUARE_QUANT'
    metaquant = metatype associated with quant
        e.g. 'arquantities'
        default (if None): METAQUANT (global variable in document_vars module, set in load_..._quantities files).
    level     = loading_level
        i.e. number of layers deep right now in the chain of get_var call(s).
        default (if None): getattr(obj, LOADING_LEVEL, 0)

    if metaquant is None, use helita.sim.document_vars.METAQUANT as default.
    returns the value in obj._quant_selected.

    if delay, set QUANT_SELECTION instead of QUANT_SELECTED.
    (if delay, it is recommended to later call quant_select_selection to update.)
    QUANT_SELECTION is maintained by document_vars.quant_tracking_top_level() wrapper
    '''
    if varname is None:
        varname = getattr(obj, VARNAME_INPUT, None)
    if metaquant is None:
        metaquant = METAQUANT
    if level is None:
        level = getattr(obj, LOADING_LEVEL, 0)
        loading_level = level
    info = QuantInfo(varname=varname, quant=quant, typequant=typequant, metaquant=metaquant, level=level)
    if delay:
        setattr(obj, QUANT_SELECTION, info)
    else:
        setattr(obj, QUANT_SELECTED, info)
        _track_quants_selected(obj, info)
    return info

def _track_quants_selected(obj, info, maxlen=QUANT_TRACKING_N):
    '''updates obj._quants_selected with info.
    if _quants_selected attr doesn't exist, make a deque.

    maxlen for deque will be obj._quant_ntracking if it exists; else value of maxlen kwarg.

    Also, updates obj._quants_by_level with info. (same info; different format.)
    '''
    # put info into QUANTS_SELECTED
    if hasattr(obj, QUANTS_SELECTED):
        getattr(obj, QUANTS_SELECTED).appendleft(info)
    else:
        maxlen = getattr(obj, QUANT_NTRACKING, maxlen)  # maxlen kwarg is default value.
        setattr(obj, QUANTS_SELECTED, collections.deque([info], maxlen))

    # put info into QUANTS_BY_LEVEL
    loading_level = getattr(obj, LOADING_LEVEL, 0)
    if not hasattr(obj, QUANTS_BY_LEVEL):
        setattr(obj, QUANTS_BY_LEVEL,
                {loading_level : collections.deque([info], maxlen=QUANT_BY_LEVEL_N)})
    else:
        qbl_dict = getattr(obj, QUANTS_BY_LEVEL)
        try:
            qbl_dict[loading_level].appendleft(info)
        except KeyError:
            qbl_dict[loading_level] = collections.deque([info], maxlen=QUANT_BY_LEVEL_N)

    # put info in QUANTS_TREE
    if hasattr(obj, QUANTS_TREE):
        getattr(obj, QUANTS_TREE).set_data(info)

    # return QUANTS_SELECTED
    return getattr(obj, QUANTS_SELECTED)

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
            # we need to save original METAQUANT now, because doing f might change METAQUANT.
            ## (quant_tracking_simple is meant to wrap a function inside a load_..._quantities file,
            ## so when that function (f) is called, METAQUANT will be the correct value.)
            if metaquant is None:
                remembered_metaquant = METAQUANT
            else:
                remembered_metaquant = metaquant
            # call f
            result = f(obj, quant, *args, **kwargs)
            # set quant_selected
            if result is not None:
                setattr_quant_selected(obj, quant, typequant, remembered_metaquant)
            # return result of f
            return result
        return f_but_quant_tracking
    return decorator

class QuantTree:
    '''use for tree representation of quants.

    Notes:
    - level should always be larger for children than for their parents.
    '''
    def __init__(self, data, level=-1):
        self.data     = data
        self.children = []
        self._level   = level
        self.hide_level = None

    def add_child(self, child, adjusted_level=False):
        '''add child to self.

        If child is QuantTree and adjusted_level=True,
            instead append a copy of child, with its _level adjusted to self._level + 1
        '''
        if isinstance(child, QuantTree):
            if adjusted_level:
                child = child.with_adjusted_base_level(self._level + 1)
            self.children.append(child)   # newest child goes at end of list.
        else:
            child = QuantTree(child, level=self._level + 1)
            self.add_child(child)
        return child

    def set_data(self, data):
        self.data = data

    def __str__(self):
        lvlstr = ' '*self._level + '(L{level}) '.format(level=self._level)
        # check hide level. if level >= hide_level, hide.
        if self.hide_level is not None:
            if self._level >= self.hide_level:
                return (lvlstr + '{}').format(repr(self))
        # << if I reach this line it means I am not hiding myself.
        # if no children, return string with level and data.
        if len(self.children) == 0:
            return (lvlstr + '{data}').format(data=self.data)
        # else, we have children, so return a string with level, data, and children
        def _child_to_str(child):
            return '\n' + child.str(self.hide_level, count_from_here=False)
        children_strs = ','.join([_child_to_str(child) for child in self.children])
        return (lvlstr + '{data} : {children}').format(data=self.data, children=children_strs)

    def __repr__(self):
        if isinstance(self.data, QuantInfo):
            qi_str = "(varname='{}', quant='{}')".format(self.data.varname, self.data.quant)
        else:
            qi_str = ''
        fmtdict = dict(qi_str=qi_str, hexid=hex(id(self)), Nnode=1 + self.count_descendants())
        return '<<QuantTree{qi_str} at {hexid}> with {Nnode} nodes.>'.format(**fmtdict)

    def str(self, hide_level=None, count_from_here=True):
        '''sets self.hide_level, returns str(self).
        restores self.hide_level to its original value before returning result.

        if count_from_here, only hides after getting hide_level more levels deep than we are now.
        (e.g. if count_from_here, and self is level 7, and hide_level is 3: hides level 10 and greater.)
        '''
        orig_hide = self.hide_level
        if count_from_here and (hide_level is not None):
            hide_level = hide_level + self._level
        self.hide_level = hide_level
        try:
            return self.__str__()
        finally:
            self.hide_level = orig_hide

    def get_child(self, i_child=0, oldest_first=True):
        '''returns the child determined by index i_child.

        self.get_child() (with no args/kwargs entered) returns the oldest child (the child added first).

        Parameters
        ----------
        i_child: int (default 0)
            index of child to get. See oldest_first kwarg for ordering convention.
        oldest_first: True (default) or False.
            Determines the ordering convention for children:
            True --> sort from oldest to youngest (0 is the oldest child (added first)).
                     Equivalent to self.children[i_child]
            False -> sort from youngest to oldest (0 is the youngest child (added most-recently)).
                     Equivalent to self.children[::-1][i_child]
        '''
        if oldest_first:
            return self.children[i_child]
        else:
            return self.children[::-1][i_child]
        
    def set_base_level(self, level):
        '''sets self._level to level; also adjusts level of all children appropriately.

        Example: self._level==2, self.children[0]._level==3;
            self.set_base_level(0) --> self._level==0, self.children[0]._level==1
        '''
        lsubtract = self._level - level
        self._adjust_base_level(lsubtract)

    def _adjust_base_level(self, l_subtract):
        '''sets self._level to self._level - l_subtract.
        Also decreases level of all children by ldiff, and all childrens' children, etc.
        '''
        self._level -= l_subtract
        for child in self.children:
            child._adjust_base_level(l_subtract)

    def with_adjusted_base_level(self, level):
        '''set_base_level(self) except it is nondestructive, i.e. sets level for a deepcopy of self.
        returns the copy with the base level set to level.
        '''
        result = copy.deepcopy(self)
        result.set_base_level(level)
        return result

    def count_descendants(self):
        '''returns total number of descendants (children, childrens' children, etc) of self.'''
        result = len(self.children)
        for child in self.children:
            result += child.count_descendants()
        return result


def _get_orig_tree(obj):
    '''gets QUANTS_TREE from obj (when LOADING_LEVEL is not -1; else, returns a new QuantTree).'''
    loading_level = getattr(obj, LOADING_LEVEL, -1) # get loading_level. Outside of f, the default is -1.
    if (loading_level== -1) or (not hasattr(obj, QUANTS_TREE)):
        orig_tree = QuantTree(None, level=-1)
    else:
        orig_tree = getattr(obj, QUANTS_TREE)
    return orig_tree

def quant_tree_tracking(f):
    '''wrapper for f which makes it track quant tree.
    
    QUANTS_TREE (attr of obj) will be a tree like:
    (L(N-1)) None :
    (L(N)) QuantInfo(var_current_layer) :
     (L(N+1)) QuantInfo(var_1, i.e. 1st var gotten while calculating var_current_layer) :
      (L(N+2)) ... (tree for var_1)
     (L(N+1)) QuantInfo(var_2, i.e. 2nd var gotten while calcualting var_current_layer) :
      (L(N+2)) ... (tree for var_2)
    
    Another way to write it; it will be a tree like:
    QuantTree(data=None, level=N-1, children= \
    [
        QuantTree(data=QuantInfo(var_current_layer), level=N, children= \
        [
            QuantTree(data=QuantInfo(var_1, level=N+1, children=...)),
            QuantTree(data=QuantInfo(var_2, level=N+1, children=...)),
            ...
        ])
    ])
    '''
    @functools.wraps(f)
    def f_but_quant_tree_tracking(obj, varname, *args, **kwargs):
        __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
        orig_tree  = _get_orig_tree(obj)
        tree_child = orig_tree.add_child(None)
        setattr(obj, QUANTS_TREE, tree_child)
        # call f.
        result = f(obj, varname, *args, **kwargs)
        # retore original tree. (The data is set by f, via _track_quants_selected and quant_tracking_top_level)
        setattr(obj, QUANTS_TREE, orig_tree)
        # return result of f.
        return result
    return f_but_quant_tree_tracking

def quant_tracking_top_level(f):
    '''decorator which improves quant tracking. (decorate _load_quantities using this.)'''
    @quant_tree_tracking
    @maintain_attrs(LOADING_LEVEL, VARNAME_INPUT, QUANT_SELECTION, QUANT_SELECTED)
    @functools.wraps(f)
    def f_but_quant_tracking_level(obj, varname, *args, **kwargs):
        __tracebackhide__ = HIDE_DECORATOR_TRACEBACKS
        setattr(obj, LOADING_LEVEL, getattr(obj, LOADING_LEVEL, -1) + 1) # increment LOADING_LEVEL.
        setattr(obj, VARNAME_INPUT, varname)      # save name of the variable which was input.
        setattr(obj, QUANT_SELECTED, QuantInfo(None))  # smash QUANT_SELECTED before doing f.
        result = f(obj, varname, *args, **kwargs)
        # even if we don't recognize this quant (because we didn't put quant tracking document_vars code for it (yet)),
        ## we should still set QUANT_SELECTED to the info we do know (i.e. the varname), with blanks for what we dont know.
        quant_info = getattr(obj, QUANT_SELECTED, QuantInfo(None))
        if quant_info.varname is None:  # f did not set varname for quant_info, so we'll do it now.
            setattr_quant_selected(obj, quant=QUANT_NOT_FOUND, typequant=QUANT_NOT_FOUND, metaquant=QUANT_NOT_FOUND,
                                   varname=varname)
        return result
    return f_but_quant_tracking_level


def get_quant_tracking_state(obj, from_internal=False):
    '''returns quant tracking state of obj.
    The state includes only the quant_tree and the quant_selected.

    from_internal: False (default) or True
        True  <-> use when we are caching due to a "with Caching(...)" block.
        False <-> use when we are caching due to the "@with_caching" wrapper.
    '''
    if not hasattr(obj, QUANTS_TREE):   # not sure if this ever happens.
        quants_tree = QuantTree(None)   #   if it does, return an empty QuantTree so we don't crash.
    elif from_internal:   # we are saving state while INSIDE quant_tree_tracking (inside _load_quantity).
        # QUANTS_TREE looks like QuantTree(None) but it is the child of the tree which will have
        ## the data that we are getting from this call of get_var (which we are inside now).
        ## Thus, when the call to get_var is completed, the data for this tree will be filled
        ## with the appropriate QuantInfo about the quant we are getting now.
        quants_tree = getattr(obj, QUANTS_TREE) 
    else:                 # we are saving state while OUTSIDE quant_tree_tracking (outside _load_quantity).
        # QUANTS_TREE looks like [None : [..., QuantTree(QuantInfo( v ))]] where
        ## v is the var we just got with the latest call to _load_quantity.
        quants_tree = getattr(obj, QUANTS_TREE).get_child(-1)  # get the newest child.
    state = dict(quants_tree    = quants_tree,
                 quant_selected = getattr(obj, QUANT_SELECTED, QuantInfo(None)),
                 _from_internal = from_internal,  # not used, but maybe helpful for debugging.
                 _ever_restored = False # whether we have ever restored this state.
                )
    return state

def restore_quant_tracking_state(obj, state):
    '''restores the quant tracking state of obj.'''
    state_tree   = state['quants_tree']
    obj_tree     = _get_orig_tree(obj)
    child_to_add = state_tree   # add state tree as child of obj_tree.
    if not state['_ever_restored']:
        state['_ever_restored'] = True
        if isinstance(child_to_add.data, QuantInfo):
            # adjust level of top QuantInfo in tree, to indicate it is from cache.
            q = child_to_add.data._asdict()
            q['level'] = str(q['level']) + ' (FROM CACHE)'
            child_to_add.data = QuantInfo(**q)
    # add child to obj_tree.
    obj_tree.add_child(child_to_add, adjusted_level=True)
    setattr(obj, QUANTS_TREE, obj_tree)
    # set QUANT_SELECTED.
    selected = state.get('quant_selected', QuantInfo(None))
    setattr(obj, QUANT_SELECTED, selected)


''' ----------------------------- quant tracking - lookup ----------------------------- '''

def gotten_vars(obj, hide_level=3, hide_interp=True, hide=[], hidef=lambda info: False,
                hide_quants=[], hide_typequants=[], hide_metaquants=[]):
    '''returns obj._quants_selected, which shows the most recent quants which get_var got.

    It is possible to hide quants from the list using the kwargs of this function.

    hide_level: integer (default 3)
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
        quant, typequant, metaquant = info.quant, info.typequant, info.metaquant
        varname, level = info.varname, info.level
        # if we should hide this one, continue.
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

def got_vars_tree(obj, as_data=False, hide_level=None, i_child=0, oldest_first=True):
    '''prints QUANTS_TREE for obj.
    This tree shows the vars which were gotten during the most recent "level 0" call to get_var.
    (Calls to get_var from outside of helita have level == 0.)

    Use as_data=True to return the QuantTree object instead of printing it.

    Use hide_level=N to hide all layers of the tree with L >= N.

    Use i_child to get a child other than children[-1] (default).
        Note that if you are calling got_vars_tree externally (not inside any calls to get_var),
        then there will be only one child, and it will be the QuantTree for the var passed to get_var.

    Use oldest_first to tell the children ordering convention:
        True --> 0 is the oldest child (added first); -1 is the newest child (added most-recently).
        False -> the order is reversed, e.g. -1 is the oldest child instead.
    '''
    # Get QUANTS_TREE attr. Since this function (got_vars_tree) is optional, and for end-user,
    ## crash elegantly if obj doesn't have QUANTS_TREE, instead of trying to handle the crash.
    quants_tree = getattr(obj, QUANTS_TREE)
    # By design, data in top level of QUANTS_TREE is always None
    ## (except inside the wrapper quant_tree_tracking, which is the code that manages the tree).
    ## Thus the top level of quants_tree is not useful data, so we go to a child instead.
    quants_tree = quants_tree.get_child(i_child, oldest_first)
    # if as_data, return. Else, print.
    if as_data:
        return quants_tree
    else:
        print(quants_tree.str(hide_level=hide_level))

def quant_lookup(obj, quant_info):
    '''returns entry in obj.vardict related to quant_info (a QuantInfo object).
    if quant_info does not have an entry in obj.vardict, return an empty dict().
    '''
    vardict = getattr(obj, VARDICT, dict())
    metaquant_dict = vardict.get(quant_info.metaquant, dict())
    typequant_dict = metaquant_dict.get(quant_info.typequant, dict())
    quant_dict     = typequant_dict.get(quant_info.quant, dict())
    return quant_dict

def get_quant_info(obj, lookup_in_vardict=False):
    '''returns QuantInfo object for the top-level quant in got_vars_tree.
    If lookup_in_vardict, also use obj.quant_lookup to look up that info in obj.vardict.
    '''
    quant_info = got_vars_tree(obj, as_data=True, i_child=0).data
    if lookup_in_vardict:
        return quant_lookup(obj, quant_info)
    else:
        return quant_info