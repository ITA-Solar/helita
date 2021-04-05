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
"""

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = '(not yet documented)'        #default documentation if none is provided.
QUANTDOC = '_DOC_QUANT'                 #key for dd.vardict[TYPE_QUANT] containing doc for what TYPE_QUANT means.
CREATING_VARDICT = '_creating_vardict'  #attribute of obj which tells if we are running get_var('') to create vardict.

# global variable which tells which quantity you are setting now.
METAQUANT = None

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

    vardict[METAQUANT] = dict()
    vardict[METAQUANT][QUANTDOC] = QUANT_DOC

def vars_documenter(obj, TYPE_QUANT, QUANT_VARS=None, QUANT_DOC=NONEDOC, rewrite=False):
    '''function factory; returns function(varname, vardoc) which writes documentation of var.
    The documentation goes to obj.vardict[METAQUANT][TYPE_QUANT].
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
        def document_var(varname, vardoc):
            '''puts documentation about var named varname into obj.vardict[TYPE_QUANT].'''
            if (QUANT_VARS is not None) and (varname not in QUANT_VARS):
                return
            vardict[TYPE_QUANT][varname] = vardoc
        # initialize documentation to NONEDOC for var in QUANT_VARS
        if QUANT_VARS is not None:
            for varname in QUANT_VARS:
                document_var(varname, vardoc=NONEDOC)
        return document_var
    else:
        # do nothing and return a function which does nothing.
        def dont_document_var(varname, vardoc):
            '''does nothing.
            (because obj.vardict[TYPE_QUANT] already existed when vars_documenter was called).
            '''
            return
        return dont_document_var

def create_vardict(obj):
    '''call obj.get_var('') but with prints turned off. Afterwards, obj.vardict will be full of documentation.'''
    setattr(obj, CREATING_VARDICT, True)
    obj.get_var('')
    setattr(obj, CREATING_VARDICT, False)

def creating_vardict(obj, default=False):
    '''return whether obj is currently creating vardict. If unsure, return <default>.'''
    return getattr(obj, CREATING_VARDICT, default)

# TODO: make something which helps tell new users how to use vardict.
#   (maybe? I mean, it's just a dictionary, which shouldn't be too complicated.)