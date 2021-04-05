"""
Created by Sam Evans on Apr 3 2021

Purpose: helper functions for documentation of variables.
"""

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = '(not yet documented)'        #default documentation if none is provided.
QUANTDOC = '_DOC_QUANT'                 #key for dd.vardict[TYPE_QUANT] containing doc for what TYPE_QUANT means.
CREATING_VARDICT = '_creating_vardict'  #attribute of obj which tells if we are running get_var('') to create vardict.

def vars_documenter(obj, TYPE_QUANT, QUANT_VARS=None, QUANT_DOC=NONEDOC, rewrite=False):
    '''function factory; returns function which documents a var for obj in obj.vardict[TYPE_QUANT].
    if QUANT_VARS is not None, also documents all the vars in varnames with vardoc=NONEDOC.

    if not rewrite, and TYPE_QUANT already in obj.vardict.keys() (when vars_documenter is called),
        instead do nothing and return a function which does nothing.

    also sets vardict[TYPE_QUANT][document_vars.QUANTDOC] = QUANT_DOC.
    '''
    # get vardict[TYPE_QUANT], creating if necessary.
    vardict = getattr(obj, VARDICT)
    write = rewrite
    if not TYPE_QUANT in vardict.keys():
        vardict[TYPE_QUANT] = dict()
        vardict[TYPE_QUANT][QUANTDOC] = QUANT_DOC
        write = True
    if write:
        # define function (which will be returned)
        def document_var(varname, vardoc):
            '''puts documentation about var named varname into obj.vardict[TYPE_QUANT].'''
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