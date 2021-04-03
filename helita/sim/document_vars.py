"""
Created by Sam Evans on Apr 3 2021

Purpose: helper functions for documentation of variables.
"""

VARDICT = 'vardict'   #name of attribute (of obj) which should store documentation about vars.
NONEDOC = None        #default documentation if none is provided.

def _vardict(obj):
    '''create obj.vardict if necessary. return obj.vardict.'''
    if not hasattr(obj, VARDICT):
        setattr(obj, VARDICT, dict())
    return getattr(obj, VARDICT)

def _TYPE_QUANT(vardict, TYPE_QUANT):
    '''create vardict[TYPE_QUANT] if necessary. return vardict[TYPE_QUANT].'''
    if not TYPE_QUANT in vardict.keys():
        vardict[TYPE_QUANT] = dict()
    return vardict[TYPE_QUANT]

def vars_documenter(obj, TYPE_QUANT, QUANT_VARS=None):
    '''function factory; returns function which documents a var for obj in obj.vardict[TYPE_QUANT].
    if QUANT_VARS is not None, also documents all the vars in varnames with vardoc=NONEDOC.
    '''
    # get vardict[TYPE_QUANT], creating if necessary.
    vardict         = _vardict(obj)
    type_quant_dict = _TYPE_QUANT(vardict, TYPE_QUANT)
    # define function (which will be returned)
    def document_var(varname, vardoc):
        '''puts documentation about var named varname into obj.vardict[TYPE_QUANT].'''
        type_quant_dict[varname] = vardoc
    # initialize documentation to NONEDOC for var in QUANT_VARS
    if QUANT_VARS is not None:
        for varname in QUANT_VARS:
            document_var(varname, vardoc=NONEDOC)
    return document_var

# TODO: make something which helps tell new users how to use vardict.
#   (maybe? I mean, it's just a dictionary, which shouldn't be too complicated.)