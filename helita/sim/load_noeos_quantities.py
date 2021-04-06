import numpy as np
from . import document_vars

def load_noeos_quantities(obj, quant, *args,  EOSTAB_QUANT=None,  **kwargs):

  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'noeosquantities', 'Computes some variables without EOS tables')

  val = get_eosparam(obj, quant, EOSTAB_QUANT=EOSTAB_QUANT)

  return val


def get_eosparam(obj, quant, EOSTAB_QUANT=None): 
  '''Computes some variables without EOS tables '''
  if (EOSTAB_QUANT == None):
      EOSTAB_QUANT = ['ne']
  
  docvar = document_vars.vars_documenter(obj, 'EOSTAB_QUANT', EOSTAB_QUANT, get_eosparam.__doc__)
  docvar('ne', "electron density [cm^-3]")
  
  if (quant == '') or not quant in EOSTAB_QUANT:
    return None

  nh = obj.get_var('rho') / obj.uni.grph
      
  return  nh + 2.*nh*(obj.uni.grph/obj.uni.m_h-1.) # this may need a better adjustment.        


