import numpy as np

def load_noeos_quantities(obj, quant, *args,  EOSTAB_QUANT=None,  **kwargs):

  quant = quant.lower()

  if not hasattr(obj, 'description'):
    obj.description = {}

  val = get_eosparam(obj, quant, EOSTAB_QUANT=EOSTAB_QUANT)

  return val


def get_eosparam(obj, quant, EOSTAB_QUANT=None): 

  if (EOSTAB_QUANT == None):
      EOSTAB_QUANT = ['ne']
      if not hasattr(obj,'description'):
          obj.description={}
  
  obj.description['EOSTAB'] = ('Electron density in cgs: ' + ', '.join(EOSTAB_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['EOSTAB']
  else:
    obj.description['ALL'] = obj.description['EOSTAB']

  if (quant == ''):
    return None

  if quant in EOSTAB_QUANT:

    nh = obj.get_var('rho') / obj.uni.grph
        
    return  nh + 2.*nh*(obj.grph/obj.m_h-1.) # this may need a better adjustment.        

  else: 

    return None
