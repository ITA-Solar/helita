import numpy as np
import os
from glob import glob
import warnings
from . import document_vars

elemlist = ['h', 'he', 'c', 'o', 'ne', 'na', 'mg', 'al', 'si', 's',
        'k', 'ca', 'cr', 'fe', 'ni']

# put quant default lists that are based on elemlist here
## (instead of calculating inside each function,)
## because it takes non-negligible time to calculate these repeatedly
CROSTAB_LIST = ['e_' + clist for clist in elemlist]
CROSTAB_LIST += [clist+'_e' for clist in elemlist]
for iel in elemlist:
  CROSTAB_LIST = CROSTAB_LIST + [
    iel + '_' + clist for clist in elemlist]
_COLFRE_QUANT = ['nu' + clist for clist in CROSTAB_LIST]
_COLFRE_QUANT += ['nu%s_mag' % clist for clist in CROSTAB_LIST]
_COLFRE_QUANT += ['nue_' + clist for clist in elemlist]
_COLFREMX_QUANT = ['numx' + clist for clist in CROSTAB_LIST]
_COLFREMX_QUANT += ['numx%s_mag' % clist for clist in CROSTAB_LIST]
_COLCOU_QUANT = ['nucou' + clist for clist in CROSTAB_LIST]
_COLCOU_QUANT += ['nucoue_' + clist for clist in elemlist]
_COLCOUMS_QUANT = ['nucou_ei', 'nucou_ii']
_COLCOUMS_QUANT += ['nucou' + clist + '_i' for clist in elemlist]
_COLFRI_QUANT = ['nu_ni', 'numx_ni', 'nu_en', 'nu_ei', 'nu_in', 'nu_ni_mag', 'nu_in_mag']
_COLFRI_QUANT += ['nu' + clist + '_i' for clist in elemlist]
_COLFRI_QUANT += ['numx' + clist + '_i' for clist in elemlist]
_COLFRI_QUANT += ['nu' + clist + '_i_mag' for clist in elemlist]
_COLFRI_QUANT += ['numx' + clist + '_i_mag' for clist in elemlist]
_COLFRI_QUANT += ['nu' + clist + '_n' for clist in elemlist]
_COLFRI_QUANT += ['numx' + clist + '_n' for clist in elemlist]
_COLFRI_QUANT += ['nu' + clist + '_n_mag' for clist in elemlist]
_COLFRI_QUANT += ['numx' + clist + '_n_mag' for clist in elemlist]
_COULOMB_COL_QUANT = ['coucol' + clist for clist in elemlist]
_GYROF_QUANT = ['gfe'] + ['gf' + clist for clist in elemlist]
_KAPPA_QUANT = ['kappa' + clist for clist in elemlist]
_IONP_QUANT = ['n' + clist + '-' for clist in elemlist]
_IONP_QUANT += ['r' + clist + '-' for clist in elemlist]

whsp = '  '

def load_quantities(obj, quant, *args, PLASMA_QUANT=None, CYCL_RES=None,
                COLFRE_QUANT=None, COLFRI_QUANT=None, IONP_QUANT=None,
                EOSTAB_QUANT=None, TAU_QUANT=None, DEBYE_LN_QUANT=None,
                CROSTAB_QUANT=None, COULOMB_COL_QUANT=None, AMB_QUANT=None, 
                HALL_QUANT=None, BATTERY_QUANT=None, SPITZER_QUANT=None, 
                KAPPA_QUANT=None, GYROF_QUANT=None, WAVE_QUANT=None, 
                FLUX_QUANT=None, CURRENT_QUANT=None, COLCOU_QUANT=None,  
                COLCOUMS_QUANT=None, COLFREMX_QUANT=None, EM_QUANT=None, **kwargs):
#                HALL_QUANT=None, SPITZER_QUANT=None, **kwargs):
  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'quantities', 'These are the single-fluid quantities')

  if EM_QUANT != '':
    val = get_em(obj, quant, EM_QUANT=EM_QUANT, **kwargs)  
  else: 
    val = None
  if val is None and COULOMB_COL_QUANT != '':
    val = get_coulomb(obj, quant, COULOMB_COL_QUANT=COULOMB_COL_QUANT, **kwargs)
  if val is None and COLFRE_QUANT != '':
    val = get_collision(obj, quant, COLFRE_QUANT=COLFRE_QUANT,**kwargs)
  if val is None and CROSTAB_QUANT != '':
    val = get_crossections(obj, quant, CROSTAB_QUANT=CROSTAB_QUANT,**kwargs)
  if val is None and COLFRI_QUANT != '':
    val = get_collision_ms(obj, quant, COLFRI_QUANT=COLFRI_QUANT, **kwargs)
  if val is None and CURRENT_QUANT != '':
    val = get_current(obj, quant, CURRENT_QUANT=CURRENT_QUANT, **kwargs)
  if val is None and FLUX_QUANT != '':
    val = get_flux(obj, quant, FLUX_QUANT=FLUX_QUANT, **kwargs)
  if val is None and PLASMA_QUANT != '':
    val = get_plasmaparam(obj, quant, PLASMA_QUANT=PLASMA_QUANT, **kwargs)
  if val is None and WAVE_QUANT != '':
    val = get_wavemode(obj, quant, WAVE_QUANT=WAVE_QUANT, **kwargs)
  if val is None and CYCL_RES != '':
    val = get_cyclo_res(obj, quant, CYCL_RES=CYCL_RES, **kwargs)
  if val is None and GYROF_QUANT != '':
    val = get_gyrof(obj, quant, GYROF_QUANT=GYROF_QUANT, **kwargs)
  if val is None and KAPPA_QUANT != '':
    val = get_kappa(obj, quant, KAPPA_QUANT=KAPPA_QUANT, **kwargs)
  if val is None and DEBYE_LN_QUANT != '':
    val = get_debye_ln(obj, quant, DEBYE_LN_QUANT=DEBYE_LN_QUANT, **kwargs)
  if val is None and IONP_QUANT != '':
    val = get_ionpopulations(obj, quant, IONP_QUANT=IONP_QUANT, **kwargs)
  if val is None and AMB_QUANT != '':
    val = get_ambparam(obj, quant, AMB_QUANT=AMB_QUANT, **kwargs)
  if val is None and HALL_QUANT != '':
    val = get_hallparam(obj, quant, HALL_QUANT=HALL_QUANT, **kwargs)
  if val is None and BATTERY_QUANT != '':
    val = get_batteryparam(obj, quant, BATTERY_QUANT=BATTERY_QUANT, **kwargs)  
  if val is None and SPITZER_QUANT != '':
    val = get_spitzerparam(obj, quant, SPITZER_QUANT=SPITZER_QUANT, **kwargs) 
  if val is None and EOSTAB_QUANT != '': 
    val = get_eosparam(obj, quant, EOSTAB_QUANT=EOSTAB_QUANT, **kwargs)
  if val is None and COLCOU_QUANT != '': 
    val = get_collcoul(obj, quant, COLCOU_QUANT=COLCOU_QUANT, **kwargs)
  if val is None and COLCOUMS_QUANT != '': 
    val = get_collcoul_ms(obj, quant, COLCOUMS_QUANT=COLCOUMS_QUANT, **kwargs)
  if val is None and COLFREMX_QUANT != '': 
    val = get_collision_maxw(obj, quant, COLFREMX_QUANT=COLFREMX_QUANT, **kwargs)
  #if np.shape(val) is ():
  #  val = get_spitzerparam(obj, quant)
  return val


def get_em(obj, quant, EM_QUANT = None,  *args, **kwargs):
  """
  Calculates emission messure (EM).

  Parameters
  ----------
  Returns
  -------
  array - ndarray
      Array with the dimensions of the 3D spatial from the simulation
      of the emission measure c.g.s units.
  """
  if EM_QUANT == '': 
        return None

  if EM_QUANT is None:
    EM_QUANT = ['emiss']
    
  unitsnorm = 1e27
  for key, value in kwargs.items():
        if key == 'unitsnorm':
            unitsnorm = value
  
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'EM_QUANT', EM_QUANT, get_em.__doc__)
    docvar('emiss',  'emission messure [cgs]')

  if (quant == '') or not quant in EM_QUANT:
    return None

  
  sel_units = obj.sel_units
  obj.sel_units = 'cgs'

  rho = obj.get_var('rho')
  en = obj.get_var('ne')  
  nh = rho / obj.uni.grph

  obj.sel_units = sel_units

  return en * (nh / unitsnorm)


def get_crossections(obj, quant, CROSTAB_QUANT=None, **kwargs):
  '''
  Computes cross section between species in cgs
  '''

  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = CROSTAB_LIST

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'CROSTAB_QUANT', CROSTAB_QUANT, get_crossections.__doc__)
 
  quant_elem = ''.join([i for i in quant if not i.isdigit()])

  if (quant == '') or not quant_elem in CROSTAB_QUANT:
    return None

  tg = obj.get_var('tg')
  elem = quant.split('_')

  #spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
  #spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
  spic1 = elem[0]
  spic2 = elem[1]
  spic1_ele = ''.join([i for i in spic1 if not i.isdigit()])
  spic2_ele = ''.join([i for i in spic2 if not i.isdigit()])

  cross_tab=None
  cross_dict = {}
  cross_dict['h1','h2']  = cross_dict['h2','h1']  = 'p-h-elast.txt'
  cross_dict['h2','h22'] = cross_dict['h22','h2'] = 'h-h2-data.txt'
  cross_dict['h2','he1'] = cross_dict['he1','h2'] = 'p-he.txt'
  cross_dict['e','he1'] = cross_dict['he1','e'] = 'e-he.txt'
  cross_dict['e','h1']  = cross_dict['h1','e']  = 'e-h.txt'
  maxwell = False

  for key, value in kwargs.items():
    if key == 'cross_tab':
      cross_tab = value
    if key == 'cross_dict': 
      cross_dict = value 
    if key == 'maxwell':
      maxwell = value

  if cross_tab == None: 
    try: 
      cross_tab = cross_dict[spic1,spic2]
    except:  
      if not(maxwell): 
        if (spic1_ele == 'h'):
          cross = obj.uni.weightdic[spic2_ele] / obj.uni.weightdic['h'] * \
              obj.uni.cross_p * np.ones(np.shape(tg))
        elif (spic2_ele == 'h'):
          cross = obj.uni.weightdic[spic1_ele] / obj.uni.weightdic['h'] * \
              obj.uni.cross_p * np.ones(np.shape(tg))
        elif (spic1_ele == 'he'):
          cross = obj.uni.weightdic[spic2_ele] / obj.uni.weightdic['he'] * \
              obj.uni.cross_he * np.ones(np.shape(tg))
        elif (spic2_ele == 'he'):
          cross = obj.uni.weightdic[spic1_ele] / obj.uni.weightdic['he'] * \
              obj.uni.cross_he * np.ones(np.shape(tg))
        else: 
          cross = obj.uni.weightdic[spic2_ele] / obj.uni.weightdic['h'] * \
              obj.uni.cross_p / (np.pi*obj.uni.weightdic[spic2_ele])**2 * \
              np.ones(np.shape(tg))

  if cross_tab != None:
    crossobj = obj.cross_sect(cross_tab=[cross_tab])
    cross = crossobj.cross_tab[0]['crossunits'] * crossobj.tab_interp(tg)

  try:
    return cross
  except Exception:
    print('(WWW) cross-section: wrong combination of species', end="\r",
            flush=True)
    return None 


def get_eosparam(obj, quant, EOSTAB_QUANT=None, **kwargs): 
  '''
  Variables from EOS table. All of them 
  are in cgs except ne which is in SI.
  '''

  if (EOSTAB_QUANT == None):
      EOSTAB_QUANT = ['ne', 'tg', 'pg', 'kr', 'eps', 'opa', 'temt', 'ent']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'EOSTAB_QUANT', EOSTAB_QUANT, get_eosparam.__doc__)
    docvar('ne',  'electron density [m^-3]')
    docvar('tg',  'Temperature [K]')
    docvar('pg',  'gas pressure [dyn/cm^2]')
    docvar('kr',  'Rosseland opacity [cm^2/g]')
    docvar('eps',  'scattering probability')
    docvar('opa',  'opacity')
    docvar('temt',  'thermal emission')
    docvar('ent',  'entropy')


  if (quant == '') or not quant in EOSTAB_QUANT:
    return None

  if quant == 'tau':
    return calc_tau(obj)

  else: 
    # bifrost_uvotrt uses SI!
    fac = 1.0
    if (quant == 'ne') and (obj.sel_units != 'cgs'):
      fac = 1.e6  # cm^-3 to m^-3

    if obj.hion and quant == 'ne':
        return obj.get_var('hionne') * fac

    sel_units = obj.sel_units
    obj.sel_units = 'cgs'
    rho = obj.get_var('rho') 
    ee = obj.get_var('e') / rho
  
    obj.sel_units = sel_units

    if obj.verbose:
        print(quant + ' interpolation...', whsp*7, end="\r", flush=True)

    return obj.rhoee.tab_interp(
      rho, ee, order=1, out=quant) * fac



def get_collision(obj, quant, COLFRE_QUANT=None, **kwargs):
  '''
  Collision frequency between different species in (cgs)
  It will assume Maxwell molecular collisions if crossection 
  tables does not exist. 
  '''
  if COLFRE_QUANT is None:
    COLFRE_QUANT = _COLFRE_QUANT

  if quant=='':  
    docvar = document_vars.vars_documenter(obj, 'COLFRE_QUANT', COLFRE_QUANT, get_collision.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in COLFRE_QUANT:
    return None

  elem = quant.split('_')
  spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
  ion1 = ''.join([i for i in elem[0] if i.isdigit()])
  spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
  ion2 = ''.join([i for i in elem[1] if i.isdigit()])
  spic1 = spic1[2:]
  
  crossarr = get_crossections(obj,'%s%s_%s%s' % (spic1,ion1,spic2,ion2), **kwargs)

  if np.any(crossarr) == None: 
    return get_collision_maxw(obj,'numx'+quant[2:], **kwargs)
  else: 

    nspic2 = obj.get_var('n%s-%s' % (spic2, ion2))
    if np.size(elem) > 2:
      nspic2 *= (1.0-obj.get_var('kappanorm_%s' % spic2))

    tg = obj.get_var('tg')
    if spic1 == 'e':
      awg1 = obj.uni.m_electron
    else:
      awg1 = obj.uni.weightdic[spic1] * obj.uni.amu
    if spic1 == 'e':
      awg2 = obj.uni.m_electron
    else:
      awg2 = obj.uni.weightdic[spic2] * obj.uni.amu
    scr1 = np.sqrt(8.0 * obj.uni.kboltzmann * tg / obj.uni.pi)

    return crossarr * np.sqrt((awg1 + awg2) / (awg1 * awg2)) *\
      scr1 * nspic2  # * (awg1 / (awg1 + awg1))


def get_collision_maxw(obj, quant, COLFREMX_QUANT=None, **kwargs):
  '''
  Maxwell molecular collision frequency 
  '''
  if COLFREMX_QUANT is None:
    COLFREMX_QUANT = _COLFREMX_QUANT
  
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'COLFREMX_QUANT', COLFREMX_QUANT, get_collision_maxw.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in COLFREMX_QUANT:
    return None

  #### ASSUMES ifluid is charged AND jfluid is neutral. ####
  #set constants. for more details, see eq2 in Appendix A of Oppenheim 2020 paper.
  CONST_MULT    = 1.96     #factor in front.
  CONST_ALPHA_N = 6.67e-31 #[m^3]    #polarizability for Hydrogen 
  e_charge= 1.602176e-19   #[C]      #elementary charge
  eps0    = 8.854187e-12   #[F m^-1] #epsilon0, standard 

  elem = quant.split('_')
  spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
  ion1 = ''.join([i for i in elem[0] if i.isdigit()])
  spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
  ion2 = ''.join([i for i in elem[1] if i.isdigit()])
  spic1 = spic1[4:] 

  polarizability_dict = {}
  polarizability_dict['h']  = 6.68E-31
  polarizability_dict['he'] = 2.05E-31
  polarizability_dict['li'] = 2.43E-29
  polarizability_dict['be'] = 5.59E-30
  polarizability_dict['b']  = 3.04E-30
  polarizability_dict['c']  = 1.67E-30
  polarizability_dict['n']  = 1.10E-30
  polarizability_dict['o']  = 7.85E-31
  polarizability_dict['f']  = 5.54E-31
  polarizability_dict['ne']  = 3.94E-31
  polarizability_dict['na']  = 2.41E-29
  polarizability_dict['mg']  = 1.06E-29
  polarizability_dict['al']  = 8.57E-30
  polarizability_dict['si']  = 5.53E-30
  polarizability_dict['p']  = 3.70E-30
  polarizability_dict['s']  = 2.87E-30
  polarizability_dict['cl']  = 2.16E-30
  polarizability_dict['ar']  = 1.64E-30
  polarizability_dict['k']  = 4.29E-29
  polarizability_dict['ca']  = 2.38E-29
  polarizability_dict['sc']  = 1.44E-29
  polarizability_dict['ti']  = 1.48E-29
  polarizability_dict['v']  = 1.29E-29
  polarizability_dict['cr']  = 1.23E-29
  polarizability_dict['mn']  = 1.01E-29
  polarizability_dict['fe']  = 9.19E-30
  polarizability_dict['co']  = 8.15E-30
  polarizability_dict['ni']  = 7.26E-30
  polarizability_dict['cu']  = 6.89E-30
  polarizability_dict['zn']  = 5.73E-30
  polarizability_dict['ga']  = 7.41E-30
  polarizability_dict['ge']  = 5.93E-30
  polarizability_dict['as']  = 4.45E-30
  polarizability_dict['se']  = 4.28E-30
  polarizability_dict['br']  = 3.11E-30
  polarizability_dict['kr']  = 2.49E-30
  polarizability_dict['rb']  = 4.74E-29
  polarizability_dict['sr']  = 2.92E-29
  polarizability_dict['y']  = 2.40E-29
  polarizability_dict['zr']  = 1.66E-29
  polarizability_dict['nb']  = 1.45E-29
  polarizability_dict['mo']  = 1.29E-29
  polarizability_dict['tc']  = 1.17E-29
  polarizability_dict['ru']  = 1.07E-29
  polarizability_dict['rh']  = 9.78E-30
  polarizability_dict['pd']  = 3.87E-30

  tg = obj.get_var('tg')
  if spic1 == 'e':
    awg1 = obj.uni.msi_e 
  else:
    awg1 = obj.uni.weightdic[spic1] * obj.uni.amusi
  if spic1 == 'e':
    awg2 = obj.uni.msi_e
  else:
    awg2 = obj.uni.weightdic[spic2] * obj.uni.amusi

  if (ion1==0 and ion2!=0):
    CONST_ALPHA_N=polarizability_dict[spic1]
    nspic2 = obj.get_var('n%s-%s' % (spic2, ion2)) / (obj.uni.cm_to_m**3)  # convert to SI.
    if np.size(elem) > 2:
      nspic2 *= (1.0-obj.get_var('kappanorm_%s' % spic2))
    return CONST_MULT * nspic2 * np.sqrt(CONST_ALPHA_N * e_charge**2 * awg2 / (eps0 * awg1 * (awg1 + awg2)))  
  elif (ion2==0 and ion1!=0):
    CONST_ALPHA_N=polarizability_dict[spic2]
    nspic1 = obj.get_var('n%s-%s' % (spic1, ion1)) / (obj.uni.cm_to_m**3)  # convert to SI.
    if np.size(elem) > 2:
      nspic1 *= (1.0-obj.get_var('kappanorm_%s' % spic2))  
    return CONST_MULT * nspic1 * np.sqrt(CONST_ALPHA_N * e_charge**2 * awg1 / (eps0 * awg2 * (awg1 + awg2)))   
  else:
    nspic2 = obj.get_var('n%s-%s' % (spic2, ion2)) / (obj.uni.cm_to_m**3)  # convert to SI.
    if np.size(elem) > 2:
      nspic2 *= (1.0-obj.get_var('kappanorm_%s' % spic2))
    return CONST_MULT * nspic2 * np.sqrt(CONST_ALPHA_N * e_charge**2 * awg2 / (eps0 * awg1 * (awg1 + awg2)))  



def get_collcoul(obj, quant, COLCOU_QUANT=None, **kwargs):
  '''
  Coulomb Collision frequency between different ionized species (cgs)
  (Hansteen et al. 1997)
  '''
  if COLCOU_QUANT is None:
    COLCOU_QUANT = _COLCOU_QUANT

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'COLCOU_QUANT', COLCOU_QUANT, get_collcoul.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in COLCOU_QUANT:
    return None

  elem = quant.split('_')
  spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
  ion1 = ''.join([i for i in elem[0] if i.isdigit()])
  spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
  ion2 = ''.join([i for i in elem[1] if i.isdigit()])
  spic1 = spic1[5:]
  nspic2 = obj.get_var('n%s-%s' % (spic2, ion2)) # scr2

  tg = obj.get_var('tg') #scr1
  nel = obj.get_var('ne') / 1e6 # it takes into account NEQ and converts to cgs
  
  coulog = 23. + 1.5 * np.log(tg/1.e6) - 0.5 * np.log(nel/1e6) # Coulomb logarithm scr4
  
  mst = obj.uni.weightdic[spic1] * obj.uni.weightdic[spic2] * obj.uni.amu / \
      (obj.uni.weightdic[spic1] + obj.uni.weightdic[spic2])

  return 1.7 * coulog/20.0 * (obj.uni.m_h/(obj.uni.weightdic[spic1] * 
        obj.uni.amu)) * (mst/obj.uni.m_h)**0.5 * \
        nspic2 / tg**1.5 * (int(ion2)-1)**2


def get_collcoul_ms(obj, quant, COLCOUMS_QUANT=None, **kwargs):
  '''
  Coulomb collision between for a specific ionized species (or electron) with 
  all ionized elements (cgs)
  '''
  if (COLCOUMS_QUANT == None):
    COLCOUMS_QUANT = _COLCOUMS_QUANT

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'COLCOUMS_QUANT', COLCOUMS_QUANT, get_collcoul_ms.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in COLCOUMS_QUANT:
    return None


  if (quant == 'nucou_ii'):
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist: 

      result += obj.uni.amu * obj.uni.weightdic[ielem] * \
              obj.get_var('n%s-1' % ielem) * \
              obj.get_var('nucou%s1_i'% (ielem))
  
    if obj.heion:
      result += obj.uni.amu * obj.uni.weightdic['he'] * obj.get_var('nhe-3') * \
          obj.get_var('nucouhe3_i')

  elif quant[-2:] == '_i':
    lvl = '2'

    elem = quant.split('_')
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist:
      if elem[0][5:] != '%s%s' % (ielem, lvl):
        result += obj.get_var('%s_%s%s' %
                (elem[0], ielem, lvl)) 

  return result


def get_collision_ms(obj, quant, COLFRI_QUANT=None, **kwargs):
  '''
  Sum of collision frequencies (cgs). 
  '''

  if (COLFRI_QUANT == None):
    COLFRI_QUANT = _COLFRI_QUANT

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'COLFRI_QUANT', COLFRI_QUANT, get_collision_ms.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in COLFRI_QUANT:
    return None

  if ((quant == 'nu_ni_mag') or (quant == 'nu_ni')):
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist: 
      if ielem in elemlist[2:] and '_mag' in quant: 
        const = (1 - obj.get_var('kappanorm_%s' % ielem)) 
        mag='_mag'
      else: 
        const = 1.0
        mag=''

      result += obj.uni.amu * obj.uni.weightdic[ielem] * \
              obj.get_var('n%s-1' % ielem) * const * \
              obj.get_var('nu%s1_i%s'% (ielem,mag), **kwargs)

      if ((ielem in elemlist[2:]) and ('_mag' in quant)): 
        result += obj.uni.amu * obj.uni.weightdic[ielem] * \
              obj.get_var('n%s-2' % ielem) * const * \
              obj.get_var('nu%s2_i%s'% (ielem,mag), **kwargs)


  if ((quant == 'numx_ni_mag') or (quant == 'numx_ni')):
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist: 
      if ielem in elemlist[2:] and '_mag' in quant: 
        const = (1 - obj.get_var('kappanorm_%s' % ielem)) 
        mag='_mag'
      else: 
        const = 1.0
        mag=''

      result += obj.uni.amu * obj.uni.weightdic[ielem] * \
              obj.get_var('n%s-1' % ielem) * const * \
              obj.get_var('numx%s1_i%s'% (ielem,mag), **kwargs)

      if ((ielem in elemlist[2:]) and ('_mag' in quant)): 
        result += obj.uni.amu * obj.uni.weightdic[ielem] * \
              obj.get_var('n%s-2' % ielem) * const * \
              obj.get_var('numx%s2_i%s'% (ielem,mag), **kwargs)                

  if ((quant == 'nu_in_mag') or (quant == 'nu_in')):
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist:
      if (ielem in elemlist[2:] and '_mag' in quant): 
        const = (1 - obj.get_var('kappanorm_%s' % ielem)) 
        mag='_mag'
      else: 
        const = 1.0
        mag=''

      result += obj.uni.amu * obj.uni.weightdic[ielem] * const * \
          obj.get_var('n%s-2' % ielem) * obj.get_var('nu%s2_n%s' % (ielem,mag), **kwargs)
    if obj.heion:
      result += obj.uni.amu * obj.uni.weightdic['he'] * obj.get_var('nhe-3') * \
          obj.get_var('nuhe3_n%s'% mag, **kwargs)


  elif quant == 'nu_ei':
    nel = obj.get_var('ne') / 1e6  # NEQ is taken into account and its converted to cgs
    culblog = 23. + 1.5 * np.log(obj.get_var('tg') / 1.e6) - \
      0.5 * np.log(nel / 1e6)

    result = 3.759 * nel / (obj.get_var('tg')**(1.5)) * culblog


  elif quant == 'nu_en':
    elem = quant.split('_')
    result = np.zeros(np.shape(obj.r))
    lvl = 1
    for ielem in elemlist:
      if ielem in ['h', 'he']:
        result += obj.get_var('%s_%s%s' %
                       ('nue', ielem, lvl), **kwargs)

  elif quant[-2:] == '_i' or quant[-2:] == '_n' or quant[-6:] == '_i_mag' or quant[-6:] == '_n_mag':
    addtxt = ''
    if quant[-4:] == '_mag':
      addtxt = '_mag'
    if '_i' in quant:
      lvl = '2'
    else:
      lvl = '1'
    elem = quant.split('_')
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist:
      if elem[0][2:] != '%s%s' % (ielem, lvl):
        result += obj.get_var('%s_%s%s%s' %
                (elem[0], ielem, lvl, addtxt), **kwargs) #* obj.uni.weightdic[ielem] /\
                #(obj.uni.weightdic[ielem] + obj.uni.weightdic[elem[0][2:-1]])
    #if obj.heion and quant[-3:] == '_i':
      #result += obj.get_var('%s_%s%s' % (elem[0], 'he3', addtxt)) * obj.uni.weightdic['he'] /\
      #        (obj.uni.weightdic['he'] + obj.uni.weightdic[elem[0][2:-1]])

  return result


def get_coulomb(obj, quant, COULOMB_COL_QUANT=None, **kwargs):
  '''
  Coulomb collision frequency in Hz
  '''
  if COULOMB_COL_QUANT is None:
    COULOMB_COL_QUANT = _COULOMB_COL_QUANT
  
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'COULOMB_COL_QUANT', COULOMB_COL_QUANT, get_coulomb.__doc__)

  if (quant == '') or not quant in COULOMB_COL_QUANT:
    return None

  iele = np.where(COULOMB_COL_QUANT == quant)
  tg = obj.get_var('tg')
  nel = np.copy(obj.get_var('ne')) # already takes into account NEQ (SI)
  elem = quant.replace('coucol', '')

  const = (obj.uni.pi * obj.uni.qsi_electron ** 4 /
           ((4.0 * obj.uni.pi * obj.uni.permsi)**2 *
            np.sqrt(obj.uni.weightdic[elem] * obj.uni.amusi *
                   (2.0 * obj.uni.ksi_b) ** 3) + 1.0e-20))

  return (const * nel.astype('Float64') *
          np.log(12.0 * obj.uni.pi * nel.astype('Float64') *
          obj.get_var('debye_ln').astype('Float64') + 1e-50) /
          (np.sqrt(tg.astype('Float64')**3) + 1.0e-20))



def get_current(obj, quant, CURRENT_QUANT=None, **kwargs):
  '''
  Calculates currents (bifrost units) or
  rotational components of the velocity
  '''
  if CURRENT_QUANT is None:
    CURRENT_QUANT = ['ix', 'iy', 'iz', 'wx', 'wy', 'wz']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'CURRENT_QUANT', CURRENT_QUANT, get_current.__doc__)
    docvar('ix',  'component x of the current')
    docvar('iy',  'component y of the current')
    docvar('iz',  'component z of the current')
    docvar('wx',  'component x of the rotational of the velocity')
    docvar('wy',  'component y of the rotational of the velocity')
    docvar('wz',  'component z of the rotational of the velocity')

  if (quant == '') or not quant in CURRENT_QUANT:
    return None

  # Calculate derivative of quantity
  axis = quant[-1]
  if quant[0] == 'i':
    q = 'b'
  else:
    q = 'u'
  try:
    var = getattr(obj, quant)
  except AttributeError:
    if axis == 'x':
      varsn = ['z', 'y']
      derv = ['dydn', 'dzdn']
    elif axis == 'y':
      varsn = ['x', 'z']
      derv = ['dzdn', 'dxdn']
    elif axis == 'z':
      varsn = ['y', 'x']
      derv = ['dxdn', 'dydn']

  # 2D or close
  if (getattr(obj, 'n' + varsn[0]) < 5) or (getattr(obj, 'n' + varsn[1]) < 5):
    return np.zeros_like(obj.r)
  else:
    return (obj.get_var('d' + q + varsn[0] + derv[0]) -
            obj.get_var('d' + q + varsn[1] + derv[1]))



def get_flux(obj, quant, FLUX_QUANT=None, **kwargs):
  '''
  Computes flux
  '''
  if FLUX_QUANT is None:
    FLUX_QUANT = ['pfx', 'pfy', 'pfz', 'pfex', 'pfey', 'pfez', 'pfwx',
                'pfwy', 'pfwz']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'FLUX_QUANT', FLUX_QUANT, get_flux.__doc__)
    docvar('pfx',  'component x of the Poynting flux')
    docvar('pfy',  'component y of the Poynting flux')
    docvar('pfz',  'component z of the Poynting flux')
    docvar('pfex',  'component x of the Flux emergence')
    docvar('pfey',  'component y of the Flux emergence')
    docvar('pfez',  'component z of the Flux emergence')
    docvar('pfwx',  'component x of the Poynting flux from "horizontal" motions')
    docvar('pfwy',  'component y of the Poynting flux from "horizontal" motions')
    docvar('pfwz',  'component z of the Poynting flux from "horizontal" motions')

  if (quant == '') or not quant in FLUX_QUANT:
    return None

  axis = quant[-1]
  if axis == 'x':
    varsn = ['z', 'y']
  elif axis == 'y':
    varsn = ['x', 'z']
  elif axis == 'z':
    varsn = ['y', 'x']
  if 'pfw' in quant or len(quant) == 3:
    var = - obj.get_var('b' + axis + 'c') * (
      obj.get_var('u' + varsn[0] + 'c') *
      obj.get_var('b' + varsn[0] + 'c') +
      obj.get_var('u' + varsn[1] + 'c') *
      obj.get_var('b' + varsn[1] + 'c'))
  else:
    var = np.zeros_like(obj.r)
  if 'pfe' in quant or len(quant) == 3:
    var += obj.get_var('u' + axis + 'c') * (
      obj.get_var('b' + varsn[0] + 'c')**2 +
      obj.get_var('b' + varsn[1] + 'c')**2)
  return var



def get_plasmaparam(obj, quant, PLASMA_QUANT=None, **kwargs):
  '''
  Adimensional parameters for single fluid
  '''
  if PLASMA_QUANT is None:
    PLASMA_QUANT = ['beta', 'va', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky',
                'kz']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'PLASMA_QUANT', PLASMA_QUANT, get_plasmaparam.__doc__)
    docvar('beta', "plasma beta")
    docvar('va', "alfven speed [simu. units]")
    docvar('cs', "sound speed [simu. units]")
    docvar('s', "entropy [log of quantities in simu. units]")
    docvar('ke', "kinetic energy density of ifluid [simu. units]")
    docvar('mn', "mach number (using sound speed)")
    docvar('man', "mach number (using alfven speed)")
    docvar('hp', "Pressure scale height")
    for var in ['vax', 'vay', 'vaz']:
      docvar(var, "{axis} component of alfven velocity [simu. units]".format(axis=var[-1]))
    for var in ['kx', 'ky', 'kz']:
      docvar(var, ("{axis} component of kinetic energy density of ifluid [simu. units]."+\
                  "(0.5 * rho * (get_var(u{axis})**2)").format(axis=var[-1]))

  if (quant == '') or not quant in PLASMA_QUANT:
    return None

  if quant in ['hp', 's', 'cs', 'beta']:
    var = obj.get_var('p')
    if quant == 'hp':
      if getattr(obj, 'nx') < 5:
        return np.zeros_like(var)
      else:
        return 1. / (cstagger.do(var, 'ddzup') + 1e-12)
    elif quant == 'cs':
      return np.sqrt(obj.params['gamma'][obj.snapInd] *
                     var / obj.get_var('r'))
    elif quant == 's':
      return (np.log(var) - obj.params['gamma'][obj.snapInd] *
              np.log(obj.get_var('r')))
    elif quant == 'beta':
      return 2 * var / obj.get_var('b2')

  if quant in ['mn', 'man']:
    var = obj.get_var('modu')
    if quant == 'mn':
      return var / (obj.get_var('cs') + 1e-12)
    else:
      return var / (obj.get_var('va') + 1e-12)

  if quant in ['va', 'vax', 'vay', 'vaz']:
    var = obj.get_var('r')
    if len(quant) == 2:
      return obj.get_var('modb') / np.sqrt(var)
    else:
      axis = quant[-1]
      return np.sqrt(obj.get_var('b' + axis + 'c') ** 2 / var)

  if quant in ['hx', 'hy', 'hz', 'kx', 'ky', 'kz']:
    axis = quant[-1]
    var = obj.get_var('p' + axis + 'c')
    if quant[0] == 'h':
      return ((obj.get_var('e') + obj.get_var('p')) /
              obj.get_var('r') * var)
    else:
      return obj.get_var('u2') * var * 0.5

  if quant in ['ke']:
    var = obj.get_var('r')
    return obj.get_var('u2') * var * 0.5


def get_wavemode(obj, quant, WAVE_QUANT=None, **kwargs):
  '''
  computes waves modes
  '''
  if WAVE_QUANT is None:
    WAVE_QUANT = ['alf', 'fast', 'long']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'WAVE_QUANT', WAVE_QUANT, get_wavemode.__doc__)
    docvar('alf', "Alfven wave component [simu units]")
    docvar('fast', "fast wave component [simu units]")
    docvar('long', "longitudinal wave component [simu units]")

  if (quant == '') or not quant in WAVE_QUANT:
    return None

  bx = obj.get_var('bxc')
  by = obj.get_var('byc')
  bz = obj.get_var('bzc')
  bMag = np.sqrt(bx**2 + by**2 + bz**2)
  bx, by, bz = bx / bMag, by / bMag, bz / bMag  # b is already centered
  # unit vector of b
  unitB = np.stack((bx, by, bz))

  if quant == 'alf':
    uperb = obj.get_var('uperb')
    uperbVect = uperb * unitB
    # cross product (uses cstagger bc no variable gets uperbVect)
    curlX = (cstagger.do(cstagger.do(uperbVect[2], 'ddydn'), 'yup') -
             cstagger.do(cstagger.do(uperbVect[1], 'ddzdn'), 'zup'))
    curlY = (-cstagger.do(cstagger.do(uperbVect[2], 'ddxdn'), 'xup')
             + cstagger.do(cstagger.do(uperbVect[0], 'ddzdn'), 'zup'))
    curlZ = (cstagger.do(cstagger.do(uperbVect[1], 'ddxdn'), 'xup') -
             cstagger.do(cstagger.do(uperbVect[0], 'ddydn'), 'yup'))
    curl = np.stack((curlX, curlY, curlZ))
    # dot product
    result = np.abs((unitB * curl).sum(0))
  elif quant == 'fast':
    uperb = obj.get_var('uperb')
    uperbVect = uperb * unitB

    result = np.abs(cstagger.do(cstagger.do(
      uperbVect[0], 'ddxdn'), 'xup') + cstagger.do(cstagger.do(
        uperbVect[1], 'ddydn'), 'yup') + cstagger.do(
          cstagger.do(uperbVect[2], 'ddzdn'), 'zup'))
  else:
    dot1 = obj.get_var('uparb')
    grad = np.stack((cstagger.do(cstagger.do(dot1, 'ddxdn'),
            'xup'), cstagger.do(cstagger.do(dot1, 'ddydn'), 'yup'),
                     cstagger.do(cstagger.do(dot1, 'ddzdn'), 'zup')))
    result = np.abs((unitB * grad).sum(0))
  return result


def get_cyclo_res(obj, quant, CYCL_RES=None, **kwargs):
  '''
  esonant cyclotron frequencies
  (only for do_helium) are (SI units)
  '''
  if (CYCL_RES is None):
    CYCL_RES = ['n6nhe2', 'n6nhe3', 'nhe2nhe3']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'CYCL_RES', CYCL_RES, get_cyclo_res.__doc__)

  if (quant == '') or not quant in CYCL_RES:
    return None

  if obj.hion and obj.heion:
    posn = ([pos for pos, char in enumerate(quant) if char == 'n'])
    q2 = quant[posn[-1]:]
    q1 = quant[:posn[-1]]
    nel = obj.get_var('ne')/1e6 # already takes into account NEQ converted to cgs
    var2 = obj.get_var(q2)
    var1 = obj.get_var(q1)
    z1 = 1.0
    z2 = float(quant[-1])
    if q1[:3] == 'n6':
      omega1 = obj.get_var('gfh2')
    else:
      omega1 = obj.get_var('gf'+q1[1:])
    omega2 = obj.get_var('gf'+q2[1:])
    return (z1 * var1 * omega2 + z2 * var2 * omega1) / nel
  else:
    raise ValueError(('get_quantity: This variable is only '
                      'avaiable if do_hion and do_helium is true'))


def get_gyrof(obj, quant, GYROF_QUANT=None, **kwargs):
  '''
  gyro freqency are (Hz)
  gf+ ionization state
  '''
  if (GYROF_QUANT is None):
    GYROF_QUANT = _GYROF_QUANT
  
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'GYROF_QUANT', GYROF_QUANT, get_gyrof.__doc__)

  if (quant == '') or not ''.join([i for i in quant if not i.isdigit()]) in GYROF_QUANT:
    return None

  if quant == 'gfe':
    return obj.get_var('modb') * obj.uni.usi_b * \
            obj.uni.qsi_electron / (obj.uni.msi_e)
  else:
    ion = float(''.join([i for i in quant if i.isdigit()]))
    return obj.get_var('modb') * obj.uni.usi_b * \
        obj.uni.qsi_electron * (ion - 1.0) / \
        (obj.uni.weightdic[quant[2:-1]] * obj.uni.amusi)



def get_kappa(obj, quant, KAPPA_QUANT=None, **kwargs):
  '''
  kappa, i.e., magnetization (adimensional)
  at the end it must have the ionization
  '''
  if (KAPPA_QUANT is None):
    KAPPA_QUANT = ['kappanorm_', 'kappae'] + _KAPPA_QUANT
        

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'KAPPA_QUANT', KAPPA_QUANT, get_kappa.__doc__)

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in KAPPA_QUANT:
    if quant == 'kappae':
      return obj.get_var('gfe') / (obj.get_var('nu_en') + 1e-28)
    else:
      elem = quant.replace('kappa', '')
      return obj.get_var('gf'+elem) / (obj.get_var('nu'+elem+'_n') + 1e-28)

  elif quant[:-1] in KAPPA_QUANT or quant[:-2] in KAPPA_QUANT:
    elem = quant.split('_')
    return obj.get_var('kappah2')**2/(obj.get_var('kappah2')**2 + 1) - \
          obj.get_var('kappa%s2' % elem[1])**2 / \
                      (obj.get_var('kappa%s2' % elem[1])**2 + 1)
  else:
    return None


def get_debye_ln(obj, quant, DEBYE_LN_QUANT=None, **kwargs):
  '''
  Computes Debye length in ... units
  '''

  if (DEBYE_LN_QUANT is None):
    DEBYE_LN_QUANT = ['debye_ln']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'DEBYE_LN_QUANT', DEBYE_LN_QUANT, get_debye_ln.__doc__)
    docvar('debye_ln', "Debye length [u.u_l]")

  if (quant == '') or not quant in DEBYE_LN_QUANT:
    return None

  tg = obj.get_var('tg')
  part = np.copy(obj.get_var('ne'))
  # We are assuming a single charge state:
  for iele in elemlist:
    part += obj.get_var('n' + iele + '-2')
  if obj.heion:
    part += 4.0 * obj.get_var('nhe3')
  # check units of n
  return np.sqrt(obj.uni.permsi / obj.uni.qsi_electron**2 /
                 (obj.uni.ksi_b * tg.astype('Float64') *
                  part.astype('Float64') + 1.0e-20))



def get_ionpopulations(obj, quant, IONP_QUANT=None, **kwargs):
  '''
  densities for specific ionized species
  '''
  if (IONP_QUANT is None):
    IONP_QUANT = _IONP_QUANT
    IONP_QUANT += ['rneu', 'rion', 'nion', 'nneu', 'nelc']
    IONP_QUANT += ['rneu_nomag', 'rion_nomag', 'nion_nomag', 'nneu_nomag']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'IONP_QUANT', IONP_QUANT, get_ionpopulations.__doc__)

  if (quant == ''):
      return None

  if ((quant in IONP_QUANT) and (quant[-3:] in ['ion', 'neu'])):
    if 'ion' in quant:
        lvl = '2'
    else:
        lvl = '1'
    result = np.zeros(np.shape(obj.r))
    for ielem in elemlist:
        result += obj.get_var(quant[0]+ielem+'-'+lvl)
    return result

  elif ((quant in IONP_QUANT) and (quant[-9:] in ['ion_nomag', 'neu_nomag'])):
    # I dont think it makes sence to have neu_nomag
    if 'ion' in quant:
        lvl = '2'
    else:
        lvl = '1'
    result = np.zeros(np.shape(obj.r))
    if quant[-7:] == 'ion_nomag':
      for ielem in elemlist[2:]:
        result += obj.get_var(quant[0]+ielem+'-'+lvl) * \
                              (1-obj.get_var('kappanorm_%s' % ielem))
    else:
      for ielem in elemlist[2:]:
        result += obj.get_var(quant[0]+ielem+'-'+lvl) * \
                              (1-obj.get_var('kappanorm_%s' % ielem))
    return result


  elif (quant == 'nelc'):
  
    result = np.zeros(np.shape(obj.r))  
    for ielem in elemlist:
      result += obj.get_var('n'+ielem+'-2')
    
    result += obj.get_var('nhe-3')*2

    return result

  elif ''.join([i for i in quant if not i.isdigit()]) in IONP_QUANT:
    elem = quant.replace('-', '')
    spic = ''.join([i for i in elem if not i.isdigit()])
    lvl = ''.join([i for i in elem if i.isdigit()])

    if obj.hion and spic[1:] == 'h':
      if quant[0] == 'n':
        mass = 1.0
      else:
        mass = obj.uni.m_h
      if lvl == '1':
        return mass * (obj.get_var('n1') + obj.get_var('n2') + obj.get_var('n3') +
                         obj.get_var('n4') + obj.get_var('n5'))
      else:
        return mass * obj.get_var('n6')

    elif obj.heion and spic[1:] == 'he':
      if quant[0] == 'n':
        mass = 1.0
      else:
        mass = obj.uni.m_he
      if obj.verbose:
        print('get_var: reading nhe%s' % lvl, whsp*5, end="\r",
              flush=True)
      return mass * obj.get_var('nhe%s' % lvl)

    else:
      sel_units = obj.sel_units
      obj.sel_units = 'cgs'
      rho = obj.get_var('rho')
      nel = np.copy(obj.get_var('ne')) # cgs
      tg = obj.get_var('tg')    
      obj.sel_units = sel_units

      if quant[0] == 'n':
        dens = False
      else:
        dens = True

      return ionpopulation(obj, rho, nel, tg, elem=spic[1:], lvl=lvl, dens=dens) # cm^3
  else:
    return None


def get_ambparam(obj, quant, AMB_QUANT=None, **kwargs):
  '''
  ambipolar velocity or related terms
  '''
  if (AMB_QUANT is None):
    AMB_QUANT = ['uambx', 'uamby', 'uambz', 'ambx', 'amby', 'ambz',
              'eta_amb1', 'eta_amb2', 'eta_amb3', 'eta_amb4', 'eta_amb5',
              'nchi', 'npsi', 'nchi_red', 'npsi_red',
              'rchi', 'rpsi', 'rchi_red', 'rpsi_red','alphai','betai']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'AMB_QUANT', AMB_QUANT, get_ambparam.__doc__)
    docvar('uambx', "component x of the ambipolar velocity")
    docvar('uamby', "component y of the ambipolar velocity")
    docvar('uambz', "component z of the ambipolar velocity")
    docvar('ambx', "component x of the ambipolar term")
    docvar('amby', "component y of the ambipolar term")
    docvar('ambz', "component z of the ambipolar term")
    docvar('eta_amb1', "ambipolar diffusion using nu_ni")
    docvar('eta_amb2', "ambipolar diffusion using nu_in")
    docvar('eta_amb3', "ambipolar diffusion using nu_ni_max and rion_nomag")
    docvar('eta_amb4', "ambipolar diffusion using Yakov for low ionization regime, Eq (20) (ref{Faraday_corr})")
    docvar('eta_amb4a', "ambipolar diffusion using Yakov for low ionization regime, Eq (20) (ref{Faraday_corr}), only the numerator")
    docvar('eta_amb4b', "ambipolar diffusion using Yakov for low ionization regime, Eq (20) (ref{Faraday_corr}), only the denumerator")
    docvar('eta_amb5', "ambipolar diffusion using Yakov for any ionization regime, 7c")
    docvar('nchi', "from Yakov notes to derive the ambipolar diff")
    docvar('npsi', "from Yakov notes to derive the ambipolar diff")
    docvar('nchi_red', "from Yakov notes to derive the ambipolar diff")
    docvar('npsi_red', "from Yakov notes to derive the ambipolar diff")
    docvar('rchi', "from Yakov notes to derive the ambipolar diff")
    docvar('rpsi', "from Yakov notes to derive the ambipolar diff")
    docvar('rchi_red', "from Yakov notes to derive the ambipolar diff")
    docvar('rpsi_red', "from Yakov notes to derive the ambipolar diff")
    docvar('alphai', "from Yakov notes to derive the ambipolar diff")
    docvar('betai', "from Yakov notes to derive the ambipolar diff")  

  if (quant == '') or not (quant in AMB_QUANT):
    return None

  if obj.sel_units == 'cgs': 
    u_b = 1.0
  else: 
    u_b = obj.uni.u_b

  axis = quant[-1]
  if quant == 'eta_amb1':  # version from other
    result = (obj.get_var('rneu') / obj.get_var('rho') * u_b)**2
    result /= (4.0 * obj.uni.pi * obj.get_var('nu_ni', **kwargs) + 1e-20)
    result *= obj.get_var('b2') #/ 1e7

  # This should be the same and eta_amb2 except that eta_amb2 has many more species involved.
  elif quant == 'eta_amb2':
    result = (obj.get_var('rneu') / obj.get_var('rho') * u_b)**2 / (
        4.0 * obj.uni.pi * obj.get_var('nu_in', **kwargs) + 1e-20)
    result *= obj.get_var('b2') #/ 1e7

  elif quant == 'eta_amb3':  # This version takes into account the magnetization
    result = ((obj.get_var('rneu') + obj.get_var('rion_nomag')) / obj.r * obj.uni.u_b)**2 / (
        4.0 * obj.uni.pi * obj.get_var('nu_ni_mag') + 1e-20)
    result *= obj.get_var('b2') #/ 1e7

  # Yakov for low ionization regime, Eq (20) (ref{Faraday_corr})
  elif quant == 'eta_amb4':
    psi = obj.get_var('npsi')
    chi = obj.get_var('nchi')

    result = obj.get_var('modb') * obj.uni.u_b * (psi / (1e2 * (psi**2 + chi**2)) - 1.0 / (
            obj.get_var('nelc')  * obj.get_var('kappae') * 1e2 + 1e-20))

  # Yakov for any ionization regime, 7c
  elif quant == 'eta_amb5': 
    psi = obj.get_var('npsi')
    chi = obj.get_var('nchi')
    
    chi = obj.r*0.0
    chif = obj.r*0.0
    psi = obj.r*0.0
    psif = obj.r*0.0      
    eta = obj.r*0.0      
    kappae = obj.get_var('kappae')

    for iele in elemlist:
      kappaiele = obj.get_var('kappa'+iele+'2')
      chi += (kappae + kappaiele) * (
          kappae - kappaiele) / (
          1.0 + kappaiele**2) / (
          1.0 + kappae**2) * obj.get_var('n'+iele+'-2')
      chif +=  obj.get_var('r'+iele+'-2') * kappaiele / (
          1.0 + kappaiele**2) 
      psif +=  obj.get_var('r'+iele+'-2') / (
          1.0 + kappaiele**2) 
      psi += (kappae + kappaiele) * (
          1.0 + kappae * kappaiele) / (
          1.0 + kappaiele**2) / (
          1.0 + kappae**2) * obj.get_var('n'+iele+'-2')
      eta += (kappae + kappaiele) * obj.get_var('n'+iele+'-2')

    result =  obj.get_var('modb') * obj.uni.u_b * ( 1.0 / ((psi**2 + chi**2) * obj.r) * (chi* chif - psi * (
          obj.get_var('rneu')+psif))- 1.0 / (eta+1e-28)) 


  elif quant == 'eta_amb4a':
    psi = obj.get_var('npsi')
    chi = obj.get_var('nchi')

    result = obj.get_var('modb') * obj.uni.u_b * (psi / (psi**2 + chi**2) + 1e-20)

  elif quant == 'eta_amb4b':

    result = obj.get_var('modb') * obj.uni.u_b * ( 1.0 / (
            obj.get_var('hionne') / 1e6 * obj.get_var('kappae') + 1e-20))

  elif quant in ['nchi','rchi']:
    result = obj.r*0.0
    kappae = obj.get_var('kappae')

    for iele in elemlist:
      result += (kappae + obj.get_var('kappa'+iele+'2')) * (
          kappae - obj.get_var('kappa'+iele+'2')) / (
          1.0 + obj.get_var('kappa'+iele+'2')**2) / (
          1.0 + kappae**2) * obj.get_var(quant[0]+iele+'-2')

  elif quant in ['npsi','rpsi']: # Yakov, Eq ()
    result = obj.r*0.0
    kappae = obj.get_var('kappae')

    for iele in elemlist:
      result += (kappae + obj.get_var('kappa'+iele+'2')) * (
          1.0 + kappae * obj.get_var('kappa'+iele+'2')) / (
          1.0 + obj.get_var('kappa'+iele+'2')**2) / (
          1.0 + kappae**2) * obj.get_var(quant[0]+iele+'-2')

  elif quant == 'alphai':
    result = obj.r*0.0
    kappae = obj.get_var('kappae')

    for iele in elemlist:
      result += (kappae + obj.get_var('kappa'+iele+'2')) * (
          kappae - obj.get_var('kappa'+iele+'2')) / (
          1.0 + obj.get_var('kappa'+iele+'2')**2) / (
          1.0 + kappae**2) 

  elif quant == 'betai': # Yakov, Eq ()
    result = obj.r*0.0

    for iele in elemlist:
      result += (obj.get_var('kappae') + obj.get_var('kappa'+iele+'2')) * (
          1.0 + obj.get_var('kappae') * obj.get_var('kappa'+iele+'2')) / (
          1.0 + obj.get_var('kappa'+iele+'2')**2) / (
          1.0 + obj.get_var('kappae')**2) 

  elif quant in ['nchi_red','rchi_red']: # alpha
    result = obj.r*0.0

    for iele in elemlist:
      result += 1.0 / (1.0 + obj.get_var('kappa'+iele+'2')**2) *\
                obj.get_var(quant[0]+iele+'-2')

  elif quant in ['npsi_red','rpsi_red']: # beta
    result = obj.r*0.0

    for iele in elemlist:
      result += obj.get_var('kappa'+iele+'2') / (
                1.0 + obj.get_var('kappa'+iele+'2')**2) * \
                obj.get_var(quant[0]+iele+'-2')
  
  elif quant[0] == 'u':
    result = obj.get_var('itimesb' + quant[-1]) / \
                         obj.get_var('modb') * obj.get_var('eta_amb')

  elif (quant[-4:-1] == 'amb' and quant[-1] in ['x','y','z'] and 
       quant[1:3] != 'chi' and quant[1:3] != 'psi'):

    axis = quant[-1]
    if axis == 'x':
      varsn = ['y', 'z']
    elif axis == 'y':
      varsn = ['z', 'y']
    elif axis == 'z':
      varsn = ['x', 'y']
    result = (obj.get_var('itimesb' + varsn[0]) *
      obj.get_var('b' + varsn[1] + 'c') -
      obj.get_var('itimesb' + varsn[1]) *
      obj.get_var('b' + varsn[0] + 'c')) / obj.get_var('b2') * obj.get_var('eta_amb')

  return  result


def get_hallparam(obj, quant, HALL_QUANT=None, **kwargs):
  '''
  Hall velocity or related terms
  '''
  if (HALL_QUANT is None):
    HALL_QUANT = ['uhallx', 'uhally', 'uhallz', 'hallx', 'hally', 'hallz',
                'eta_hall', 'eta_hallb']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'HALL_QUANT', HALL_QUANT, get_hallparam.__doc__)
    docvar('uhallx', "component x of the Hall velocity")
    docvar('uhally', "component y of the Hall velocity")
    docvar('uhallz', "component z of the Hall velocity")
    docvar('hallx', "component x of the Hall term")
    docvar('hally', "component y of the Hall term")
    docvar('hallz', "component z of the Hall term")
    docvar('eta_hall', "Hall term ")
    docvar('eta_hallb', "Hall term / B")

  if (quant == '') or not (quant in HALL_QUANT):
    return None

  if quant[0] == 'u':
    try:
      result = obj.get_var('i' + quant[-1])
    except:
      result = obj.get_var('rotb' + quant[-1])   
  elif quant == 'eta_hall':
    nel = obj.get_var('nel')
    result =  (obj.uni.clight)*(obj.uni.u_b) / (4.0 * obj.uni.pi * obj.uni.q_electron * nel)
    result = obj.get_var('modb')*result /obj.uni.u_l/obj.uni.u_l*obj.uni.u_t 
  
  elif quant == 'eta_hallb':
    nel = obj.get_var('nel')
    result =  (obj.uni.clight)*(obj.uni.u_b) / (4.0 * obj.uni.pi * obj.uni.q_electron * nel)
    result = result /obj.uni.u_l/obj.uni.u_l*obj.uni.u_t 

  else:
    result = obj.get_var('itimesb_' + quant[-1]) / obj.get_var('modb')

  return result #obj.get_var('eta_hall') * result


def get_batteryparam(obj, quant, BATTERY_QUANT=None, **kwargs):
  '''
  Related battery terms
  '''
  if (BATTERY_QUANT is None):
    BATTERY_QUANT = ['bb_constqe', 'dxpe', 'dype', 'dzpe', 'bb_batx',
                    'bb_baty', 'bb_batz']
  
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'BATTERY_QUANT', BATTERY_QUANT, get_batteryparam.__doc__)
    docvar('bb_constqe', "constant coefficient involved in the battery term")
    docvar('dxpe', "Gradient of electron pressure in the x direction [simu.u_p/simu.u_l]")
    docvar('dype', "Gradient of electron pressure in the y direction [simu.u_p/simu.u_l]")
    docvar('dzpe', "Gradient of electron pressure in the z direction [simu.u_p/simu.u_l]")
    docvar('bb_batx', "Component of the battery term  in the x direction, (1/ne qe)*dx(pe)")
    docvar('bb_baty', "Component of the battery term  in the y direction, (1/ne qe)*dy(pe)")
    docvar('bb_batz', "Component of the battery term  in the z direction, (1/ne qe)*dz(pe)")

  if (quant == '') or not (quant in BATTERY_QUANT):
    return None
    
  if quant == 'bb_constqe':
    const = (obj.uni.usi_p / obj.uni.qsi_electron / (1.0/((obj.uni.cm_to_m)**3)) / obj.uni.usi_l / (obj.uni.usi_b * obj.uni.usi_l/obj.uni.u_t))#/obj.uni.u_p
    result = const

  if quant == 'bb_batx':
    gradx_pe = obj.get_var('dpedxup')#obj.get_var('d' + pe + 'dxdn')
    nel     =  obj.get_var('nel')
    bb_constqe = obj.uni.usi_p / obj.uni.qsi_electron / (1.0/((obj.uni.cm_to_m)**3)) / obj.uni.usi_l / (obj.uni.usi_b * obj.uni.usi_l/obj.uni.u_t)#/obj.uni.u_p
    bb_batx = gradx_pe / (nel * bb_constqe)
    result  = bb_batx 

  if quant == 'bb_baty':
    grady_pe = obj.get_var('dpedyup')#obj.get_var('d' + pe + 'dxdn')
    nel     =  obj.get_var('nel')
    bb_constqe = obj.uni.usi_p / obj.uni.qsi_electron / (1.0/((obj.uni.cm_to_m)**3)) / obj.uni.usi_l / (obj.uni.usi_b * obj.uni.usi_l/obj.uni.u_t)#/obj.uni.u_p
    bb_baty = grady_pe / (nel * bb_constqe)
    result  = bb_baty

  if quant == 'bb_batz':
    gradz_pe = obj.get_var('dpedzup')#obj.get_var('d' + pe + 'dxdn')
    nel     =  obj.get_var('nel')
    bb_constqe = obj.uni.usi_p / obj.uni.qsi_electron / (1.0/((obj.uni.cm_to_m)**3)) / obj.uni.usi_l / (obj.uni.usi_b * obj.uni.usi_l/obj.uni.u_t)#/obj.uni.u_p
    bb_batz = gradz_pe / (nel * bb_constqe)
    result  = bb_batz  
  return result


def get_spitzerparam(obj, quant, SPITZER_QUANT=None, **kwargs):
  '''
  Spitzer related term
  '''

  if (SPITZER_QUANT is None):
    SPITZER_QUANT = ['fcx','fcy','fcz','qspitz']

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'SPITZER_QUANT', SPITZER_QUANT, get_spitzerparam.__doc__)
    docvar('fcx', "X component of the anisotropic electron heat flux, i.e., (kappae(B)*grad(Te))_x")
    docvar('fcy', "Y component of the anisotropic electron heat flux, i.e., (kappae(B)*grad(Te))_y")
    docvar('fcz', "Z component of the anisotropic electron heat flux, i.e., (kappae(B)*grad(Te))_z")
    docvar('qspitz', "Electron heat flux, i.e., Qspitz [simu.u_e/simu.u_t] erg.s-1")

  if (quant == '') or not (quant in SPITZER_QUANT):
    return None
   
  if (quant == 'fcx'):  
    kappaq = obj.get_var('kappaq')
    gradx_Te = obj.get_var('detgdxup')
    bx =   obj.get_var('bx')
    by =   obj.get_var('by')
    bz =   obj.get_var('bz')
    rhs =  obj.get_var('rhs')
    bmin = 1E-5 

    normb = np.sqrt(bx**2+by**2+bz**2)
    norm2bmin = bx**2+by**2+bz**2+bmin**2

    bbx = bx/normb

    bm = (bmin**2)/norm2bmin

    fcx = kappaq * (bbx*rhs+bm*gradx_Te)

    result = fcx

  if (quant == 'fcy'):  
    kappaq = obj.get_var('kappaq')
    grady_Te = obj.get_var('detgdyup')
    bx =   obj.get_var('bx')
    by =   obj.get_var('by')
    bz =   obj.get_var('bz')
    rhs =  obj.get_var('rhs')
    bmin = 1E-5 

    normb = np.sqrt(bx**2+by**2+bz**2)
    norm2bmin = bx**2+by**2+bz**2+bmin**2

    bby = by/normb

    bm = (bmin**2)/norm2bmin

    fcy = kappaq * (bby*rhs+bm*grady_Te)

    result = fcy

  if (quant == 'fcz'): 
    kappaq = obj.get_var('kappaq')
    gradz_Te = obj.get_var('detgdzup')
    bx =   obj.get_var('bx')
    by =   obj.get_var('by')
    bz =   obj.get_var('bz')
    rhs =  obj.get_var('rhs')
    bmin = 1E-5 

    normb = np.sqrt(bx**2+by**2+bz**2)
    norm2bmin = bx**2+by**2+bz**2+bmin**2

    bbz = bz/normb

    bm = (bmin**2)/norm2bmin

    fcz = kappaq * (bbz*rhs+bm*gradz_Te)

    result = fcz     

  if (quant == 'qspitz'):  
    dxfcx = obj.get_var('dfcxdxup')
    dyfcy = obj.get_var('dfcydyup')
    dzfcz = obj.get_var('dfczdzup')
    result = dxfcx + dyfcy + dzfcz

  return result 
 

def calc_tau(obj):
  """
  Calculates optical depth.

  """
  warnings.warn("Use of calc_tau is discouraged. It is model-dependent, "
                "inefficient and slow.")

  # grph = 2.38049d-24 uni.GRPH
  # bk = 1.38e-16 uni.KBOLTZMANN
  # EV_TO_ERG=1.60217733E-12 uni.EV_TO_ERG
  

  units_temp=obj.transunits 

  nel = obj.trans2comm('ne')
  tg = obj.trans2comm('tg')
  rho = obj.trans2comm('rho') 

  tau = np.zeros((obj.nx, obj.ny, obj.nz)) + 1.e-16
  xhmbf = np.zeros((obj.nz))
  const = (1.03526e-16 / obj.uni.grph) * 2.9256e-17 
  for iix in range(obj.nx):
      for iiy in range(obj.ny):
          for iiz in range(obj.nz):
              xhmbf[iiz] = const * nel[iix, iiy, iiz] / \
                  tg[iix, iiy, iiz]**1.5 * np.exp(0.754e0 *
                  obj.uni.ev_to_erg / obj.uni.kboltzmann /
                  tg[iix, iiy, iiz]) * rho[iix, iiy, iiz]

          for iiz in range(obj.nz-1,0,-1):
              tau[iix, iiy, iiz] = tau[iix, iiy, iiz - 1] + 0.5 *\
                  (xhmbf[iiz] + xhmbf[iiz - 1]) *\
                  np.abs(obj.dz1d[iiz]) 

  if not units_temp: 
    obj.trans2noncommaxes

  return tau

def ionpopulation(obj, rho, nel, tg, elem='h', lvl='1', dens=True, **kwargs):
  '''
  rho is cgs.
  tg in [K]
  nel in cgs. 
  The output, is in cgs
  '''

  print('ionpopulation: reading species %s and level %s' % (elem, lvl), whsp,
      end="\r", flush=True)
  '''
  fdir = '.'
  try:
    tmp = find_first_match("*.idl", fdir)
  except IndexError:
    try:
      tmp = find_first_match("*idl.scr", fdir)
    except IndexError:
      try:
        tmp = find_first_match("mhd.in", fdir)
      except IndexError:
        tmp = ''
        print("(WWW) init: no .idl or mhd.in files found." +
              "Units set to 'standard' Bifrost units.")
  '''
  uni = obj.uni

  totconst = 2.0 * uni.pi * uni.m_electron * uni.k_b / \
      uni.hplanck / uni.hplanck
  abnd = np.zeros(len(uni.abnddic))
  count = 0

  for ibnd in uni.abnddic.keys():
    abnddic = 10**(uni.abnddic[ibnd] - 12.0)
    abnd[count] = abnddic * uni.weightdic[ibnd] * uni.amu
    count += 1

  abnd = abnd / np.sum(abnd)
  phit = (totconst * tg)**(1.5) * 2.0 / nel
  kbtg = uni.ev_to_erg / uni.k_b / tg
  n1_n0 = phit * uni.u1dic[elem] / uni.u0dic[elem] * np.exp(
      - uni.xidic[elem] * kbtg)
  c2 = abnd[uni.atomdic[elem] - 1] * rho
  ifracpos = n1_n0 / (1.0 + n1_n0)

  if dens:
    if lvl == '1':
      return (1.0 - ifracpos) * c2 
    else:
      return ifracpos * c2 

  else:
    if lvl == '1':
      return (1.0 - ifracpos) * c2  / uni.weightdic[elem] / uni.amu
    else:
      return ifracpos * c2 /uni.weightdic[elem] / uni.amu


def find_first_match(name, path,incl_path=False, **kwargs):
  '''
  This will find the first match,
  name : string, e.g., 'patern*'
  incl_root: boolean, if true will add full path, otherwise, the name.
  path : sring, e.g., '.'
  '''
  originalpath=os.getcwd()
  os.chdir(path)
  for file in glob(name):
    if incl_path:
      os.chdir(originalpath)
      return os.path.join(path, file)
    else:
      os.chdir(originalpath)
      return file
  os.chdir(originalpath)
