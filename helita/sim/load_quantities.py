import numpy as np
import os
from glob import glob

elemlist = ['h', 'he', 'c', 'o', 'ne', 'na', 'mg', 'al', 'si', 's',
        'k', 'ca', 'cr', 'fe', 'ni']

CROSTAB_LIST = ['h_' + clist for clist in elemlist]
CROSTAB_LIST += ['e_' + clist for clist in elemlist]
for iel in elemlist:
  CROSTAB_LIST = CROSTAB_LIST + [
    iel + '_' + clist for clist in elemlist]

whsp = '  '

def load_quantities(obj, quant, *args, PLASMA_QUANT=None, CYCL_RES=None,
                COLFRE_QUANT=None, COLFRI_QUANT=None, IONP_QUANT=None,
                EOSTAB_QUANT=None, TAU_QUANT=None, DEBYE_LN_QUANT=None,
                CROSTAB_QUANT=None, COULOMB_COL_QUANT=None, AMB_QUANT=None, 
                HALL_QUANT=None, BATTERY_QUANT=None, SPITZER_QUANT=None, 
                KAPPA_QUANT=None, GYROF_QUANT=None, WAVE_QUANT=None, 
                FLUX_QUANT=None, CURRENT_QUANT=None, COLCOU_QUANT=None,  
                COLCOUMS_QUANT=None, COLFREMX_QUANT=None, **kwargs):
#                HALL_QUANT=None, SPITZER_QUANT=None, **kwargs):
  quant = quant.lower()

  if not hasattr(obj, 'description'):
    obj.description = {}

  val = get_coulomb(obj, quant, COULOMB_COL_QUANT=COULOMB_COL_QUANT)
  if np.shape(val) is ():
    val = get_collision(obj, quant, COLFRE_QUANT=COLFRE_QUANT)
  if np.shape(val) is ():
    val = get_crossections(obj, quant, CROSTAB_QUANT=CROSTAB_QUANT)
  if np.shape(val) is ():
    val = get_collision_ms(obj, quant, COLFRI_QUANT=COLFRI_QUANT)
  if np.shape(val) is ():
    val = get_current(obj, quant, CURRENT_QUANT=CURRENT_QUANT)
  if np.shape(val) is ():
    val = get_flux(obj, quant, FLUX_QUANT=FLUX_QUANT)
  if np.shape(val) is ():
    val = get_plasmaparam(obj, quant, PLASMA_QUANT=PLASMA_QUANT)
  if np.shape(val) is ():
    val = get_wavemode(obj, quant, WAVE_QUANT=WAVE_QUANT)
  if np.shape(val) is ():
    val = get_cyclo_res(obj, quant, CYCL_RES=CYCL_RES)
  if np.shape(val) is ():
    val = get_gyrof(obj, quant, GYROF_QUANT=GYROF_QUANT)
  if np.shape(val) is ():
    val = get_kappa(obj, quant, KAPPA_QUANT=KAPPA_QUANT)
  if np.shape(val) is ():
    val = get_debye_ln(obj, quant, DEBYE_LN_QUANT=DEBYE_LN_QUANT)
  if np.shape(val) is ():
    val = get_ionpopulations(obj, quant, IONP_QUANT=IONP_QUANT)
  if np.shape(val) is ():
    val = get_ambparam(obj, quant, AMB_QUANT=AMB_QUANT)
  if np.shape(val) is ():
    val = get_hallparam(obj, quant, HALL_QUANT=HALL_QUANT)
  if np.shape(val) is ():
    val = get_batteryparam(obj, quant, BATTERY_QUANT=BATTERY_QUANT)  
  if np.shape(val) is ():
    val = get_spitzerparam(obj, quant, SPITZER_QUANT=SPITZER_QUANT) 
  if np.shape(val) is (): 
    val = get_eosparam(obj, quant, EOSTAB_QUANT=EOSTAB_QUANT)
  if np.shape(val) is (): 
    val = get_collcoul(obj, quant, COLCOU_QUANT=COLCOU_QUANT)
  if np.shape(val) is (): 
    val = get_collcoul_ms(obj, quant, COLCOUMS_QUANT=COLCOUMS_QUANT)
  if np.shape(val) is (): 
    val = get_collision_maxw(obj, quant, COLFREMX_QUANT=COLFREMX_QUANT)
  #if np.shape(val) is ():
  #  val = get_spitzerparam(obj, quant)
  return val


def get_crossections(obj, quant, CROSTAB_QUANT=None):

  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = CROSTAB_LIST
    CROSTAB_QUANT += ['p_h','h_h2','p_h2']
  obj.description['CROSTAB'] = ('Cross section between species'
    '(in cgs): ' + ', '.join(CROSTAB_QUANT))
 
  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['CROSTAB']
  else:
    obj.description['ALL'] = obj.description['CROSTAB']

  if (quant == ''):
    return None

  if quant in CROSTAB_QUANT:
    tg = obj.get_var('tg')
    elem = quant.split('_')

    #spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
    #spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
    spic1 = elem[0]
    spic2 = elem[1]

    cross_tab = ''
    crossunits = 2.8e-17

    if (([spic1, spic2] == ['p', 'h']) or ([spic1, spic2] == ['h', 'p'])):
        cross_tab = 'p-h-elast.txt'
    elif ([spic1, spic2] == ['h', 'h']): 
        cross_tab = 'h-h-data2.txt'
    elif ([spic1, spic2] == ['h', 'h2']): 
        cross_tab = 'h-h2-data.txt'
    elif ([spic1, spic2] == ['p', 'h2']): 
        cross_tab = 'h-h2-data.txt'
    elif (([spic1, spic2] == ['h', 'he']) or ([spic2, spic1] == ['h', 'he'])):
      cross_tab = 'p-he.txt'
    elif ([spic1, spic2] == ['he', 'he']):
      cross_tab = 'he-he.txt'
    elif (([spic1, spic2] == ['e', 'he']) or ([spic2, spic1] == ['e', 'he'])):
      cross_tab = 'e-he.txt'
    elif (([spic1, spic2] == ['e', 'h']) or ([spic2, spic1] == ['e', 'h']) or ([spic1, spic2] == ['e', 'p']) or  or ([spic2, spic1] == ['e', 'p'])):
      cross_tab = 'e-h.txt'
    elif (spic1 == 'h'):
      cross = obj.uni.weightdic[spic2] / obj.uni.weightdic['h'] * \
            obj.uni.cross_p * np.ones(np.shape(tg))
    elif (spic2 == 'h'):
      cross = obj.uni.weightdic[spic1] / obj.uni.weightdic['h'] * \
            obj.uni.cross_p * np.ones(np.shape(tg))
    elif (spic1 == 'he'):
      cross = obj.uni.weightdic[spic2] / obj.uni.weightdic['he'] * \
            obj.uni.cross_he * np.ones(np.shape(tg))
    elif (spic2 == 'he'):
      cross = obj.uni.weightdic[spic1] / obj.uni.weightdic['he'] * \
            obj.uni.cross_he * np.ones(np.shape(tg))
    else: 
       cross = obj.uni.weightdic[spic2] / obj.uni.weightdic['h'] * \
            obj.uni.cross_p / (np.pi*obj.uni.weightdic[spic2])**2 * \
            np.ones(np.shape(tg))

    if cross_tab != '':
      crossobj = obj.cross_sect(cross_tab=[cross_tab])
      cross = crossobj.cross_tab[0]['crossunits'] * crossobj.tab_interp(tg)

    try:
      return cross
    except Exception:
      print('(WWW) cross-section: wrong combination of species')
  else:
    return None

def get_eosparam(obj, quant, EOSTAB_QUANT=None): 

  if (EOSTAB_QUANT == None):
      EOSTAB_QUANT = ['ne', 'tg', 'pg', 'kr', 'eps', 'opa', 'temt', 'ent']
      if not hasattr(obj,'description'):
          obj.description={}
  
  obj.description['EOSTAB'] = ('Variables from EOS table. All of them '
          'are in cgs except ne which is in SI. The electron density '
          '[m^-3], temperature [K], pressure [dyn/cm^2], Rosseland opacity '
          '[cm^2/g], scattering probability, opacity, thermal emission and '
          'entropy are as follows: ' + ', '.join(EOSTAB_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['EOSTAB']
  else:
    obj.description['ALL'] = obj.description['EOSTAB']

  if (quant == ''):
    return None


  if quant in EOSTAB_QUANT:
    # unit conversion to SI
    # to g/cm^3
    ur = obj.params['u_r'][obj.snapInd]
    ue = obj.params['u_ee'][obj.snapInd]        # to erg/g
    if obj.hion and quant == 'ne':
        return obj.get_var('hionne')
    rho = obj.get_var('r')
    rho = rho * ur
    ee = obj.get_var('ee')
    ee = ee * ue
    if obj.verbose:
        print(quant + ' interpolation...', whsp*7, end="\r", flush=True)

    fac = 1.0
    # JMS Why SI?? SI seems to work with bifrost_uvotrt.
    if quant == 'ne':
      fac = 1.e6  # cm^-3 to m^-3

    return obj.rhoee.tab_interp(
      rho, ee, order=1, out=quant) * fac

  elif quant == 'tau':
    return obj.calc_tau()
  else: 
   return None





def get_collision(obj, quant, COLFRE_QUANT=None):

  if COLFRE_QUANT is None:
    COLFRE_QUANT = ['nu' + clist for clist in CROSTAB_LIST]
    COLFRE_QUANT += ['nu%s_mag' % clist for clist in CROSTAB_LIST]
    COLFRE_QUANT += ['nue_' + clist for clist in elemlist]
  
  obj.description['COLFRE'] = ('Collision frequency (elastic and charge'
        'exchange) between different species in (cgs): ' +
        ', '.join(COLFRE_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COLFRE']
  else:
    obj.description['ALL'] = obj.description['COLFRE']


  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in COLFRE_QUANT:

    elem = quant.split('_')
    spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
    ion1 = ''.join([i for i in elem[0] if i.isdigit()])
    spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
    ion2 = ''.join([i for i in elem[1] if i.isdigit()])
    spic1 = spic1[2:]
    
    if ((spic1 == 'h') and (int(ion1) > 1)): 
      spic1 = 'p'
    if ((spic2 == 'h') and (int(ion2) > 1)): 
      spic2 = 'p'

    crossarr = obj.get_var('%s_%s' % (spic1, spic2))
    if spic1 == 'p': 
      spic1 = 'h'
    if spic2 == 'p': 
      spic2 = 'h'
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
  else:
    return None



def get_collision_maxw(obj, quant, COLFREMX_QUANT=None):
  '''
  MAxwell molecular collision
  '''
  if COLFREMX_QUANT is None:
    COLFREMX_QUANT = ['numx' + clist for clist in CROSTAB_LIST]
    COLFREMX_QUANT += ['numx%s_mag' % clist for clist in CROSTAB_LIST]
  
  obj.description['COLFREMX'] = ('Collision frequency (elastic and charge'
        'exchange) between different species in (cgs): ' +
        ', '.join(COLFREMX_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COLFREMX']
  else:
    obj.description['ALL'] = obj.description['COLFREMX']

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in COLFREMX_QUANT:

    #### ASSUMES ifluid is charged AND jfluid is neutral. ####
    #set constants. for more details, see eq2 in Appendix A of Oppenheim 2020 paper.
    CONST_MULT    = 1.96     #factor in front.
    CONST_ALPHA_N = 6.67e-31 #[m^3]    #polarizability for Hydrogen #unsure of units.
    e_charge= 1.602176e-19   #[C]      #elementary charge
    eps0    = 8.854187e-12   #[F m^-1] #epsilon0, standard 

    elem = quant.split('_')
    spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
    ion1 = ''.join([i for i in elem[0] if i.isdigit()])
    spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
    ion2 = ''.join([i for i in elem[1] if i.isdigit()])
    spic1 = spic1[4:]
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

    return CONST_MULT * nspic2 * np.sqrt(CONST_ALPHA_N * e_charge**2 * awg2 / (eps0 * obj.uni.amusi * awg1 * (awg1 + awg2)))

  else:
    return None


def get_collcoul(obj, quant, COLCOU_QUANT=None):

  if COLCOU_QUANT is None:
    COLCOU_QUANT = ['nucou' + clist for clist in CROSTAB_LIST]
    COLCOU_QUANT += ['nucoue_' + clist for clist in elemlist]
  
  obj.description['COLCOU'] = ('Coulomb Collision frequency between different'+
        'ionized species in (cgs) (Hansteen et al. 1997): ' +
        ', '.join(COLCOU_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COLCOU']
  else:
    obj.description['ALL'] = obj.description['COLCOU']

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in COLCOU_QUANT:

    elem = quant.split('_')
    spic1 = ''.join([i for i in elem[0] if not i.isdigit()])
    ion1 = ''.join([i for i in elem[0] if i.isdigit()])
    spic2 = ''.join([i for i in elem[1] if not i.isdigit()])
    ion2 = ''.join([i for i in elem[1] if i.isdigit()])
    spic1 = spic1[5:]
    nspic2 = obj.get_var('n%s-%s' % (spic2, ion2)) # scr2

    tg = obj.get_var('tg') #scr1
    if obj.hion:
      nel = obj.get_var('hionne')
    else:
      nel = obj.get_var('ne')
    
    coulog = 23. + 1.5 * np.log(tg/1.e6) - 0.5 * np.log(nel/1e6) # Coulomb logarithm scr4
    
    mst = obj.uni.weightdic[spic1] * obj.uni.weightdic[spic2] * obj.uni.amu / \
        (obj.uni.weightdic[spic1] + obj.uni.weightdic[spic2])

    return 1.7 * coulog/20.0 * (obj.uni.m_h/(obj.uni.weightdic[spic1] * 
          obj.uni.amu)) * (mst/obj.uni.m_h)**0.5 * \
          nspic2 / tg**1.5 * (int(ion2)-1)**2

  else:
    return None

def get_collcoul_ms(obj, quant, COLCOUMS_QUANT=None):

  if (COLCOUMS_QUANT == None):
    COLCOUMS_QUANT = ['nucou_ei', 'nucou_ii']
    COLCOUMS_QUANT += ['nucou' + clist + '_i' for clist in elemlist]

  obj.description['COLCOUMS'] = ('Coulomb collision between ionized fluids' +
                  ' (adding) in (cgs): ' + ', '.join(COLCOUMS_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COLCOUMS']
  else:
    obj.description['ALL'] = obj.description['COLCOUMS']

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in COLCOUMS_QUANT:

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
  else:
    return None

def get_collision_ms(obj, quant, COLFRI_QUANT=None):

  if (COLFRI_QUANT == None):
    COLFRI_QUANT = ['nu_ni', 'numx_ni', 'nu_en', 'nu_ei', 'nu_in', 'nu_ni_mag', 'nu_in_mag']
    COLFRI_QUANT += ['nu' + clist + '_i' for clist in elemlist]
    COLFRI_QUANT += ['numx' + clist + '_i' for clist in elemlist]
    COLFRI_QUANT += ['nu' + clist + '_i_mag' for clist in elemlist]
    COLFRI_QUANT += ['numx' + clist + '_i_mag' for clist in elemlist]
    COLFRI_QUANT += ['nu' + clist + '_n' for clist in elemlist]
    COLFRI_QUANT += ['numx' + clist + '_n' for clist in elemlist]
    COLFRI_QUANT += ['nu' + clist + '_n_mag' for clist in elemlist]
    COLFRI_QUANT += ['numx' + clist + '_n_mag' for clist in elemlist]

  obj.description['COLFRI'] = ('Collision frequency (elastic and charge'
        'exchange) between fluids in (cgs): ' + ', '.join(COLFRI_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COLFRI']
  else:
    obj.description['ALL'] = obj.description['COLFRI']

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in COLFRI_QUANT:

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
                obj.get_var('nu%s1_i%s'% (ielem,mag))

        if ((ielem in elemlist[2:]) and ('_mag' in quant)): 
          result += obj.uni.amu * obj.uni.weightdic[ielem] * \
                obj.get_var('n%s-2' % ielem) * const * \
                obj.get_var('nu%s2_i%s'% (ielem,mag))


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
                obj.get_var('numx%s1_i%s'% (ielem,mag))

        if ((ielem in elemlist[2:]) and ('_mag' in quant)): 
          result += obj.uni.amu * obj.uni.weightdic[ielem] * \
                obj.get_var('n%s-2' % ielem) * const * \
                obj.get_var('numx%s2_i%s'% (ielem,mag))                

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
            obj.get_var('n%s-2' % ielem) * obj.get_var('nu%s2_n%s' % (ielem,mag))
      if obj.heion:
        result += obj.uni.amu * obj.uni.weightdic['he'] * obj.get_var('nhe-3') * \
            obj.get_var('nuhe3_n%s'% mag)


    elif quant == 'nu_ei':
      if obj.hion:
        nel = obj.get_var('hionne')
      else:
        nel = obj.get_var('ne')
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
                         ('nue', ielem, lvl))

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
                  (elem[0], ielem, lvl, addtxt)) #* obj.uni.weightdic[ielem] /\
                  #(obj.uni.weightdic[ielem] + obj.uni.weightdic[elem[0][2:-1]])
      #if obj.heion and quant[-3:] == '_i':
        #result += obj.get_var('%s_%s%s' % (elem[0], 'he3', addtxt)) * obj.uni.weightdic['he'] /\
        #        (obj.uni.weightdic['he'] + obj.uni.weightdic[elem[0][2:-1]])

    return result
  else:
    return None


def get_coulomb(obj, quant, COULOMB_COL_QUANT=None):

  if COULOMB_COL_QUANT is None:
    COULOMB_COL_QUANT = ['coucol' + clist for clist in elemlist]
  obj.description['COULOMB_COL'] = ('Coulomb collision frequency in Hz'
        'units: ' + ', '.join(COULOMB_COL_QUANT))
  
  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['COULOMB_COL']
  else:
    obj.description['ALL'] = obj.description['COULOMB_COL']

  if (quant == ''):
    return None

  if quant in COULOMB_COL_QUANT:
    iele = np.where(COULOMB_COL_QUANT == quant)
    tg = obj.get_var('tg')
    if obj.hion:
      nel = np.copy(obj.get_var('hionne'))
    else:
      nel = np.copy(obj.get_var('ne'))
    elem = quant.replace('coucol', '')

    const = (obj.uni.pi * obj.uni.qsi_electron ** 4 /
             ((4.0 * obj.uni.pi * obj.uni.permsi)**2 *
              np.sqrt(obj.uni.weightdic[elem] * obj.uni.amusi *
                     (2.0 * obj.uni.ksi_b) ** 3) + 1.0e-20))

    return (const * nel.astype('Float64') *
            np.log(12.0 * obj.uni.pi * nel.astype('Float64') *
            obj.get_var('debye_ln').astype('Float64') + 1e-50) /
            (np.sqrt(tg.astype('Float64')**3) + 1.0e-20))
  else:
    return None


def get_current(obj, quant, CURRENT_QUANT=None):

  if CURRENT_QUANT is None:
    CURRENT_QUANT = ['ix', 'iy', 'iz', 'wx', 'wy', 'wz']
  obj.description['CURRENT'] = ('Calculates currents (bifrost units) or'
        'rotational components of the velocity as follows ' +
        ', '.join(CURRENT_QUANT))

  if 'ALL' in obj.description.keys():
      obj.description['ALL'] += "\n" + obj.description['CURRENT']
  else:
      obj.description['ALL'] = obj.description['CURRENT']

  if (quant == ''):
    return None

  if quant in CURRENT_QUANT:
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

  else:
    return None


def get_flux(obj, quant, FLUX_QUANT=None):

  if FLUX_QUANT is None:
    FLUX_QUANT = ['pfx', 'pfy', 'pfz', 'pfex', 'pfey', 'pfez', 'pfwx',
                'pfwy', 'pfwz']
  obj.description['FLUX'] = ('Poynting flux, Flux emergence, and'
      'Poynting flux from "horizontal" motions: ' +
      ', '.join(FLUX_QUANT))

  if 'ALL' in obj.description.keys():
      obj.description['ALL'] += "\n" + obj.description['FLUX']
  else:
      obj.description['ALL'] = obj.description['FLUX']


  if (quant == ''):
    return None

  if quant in FLUX_QUANT:
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
  else:
    return None


def get_plasmaparam(obj, quant, PLASMA_QUANT=None):

  if PLASMA_QUANT is None:
    PLASMA_QUANT = ['beta', 'va', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky',
                'kz']
  obj.description['PLASMA'] = ('Plasma beta, alfven velocity (and its'
        'components), sound speed, entropy, kinetic energy flux'
        '(and its components), magnetic and sonic Mach number'
        'pressure scale height, and each component of the total energy'
        'flux (if applicable, Bifrost units): ' +
        ', '.join(PLASMA_QUANT))

  if 'ALL' in obj.description.keys():
      obj.description['ALL'] += "\n" + obj.description['PLASMA']
  else:
      obj.description['ALL'] = obj.description['PLASMA']

  if (quant == ''):
    return None

  if quant in PLASMA_QUANT:
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
  else:
    return None


def get_wavemode(obj, quant, WAVE_QUANT=None):

  if WAVE_QUANT is None:
    WAVE_QUANT = ['alf', 'fast', 'long']
  obj.description['WAVE'] = ('Alfven, fast and longitudinal wave'
        'components (Bifrost units): ' + ', '.join(WAVE_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['WAVE']
  else:
    obj.description['ALL'] = obj.description['WAVE']

  if (quant == ''):
    return None

  if quant in WAVE_QUANT:
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
  else:
    return None


def get_cyclo_res(obj, quant, CYCL_RES=None):

  if (CYCL_RES is None):
    CYCL_RES = ['n6nhe2', 'n6nhe3', 'nhe2nhe3']
  obj.description['CYCL_RES'] = ('Resonant cyclotron frequencies'
        '(only for do_helium) are (SI units): ' + ', '.join(CYCL_RES))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['CYCL_RES']
  else:
    obj.description['ALL'] = obj.description['CYCL_RES']

  if (quant == ''):
    return None

  if quant in CYCL_RES:
    if obj.hion and obj.heion:
      posn = ([pos for pos, char in enumerate(quant) if char == 'n'])
      q2 = quant[posn[-1]:]
      q1 = quant[:posn[-1]]
      if obj.hion:
        nel = obj.get_var('hionne')
      else:
        nel = obj.get_var('ne')
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
  else:
    return None


def get_gyrof(obj, quant, GYROF_QUANT=None):

  if (GYROF_QUANT is None):
    GYROF_QUANT = ['gfe'] + ['gf' + clist for clist in elemlist]
  obj.description['GYROF'] = ('gyro freqency are (Hz): ' +
        ', '.join(GYROF_QUANT) + ' at the end it must have the ionization' +
        'state, e,g, gfh2 is for ionized hydrogen')

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['GYROF']
  else:
    obj.description['ALL'] = obj.description['GYROF']

  if (quant == ''):
    return None

  if ''.join([i for i in quant if not i.isdigit()]) in GYROF_QUANT:
    if quant == 'gfe':
      return obj.get_var('modb') * obj.uni.usi_b * \
              obj.uni.qsi_electron / (obj.uni.msi_e)
    else:
      ion = float(''.join([i for i in quant if i.isdigit()]))
      return obj.get_var('modb') * obj.uni.usi_b * \
          obj.uni.qsi_electron * (ion - 1.0) / \
          (obj.uni.weightdic[quant[2:-1]] * obj.uni.amusi)
  else:
    return None


def get_kappa(obj, quant, KAPPA_QUANT=None):

  if (KAPPA_QUANT is None):
    KAPPA_QUANT = ['kappanorm_', 'kappae'] + \
        ['kappa' + clist for clist in elemlist]
  obj.description['KAPPA'] = ('kappa, i.e., magnetization are (adimensional): ' +
        ', '.join(KAPPA_QUANT) + ' at the end it must have the ionization' +
        'state, e,g, kappah2 is for protons')

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['KAPPA']
  else:
    obj.description['ALL'] = obj.description['KAPPA']

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


def get_debye_ln(obj, quant, DEBYE_LN_QUANT=None):

  if (DEBYE_LN_QUANT is None):
    DEBYE_LN_QUANT = ['debye_ln']
  obj.description['DEBYE'] = ('Debye length in ... units:' +
        ', '.join(DEBYE_LN_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['DEBYE']
  else:
    obj.description['ALL'] = obj.description['DEBYE']

  if (quant == ''):
    return None

  if quant in DEBYE_LN_QUANT:
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
  else:
    return None


def get_ionpopulations(obj, quant, IONP_QUANT=None):

  if (IONP_QUANT is None):
    IONP_QUANT = ['n' + clist + '-' for clist in elemlist]
    IONP_QUANT += ['r' + clist + '-' for clist in elemlist]
    IONP_QUANT += ['rneu', 'rion', 'nion', 'nneu', 'nelc']
    IONP_QUANT += ['rneu_nomag', 'rion_nomag', 'nion_nomag', 'nneu_nomag']
  obj.description['IONP'] = ('densities for specific ionized species as'
        'follow (in SI): ' + ', '.join(IONP_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['IONP']
  else:
    obj.description['ALL'] = obj.description['IONP']

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
      tg = obj.get_var('tg')
      r = obj.get_var('r')
      if obj.hion:
        nel = np.copy(obj.get_var('hionne'))
      else:
        nel = np.copy(obj.get_var('ne')) 

      if quant[0] == 'n':
        dens = False
      else:
        dens = True
      return ionpopulation(obj, r, nel, tg, elem=spic[1:], lvl=lvl, dens=dens) /1e6 # to convert in cm^3
  else:
    return None


def get_ambparam(obj, quant, AMB_QUANT=None):

  if (AMB_QUANT is None):
    AMB_QUANT = ['uambx', 'uamby', 'uambz', 'ambx', 'amby', 'ambz',
              'eta_amb1', 'eta_amb2', 'eta_amb3', 'eta_amb4', 'eta_amb5','eta_amb6',
              'eta_amb5a', 'eta_amb5b', 'nchi', 'npsi', 'nchi_red', 'npsi_red',
              'rchi', 'rpsi', 'rchi_red', 'rpsi_red','alphai','betai']

  obj.description['AMB'] = ('ambipolar velocity or term as'
          'follow (in Bifrost units): ' + ', '.join(AMB_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['AMB']
  else:
      obj.description['ALL'] = obj.description['AMB']

  if (quant == ''):
    return None

  if (quant in AMB_QUANT):
    axis = quant[-1]
    if quant == 'eta_amb1':  # version from other
      result = (obj.get_var('rneu') / obj.get_var('r') * obj.uni.u_b)**2
      result /= (4.0 * obj.uni.pi * obj.get_var('nu_ni') + 1e-20)
      result *= obj.get_var('b2') #/ 1e7

    elif quant == 'eta_amb6':  # version from other
      result = (obj.get_var('rneu') / obj.get_var('r') * obj.uni.u_b)**2
      bla=obj.get_var('numx_ni')
      result /= (4.0 * obj.uni.pi * obj.get_var('numx_ni') + 1e-20)
      result *= obj.get_var('b2') #/ 1e7

    # This should be the same and eta_amb2 except that eta_amb2 has many more species involved.
    elif quant == 'eta_amb2':
      result = (obj.get_var('rneu') / obj.r * obj.uni.u_b)**2 / (
          4.0 * obj.uni.pi * obj.get_var('nu_in') + 1e-20)
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
      psi = obj.get_var('npsi')
      chi = obj.get_var('nchi')

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
      result = obj.get_var('jxb' + quant[-1]) / \
                           dd.get_var('modb') * dd.get_var('eta_amb')

    elif (quant[-4:-1] == 'amb' and quant[-1] in ['x','y','z'] and 
         quant[1:3] != 'chi' and quant[1:3] != 'psi'):

      axis = quant[-1]
      if axis == 'x':
        varsn = ['y', 'z']
      elif axis == 'y':
        varsn = ['z', 'y']
      elif axis == 'z':
        varsn = ['x', 'y']
      result = (obj.get_var('jxb' + varsn[0]) *
        obj.get_var('b' + varsn[1] + 'c') -
        obj.get_var('jxb' + varsn[1]) *
        obj.get_var('b' + varsn[0] + 'c')) / dd.get_var('b2') * dd.get_var('eta_amb')

    return  result
  else:
    return None

def get_hallparam(obj, quant, HALL_QUANT=None):

  if (HALL_QUANT is None):
    HALL_QUANT = ['uhallx', 'uhally', 'uhallz', 'hallx', 'hally', 'hallz',
                'eta_hall', 'eta_hallb']
  obj.description['HALL'] = ('Hall velocity or term as'
        'follow (in Bifrost units): ' + ', '.join(HALL_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['HALL']
  else:
      obj.description['ALL'] = obj.description['HALL']


  if (quant == ''):
    return None

  if (quant in HALL_QUANT):
    if quant[0] == 'u':
      try:
        result = obj.get_var('j' + quant[-1])
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
      result = obj.get_var('jxb_' + quant[-1]) / obj.get_var('modb')

    return result #obj.get_var('eta_hall') * result
  else:
    return None

def get_batteryparam(obj, quant, BATTERY_QUANT=None):
  if (BATTERY_QUANT is None):
    BATTERY_QUANT = ['bb_constqe', 'dxpe', 'dype', 'dzpe', 'bb_batx',
                    'bb_baty', 'bb_batz']
  
  obj.description['BATTERY'] = ('Battery term ' + ', '.join(BATTERY_QUANT))

  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['BATTERY']
  else:
    obj.description['ALL'] = obj.description['BATTERY']

  if (quant == ''):
    return None
      
  if (quant in BATTERY_QUANT):
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
  else:
    return None


def get_spitzerparam(obj, quant, SPITZER_QUANT=None):
  if (SPITZER_QUANT is None):
    SPITZER_QUANT = ['fcx','fcy','fcz','qspitz']

  obj.description['SPITZER'] = ('Spitzer term ' + ', '.join(SPITZER_QUANT))
  
  if 'ALL' in obj.description.keys():
    obj.description['ALL'] += "\n" + obj.description['SPITZER']
  else:
    obj.description['ALL'] = obj.description['SPITZER']

  if (quant == ''):
    return None
   
  if (quant in SPITZER_QUANT): 
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
  else:
    return None  


def ionpopulation(obj, rho, nel, tg, elem='h', lvl='1', dens=True):
  '''
  rho is supposed to be in Bifrost units.
  tg in [K]
  nel in SI. 
  The output, is in SI
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
  nelcgs = nel * 1e-6
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
      return (1.0 - ifracpos) * c2 * uni.usi_r 
    else:
      return ifracpos * c2 * uni.usi_r

  else:
    if lvl == '1':
      return (1.0 - ifracpos) * c2 * (uni.usi_r / (uni.weightdic[elem] *
                                                   uni.amusi))
    else:
      return ifracpos * c2 * (uni.usi_r / (uni.weightdic[elem] *
                                           uni.amusi))


def find_first_match(name, path,incl_path=False):
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
