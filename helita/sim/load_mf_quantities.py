import numpy as np


def load_mf_quantities(obj, quant, *args, PLASMA_QUANT=None, CYCL_RES=None,
                       COLFRE_QUANT=None, COLFRI_QUANT=None, IONP_QUANT=None,
                       EOSTAB_QUANT=None, TAU_QUANT=None, DEBYE_LN_QUANT=None,
                       CROSTAB_QUANT=None, COULOMB_COL_QUANT=None, AMB_QUANT=None,
                       HALL_QUANT=None, **kwargs):
  quant = quant.lower()

  if not hasattr(obj, 'mf_description'):
    obj.mf_description = {}

  val = get_global_var(obj, quant)
  if np.shape(val) is ():
    val = get_mf_ndens(obj, quant)
  if np.shape(val) is ():
    val = get_mf_colf(obj, quant)
  if np.shape(val) is ():
    val = get_mf_driftvar(obj, quant)
  if np.shape(val) is ():
    val = get_mf_cross(obj, quant)
  return val

  '''
  ADD HERE Stuff from quantity that needs to be modified since it is MFMS:
  PLASMA_QUANT = ['beta', 'va', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                  'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky',
                  'kz']
  CYCL_RES = ['n6nhe2', 'n6nhe3', 'nhe2nhe3']
  COLFRE_QUANT = ['nu' + clist for clist in CROSTAB_QUANT]
  COLFRI_QUANT = ['nu_ni', 'nu_en', 'nu_ei']
  COLFRI_QUANT = COLFRI_QUANT + \
      ['nu' + clist + '_i' for clist in elemlist]
  COLFRI_QUANT = COLFRI_QUANT + \
      ['nu' + clist + '_n' for clist in elemlist]

  '''


def get_global_var(obj, var, GLOBAL_QUANT=None):
  if GLOBAL_QUANT is None:
      GLOBAL_QUANT = ['totr', 'grph', 'tot_part', 'mu', 'nel']

  obj.mf_description['GLOBAL_QUANT'] = ('These variables are calculate looping'
                                        'either speciess or levels' +
                                        ', '.join(GLOBAL_QUANT))
  if 'ALL' in obj.mf_description.keys():
      obj.mf_description['ALL'] += "\n" + obj.mf_description['GLOBAL_QUANT']
  else:
      obj.mf_description['ALL'] = obj.mf_description['GLOBAL_QUANT']

  if (var == ''):
      return None

  if var in GLOBAL_QUANT:
    output = np.zeros(np.shape(obj.r))    
    if var == 'totr':  # total density
      for ispecies in range(0, obj.mf_nspecies):
        nlevels = obj.att[ispecies].params.nlevel
        for ilevel in range(1,nlevels+1):
          ouput += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)
      return ouput

    elif var == 'nel':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
                 obj.uni.amu / obj.uni.u_r
        for ilevel in range(1,nlevels+1):
          obj.att[ispecies].params.nlevel
          output += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=ilevel) / weight * (obj.att[ispecies].params.levels['stage'][ilevel-1]-1)

    elif var == 'grph':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
          obj.uni.amu / obj.uni.u_r

        for ilevel in range(1,nlevels+1):
          total_hpart += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=ilevel) / weight

        for mf_ispecies in obj.att:
          nlevels = obj.att[ispecies].params.nlevel
          weight = obj.att[ispecies].params.atomic_weight * \
              obj.uni.amu / obj.uni.u_r

          for ilevel in range(1,nlevels+1):
            ouput += obj.get_var('r', mf_ispecies=ispecies,
                mf_ilevel=ilevel) / mf_total_hpart * u_r

    elif var == 'tot_part':
      for mf_ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
              obj.uni.amu / obj.uni.u_r
        for ilevel in range(1,nlevels+1):
          ouput += obj.get_var('r', mf_ispecies=mf_ispecies,
              mf_ilevel=ilevel) / weight * (obj.att[ispecies].params.levels[ilevel-1]+1)

    elif var == 'mu':
      for mf_ispecies in obj.att:
        nlevels = obj.att[mf_ispecies].params.nlevel
        for mf_ilevel in range(1,nlevels+1):
          ouput += obj.get_var('r', mf_ispecies=mf_ispecies,
              mf_ilevel=mf_ilevel)

    return output
  else:
    return None


def get_mf_ndens(obj, var, NDENS_QUANT=None):
  if NDENS_QUANT is None:
    NDENS_QUANT = ['n_i']

  obj.mf_description['NDENS_QUANT'] = ('These variables are calculate looping'
                                       'either speciess or levels' +
                                       ', '.join(NDENS_QUANT))
  if 'ALL' in obj.mf_description.keys():
    obj.mf_description['ALL'] += "\n" + obj.mf_description['NDENS_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['NDENS_QUANT']

  if (var == ''):
    return None

  if var in NDENS_QUANT:
    return obj.get_var('r') * obj.params['u_r'][0] / (obj.uni.amu * obj.att[obj.mf_ispecies].params.atomic_weight)
  else:
    return None


def get_mf_colf(obj, var, COLFRE_QUANT=None):
  if COLFRE_QUANT is None:
    COLFRE_QUANT = ['C_tot_per_vol', '1dcolslope',
                    'nu_ij']  # JMS in obj.mf_description
    # you could describe what is what with the detail or definitions that you desire.

  obj.mf_description['COLFRE_QUANT'] = ('Collisional quantities for mf_ispecies '
                                        'and mf_jspecies: ' +
                                        ', '.join(COLFRE_QUANT)+'.\n'
                                        'nu_ij := mu_ji times the frequency with which a single, specific '
                                        'particle of species i will collide with ANY particle of species j, '
                                        'where mu_ji = m_j / (m_i + m_j).\n'
                                        '1dcolslope := -(nu_ij + nu_ji).\n'
                                        'C_tot_per_vol := number of collisions per volume = '
                                        'nu_ij * n_j / mu_ji = nu_ji * n_i / mu_ij.')

  if 'ALL' in obj.mf_description.keys():
    if not obj.mf_description['COLFRE_QUANT'] in obj.mf_description['ALL']:
    #SE added ^this^ line so that info will only be added to ALL if not in ALL already.
        obj.mf_description['ALL'] += "\n" + obj.mf_description['COLFRE_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['COLFRE_QUANT']

  if (var == ''):
    return None
  if var in COLFRE_QUANT:
    if var == "C_tot_per_vol":
      (s_i, l_i) = (obj.mf_ispecies, obj.mf_ilevel)
      (s_j, l_j) = (obj.mf_jspecies, obj.mf_jlevel)
      m_i = obj.att[obj.mf_ispecies].params.atomic_weight
      m_j = obj.att[obj.mf_jspecies].params.atomic_weight
      value = obj.get_var("nu_ij") / (m_j / (m_i + m_j)) * \
          obj.get_var("n_i", mf_ispecies=s_j, mf_ilevel=l_j) 
          # SE added /mu_ji -> C_tot_per_vol == collisions/volume
      obj.set_mfi(s_i, l_i)
      obj.set_mfj(s_j, l_j) #SE: mfj should be unchanged anyway. included for readability.
      return value

    elif var == "nu_ij":
      cross = obj.get_var('cross')  # units are in cm^2.
      m_i   = obj.att[obj.mf_ispecies].params.atomic_weight
      m_j   = obj.att[obj.mf_jspecies].params.atomic_weight
      mu    = obj.uni.amu * m_i * m_j / (m_i + m_j)
      tg    = obj.get_var('mfe_tg')
      #get n_j:
      (s_i, l_i) = (obj.mf_ispecies, obj.mf_ilevel)
      (s_j, l_j) = (obj.mf_jspecies, obj.mf_jlevel)
      n_j   = obj.get_var("n_i", mf_ispecies=s_j, mf_ilevel=l_j)
      obj.set_mfi(s_i, l_i)
      obj.set_mfj(s_j, l_j) #SE: mfj should be unchanged anyway. included for readability.
      #calculate & return nu_ij:
      return n_j * m_j / (m_i + m_j) * cross * np.sqrt(8 * obj.uni.kboltzmann * tg / (np.pi * mu))
      # JMS Added here m_j / (m_i + m_j), I prefer to have mu in the collision frequency instead of spearated.
      # SE (4/9/20 corrected using "n_i" to now properly use "n_j" instead.
      
    elif var == "1dcolslope":
      (s_i, l_i) = (obj.mf_ispecies, obj.mf_ilevel)
      (s_j, l_j) = (obj.mf_jspecies, obj.mf_jlevel)
      value = -(obj.get_var("nu_ij",
                            mf_ispecies=s_i, mf_ilevel=l_i,
                            mf_jspecies=s_j, mf_jlevel=l_j) +
                obj.get_var("nu_ij",
                            mf_ispecies=s_j, mf_ilevel=l_j,
                            mf_jspecies=s_i, mf_jlevel=l_i))
      obj.set_mfi(s_i, l_i)
      obj.set_mfj(s_j, l_j)
      return value

    else:
      print('ERROR: under construction, the idea is to include here quantity vars specific for species/levels')
      return obj.r * 0.0

  else:
    return None


def get_mf_driftvar(obj, var, DRIFT_QUANT=None):
  if DRIFT_QUANT is None:
    DRIFT_QUANT = ['ud', 'pd', 'ed', 'rd', 'tgd']

  obj.mf_description['DRIFT_QUANT'] = (
      'Drift between two fluids ' + ', '.join(DRIFT_QUANT))

  if 'ALL' in obj.mf_description.keys():
    obj.mf_description['ALL'] += "\n" + obj.mf_description['DRIFT_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['DRIFT_QUANT']

  if (var == ''):
    return None

  if var[:-1] in DRIFT_QUANT:
    axis = var[-1]
    varn = var[:-2]
    return (obj.get_var(varn + axis, mf_ispecies=obj.mf_ispecies, mf_ilevel=obj.mf_ilevel) -
            obj.get_var(varn + axis, mf_ispecies=obj.mf_jspecies, mf_ilevel=obj.mf_jlevel))
  elif var in DRIFT_QUANT:
    return (obj.get_var(var[:-1], mf_ispecies=obj.mf_ispecies, mf_ilevel=obj.mf_ilevel) -
            obj.get_var(var[:-1], mf_ispecies=obj.mf_jspecies, mf_ilevel=obj.mf_jlevel))
  else:
    return None


def get_mf_cross(obj, var, CROSTAB_QUANT=None):
  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = ['cross']

  obj.mf_description['CROSTAB_QUANT'] = ('Cross section between species'
                                         '(in cgs): ' + ', '.join(CROSTAB_QUANT))

  if 'ALL' in obj.mf_description.keys():
    obj.mf_description['ALL'] += "\n" + obj.mf_description['CROSTAB_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['CROSTAB_QUANT']

  if (var == ''):
    return None

  if var in CROSTAB_QUANT:
    tg = obj.get_var('mfe_tg')
    if (obj.mf_ispecies < 0):
      spic1 = 'e'
    else:
      spic1 = obj.att[obj.mf_ispecies].params.element
    if (obj.mf_jspecies < 0):
      spic2 = 'e'
    else:
      spic2 = obj.att[obj.mf_jspecies].params.element
    cross_tab = ''
    crossunits = 2.8e-17
    if ([spic1, spic2] == ['h', 'h']):
      cross_tab = 'p-h-elast.txt'
    elif (([spic1, spic2] == ['h', 'he']) or
          ([spic2, spic1] == ['h', 'he'])):
      cross_tab = 'p-he.txt'
    elif ([spic1, spic2] == ['he', 'he']):
      cross_tab = 'he-he.txt'
    elif (([spic1, spic2] == ['e', 'he']) or
          ([spic2, spic1] == ['e', 'he'])):
      cross_tab = 'e-he.txt'
    elif (([spic1, spic2] == ['e', 'h']) or ([spic2, spic1] == ['e', 'h'])):
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

    if cross_tab != '':
      crossobj = obj.cross_sect(cross_tab=[cross_tab])
      crossunits = crossobj.cross_tab[0]['crossunits']
      cross = crossunits * crossobj.tab_interp(tg)

    try:
        return cross
    except Exception:
        print('(WWW) cross-section: wrong combination of species')
  else:
      return None
