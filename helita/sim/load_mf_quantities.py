import numpy as np


def load_mf_quantities(obj, quant, *args, GLOBAL_QUANT=None, COLFRE_QUANT=None, 
                      NDENS_QUANT=None, CROSTAB_QUANT=None, LOGCUL_QUANT=None, 
                      SPITZERTERM_QUANT=None, PLASMA_QUANT=None, DRIFT_QUANT=None, 
                      **kwargs):

  quant = quant.lower()

  if not hasattr(obj, 'mf_description'):
    obj.mf_description = {}

  val = get_global_var(obj, quant, GLOBAL_QUANT=GLOBAL_QUANT)
  if np.shape(val) is ():
    val = get_mf_ndens(obj, quant, NDENS_QUANT=NDENS_QUANT)
  if np.shape(val) is ():
    val = get_mf_colf(obj, quant, COLFRE_QUANT=COLFRE_QUANT)
  if np.shape(val) is ():
    val = get_mf_logcul(obj, quant, LOGCUL_QUANT=LOGCUL_QUANT)
  if np.shape(val) is ():
    val = get_mf_driftvar(obj, quant, DRIFT_QUANT=DRIFT_QUANT)
  if np.shape(val) is ():
    val = get_mf_cross(obj, quant, CROSTAB_QUANT=CROSTAB_QUANT)
  if np.shape(val) is ():
    val = get_spitzerterm(obj, quant, SPITZERTERM_QUANT=SPITZERTERM_QUANT)  
  if np.shape(val) is (): 
    val = get_mf_plasmaparam(obj, quant, PLASMA_QUANT=PLASMA_QUANT)
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
      GLOBAL_QUANT = ['totr', 'grph', 'tot_part', 'mu', 'nel', 'pe', 'rc','rne']

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
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        for ilevel in range(1,nlevels+1):
          output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)
      return output

    elif var == 'rc':  # total ionized density
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        for ilevel in range(1,nlevels+1):
          if (obj.att[ispecies].params.levels['stage'][ilevel-1] > 1): 
            output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)
      return output

    elif var == 'rneu':  # total ionized density
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        for ilevel in range(1,nlevels+1):
          if (obj.att[ispecies].params.levels['stage'][ilevel-1] == 1): 
            output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)
      return output

    elif var == 'nel':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
                 obj.uni.amu / obj.uni.u_r
        for ilevel in range(1,nlevels+1):
          obj.att[ispecies].params.nlevel
          output += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=ilevel) / weight * (obj.att[ispecies].params.levels['stage'][ilevel-1]-1)
          
    elif var == 'pe':
      output = (obj.uni.gamma-1) * obj.get_var('e', mf_ispecies=-1) 

    elif var == 'grph':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
          obj.uni.amu / obj.uni.u_r

        for ilevel in range(1,nlevels+1):
          total_hpart += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=ilevel) / weight

        for ispecies in obj.att:
          nlevels = obj.att[ispecies].params.nlevel
          weight = obj.att[ispecies].params.atomic_weight * \
              obj.uni.amu / obj.uni.u_r

          for ilevel in range(1,nlevels+1):
            output += obj.get_var('r', mf_ispecies=ispecies,
                mf_ilevel=ilevel) / mf_total_hpart * u_r

    elif var == 'tot_part':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        weight = obj.att[ispecies].params.atomic_weight * \
              obj.uni.amu / obj.uni.u_r
        for ilevel in range(1,nlevels+1):
          output += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=ilevel) / weight * (obj.att[ispecies].params.levels[ilevel-1]+1)

    elif var == 'mu':
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        for mf_ilevel in range(1,nlevels+1):
          output += obj.get_var('r', mf_ispecies=ispecies,
              mf_ilevel=mf_ilevel)

    return output
  else:
    return None


def get_mf_ndens(obj, var, NDENS_QUANT=None):
  if NDENS_QUANT is None:
    NDENS_QUANT = ['nr']

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

def get_spitzerterm(obj, var, SPITZERTERM_QUANT=None):
  if SPITZERTERM_QUANT is None:
    SPITZERTERM_QUANT = ['kappaq','dxTe','dyTe','dzTe','rhs']

  obj.mf_description['SPITZERTERM_QUANT'] = ('These variables are calculate spitzer conductivities'
                                       'either speciess or levels' +
                                       ', '.join(SPITZERTERM_QUANT))
  if 'ALL' in obj.mf_description.keys():
    obj.mf_description['ALL'] += "\n" + obj.mf_description['SPITZERTERM_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['SPITZERTERM_QUANT']

  if (var == ''):
    return None

  if var in SPITZERTERM_QUANT:
    if (var == 'kappaq'):
      spitzer_amp = 1.0
      kappa_e = 1.1E-25
      kappaq0 = kappa_e * spitzer_amp
      te  = obj.get_var('etg')
      result = kappaq0*(te)**(5.0/2.0)

    if (var == 'dxTe'):     
      gradx_Te = obj.get_var('detgdxup')
      result = gradx_Te

    if (var == 'dyTe'):
      grady_Te = obj.get_var('detgdyup')
      result = grady_Te
    
    if (var == 'dzTe'):
      gradz_Te = obj.get_var('detgdzup')
      result = gradz_Te

    if (var == 'rhs'):  
      bx =   obj.get_var('bx')
      by =   obj.get_var('by')
      bz =   obj.get_var('bz')
      gradx_Te = obj.get_var('detgdxup')
      grady_Te = obj.get_var('detgdyup')
      gradz_Te = obj.get_var('detgdzup')

      bmin = 1E-5 

      normb = np.sqrt(bx**2+by**2+bz**2)
      norm2bmin = bx**2+by**2+bz**2+bmin**2

      bbx = bx/normb
      bby = by/normb
      bbz = bz/normb

      bm = (bmin**2)/norm2bmin

      rhs = bbx*gradx_Te + bby*grady_Te + bbz*gradz_Te
      result = rhs

    return result
  else:
    return None



def get_mf_colf(obj, var, COLFRE_QUANT=None):
  if COLFRE_QUANT is None:
    COLFRE_QUANT = ['c_tot_per_vol', '1dcolslope',
                    'nu_ij','nu_en','nu_ei','nu_ij_mx']  

    # JMS in obj.mf_description
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
  
  print(var, COLFRE_QUANT)
  if var in COLFRE_QUANT:
    if var == "c_tot_per_vol":
      (s_i, l_i) = (obj.mf_ispecies, obj.mf_ilevel)
      (s_j, l_j) = (obj.mf_jspecies, obj.mf_jlevel)
      m_i = obj.att[obj.mf_ispecies].params.atomic_weight
      m_j = obj.att[obj.mf_jspecies].params.atomic_weight
      value = obj.get_var("nu_ij") * \
          obj.get_var("nr", mf_ispecies=s_j, mf_ilevel=l_j) 
          # SE added /mu_ji -> C_tot_per_vol == collisions/volume
      obj.set_mfi(s_i, l_i)
      obj.set_mfj(s_j, l_j) #SE: mfj should be unchanged anyway. included for readability.
      return value

    elif var == "nu_ij":
      ispecies = obj.mf_ispecies
      jspecies = obj.mf_jspecies
      ilevel = obj.mf_ilevel
      jlevel = obj.mf_jlevel
      #get n_j:
      n_j   = obj.get_var("nr", mf_ispecies=jspecies, mf_ilevel=jlevel)
      #restore original i & j species & levels
      obj.set_mfi(ispecies, ilevel)
      obj.set_mfj(jspecies, jlevel) #SE: mfj should be unchanged anyway. included for readability.
      if (ispecies < 0):
        m_i   = obj.uni.m_electron/obj.uni.amu
        tgi   = obj.get_var('etg',    mf_ispecies=ispecies, mf_ilevel=ilevel)
      else: 
        m_i   = obj.att[ispecies].params.atomic_weight
        tgi   = obj.get_var('mfe_tg', mf_ispecies=ispecies, mf_ilevel=ilevel)
      #get m_j, tgj:
      if (jspecies < 0):
        m_j   = obj.uni.m_electron/obj.uni.amu
        tgj   = obj.get_var('etg',    mf_ispecies=jspecies, mf_ilevel=jlevel)
      else:
        m_j   = obj.att[jspecies].params.atomic_weight
        tgj   = obj.get_var('mfe_tg', mf_ispecies=jspecies, mf_ilevel=jlevel)
      #more variables
      mu    = obj.uni.amu * m_i * m_j / (m_i + m_j)

      #get tgj:
      obj.set_mfi(jspecies, jlevel)
      tgj = obj.get_var('etg') if jspecies < 0 else obj.get_var('mfe_tg')
      #restore original i & j species & levels
      obj.set_mfi(ispecies, ilevel)
      obj.set_mfj(jspecies, jlevel) #SE: mfj should be unchanged anyway. included for readability.      
      #calculate tgij; the mass-weighted temperature used in nu_ij calculation.
      tgij = (m_i * tgj + m_j * tgi) / (m_i + m_j)
      if not(ispecies < 0) and not(jspecies < 0): 
        if ((obj.att[ispecies].params.levels[ilevel-1]['stage'] > 1) and (
             obj.att[jspecies].params.levels[jlevel-1]['stage'] > 1)):
          m_h = obj.uni.m_h
          logcul = obj.get_var('logcul')
          return 1.7 * logcul/20.0 * (m_h/(m_i * obj.uni.amu)) * (mu/m_h)**0.5 * \
               n_j / tgij**1.5 * (obj.att[jspecies].params.levels[jlevel-1]['stage'] - 1.0)
        
      else: 
        #restore original i & j species & levels
        obj.set_mfi(ispecies, ilevel)
        obj.set_mfj(jspecies, jlevel) #SE: mfj should be unchanged anyway. included for readability.
        cross = obj.get_var('cross')  # units are in cm^2.
        #calculate & return nu_ij:
        return n_j * m_j / (m_i + m_j) * cross * np.sqrt(8 * obj.uni.kboltzmann * tgij / (np.pi * mu))
    
    elif var == "nu_ij_mx":
      #### ASSUMES ifluid is charged AND jfluid is neutral. ####
      #set constants. for more details, see eq2 in Appendix A of Oppenheim 2020 paper.
      CONST_MULT    = 1.96     #factor in front.
      CONST_ALPHA_N = 6.67e-31 #[m^3]    #polarizability for Hydrogen #unsure of units.
      e_charge= 1.602176e-19   #[C]      #elementary charge
      eps0    = 8.854187e-12   #[F m^-1] #epsilon0, standard definition
      #get variables.
      (ispec, ilvl) = (obj.mf_ispecies, obj.mf_ilevel)
      (jspec, jlvl) = (obj.mf_jspecies, obj.mf_jlevel)
      n_j = obj.get_var("nr", mf_ispecies=jspec, mf_ilevel=jlvl) /(obj.uni.cm_to_m**3)      #number density [m^-3]
      m_i = obj.uni.msi_e/obj.uni.amusi if ispec<0 else obj.att[ispec].params.atomic_weight #mass [amu]
      #m_j = obj.uni.msi_e/obj.uni.amusi if jspec<0 else obj.att[jspec].params.atomic_weight #mass [amu]
      m_j = obj.att[jspec].params.atomic_weight #mass [amu]      #jfluid is assumed to be neutral --> can't be electrons.
      #restore original i & j species & levels
      obj.set_mfi(ispec, ilvl)
      obj.set_mfj(jspec, jlvl) #SE: mfj should be unchanged anyway. included for readability
      #calculate & return nu_ij_test:
      return CONST_MULT * n_j * np.sqrt(CONST_ALPHA_N * e_charge**2 * m_j / (eps0 * obj.uni.amusi * m_i *(m_i + m_j)))

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
      return value #* m_j / (m_i + m_j)

    elif var == 'nu_ei':
      result = np.zeros(np.shape(obj.r))
      ispecies = obj.mf_ispecies
      ilevel = obj.mf_ilevel      
      nel = obj.get_var('nel')
      nr = obj.get_var('nr',mf_ispecies=ispecies,mf_ilevel=ilevel)
      obj.set_mfi(ispecies,ilevel)
      #for ispecies in obj.att:
      #nlevels = obj.att[ispecies].params.nlevel
      #for ilevel in range(1,nlevels+1):
      
      if (obj.att[ispecies].params.levels['stage'][ilevel-1] > 1):
        mst = obj.att[ispecies].params.atomic_weight*obj.uni.amu * obj.uni.m_electron / (
              obj.att[ispecies].params.atomic_weight*obj.uni.amu + (obj.uni.m_electron))

        #mst = obj.uni.m_electron

        etg = obj.get_var('etg')
        tg1 = obj.get_var('mfe_tg',mf_ispecies=ispecies,mf_ilevel=ilevel)

        tg =  ((obj.uni.m_electron/obj.uni.amu) * tg1 + 
              obj.att[ispecies].params.atomic_weight * etg)/((
              obj.uni.m_electron/obj.uni.amu)+obj.att[ispecies].params.atomic_weight) 

        culblog = 23. + 1.5 * np.log(etg / 1.e6) - \
            0.5 * np.log(nel / 1e6)
        result = 1.7 * culblog/20. * (obj.uni.m_h/obj.uni.m_electron) * nr * \
            (mst/obj.uni.m_h)**0.5 / tg**1.5 * (obj.att[ispecies].params.levels['stage'][ilevel-1]-1)
      obj.set_mfi(ispecies,ilevel)
      return result 

    elif var == 'nu_en':
      result = np.zeros(np.shape(obj.r))
      lvl = 1
      for ispecies in obj.att:
        nlevels = obj.att[ispecies].params.nlevel
        for ilevel in range(1,nlevels+1):
          if (obj.att[ispecies].params.levels['stage'][ilevel-1] == 1):
            result += obj.get_var('nu_ij', mf_ispecies=-1,mf_jspeces=ispecies,mf_jlevel=ilevel)
      
      return result 

    else:
      print('ERROR: under construction, the idea is to include here quantity vars specific for species/levels')
      return obj.r * 0.0

  else:
    return None



def get_mf_logcul(obj, var, LOGCUL_QUANT=None):
  if LOGCUL_QUANT is None:
    LOGCUL_QUANT = ['logcul']  

    # JMS in obj.mf_description
    # you could describe what is what with the detail or definitions that you desire.

  obj.mf_description['LOGCUL_QUANT'] = ('Logcul')

  if 'ALL' in obj.mf_description.keys():
    if not obj.mf_description['LOGCUL_QUANT'] in obj.mf_description['ALL']:
    #SE added ^this^ line so that info will only be added to ALL if not in ALL already.
        obj.mf_description['ALL'] += "\n" + obj.mf_description['LOGCUL_QUANT']
  else:
    obj.mf_description['ALL'] = obj.mf_description['LOGCUL_QUANT']

  if (var == ''):
    return None
  
  if var in LOGCUL_QUANT:
    if var == "logcul":
      ispecies = obj.mf_ispecies
      ilevel = obj.mf_ilevel
      etg = obj.get_var('etg')
      nel = obj.get_var('nel')
      obj.set_mfi(ispecies,ilevel)
      return 23. + 1.5 * np.log(etg / 1.e6) - \
            0.5 * np.log(nel / 1e6)

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
    ispecies = obj.mf_ispecies
    jspecies = obj.mf_jspecies
    ilevel = obj.mf_ilevel
    jlevel = obj.mf_jlevel
    if (obj.mf_ispecies < 0):
      spic1 = 'e'
      tg1 = obj.get_var('etg') 
      wgt1 = obj.uni.m_electron/obj.uni.amu
    else:
      spic1 = obj.att[ispecies].params.element
      tg1 = obj.get_var('mfe_tg',mf_ispecies=ispecies,mf_ilevel=ilevel)
      wgt1 = obj.att[ispecies].params.atomic_weight

    if (obj.mf_jspecies < 0):
      spic2 = 'e'
      tg2 = obj.get_var('etg')  
      wgt2 = obj.uni.m_electron/obj.uni.amu
    else:
      spic2 = obj.att[jspecies].params.element
      tg2 = obj.get_var('mfe_tg',mf_ispecies=jspecies,mf_ilevel=jlevel)
      wgt2 = obj.att[jspecies].params.atomic_weight
    obj.set_mfi(ispecies,ilevel)
    obj.set_mfj(jspecies,jlevel)

    tg = (tg1*wgt2 + tg2*wgt1)/(wgt1+wgt2)
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



def get_mf_plasmaparam(obj, quant, PLASMA_QUANT=None):

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
    obj.description['ALL'] += "\n" + obj.description['PLASMA']

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
                       var / obj.get_var('totr'))
      elif quant == 's':
        return (np.log(var) - obj.params['gamma'][obj.snapInd] *
                np.log(obj.get_var('totr')))
      elif quant == 'beta':
        return 2 * var / obj.get_var('b2')

    if quant in ['mn', 'man']:
      var = obj.get_var('modu')
      if quant == 'mn':
        return var / (obj.get_var('cs') + 1e-12)
      else:
        return var / (obj.get_var('va') + 1e-12)

    if quant in ['va', 'vax', 'vay', 'vaz']:
      var = obj.get_var('totr')
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
