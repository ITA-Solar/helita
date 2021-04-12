# import builtins
import warnings

# import internal modules
from . import document_vars

# import external public modules
import numpy as np

# import external private modules
try:
  from at_tools import fluids as fl
except ImportError:
  warnings.warn('failed to import at_tools.fluids; some functions in helita.sim.load_mf_quantities may crash')


def load_mf_quantities(obj, quant, *args, GLOBAL_QUANT=None, COLFRE_QUANT=None, 
                      NDENS_QUANT=None, CROSTAB_QUANT=None, LOGCUL_QUANT=None, 
                      SPITZERTERM_QUANT=None, PLASMA_QUANT=None, DRIFT_QUANT=None, 
                      ELECTRON_QUANT=None,
                      **kwargs):

  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'mf_quantities', 'These are the multi-fluid quantities; only used by ebysus.')

  val = get_global_var(obj, quant, GLOBAL_QUANT=GLOBAL_QUANT)
  if val is None:
    val = get_electron_var(obj, quant, ELECTRON_QUANT=ELECTRON_QUANT)
  if val is None:
    val = get_mf_ndens(obj, quant, NDENS_QUANT=NDENS_QUANT)
  if val is None:
    val = get_mf_colf(obj, quant, COLFRE_QUANT=COLFRE_QUANT)
  if val is None:
    val = get_mf_logcul(obj, quant, LOGCUL_QUANT=LOGCUL_QUANT)
  if val is None:
    val = get_mf_driftvar(obj, quant, DRIFT_QUANT=DRIFT_QUANT)
  if val is None:
    val = get_mf_cross(obj, quant, CROSTAB_QUANT=CROSTAB_QUANT)
  if val is None:
    val = get_spitzerterm(obj, quant, SPITZERTERM_QUANT=SPITZERTERM_QUANT)  
  if val is None: 
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
  '''Variables which are calculated by looping through species or levels.'''
  if GLOBAL_QUANT is None:
      GLOBAL_QUANT = ['totr', 'rc', 'rneu', 'tot_e', 'tot_ke', 'grph', 'tot_part', 'mu', 'pe', ]

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'GLOBAL_QUANT', GLOBAL_QUANT, get_global_var.__doc__)
    docvar('totr', 'sum of mass densities of all fluids [simu. mass density units]')
    docvar('rc',   'sum of mass densities of all ionized fluids [simu. mass density units]')
    docvar('rneu', 'sum of mass densities of all neutral species [simu. mass density units]')
    docvar('tot_e',  'sum of internal energy densities of all fluids [simu. energy density units]')
    docvar('tot_ke', 'sum of kinetic  energy densities of all fluids [simu. energy density units]')
    docvar('grph',  'grams per hydrogen atom')
    docvar('tot_part', 'total number of particles, including free electrons [cm^-3]')
    docvar('mu', 'ratio of total number of particles without free electrong / tot_part')

  if (var == '') or var not in GLOBAL_QUANT:
      return None

  output = np.zeros(np.shape(obj.r))    
  if var == 'totr':  # total density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

  elif var == 'rc':  # total ionized density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        if (obj.att[ispecies].params.levels['stage'][ilevel-1] > 1): 
          output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

  elif var == 'rneu':  # total neutral density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        if (obj.att[ispecies].params.levels['stage'][ilevel-1] == 1): 
          output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

  elif var == 'tot_e':
    warnings.warn('summing e for all non-electron fluids; e for electrons not yet added to this term.')
    # TODO: add electron internal energy density to output; remove warning above.
    for fluid in fl.Fluids(dd=obj):
      output += obj.get_var('e', ifluid=fluid.SL) # internal energy density of fluid

  elif var == 'tot_ke':
    output = 0.5 * obj.get_var('re') * obj.get_var('ue2')   # kinetic energy density of electrons
    for fluid in fl.Fluids(dd=obj):
      output += 0.5 * obj.get_var('r', ifluid=fluid.SL) * obj.get_var('u2')  # kinetic energy density of fluid
  
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
      weight = obj.att[ispecies].params.atomic_weight * \
            obj.uni.amu / obj.uni.u_r
      for mf_ilevel in range(1,nlevels+1):
        output += obj.get_var('r', mf_ispecies=ispecies,
            mf_ilevel=mf_ilevel) / weight 
    output = output / obj.get_var('tot_part')

  return output

def get_electron_var(obj, var, ELECTRON_QUANT=None):
  '''variables related to electrons (requires looping over ions to calculate).'''

  if ELECTRON_QUANT is None:
    ELECTRON_QUANT = ['nel', 're', 'ue2', 'uex', 'uey', 'uez', 'eke']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'ELECTRON_QUANT', ELECTRON_QUANT, get_electron_var.__doc__)
    docvar('nel',  'electron number density [cm^-3]')
    docvar('re',   'mass density of electrons [simu. mass density units]')
    untested_warning = \
      ' Tested uex agrees between helita (uex) & ebysus (eux), for one set of units, for current=0. - SE Apr 4 2021.'
    docvar('ue2',   'electron speed (magnitude of velocity) SQUARED [simu. velocity units SQUARED]' + untested_warning)
    for v in 'uex', 'uey', 'uez':
      docvar(v, '{}-component of electron velocity [simu. velocity units]'.format(v[-1]) + untested_warning)
    docvar('eke',  'electron kinetic energy density [simu. energy density units]')

  if (var == '') or (var not in ELECTRON_QUANT):
    return None

  output = np.zeros_like(obj.r)

  if var == 'nel': # number density of electrons [cm^-3]
    for fluid in fl.Fluids(dd=obj).ions():
      output += obj.get_var('nr', ifluid=fluid.SL) * fluid.ionization    #[cm^-3]

  elif var == 're': # mass density of electrons [simu. mass density units]
    for fluid in fl.Fluids(dd=obj).ions():
      nr = obj.get_var('r', ifluid=fluid.SL) / fluid.atomic_weight  #[simu mass density units / amu]
      output += nr * fluid.ionization       #[(simu mass density units / amu) * elementary charge]
    output = output * (obj.uni.msi_e / obj.uni.amusi)

  elif var == 'ue2': # electron speed [simu. velocity units]
    output = obj.get_var('uex')**2 + obj.get_var('uey')**2 + obj.get_var('uez')**2
  elif var.startswith('ue'): # electron velocity [simu. velocity units]
    # using the formula:
    ## ne qe ue = sum_j(nj uj qj) - i,   where i = current area density (charge per unit area per unit time)
    axis   = var[-1]
    i_uni  = obj.uni.u_r / (obj.uni.u_b * obj.uni.u_t * obj.uni.q_electron)     # (see unit conversion table on wiki.)
    nel    = np.zeros_like(obj.r)                   # calculate nel as we loop through fluids below, to improve efficiency.
    output = -1 * obj.get_var('i'+axis) * i_uni     # [simu velocity units * cm^-3]
    if not np.all(output == 0): 
      # remove this warning once this code has been tested.
      warnings.warn("Nonzero current has not been tested to confirm it matches between helita & ebysus. "+\
                    "You can test it by saving 'eux' via aux, and comparing get_var('eux') to get_var('uex').")
    for fluid in fl.Fluids(dd=obj).ions():
      nr   = obj.get_var('nr', ifluid=fluid.SL)     # [cm^-3]
      nel  += nr * fluid.ionization                 # [cm^-3]
      u    = obj.get_var('u'+axis, ifluid=fluid.SL) # [simu velocity units]
      output += nr * u * fluid.ionization           # [simu velocity units * cm^-3]
    output = output / nel                        # [simu velocity units]

  elif var == 'eke': #electron kinetic energy density [simu. energy density units]
    return 0.5 * obj.get_var('re') * obj.get_var('ue2')


  return output


def get_mf_ndens(obj, var, NDENS_QUANT=None):
  '''number density'''
  if NDENS_QUANT is None:
    NDENS_QUANT = ['nr']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'NDENS_QUANT', NDENS_QUANT, get_mf_ndens.__doc__)
    docvar('nr', 'number density [cm^-3]')

  if (var == '') or var not in NDENS_QUANT:
    return None

  if var == 'nr':
    return obj.get_var('r') * obj.uni.u_r / (obj.uni.amu * obj.att[obj.mf_ispecies].params.atomic_weight)

def get_spitzerterm(obj, var, SPITZERTERM_QUANT=None):
  '''spitzer conductivies'''
  if SPITZERTERM_QUANT is None:
    SPITZERTERM_QUANT = ['kappaq','dxTe','dyTe','dzTe','rhs']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'SPITZTERM_QUANT', SPITZERTERM_QUANT, get_spitzerterm.__doc__)
    #docvar('rhs', 'Someone who knows what this means should put a description here.')
    #docvar('kappaq', '???')

  if (var == '') or var not in SPITZERTERM_QUANT:
    return None

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



def get_mf_colf(obj, var, COLFRE_QUANT=None):
  '''quantities related to collision frequency'''
  if COLFRE_QUANT is None:
    COLFRE_QUANT = ['c_tot_per_vol', '1dcolslope',
                    'nu_ij','nu_en','nu_ei','nu_ij_mx']  

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'COLFRE_QUANT', COLFRE_QUANT, get_mf_colf.__doc__)
    momtrans_start = 'momentum transfer collision frequenc{:} [s^-1] between '
    colfreqnote = ' Note: m_a  n_a  nu_ab  =  m_b  n_b  nu_ba'  #identity for momentum transfer col. freq.s
    docvar('nu_ij', momtrans_start.format('y') + 'ifluid & jfluid. Use species<0 for electrons.' + colfreqnote)
    docvar('nu_ei', momtrans_start.format('y') + 'electrons & a single ion ifluid.' + colfreqnote)
    docvar('nu_en', 'sum of ' + momtrans_start.format('ies') + 'electrons & neutral fluids.' + colfreqnote)
    docvar('1dcolslope', '-(nu_ij + nu_ji)')
    docvar('c_tot_per_vol', 'number of collisions per volume; might be off by a factor of mass ratio... -SE Apr 5 21')

  if (var == '') or var not in COLFRE_QUANT:
    return None

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
      
    #restore original i & j species & levels
    obj.set_mfi(ispecies, ilevel)
    obj.set_mfj(jspecies, jlevel) #SE: mfj should be unchanged anyway. included for readability.
    cross = obj.get_var('cross')  # units are in cm^2.
    #calculate & return nu_ij:
    return 8./3. * n_j * m_j / (m_i + m_j) * cross * np.sqrt(8 * obj.uni.kboltzmann * tgij / (np.pi * mu))
  
  elif var == "nu_ij_mx":
    #### ASSUMES one fluid is charged & other is neutral. ####
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
    m_j = obj.uni.msi_e/obj.uni.amusi if jspec<0 else obj.att[jspec].params.atomic_weight #mass [amu]
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




def get_mf_logcul(obj, var, LOGCUL_QUANT=None):
  '''coulomb logarithm'''
  if LOGCUL_QUANT is None:
    LOGCUL_QUANT = ['logcul']  

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'LOGCUL_QUANT', LOGCUL_QUANT, get_mf_logcul.__doc__)
    docvar('logcul', 'Coulomb Logarithmic used for Coulomb collisions.')

  if (var == '') or var not in LOGCUL_QUANT:
    return None
  
  if var == "logcul":
    ispecies = obj.mf_ispecies
    ilevel = obj.mf_ilevel
    etg = obj.get_var('etg')
    nel = obj.get_var('nel')
    obj.set_mfi(ispecies,ilevel)
    return 23. + 1.5 * np.log(etg / 1.e6) - \
          0.5 * np.log(nel / 1e6)


def get_mf_driftvar(obj, var, DRIFT_QUANT=None):
  '''var drift between fluids. I.e. var_ifluid - var_jfluid'''
  if DRIFT_QUANT is None:
    DRIFT_QUANT = ['ud', 'pd', 'ed', 'rd', 'tgd']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'DRIFT_QUANT', DRIFT_QUANT, get_mf_driftvar.__doc__)
    def doc_start(var):
      return '"drift" for quantity ("{var}"). I.e. ({va_} for ifluid) - ({va_} for jfluid). '.format(var=var, va_=var[:-1])
    def doc_axis(var):
      return ' Must append x, y, or z; e.g. {var}x for (ifluid {va_}x) - (jfluid {va_}x).'.format(var=var, va_=var[:-1])
    docvar('ud', doc_start(var='ud') + 'u = velocity [simu. units].' + doc_axis(var='ud'))
    docvar('pd', doc_start(var='pd') + 'p = momentum density [simu. units].' + doc_axis(var='pd'))
    docvar('ed', doc_start(var='ed') + 'e = energy (density??) [simu. units].')
    docvar('rd', doc_start(var='rd') + 'r = mass density [simu. units].')
    docvar('tgd', doc_start(var='tgd') + 'tg = temperature [K].')

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
  '''cross section between species'''
  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = ['cross']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'CROSTAB_QUANT', CROSTAB_QUANT, get_mf_cross.__doc__)
    docvar('cross', 'cross section between ifluid and jfluid [cgs]. Use species < 0 for electrons.')

  if var=='' or var not in CROSTAB_QUANT:
    return None

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
    cross_tab = 'h-p-bruno-fits.txt'
  elif (([spic1, spic2] == ['h', 'he']) or
        ([spic2, spic1] == ['h', 'he'])):
    cross_tab = 'he-p-bruno-fits.txt'
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



def get_mf_plasmaparam(obj, quant, PLASMA_QUANT=None):
  '''plasma parameters, e.g. plasma beta, sound speed, pressure scale height'''
  if PLASMA_QUANT is None:
    PLASMA_QUANT = ['beta', 'va', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky',
                'kz']
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'PLASMA_QUANT', PLASMA_QUANT, get_mf_plasmaparam.__doc__)
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
    return None

  if quant=='' or quant not in PLASMA_QUANT:
    return None

  if quant in ['hp', 's', 'cs', 'beta']:
    var = obj.get_var('mfe_p')
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
