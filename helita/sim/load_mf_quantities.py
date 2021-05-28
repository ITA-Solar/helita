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
                      CROSTAB_QUANT=None, LOGCUL_QUANT=None, 
                      SPITZERTERM_QUANT=None, PLASMA_QUANT=None, DRIFT_QUANT=None, 
                      ONEFLUID_QUANT=None, ELECTRON_QUANT=None, HEATING_QUANT=None,
                      CFL_QUANT=None,
                      WAVE_QUANT=None, FB_INSTAB_QUANT=None,
                      **kwargs):

  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'mf_quantities',
      ("These are the multi-fluid quantities; only used by ebysus.\n"
       "nfluid means 'number of fluids used to read the quantity'.\n"
       "  2  -> uses obj.ifluid and obj.jfluid. (e.g. 'nu_ij')\n"
       "  1  -> uses obj.ifluid (but not jfluid). (e.g. 'ux', 'tg')\n"
       "  0  -> does not use ifluid nor jfluid. (e.g. 'bx', 'nel', 'tot_e'))\n")
                              )

  val = get_global_var(obj, quant, GLOBAL_QUANT=GLOBAL_QUANT)
  if val is None:
    val = get_electron_var(obj, quant, ELECTRON_QUANT=ELECTRON_QUANT)
  if val is None:
    val = get_onefluid_var(obj, quant, ONEFLUID_QUANT=ONEFLUID_QUANT)
  if val is None:
    val = get_heating_quant(obj, quant, HEATING_QUANT=HEATING_QUANT)
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
    val = get_cfl_quant(obj, quant, CFL_QUANT=CFL_QUANT)
  if val is None: 
    val = get_mf_plasmaparam(obj, quant, PLASMA_QUANT=PLASMA_QUANT)
  if val is None:
    val = get_mf_wavequant(obj, quant, WAVE_QUANT=WAVE_QUANT)
  if val is None:
    val = get_fb_instab_quant(obj, quant, FB_INSTAB_QUANT=FB_INSTAB_QUANT)
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
      GLOBAL_QUANT = ['totr', 'rc', 'rions', 'rneu',
                      'tot_e', 'tot_ke', 'e_ef', 'e_b', 'total_energy',
                      'tot_px', 'tot_py', 'tot_pz',
                      'grph', 'tot_part', 'mu',
                      'jx', 'jy', 'jz', 'efx', 'efy', 'efz',
                      ]

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'GLOBAL_QUANT', GLOBAL_QUANT, get_global_var.__doc__, nfluid=0)
    docvar('totr', 'sum of mass densities of all fluids [simu. mass density units]')
    for rc in ['rc', 'rions']:
      docvar(rc,   'sum of mass densities of all ionized fluids [simu. mass density units]')
    docvar('rneu', 'sum of mass densities of all neutral species [simu. mass density units]')
    docvar('tot_e',  'sum of internal energy densities of all fluids [simu. energy density units]')
    docvar('tot_ke', 'sum of kinetic  energy densities of all fluids [simu. energy density units]')
    docvar('e_ef', 'energy density in electric field [simu. energy density units]')
    docvar('e_b', 'energy density in magnetic field [simu. energy density units]')
    docvar('total_energy', 'total energy density. tot_e + tot_ke + e_ef + e_b [simu units].')
    for axis in ['x', 'y', 'z']:
      docvar('tot_p'+axis, 'sum of '+axis+'-momentum densities of all fluids [simu. mom. dens. units] ' +\
                           'NOTE: does not include "electron momentum" which is assumed to be ~= 0.')
    docvar('grph',  'grams per hydrogen atom')
    docvar('tot_part', 'total number of particles, including free electrons [cm^-3]')
    docvar('mu', 'ratio of total number of particles without free electrong / tot_part')
    for axis in ['x', 'y', 'z']:
      docvar('j'+axis, 'sum of '+axis+'-component of current per unit area [simu. current per area units]')
    for axis in ['x', 'y', 'z']:
      docvar('ef'+axis, axis+'-component of electric field [simu. E-field units] ' +\
                          '== [simu. B-field units * simu. velocity units]')
    return None

  if var not in GLOBAL_QUANT:
      return None

  output = np.zeros(np.shape(obj.r))    
  if var == 'totr':  # total density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

  elif var in ['rc', 'rions']:  # total ionized density
    for fluid in fl.Fluids(dd=obj).ions():
      output += obj.get_var('r', ifluid=fluid)

  elif var == 'rneu':  # total neutral density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        if (obj.att[ispecies].params.levels['stage'][ilevel-1] == 1): 
          output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

  elif var == 'tot_e':
    output += obj.get_var('e', mf_ispecies= -1) # internal energy density of electrons
    for fluid in fl.Fluids(dd=obj):
      output += obj.get_var('e', ifluid=fluid.SL) # internal energy density of fluid

  elif var == 'tot_ke':
    output = obj.get_var('eke')   # kinetic energy density of electrons
    for fluid in fl.Fluids(dd=obj):
      output += obj.get_var('ke', ifluid=fluid.SL)  # kinetic energy density of fluid

  elif var == 'e_ef':
    ef2  = obj.get_var('ef2')   # |E|^2  [simu E-field units, squared]
    eps0 = obj.uni.permsi       # epsilon_0 [SI units]
    units = obj.uni.usi_ef**2 / obj.uni.usi_e   # convert ef2 * eps0 to [simu energy density units]
    return (0.5 * eps0 * units) * ef2

  elif var == 'e_b':
    b2   = obj.get_var('b2')    # |B|^2  [simu B-field units, squared]
    mu0  = obj.uni.mu0si        # mu_0 [SI units]
    units = obj.uni.usi_b**2 / obj.uni.usi_e    # convert b2 * mu0 to [simu energy density units]
    return (0.5 * mu0 * units) * b2

  elif var == 'total_energy':
    output  = obj.get_var('tot_e')
    output += obj.get_var('tot_ke')
    output += obj.get_var('e_ef')
    output += obj.get_var('e_b')
  
  elif var.startswith('tot_p'):  # note: must be tot_px, tot_py, or tot_pz.
    axis = var[-1]
    for fluid in fl.Fluids(dd=obj):
      output += obj.get_var('p'+axis, ifluid=fluid.SL)   # momentum density of fluid

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

  elif var in ['jx', 'jy', 'jz']:
    # J = curl (B) / mu_0
    warnings.warn('j does not (yet) add contribution from imposed current if it exists.')
    x = var[-1]
    # imposed current (imposed "additional" current, added artificially to system)
    if obj.get_param('do_imposed_current', 0) > 0:
      ic_units = obj.get_param('ic_units', 'ebysus')
      ic_ix    = obj.get_param('ic_i'+x, 0)      # ic_ix [ic_units]
      if   ic_units.strip().lower() == 'si':
        ic_ix /= obj.uni.usi_i     # ic_ix [simu. units]
      elif ic_units.strip().lower() == 'cgs':
        ic_ix /= obj.uni.u_i       # ic_ix [simu. units]
    else:
      ic_ix = 0
    # calculated current
    curlb_x =  obj.get_var('curvec'+'b'+x) * obj.uni.usi_b / obj.uni.usi_l  # (curl b)_x [si units]
    jx = curlb_x / obj.uni.mu0si   # j [si units]
    jx = jx / obj.uni.usi_i        # j [simu. units]
    return ic_ix + jx              # j [simu. units]

  elif var in ['efx', 'efy', 'efz']:
    # E = - ue x B + (ne |qe|)^-1 * ( grad(pressure_e) + (ion & rec terms) + sum_j(R_e^(ej)) )
    # ----- calculate the necessary component of -ue x B (== B x ue) ----- #
    # There is a flag, "do_hall", when "false", we don't let the contribution
    ## from current to ue to enter in to the B x ue for electric field.
    if obj.get_param('do_hall', default="false")=="false":
      ue = 'uep'  # include only the momentum contribution in ue, in our ef calculation.
      warnings.warn('do_hall=="false", so we are dropping the j (current) contribution to ef (E-field)')
    else:
      ue = 'ue'   # include the full ue term, in our ef calculation.
    # we will need to do a cross product, with extra care to interpolate correctly.
    ## we name the axes variables x,y,z to make it easier to understand the code.
    x    = var[-1]  # axis; 'x', 'y', or 'z'
    y, z = dict(x=('y', 'z'), y=('z', 'x'), z=('x', 'y'))[x]
    # make sure we get the interpolation correct:
    ## B and ue are face-centered vectors.
    ## Thus we use _facecross_ from load_arithmetic_quantities.
    B_cross_ue__x = obj.get_var('b_facecross_'+ue+x)
    # ----- calculate grad pressure ----- #
    ## efx is at (0, -1/2, -1/2).
    ## P is at (0,0,0).
    ## dpdxup is at (1/2, 0, 0).
    ## dpdxup xdn ydn zdn is at (0, -1/2, -1/2) --> aligned with efx.
    interp = 'xdnydnzdn'
    gradPe_x = obj.get_var('dpd'+x+'up'+interp, mf_ispecies=-1) # [simu. energy density units]
    # ----- calculate ionization & recombination effects ----- #
    if obj.get_param('do_recion', default=False):
      warnings.warn('E-field contribution from ionization & recombination have not yet been added.')
    # ----- calculate collisional effects (only if do_ohm_ecol) ----- #
    sum_rejx = 0.
    if obj.params['do_ohm_ecol'][obj.snapInd]:
      # efx is at (0, -1/2, -1/2)
      ## rijx is at (-1/2, 0, 0)    (same as ux)
      ## --> to align with efx, we shift rijx by xup ydn zdn
      interp = x+'up'+y+'dn'+z+'dn'
      for fluid in fl.Fluids(dd=obj):
        sum_rejx += obj.get_var('rij'+x + interp, mf_ispecies=-1, jfluid=fluid.SL)
      ## sum_rejx has units [simu. momentum density units / simu. time units]
    # ----- calculate ne qe ----- #
    ## efx is at (0, -1/2, -1/2)
    ## ne is at (0, 0, 0)
    ## to align with efx, we shift ne by ydn zdn
    interp = y+'dn'+z+'dn'
    ne = obj.get_var('nr'+interp, mf_ispecies=-1)   # [simu. number density units]
    neqe = ne * obj.uni.simu_qsi_e                  # [simu. charge density units]
    ## we used simu_qsi_e because we are using here the SI equation for E-field.
    ## if we wanted to use simu_q_e we would have to use the cgs equation instead.
    # ----- calculate efx ----- #
    efx = B_cross_ue__x + (gradPe_x + sum_rejx) / neqe # [simu. E-field units] 
    output = efx

  return output


def get_onefluid_var(obj, var, ONEFLUID_QUANT=None):
  '''variables related to information about a single fluid.
  Use mf_ispecies= -1 to refer to electrons.
  Intended to contain only "simple" physical quantities.
  For more complicated "plasma" quantities such as gryofrequncy, see PLASMA_QUANT.

  Quantities with 'i' are "generic" version of that quantity,
  meaning it works with electrons or another fluid for ifluid.
  For example, obj.get_var('uix') is equivalent to:
    obj.get_var('uex') when obj.mf_ispecies < 0
    obj.get_var('ux') otherwise.
  '''
  if ONEFLUID_QUANT is None:
    ONEFLUID_QUANT = ['nr', 'nq', 'p', 'pressure', 'tg', 'temperature', 'ke', 'vtherm',
                      'ri', 'uix', 'uiy', 'uiz', 'pix', 'piy', 'piz']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'ONEFLUID_QUANT', ONEFLUID_QUANT, get_onefluid_var.__doc__, nfluid=1)
    docvar('nr', 'number density of ifluid [simu. number density units]')
    docvar('nq', 'charge density of ifluid [simu. charge density units]')
    for tg in ['tg', 'temperature']:
      docvar(tg, 'temperature of ifluid [K]')
    for p in ['p', 'pressure']:
      docvar(p, 'pressure of ifluid [simu. energy density units]')
    docvar('ke', 'kinetic energy density of ifluid [simu. units]')
    _equivstr = " Equivalent to obj.get_var('{ve:}') when obj.mf_ispecies < 0; obj.get_var('{vf:}'), otherwise."
    equivstr = lambda v: _equivstr.format(ve=v.replace('i', 'e'), vf=v.replace('i', ''))
    docvar('vtherm', 'thermal speed of ifluid [simu. velocity units]. = sqrt (8 * k_b * T_i / (pi * m_i) )')
    docvar('ri', 'mass density of ifluid [simu. mass density units]. '+equivstr('ri'))
    for uix in ['uix', 'uiy', 'uiz']:
      docvar(uix, 'velocity of ifluid [simu. velocity units]. '+equivstr(uix))
    for pix in ['pix', 'piy', 'piz']:
      docvar(pix, 'momentum density of ifluid [simu. momentum density units]. '+equivstr(pix))
    return None

  if var not in ONEFLUID_QUANT:
    return None

  if var == 'nr':
    if obj.mf_ispecies < 0: # electrons
      return obj.get_var('nel') / obj.uni.u_nr
    else:                   # not electrons
      mass = obj.get_mass(obj.mf_ispecies, units='simu') # [simu. mass units]
      return obj.get_var('r') / mass   # [simu number density units]

  elif var == 'nq':
    charge = obj.get_charge(obj.ifluid, units='simu') # [simu. charge units]
    if charge == 0:
      return np.zeros(obj.r.shape)
    else:
      return charge * obj.get_var('nr')

  elif var in ['p', 'pressure']:
    gamma = obj.uni.gamma
    return (gamma - 1) * obj.get_var('e')          # p = (gamma - 1) * internal energy

  elif var in ['tg', 'temperature']:
    p  = obj.get_var('p') * obj.uni.u_e    # [cgs units]
    nr = obj.get_var('nr') * obj.uni.u_nr  # [cgs units]
    return p / (nr * obj.uni.k_b)          # [K]         # p = n k T

  elif var == 'ke':
    return 0.5 * obj.get_var('ri') * obj.get_var('ui2')

  elif var == 'vtherm':
    Ti     = obj.get_var('tg')                           # [K]
    mi     = obj.get_mass(obj.mf_ispecies, units='si')   # [kg]
    vtherm = np.sqrt(obj.uni.ksi_b * Ti / mi)            # [m / s]
    consts = np.sqrt(8 / np.pi)
    return consts * vtherm / obj.uni.usi_u                   # [simu. velocity units]

  else:
    if var in ['ri', 'uix', 'uiy', 'uiz', 'pix', 'piy', 'piz']:
      if obj.mf_ispecies < 0:  # electrons
        e_var = var.replace('i', 'e')
        return obj.get_var(e_var)
      else:                    # not electrons
        f_var = var.replace('i', '')
        return obj.get_var(f_var)


def get_heating_quant(obj, var, HEATING_QUANT=None):
  '''terms related to heating of fluids.

  Note that the code in this section is written for maximum readability, not maximum efficiency.
  For most vars in this section the code would run a bit faster if you write use-case specific code.

  For example, qcolj gets qcol_uj + qcol_tgj, however each of those will separately calculate
  number density (nr) and collision frequency (nu_ij); so the code will calculate the same value
  of nr and nu_ij two separate times. It would be more efficient to calculate these only once.

  As another example, qjoulei will re-calculate the electric field efx, efy, and efz
  each time it is called; if you are doing a sum of multiple qjoulei terms it would be more
  efficient to calculate each of these only once.

  Thus, if you feel that code in this section is taking too long, you can speed it up by writing
  your own code which reduces the number of times calculations are repeated.
  (Note, if you had N_memmap = 0 or fast=False, first try using N_memmap >= 200, and fast=True.)
  '''
  if HEATING_QUANT is None:
    HEATING_QUANT = ['qcol_uj', 'qcol_tgj', 'qcolj',
                     'qcol_u', 'qcol_tg', 'qcol',
                     'qjoulei']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'HEATING_QUANT', HEATING_QUANT, get_heating_quant.__doc__)
    units = '[simu. energy density per time]'
    heati = 'heating of ifluid '+units
    docvar('qcol_uj',  heati + ' due to jfluid, due to collisions and velocity drifts.', nfluid=2)
    docvar('qcol_tgj', heati + ' due to jfluid, due to collisions and temperature differences.', nfluid=2)
    docvar('qcolj',    'total '+heati+' due to jfluid.', nfluid=2)
    docvar('qcol_u',   heati + ' due to collisions and velocity drifts.', nfluid=1)
    docvar('qcol_tg',  heati + ' due to collisions and temperature differences.', nfluid=1)
    docvar('qcol',     'total '+heati+'.', nfluid=1)
    docvar('qjoulei',  heati + ' due to Ji dot E. (Ji = qi ni ui).', nfluid=1)
    return None

  if var not in HEATING_QUANT:
    return None

  # qcol terms

  if var in ['qcol_uj', 'qcol_tgj']:
    ni = obj.get_var('nr')             # [simu. units]
    mi = obj.get_mass(obj.mf_ispecies) # [amu]
    mj = obj.get_mass(obj.mf_jspecies) # [amu]
    nu_ij = obj.get_var('nu_ij')       # [simu. units]
    coeff = (mi / (mi + mj)) * ni * nu_ij   # [simu units: length^-3 time^-1]
    if var == 'qcol_uj':
      mj_simu = obj.get_mass(obj.mf_jspecies, units='simu') # [simu mass]
      energy = mj_simu * obj.get_var('uid2')                # [simu energy]
    elif var == 'qcol_tgj':
      simu_kB = obj.uni.ksi_b * (obj.uni.usi_nr / obj.uni.usi_e)   # kB [simu energy / K]
      tgi = obj.get_var('tg')                       # [K]
      tgj = obj.get_var('tg', ifluid=obj.jfluid)    # [K]
      energy = (2 / (obj.uni.gamma - 1)) * simu_kB * (tgj - tgi)
    return coeff * energy  # [simu energy density / time]

  elif var == 'qcolj':
    return obj.get_var('qcol_uj') + obj.get_var('qcol_tgj')

  elif var in ['qcol_u', 'qcol_tg']:
    varj   = var + 'j'   # qcol_uj or qcol_tgj
    output = obj.get_var(varj, jS=-1)   # get varj for j = electrons
    for fluid in fl.Fluids(dd=obj):
      if fluid.SL != obj.ifluid:        # exclude varj for j = i
        output += obj.get_var(varj, jfluid=fluid)
    return output

  elif var == 'qcol':
    return obj.get_var('qcol_u') + obj.get_var('qcol_tg')

  # other terms

  elif var == 'qjoulei':
    # qjoulei = qi * ni * \vec{ui} dot \vec{E}
    # ui is on grid cell faces while E is on grid cell edges.
    # We must interpolate to align with energy density e, which is at center of grid cells.
    # uix is at (-0.5, 0, 0) while Ex is at (0, -0.5, -0.5)
    # --> we shift uix by xup, and Ex by yup zup
    result = np.zeros(obj.r.shape)
    qi = obj.get_charge(obj.ifluid, units='simu')    # [simu charge]
    if qi == 0:
      return result    # there is no contribution if qi is 0.
    # else
    ni = obj.get_var('nr')                           # [simu number density]
    for x, y, z in [('x', 'y', 'z'), ('y', 'z', 'x'), ('z', 'x', 'y')]:
      uix = obj.get_var('ui' + x + x+'up')           # [simu velocity]
      efx = obj.get_var('ef' + x + y+'up' + z+'up')  # [simu electric field]
      result += uix * efx
    # << at this point, result = ui dot ef
    return qi * ni * result





def get_electron_var(obj, var, ELECTRON_QUANT=None):
  '''variables related to electrons (requires looping over ions to calculate).'''

  if ELECTRON_QUANT is None:
    ELECTRON_QUANT = ['nel', 're', 'eke', 'pe']
    ELECTRON_QUANT += [ue + x for ue in ['ue', 'pe', 'uej', 'uep'] for x in ['x', 'y', 'z']]

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'ELECTRON_QUANT', ELECTRON_QUANT, get_electron_var.__doc__, nfluid=0)
    docvar('nel',  'electron number density [cm^-3]')
    docvar('re',   'mass density of electrons [simu. mass density units]')
    docvar('eke',  'electron kinetic energy density [simu. energy density units]')
    docvar('pe',   'electron pressure [simu. pressure units]')
    AXES = ['x', 'y', 'z']
    for x in AXES:
      docvar('ue'+x, '{}-component of electron velocity [simu. velocity units]'.format(x))
    for x in AXES:
      docvar('pe'+x, '{}-component of electron momentum density [simu. momentum density units]'.format(x))
    for x in AXES:
      docvar('uej'+x, '{}-component of current contribution to electron velocity [simu. velocity units]'.format(x))
    for x in AXES:
      docvar('uep'+x, '{}-component of species velocities contribution to electron velocity [simu. velocity units]'.format(x))
    return None

  if (var not in ELECTRON_QUANT):
    return None

  output = np.zeros(obj.r.shape)

  if var == 'nel': # number density of electrons [cm^-3]
    for fluid in fl.Fluids(dd=obj).ions():
      output += obj.get_var('nr', ifluid=fluid.SL) * fluid.ionization   #[simu. number density units]
    return output * obj.uni.u_nr    # [cm^-3]

  elif var == 're': # mass density of electrons [simu. mass density units]
    return obj.get_var('nr') * obj.uni.simu_m_e

  elif var == 'eke': #electron kinetic energy density [simu. energy density units]
    return obj.get_var('ke', mf_ispecies=-1)

  elif var == 'pe':
    return (obj.uni.gamma-1) * obj.get_var('e', mf_ispecies=-1) 

  elif var in ['uepx', 'uepy', 'uepz']: # electron velocity (contribution from momenta)
    # i = sum_j (nj uj qj) + ne qe ue
    ## --> ue = (i - sum_j(nj uj qj)) / (ne qe)
    x = var[-1] # axis; 'x', 'y', or 'z'.
    # get component due to velocities:
    ## r is in center of cells, while u is on faces, so we need to interpolate.
    ## r is at (0, 0, 0); ux is at (-0.5, 0, 0)
    ## ---> to align with ux, we shift r by xdn
    interp = x+'dn'
    nqe    = np.zeros(obj.r.shape)  # charge density of electrons.
    for fluid in fl.Fluids(dd=obj).ions():
      nq   = obj.get_var('nq' + interp, ifluid=fluid.SL)  # [simu. charge density units]
      ux   = obj.get_var('u'+x, ifluid=fluid.SL)          # [simu. velocity units]
      output -= nq * ux                                   # [simu. current per area units]
      nqe    -= nq                                        # [simu. charge density units]
    return output / nqe  # [simu velocity units]

  elif var in ['uejx', 'uejy', 'uejz']: # electron velocity (contribution from current)
    # i = sum_j (nj uj qj) + ne qe ue
    ## --> ue = (i - sum_j(nj uj qj)) / (ne qe)
    x = var[-1] # axis; 'x', 'y', or 'z'.
    # get component due to current:
    ## i is on edges of cells, while u is on faces, so we need to interpolate.
    ## ix is at (0, -0.5, -0.5); ux is at (-0.5, 0, 0)
    ## ---> to align with ux, we shift ix by xdn yup zup
    y, z    = tuple(set(('x', 'y', 'z')) - set((x)))
    interpj = x+'dn' + y+'up' + z+'up'
    jx      = obj.get_var('j'+x + interpj)   # [simu current per area units]
    ## r (nq) is in center of cells, while u is on faces, so we need to interpolate.
    ## r is at (0, 0, 0); ux is at (-0.5, 0, 0)
    ## ---> to align with ux, we shift r by xdn
    interpn = x+'dn'
    nqe = obj.get_var('nq' + interpn, iS= -1)     # [simu charge density units]
    return jx / nqe  # [simu velocity units]

  elif var in ['uex', 'uey', 'uez']: # electron velocity [simu. velocity units]
    # i = sum_j (nj uj qj) + ne qe ue
    ## --> ue = (i - sum_j(nj uj qj)) / (ne qe)
    x = var[-1] # axis; 'x', 'y', or 'z'.
    # get component due to current:
    ## i is on edges of cells, while u is on faces, so we need to interpolate.
    ## ix is at (0, -0.5, -0.5); ux is at (-0.5, 0, 0)
    ## ---> to align with ux, we shift ix by xdn yup zup
    y, z   = tuple(set(('x', 'y', 'z')) - set((x)))
    interp = x+'dn' + y+'up' + z+'up'
    output = obj.get_var('j'+x + interp)   # [simu current per area units]
    # get component due to velocities:
    ## r is in center of cells, while u is on faces, so we need to interpolate.
    ## r is at (0, 0, 0); ux is at (-0.5, 0, 0)
    ## ---> to align with ux, we shift r by xdn
    interp = x+'dn'
    nqe    = np.zeros(obj.r.shape)  # charge density of electrons.
    for fluid in fl.Fluids(dd=obj).ions():
      nq   = obj.get_var('nq' + interp, ifluid=fluid.SL)  # [simu. charge density units]
      ux   = obj.get_var('u'+x, ifluid=fluid.SL)          # [simu. velocity units]
      output -= nq * ux                                   # [simu. current per area units]
      nqe    -= nq                                        # [simu. charge density units]
    return output / nqe  # [simu velocity units]

  elif var in ['pex', 'pey', 'pez']: # electron momentum density [simu. momentum density units]
    # p = r * u.
    ## u is on faces of cells, while r is in center, so we need to interpolate.
    ## px and ux are at (-0.5, 0, 0); r is at (0, 0, 0)
    ## ---> to align with ux, we shift r by xdn
    x = var[-1] # axis; 'x', 'y', or 'z'.
    interp = x+'dn'
    re  = obj.get_var('re'+interp)  # [simu. mass density units]
    uex = obj.get_var('ue'+x)       # [simu. velocity units]
    return re * uex                 # [simu. momentum density units]


def get_spitzerterm(obj, var, SPITZERTERM_QUANT=None):
  '''spitzer conductivies'''
  if SPITZERTERM_QUANT is None:
    SPITZERTERM_QUANT = ['kappaq','dxTe','dyTe','dzTe','rhs']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'SPITZTERM_QUANT', SPITZERTERM_QUANT, get_spitzerterm.__doc__, nfluid='???')
    docvar('kappaq', 'Electron thermal diffusivity coefficient [Ebysus units], in SI: W.m-1.K-1')
    docvar('dxTe',   'Gradient of electron temperature in the x direction [simu.u_te/simu.u_l] in SI: K.m-1')
    docvar('dyTe',   'Gradient of electron temperature in the y direction [simu.u_te/simu.u_l] in SI: K.m-1')
    docvar('dzTe',   'Gradient of electron temperature in the z direction [simu.u_te/simu.u_l] in SI: K.m-1')
    docvar('rhs',    'Anisotropic gradient of electron temperature following magnetic field, i.e., bb.grad(Te), [simu.u_te/simu.u_l] in SI: K.m-1')
    return None

  if var not in SPITZERTERM_QUANT:
    return None

  if (var == 'kappaq'):
    spitzer_amp = 1.0
    kappa_e = 1.1E-25
    kappaq0 = kappa_e * spitzer_amp
    te  = obj.get_var('tg', mf_ispecies=-1) #obj.get_var('etg')
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
  '''quantities related to collision frequency.

  Note the collision frequencies here are the momentum transer collision frequencies.
  These obey the identity m_a  n_a  nu_ab  =  m_b  n_b  nu_ba.
  This identity ensures total momentum (sum over all species) does not change due to collisions.
  '''

  if COLFRE_QUANT is None:
    COLFRE_QUANT = ['nu_ij','nu_sj','rijx','rijy','rijz',      # basics: frequencies & mom exchange
                    'nu_si','nu_sn','nu_ei','nu_en',           # sum of frequencies
                    'nu_ij_el', 'nu_ij_mx', 'nu_ij_cl',        # colfreq by type
                    'nu_ij_res', 'nu_se_spitzcoul', 'nu_ij_capcoul', # alternative colfreq formulae
                    'nu_ij_to_ji', 'nu_sj_to_js',              # conversion factor nu_ij --> nu_ji
                    'c_tot_per_vol', '1dcolslope',]            # misc.

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'COLFRE_QUANT', COLFRE_QUANT, get_mf_colf.__doc__)
    mtra = 'momentum transfer collision frequency [simu. frequency units] between ifluid & jfluid. '
    for nu_ij in ['nu_ij', 'nu_sj']:
      docvar(nu_ij, mtra + 'Use species<0 for electrons.', nfluid=2)
    for x in ['x', 'y', 'z']:
      docvar('rij'+x, ('{x:}-component of momentum density exchange between ifluid and jfluid ' +\
                       '[simu. momentum density units / simu. time units]. ' +\
                       'rij{x:} = R_i^(ij) {x:} = mi ni nu_ij * (u{x:}_j - u{x:}_i)').format(x=x), nfluid=2)
    sstr = 'sum of momentum transfer collision frequencies [simu. frequency units] between {} & {}.'
    docvar('nu_si', sstr.format('ifluid', 'ion fluids (excluding ifluid)'), nfluid=1)
    docvar('nu_sn', sstr.format('ifluid', 'neutral fluids (excluding ifluid)'), nfluid=1)
    docvar('nu_ei', sstr.format('electrons', 'ion fluids'), nfluid=0)
    docvar('nu_en', sstr.format('electrons', 'neutral fluids'), nfluid=0)
    docvar('nu_ij_el', 'Elastic ' + mtra, nfluid=2)
    docvar('nu_ij_mx', 'Maxwell ' + mtra + 'NOTE: assumes maxwell molecules; result independent of temperatures. '+\
                        'presently, only properly implemented when ifluid=H or jfluid=H.', nfluid=2)
    docvar('nu_ij_cl', 'Coulomb ' + mtra, nfluid=2)
    docvar('nu_ij_res', 'resonant collisions between ifluid & jfluid. '+\
                        'presently, only properly implemented for ifluid=H+, jfluid=H.', nfluid=2)
    docvar('nu_se_spitzcoul', 'coulomb collisions between s & e-, including spitzer correction. ' +\
                              'Formula in Oppenheim et al 2020 appendix A eq 4. [simu freq]', nfluid=1)
    docvar('nu_ij_capcoul', 'coulomb collisions using Capitelli 2013 formulae. [simu freq]', nfluid=2)
    docvar('nu_ij_to_ji', 'nu_ij_to_ji * nu_ij = nu_ji.  nu_ij_to_ji = m_i * n_i / (m_j * n_j) = r_i / r_j', nfluid=2)
    docvar('nu_sj_to_js', 'nu_sj_to_js * nu_sj = nu_js.  nu_sj_to_js = m_s * n_s / (m_j * n_j) = r_s / r_j', nfluid=2)
    docvar('1dcolslope', '-(nu_ij + nu_ji)', nfluid=2)
    docvar('c_tot_per_vol', 'number density of collisions per volume per time '
                            '[simu. number density * simu. frequency] between ifluid and jfluid.', nfluid=2)
    return None

  if var not in COLFRE_QUANT:
    return None

  # collision frequency between ifluid and jfluid
  if var in ['nu_ij', 'nu_sj']:
    coll_type = obj.get_coll_type()   # gets 'EL', 'MX', 'CL', or None
    if coll_type is None:
      errmsg = ("Found no valid coll_keys for ifluid={}, jfluid={}. "
        "looked for 'CL' for coulomb collisions, or 'EL' or 'MX' for other collisions. "
        "You can enter coll_keys in the COLL_KEYS section in mf_param_file='{}'.")
      mf_param_file = obj.get_param('mf_param_file', default='mf_params.in')
      raise ValueError(errmsg.format(obj.ifluid, obj.jfluid, mf_param_file))
    else:
      if coll_type[0] == 'EE':     # electrons --> use "implied" coll type.
        coll_type = coll_type[1]   # TODO: add coll_keys to mf_eparams.in??
      nu_ij_varname = 'nu_ij_{}'.format(coll_type.lower())  # nu_ij_el, nu_ij_mx, or nu_ij_cl
      return obj.get_var(nu_ij_varname)

  # collision frequency - elastic or coulomb
  if var in ['nu_ij_el', 'nu_ij_cl']:
    iSL = obj.ifluid
    jSL = obj.jfluid
    # get ifluid info
    tgi  = obj.get_var('tg', ifluid=iSL)      # [K]
    m_i  = obj.get_mass(iSL[0])               # [amu]
    # get jfluid info, then restore original iSL & jSL
    with obj.MaintainFluids():
      n_j   = obj.get_var('nr', ifluid=jSL) * obj.uni.u_nr # [cm^-3]
      tgj   = obj.get_var('tg', ifluid=jSL)                # [K]
      m_j   = obj.get_mass(jSL[0])                         # [amu]

    # compute some values:
    m_jfrac = m_j / (m_i + m_j)                      # [(dimensionless)]
    m_ij    = m_i * m_jfrac                          # [amu]
    tgij    = (m_i * tgj + m_j * tgi) / (m_i + m_j)  # [K]
    
    # coulomb collisions:
    if var.endswith('cl'):
      icharge = obj.get_charge(iSL)   # [elementary charge == 1]
      jcharge = obj.get_charge(jSL)   # [elementary charge == 1]
      m_h = obj.uni.m_h / obj.uni.amu            # [amu]
      logcul = obj.get_var('logcul')
      scalars = 1.7 * 1/20.0 * (m_h/m_i) * (m_ij/m_h)**0.5 * icharge**2 * jcharge**2 / obj.uni.u_hz
      return scalars * logcul * n_j / tgij**1.5  # [ simu frequency units]
      
    # elastic collisions:
    elif var.endswith('el'):
      cross    = obj.get_var('cross')    # [cm^2]
      tg_speed = np.sqrt(8 * (obj.uni.kboltzmann/obj.uni.amu) * tgij / (np.pi * m_ij)) # [cm s^-1]
      return 4./3. * n_j * m_jfrac * cross * tg_speed / obj.uni.u_hz  # [simu frequency units]

  # collision frequency - maxwell
  elif var == 'nu_ij_mx':
    #set constants. for more details, see eq2 in Appendix A of Oppenheim 2020 paper.
    CONST_MULT    = 1.96     #factor in front.
    CONST_ALPHA_N = 6.67e-31 #[m^3]    #polarizability for Hydrogen   #(should be different for different species)
    e_charge= obj.uni.qsi_electron  #[C]      #elementary charge
    eps0    = 8.854187e-12   #[kg^-1 m^-3 s^4 (C^2 s^-2)] #epsilon0, standard definition
    CONST_RATIO   = (e_charge / obj.uni.amusi) * (e_charge / eps0) * CONST_ALPHA_N   #[C^2 kg^-1 [eps0]^-1 m^3]
    # units of CONST_RATIO: [C^2 kg^-1 (kg^1 m^3 s^-2 C^-2) m^-3] = [s^-2]
    #get variables.
    with obj.MaintainFluids():
      n_j = obj.get_var('nr', ifluid=obj.jfluid) * obj.uni.usi_nr   #number density [m^-3]
    m_i = obj.get_mass(obj.mf_ispecies)  #mass [amu]
    m_j = obj.get_mass(obj.mf_jspecies)  #mass [amu]
    #calculate & return nu_ij_test:
    return CONST_MULT * n_j * np.sqrt(CONST_RATIO * m_j / ( m_i * (m_i + m_j))) / obj.uni.usi_hz

  # momentum transfer terms
  elif var in ['rijx', 'rijy', 'rijz']:
    if obj.ifluid==obj.jfluid:      # when ifluid==jfluid, u_j = u_i, so rij = 0.
      return np.zeros(obj.r.shape)   # save time by returning 0 without reading any data.
    x = var[-1]  # axis; x= 'x', 'y', or 'z'.
    # rij = mi ni nu_ij * (u_j - u_i) = ri nu_ij * (u_j - u_i)
    nu_ij = obj.get_var('nu_ij')
    ri  = obj.get_var('ri')
    uix = obj.get_var('ui'+x)
    ujx = obj.get_var('ui'+x, ifluid=obj.jfluid)
    return ri * nu_ij * (ujx - uix)

  # sum of collision frequencies: sum_{i in ions} (nu_{ifluid, i})
  elif var == 'nu_si':
    ifluid = obj.ifluid
    result = np.zeros(np.shape(obj.r))
    for fluid in fl.Fluids(dd=obj).ions():
      if fluid.SL != ifluid:
        result += obj.get_var('nu_ij', jfluid=fluid.SL)
    return result

  # sum of collision frequencies: sum_{n in neutrals} (nu_{ifluid, n})
  elif var == 'nu_sn':
    ifluid = obj.ifluid
    result = np.zeros(np.shape(obj.r))
    for fluid in fl.Fluids(dd=obj).neutrals():
      if fluid.SL != ifluid:
        result += obj.get_var('nu_ij', jfluid=fluid.SL)
    return result 

  elif var == 'nu_ei':
    return obj.get_var('nu_si', mf_ispecies=-1)

  elif var == 'nu_en':
    return obj.get_var('nu_sn', mf_ispecies=-1)

  # collision frequency - resonant charge exchange for H, H+
  elif var == 'nu_ij_res':
    # formula assumes we are doing nu_{H+, H} collisions.
    ## it also happens to be valid for nu_{H, H+},
    ## because nu_ij_to_ji for H, H+ is the ratio nH / nH+.
    with obj.MaintainFluids():
      nH = obj.get_var('nr', ifluid=obj.jfluid) * obj.uni.usi_nr # [m^-3]
    tg = 0.5 * (obj.get_var('tg') + obj.get_var('tg', ifluid=obj.jfluid)) # [K]
    return 2.65e-16 * nH * np.sqrt(tg) * (1 - 0.083 * np.log10(tg))**2 / obj.uni.usi_hz

  # collision frequency - spitzer coulomb formula
  elif var == 'nu_se_spitzcoul':
    icharge = obj.get_charge(obj.ifluid)
    assert icharge > 0, "ifluid must be ion, but got charge={} (ifluid={})".format(icharge, obj.ifluid)
    #nuje = me pi ne e^4 ln(12 pi ne ldebye^3) / ( ms (4 pi eps0)^2 sqrt(ms (2 kb T)^3) )
    ldebye = obj.get_var('ldebye') * obj.uni.usi_l
    me   = obj.uni.msi_e
    tg   = obj.get_var('tg')
    ms   = obj.get_mass(obj.mf_ispecies, units='si')
    eps0 = obj.uni.permsi
    kb   = obj.uni.ksi_b
    qe   = obj.uni.qsi_electron
    ne   = obj.get_var('nr', mf_ispecies=-1) * obj.uni.usi_nr  # [m^-3]
    # combine numbers in a way that will prevent extremely large or small values:
    const = (1 / (16 * np.pi)) * (qe / eps0)**2 * (qe / kb) * (qe / np.sqrt(kb))
    mass_ = me / ms  *  1/ np.sqrt(ms)
    ln_   = np.log(12 * np.pi * ne) + 3 * np.log(ldebye)
    nuje0 = (const * ne) * mass_ * ln_ / (2 * tg)**(3/2)

    # try again but with logs. Run this code to confirm that the above code is correct.
    run_confirmation_routine = False   # change to True to run this code.
    if run_confirmation_routine:
      ln = np.log
      tmp1 = ln(me) + ln(np.pi) + ln(ne) + 4*ln(qe) + ln(ln(12) + ln(np.pi) + ln(ne) + 3*ln(ldebye))  # numerator
      tmp2 = ln(ms) + 2*(ln(4) + ln(np.pi) + ln(eps0)) + 0.5*(ln(ms) + 3*(ln(2) + ln(kb) + ln(tg)))  # denominator
      tmp = tmp1 - tmp2
      nuje1 = np.exp(tmp)
      print('we expect these to be approximately equal:', nuje0.mean(), nuje1.mean())
    return nuje0 / obj.uni.usi_hz

  # collision frequency - capitelli coulomb formula
  elif var == 'nu_ij_capcoul':
    iSL = obj.ifluid
    jSL = obj.jfluid
    icharge = obj.get_charge(iSL, units='si') #[C]
    jcharge = obj.get_charge(jSL, units='si') #[C]
    assert icharge != 0 and jcharge != 0, 'we require i & j both charged' +\
      ' but got icharge={}, jcharge={}'.format(icharge, jcharge)

    # get ifluid info
    tgi  = obj.get_var('tg', ifluid=iSL)      # [K]
    m_i  = obj.get_mass(iSL[0])  # [amu]
    # get jfluid info, then restore original iSL & jSL
    with obj.MaintainFluids():
      n_j   = obj.get_var('nr', ifluid=jSL) * obj.uni.usi_nr # [m^-3]
      tgj   = obj.get_var('tg', ifluid=jSL)                # [K]
      m_j   = obj.get_mass(jSL[0])            # [amu]

    # compute some values:
    m_jfrac = m_j / (m_i + m_j)                      # [(dimensionless)]
    m_ij    = m_i * m_jfrac                          # [amu]   # e.g. for H, H+, m_ij = 0.5.
    tgij    = (m_i * tgj + m_j * tgi) / (m_i + m_j)  # [K]

    tg_speed = np.sqrt(8 * (obj.uni.ksi_b/obj.uni.amusi) * tgij / (np.pi * m_ij)) # [m s^-1]
    E_alpha  = 0.5 * (m_ij * obj.uni.amusi) * tg_speed**2
      
    euler_constant = 0.577215
    b_0      = abs(icharge*jcharge)/(4 * np.pi * obj.uni.permsi * E_alpha)  # [m]  # permsi == epsilon_0
    #b_0      = abs(icharge*jcharge)/(2 * obj.uni.ksi_b*obj.uni.permsi * tgij)  # [m]   # permsi == epsilon_0
    cross    = np.pi*2.0*(b_0**2)*(np.log(2.0*obj.get_var('ldebye')*obj.uni.usi_l/b_0)-0.5-2.0*euler_constant) # [m2]

    #calculate & return nu_ij:
    nu_ij = 4./3. * n_j * m_jfrac * cross * tg_speed / obj.uni.u_hz  # [simu frequency units]
    return nu_ij

  # collision frequency conversion factor: nu_ij to nu_ji
  elif var in ['nu_ij_to_ji', 'nu_sj_to_js']:
    mi_ni = obj.get_var('ri', ifluid=obj.ifluid)  # mi * ni = ri
    mj_nj = obj.get_var('ri', ifluid=obj.jfluid)  # mj * nj = rj
    return mi_ni / mj_nj

  elif var == "c_tot_per_vol":
    m_i = obj.get_mass(obj.mf_ispecies)   # [amu]
    m_j = obj.get_mass(obj.mf_jspecies)   # [amu]
    return obj.get_var("nr", ifluid=obj.jfluid) * obj.get_var("nu_ij") / (m_j / (m_i + m_j))

  elif var == "1dcolslope":
    warnings.warn(DeprecationWarning('1dcolslope will be removed at some point in the future.'))
    return -1 * obj.get_var("nu_ij") * (1 + obj.get_var('nu_ij_to_ji'))


def get_mf_logcul(obj, var, LOGCUL_QUANT=None):
  '''coulomb logarithm'''
  if LOGCUL_QUANT is None:
    LOGCUL_QUANT = ['logcul']  

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'LOGCUL_QUANT', LOGCUL_QUANT, get_mf_logcul.__doc__)
    docvar('logcul', 'Coulomb Logarithmic used for Coulomb collisions.', nfluid=0)
    return None

  if var not in LOGCUL_QUANT:
    return None
  
  if var == "logcul":
    etg = obj.get_var('tg', mf_ispecies=-1)
    nel = obj.get_var('nel')
    return 23. + 1.5 * np.log(etg / 1.e6) - \
          0.5 * np.log(nel / 1e6)


def get_mf_driftvar(obj, var, DRIFT_QUANT=None):
  '''var drift between fluids. I.e. var_ifluid - var_jfluid'''
  if DRIFT_QUANT is None:
    DRIFT_QUANT = ['ud', 'pd', 'ed', 'rd', 'tgd', 'uid']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'DRIFT_QUANT', DRIFT_QUANT, get_mf_driftvar.__doc__, nfluid=2)
    def doc_start(var):
      return '"drift" for quantity ("{var}"). I.e. ({va_} for ifluid) - ({va_} for jfluid). '.format(var=var, va_=var[:-1])
    def doc_axis(var):
      return ' Must append x, y, or z; e.g. {var}x for (ifluid {va_}x) - (jfluid {va_}x).'.format(var=var, va_=var[:-1])
    docvar('ud', doc_start(var='ud') + 'u = velocity [simu. units].' + doc_axis(var='ud'))
    docvar('uid', doc_start(var='uid') + 'ui = velocity [simu. units].' + doc_axis(var='uid'))
    docvar('pd', doc_start(var='pd') + 'p = momentum density [simu. units].' + doc_axis(var='pd'))
    docvar('ed', doc_start(var='ed') + 'e = energy (density??) [simu. units].')
    docvar('rd', doc_start(var='rd') + 'r = mass density [simu. units].')
    docvar('tgd', doc_start(var='tgd') + 'tg = temperature [K].')
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
  '''cross section between species.'''
  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = ['cross']

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'CROSTAB_QUANT', CROSTAB_QUANT, get_mf_cross.__doc__, nfluid=2)
    docvar('cross', 'cross section between ifluid and jfluid [cgs]. Use species < 0 for electrons.')
    return None

  if var not in CROSTAB_QUANT:
    return None

  # get masses & temperatures, then restore original obj.ifluid and obj.jfluid values.
  with obj.MaintainFluids():
    m_i = obj.get_mass(obj.mf_ispecies)
    m_j = obj.get_mass(obj.mf_jspecies)
    tgi = obj.get_var('tg', ifluid = obj.ifluid)
    tgj = obj.get_var('tg', ifluid = obj.jfluid)

  # temperature, weighted by mass of species
  tg = (tgi*m_j + tgj*m_i)/(m_i + m_j)

  # look up cross table and get cross section
  #crossunits = 2.8e-17  
  crossobj = obj.get_cross_sect(ifluid=obj.ifluid, jfluid=obj.jfluid)
  crossunits = crossobj.cross_tab[0]['crossunits']
  cross = crossunits * crossobj.tab_interp(tg)

  return cross


def get_cfl_quant(obj, quant, CFL_QUANT=None):
  '''CFL quantities. All are in simu. frequency units.'''
  if CFL_QUANT is None:
    CFL_QUANTS = ['ohm']
    CFL_QUANT = ['cfl_' + q for q in CFL_QUANTS]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'CFL_QUANT', CFL_QUANT, get_cfl_quant.__doc__)
    docvar('cfl_ohm', 'cfl condition for ohmic module. (me / ms) ((qs / qe) + (ne / ns)) nu_es', nfluid=1)
    return None

  _, cfl_, quant = quant.partition('cfl_')
  if quant=='':
    return None

  elif quant=='ohm':
    fluid = obj.ifluid
    nrat  = obj.get_var('nr', iS=-1) / obj.get_var('nr', ifluid=fluid)   # ne / ns
    mrat  = obj.uni.msi_electron / obj.get_mass(fluid, units='si')       # me / ms
    qrat  = obj.get_charge(fluid) / -1                                  # qs / qe
    nu_es = obj.get_var('nu_ij', iS=-1, jfluid=fluid)                   # nu_es
    return mrat * (qrat + nrat) * nu_es


def get_mf_plasmaparam(obj, quant, PLASMA_QUANT=None):
  '''plasma parameters, e.g. plasma beta, sound speed, pressure scale height'''
  if PLASMA_QUANT is None:
    PLASMA_QUANT = ['beta', 'beta_ions', 'va', 'va_ions', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky', 'kz',
                'sgyrof', 'gyrof', 'skappa', 'kappa', 'ldebye', 'ldebyei']
  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'PLASMA_QUANT', PLASMA_QUANT, get_mf_plasmaparam.__doc__)
    docvar('beta', "plasma beta", nfluid='???') #nfluid= 1 if mfe_p is pressure for ifluid; 0 if it is sum of pressures.
    docvar('beta_ions', "plasma beta using sum of ion pressures. P / (B^2 / (2 mu0)).", nfluid=0)
    docvar('va', "alfven speed [simu. units]", nfluid=0)
    docvar('va_ions', "alfven speed [simu. units], using density := density of ions.", nfluid=0)
    docvar('cs', "sound speed [simu. units]", nfluid='???')
    docvar('s', "entropy [log of quantities in simu. units]", nfluid='???')
    docvar('mn', "mach number (using sound speed)", nfluid=1)
    docvar('man', "mach number (using alfven speed)", nfluid=1)
    docvar('hp', "Pressure scale height", nfluid='???')
    for vax in ['vax', 'vay', 'vaz']:
      docvar(vax, "{axis} component of alfven velocity [simu. units]".format(axis=vax[-1]), nfluid=0)
    for kx in ['kx', 'ky', 'kz']:
      docvar(kx, ("{axis} component of kinetic energy density of ifluid [simu. units]."+\
                  "(0.5 * rho * (get_var(u{axis})**2)").format(axis=kx[-1]), nfluid=1)
    docvar('sgyrof', "signed gryofrequency for ifluid. I.e. qi * |B| / mi. [1 / (simu. time units)]", nfluid=1)
    docvar('gyrof', "gryofrequency for ifluid. I.e. abs(qi * |B| / mi). [1 / (simu. time units)]", nfluid=1)
    kappanote = ' "Highly magnetized" when kappa^2 >> 1.'
    docvar('skappa', "signed magnetization for ifluid. I.e. sgryof/nu_sn." + kappanote, nfluid=1)
    docvar('kappa', "magnetization for ifluid. I.e. gyrof/nu_sn." + kappanote, nfluid=1)
    docvar('ldebyei', "debye length of ifluid [simu. length units]. sqrt(kB eps0 q^-2 Ti / ni)", nfluid=1)
    docvar('ldebye', "debye length of plasma [simu. length units]. " +\
                     "sqrt(kB eps0 e^-2 / (ne/Te + sum_j(Zj^2 * nj / Tj)) ); Zj = qj/e"+\
                     "1/sum_j( (1/ldebye_j) for j in fluids and electrons)", nfluid=0)
    return None

  if quant not in PLASMA_QUANT:
    return None

  if quant in ['hp', 's', 'cs', 'beta']:
    var = obj.get_var('mfe_p')  # is mfe_p pressure for ifluid, or sum of all fluid pressures? - SE Apr 19 2021
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

  if quant == 'beta_ions':
    p = np.zeros(obj.r.shape)
    for fluid in fl.Fluids(dd=obj).ions():
      p += obj.get_var('p', ifluid=fluid)
    bp = obj.get_var('b2') / 2    # (dd.uni.usi_b**2 / dd.uni.mu0si) == 1 by def'n of b in ebysus.
    return p / bp

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

  if quant in ['va_ions']:
    r = obj.get_var('rions')
    return obj.get_var('modb') / np.sqrt(r)

  if quant in ['hx', 'hy', 'hz', 'kx', 'ky', 'kz']:
    axis = quant[-1]
    var = obj.get_var('p' + axis + 'c')
    if quant[0] == 'h':
      # anyone can delete this warning once you have confirmed that get_var('hx') does what you think it should:
      warnmsg = ('get_var(hx) (or hy or hz) uses get_var(p), and used it since before get_var(p) was implemented. '
                 'Maybe should be using get_var(mfe_p) instead? '
                 'You should not trust results until you check this.  - SE Apr 19 2021.')
      warnings.warn(warnmsg)
      return ((obj.get_var('e') + obj.get_var('p')) /
              obj.get_var('r') * var)
    else:
      return obj.get_var('u2') * var * 0.5

  if quant == 'sgyrof':
    B = obj.get_var('modb')                       # magnitude of B [simu. B-field units]
    q = obj.get_charge(obj.ifluid, units='simu')     #[simu. charge units]
    m = obj.get_mass(obj.mf_ispecies, units='simu')  #[simu. mass units]
    return q * B / m

  if quant == 'gyrof':
    return np.abs(obj.get_var('sgyrof'))

  if quant == 'skappa':
    gyrof = obj.get_var('sgyrof') #[simu. freq.]
    nu_sn = obj.get_var('nu_sn')  #[simu. freq.]
    return gyrof / nu_sn 

  if quant == 'kappa':
    return np.abs(obj.get_var('skappa'))

  elif quant == 'ldebyei':
    Zi2 = obj.get_charge(obj.ifluid)**2
    if Zi2 == 0:
      return np.zeros(obj.r.shape)
    const = obj.uni.permsi * obj.uni.ksi_b / obj.uni.qsi_electron**2
    tg = obj.get_var('tg')                     # [K]
    nr = obj.get_var('nr') * obj.uni.usi_nr    # [m^-3]
    ldebsi = np.sqrt(const * tg / (nr * Zi2))  # [m]
    return ldebsi / obj.uni.usi_l  # [simu. length units]

  elif quant == 'ldebye':
    # ldebye = 1/sum_j( (1/ldebye_j) for j in fluids and electrons)
    ldeb_inv_sum = 1/obj.get_var('ldebyei', mf_ispecies=-1)
    for fluid in fl.Fluids(dd=obj).ions():
      ldeb_inv_sum += 1/obj.get_var('ldebyei', ifluid=fluid.SL)
    return 1/ldeb_inv_sum


def get_mf_wavequant(obj, quant, WAVE_QUANT=None):
  '''quantities related most directly to waves in plasmas.'''
  if WAVE_QUANT is None:
    WAVE_QUANT = ['ci', 'kmaxx', 'kmaxy', 'kmaxz']

  if quant == '':
    docvar = document_vars.vars_documenter(obj, 'WAVE_QUANT', WAVE_QUANT, get_mf_wavequant.__doc__)
    docvar('ci', "ion acoustic speed for ifluid (must be ionized) [simu. velocity units]", nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('kmax'+x, "maximum resolvable wavevector in "+x+" direction. Determined via 2*pi/obj.d"+x+"1d", nfluid=0)
    return None

  if quant == 'ci':
    assert obj.mf_ispecies != -1, "ifluid {} must be ion to get ci, but got electron.".format(obj.ifluid)
    fluids = fl.Fluids(dd=obj)
    ion = fluids[obj.ifluid]
    assert ion.ionization >= 1, "ifluid {} is not ionized; cannot get ci (==ion acoustic speed).".format(obj.ifluid)
    # (we only want to get ion acoustic speed for ions; it doesn't make sense to get it for neutrals.)
    itg = obj.get_var('tg')                  # [K] temperature of fluid
    etg = obj.get_var('tg', mf_ispecies=-1)  # [K] temperature of electrons
    igamma = obj.uni.gamma        # gamma (ratio of specific heats) of fluid
    egamma = obj.uni.gamma        # gamma (ratio of specific heats) of electrons
    ci2_p = ((ion.ionization * igamma * itg + egamma * etg) / ion.atomic_weight) # ci**2 if kB=amu=1
    ci_cgs = np.sqrt((obj.uni.k_b / obj.uni.amu) * ci2_p ) # ci [cm/s]
    ci_sim = ci_cgs / obj.uni.u_u                          # ci [simu. velocity units]
    return ci_sim

  elif quant in ['kmaxx', 'kmaxy', 'kmaxz']:
    x = quant[-1] # axis; 'x', 'y', or 'z'.
    xidx = dict(x=0, y=1, z=2)[x]  # axis; 0, 1, or 2.
    dx1d = getattr(obj, 'd'+x+'1d')  # 1D; needs dims to be added. add dims below.
    dx1d = np.expand_dims(dx1d, axis=tuple(set((0,1,2)) - set([xidx])))
    return (2 * np.pi / dx1d) + np.zeros_like(obj.r)


def get_fb_instab_quant(obj, quant, FB_INSTAB_QUANT=None):
  '''very specific quantities which are related to the Farley-Buneman instability.'''
  if FB_INSTAB_QUANT is None:
    FB_INSTAB_QUANT = ['psi0', 'psii', 'vde', 'fb_ssi_vdtrigger', 'fb_ssi_possible',
                       'fb_ssi_freq', 'fb_ssi_growth_rate']
    vecs = ['fb_ssi_freq_max', 'fb_ssi_growth_rate_max', 'fb_ssi_growth_time_min']
    FB_INSTAB_QUANT += [v+x for v in vecs for x in ['x', 'y', 'z']]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'FB_INSTAB_QUANT', FB_INSTAB_QUANT, get_fb_instab_quant.__doc__)
    for psi in ['psi0', 'psii']:
      docvar(psi, 'psi_i when k_parallel==0. equals to: (kappa_e * kappa_i)^-1.', nfluid=1)
    docvar('vde', 'electron drift velocity. equals to: |E|/|B|. [simu. velocity units]', nfluid=0)
    docvar('fb_ssi_vdtrigger', 'minimum vde [in simu. velocity units] above which the FB instability can grow, ' +\
             'in the case of SSI (single-species-ion). We assume ifluid is the single ion species.', nfluid=1)
    docvar('fb_ssi_possible', 'whether SSI Farley Buneman instability can occur (vde > fb_ssi_vdtrigger). ' +\
             'returns an array of booleans, with "True" meaning "can occur at this point".', nfluid=1)
    docvar('fb_ssi_freq', 'SSI FB instability wave frequency (real part) divided by wavenumber (k). ' +\
             'assumes wavevector in E x B direction. == Vd / (1 + psi0). ' +\
             'result is in units of [simu. frequency * simu. length].', nfluid=2)
    docvar('fb_ssi_growth_rate', 'SSI FB instability growth rate divided by wavenumber (k) squared. ' +\
             'assumes wavevector in E x B direction. == (Vd^2/(1+psi0)^2 - Ci^2)/(nu_in*(1+1/psi0)). ' +\
             'result is in units of [simu. frequency * simu. length].', nfluid=2)
    for x in ['x', 'y', 'z']:
      docvar('fb_ssi_freq_max'+x, 'SSI FB instability max frequency in '+x+' direction ' +\
               '[simu. frequency units]. calculated using fb_ssi_freq * kmax'+x, nfluid=2)
    for x in ['x', 'y', 'z']:
      docvar('fb_ssi_growth_rate_max'+x, 'SSI FB instability max growth rate in '+x+' direction ' +\
               '[simu. frequency units]. calculated using fb_ssi_growth_rate * kmax'+x, nfluid=2)
    for x in ['x', 'y', 'z']:
      docvar('fb_ssi_growth_time_min'+x, 'SSI FB instability min growth time in '+x+' direction ' +\
               '[simu. time units]. This is the amount of time it takes for the wave amplitude for the wave ' +\
               'with the largest wave vector to grow by a factor of e. == 1/fb_ssi_growth_rate_max'+x, nfluid=2)

    return None

  if quant not in FB_INSTAB_QUANT:
    return None

  elif quant in ['psi0', 'psii']:
    kappa_i = obj.get_var('kappa')
    kappa_e = obj.get_var('kappa', mf_ispecies=-1)
    return 1./(kappa_i * kappa_e)

  elif quant == 'vde':
    modE = obj.get_var('modef') # [simu. E-field units]
    modB = obj.get_var('modb') # [simu. B-field units]
    return modE / modB         # [simu. velocity units]

  elif quant == 'fb_ssi_vdtrigger':
    icharge = obj.get_charge(obj.ifluid)
    assert icharge > 0, "expected ifluid to be an ion but got ifluid charge == {}".format(icharge)
    ci   = obj.get_var('ci')   # [simu. velocity units]
    psi0 = obj.get_var('psi0')
    return ci * (1 + psi0)     # [simu. velocity units]

  elif quant == 'fb_ssi_possible':
    return obj.get_var('vde') > obj.get_var('fb_ssi_vdtrigger')

  elif quant == 'fb_ssi_freq':
    icharge = obj.get_charge(obj.ifluid)
    assert icharge > 0, "expected ifluid to be an ion but got ifluid charge == {}".format(icharge)
    jcharge = obj.get_charge(obj.jfluid)
    assert jcharge == 0, "expected jfluid to be neutral but got jfluid charge == {}".format(jcharge)
    # freq (=real part of omega) = Vd * k_x / (1 + psi0)
    Vd    = obj.get_var('vde')
    psi0  = obj.get_var('psi0')
    return Vd / (1 + psi0)

  elif quant == 'fb_ssi_growth_rate':
    # growth rate = ((omega_r/k_x)^2 - Ci^2) * (k_x)^2/(nu_in*(1+1/psi0))
    w_r_k = obj.get_var('fb_ssi_freq')  # omega_r / k_x
    psi0  = obj.get_var('psi0')
    Ci    = obj.get_var('ci')
    nu_in = obj.get_var('nu_ij')
    return (w_r_k**2 - Ci**2) / (nu_in * (1 + 1/psi0))

  elif quant in ['fb_ssi_freq_max'+x for x in ['x', 'y', 'z']]:
    x = quant[-1]
    freq        = obj.get_var('fb_ssi_freq')
    kmaxx       = obj.get_var('kmax'+x)
    return kmaxx**2 * freq

  elif quant in ['fb_ssi_growth_rate_max'+x for x in ['x', 'y', 'z']]:
    x = quant[-1]
    growth_rate = obj.get_var('fb_ssi_growth_rate')
    kmaxx       = obj.get_var('kmax'+x)
    return kmaxx**2 * growth_rate

  elif quant in ['fb_ssi_growth_time_min'+x for x in ['x', 'y', 'z']]:
    x = quant[-1]
    return 1/obj.get_var('fb_ssi_growth_rate_max'+x)