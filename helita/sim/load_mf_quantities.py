# import builtins
import warnings

# import internal modules
from . import document_vars
from .file_memory import Caching   # never alters results, but caches them for better efficiency.
                                   # use sparingly on "short" calculations; apply liberally to "long" calculations.
                                   # see also cache_with_nfluid and cache kwargs of get_var.

# import external public modules
import numpy as np

# import external private modules
try:
  from at_tools import fluids as fl
except ImportError:
  warnings.warn('failed to import at_tools.fluids; some functions in helita.sim.load_mf_quantities may crash')

# set constants
MATCH_PHYSICS = 0  # don't change this value.  # this one is the default (see ebysus.py)
MATCH_AUX     = 1  # don't change this value.


def load_mf_quantities(obj, quant, *args, GLOBAL_QUANT=None,
                       ONEFLUID_QUANT=None, ELECTRON_QUANT=None, MOMENTUM_QUANT=None,
                       HEATING_QUANT=None, SPITZERTERM_QUANT=None,
                       COLFRE_QUANT=None, LOGCUL_QUANT=None, CROSTAB_QUANT=None, 
                       DRIFT_QUANT=None, CFL_QUANT=None, PLASMA_QUANT=None,
                       WAVE_QUANT=None, FB_INSTAB_QUANT=None, THERMAL_INSTAB_QUANT=None,
                       **kwargs):
  __tracebackhide__ = True  # hide this func from error traceback stack.

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
    val = get_onefluid_var(obj, quant, ONEFLUID_QUANT=ONEFLUID_QUANT)
  if val is None:
    val = get_electron_var(obj, quant, ELECTRON_QUANT=ELECTRON_QUANT)
  if val is None:
    val = get_momentum_quant(obj, quant, MOMENTUM_QUANT=MOMENTUM_QUANT)
  if val is None:
    val = get_heating_quant(obj, quant, HEATING_QUANT=HEATING_QUANT)
  if val is None:
    val = get_spitzerterm(obj, quant, SPITZERTERM_QUANT=SPITZERTERM_QUANT) 
  if val is None:
    val = get_mf_colf(obj, quant, COLFRE_QUANT=COLFRE_QUANT)
  if val is None:
    val = get_mf_logcul(obj, quant, LOGCUL_QUANT=LOGCUL_QUANT)
  if val is None:
    val = get_mf_cross(obj, quant, CROSTAB_QUANT=CROSTAB_QUANT)
  if val is None:
    val = get_mf_driftvar(obj, quant, DRIFT_QUANT=DRIFT_QUANT)
  if val is None:
    val = get_cfl_quant(obj, quant, CFL_QUANT=CFL_QUANT)
  if val is None: 
    val = get_mf_plasmaparam(obj, quant, PLASMA_QUANT=PLASMA_QUANT)
  if val is None:
    val = get_mf_wavequant(obj, quant, WAVE_QUANT=WAVE_QUANT)
  if val is None:
    val = get_fb_instab_quant(obj, quant, FB_INSTAB_QUANT=FB_INSTAB_QUANT)
  if val is None:
    val = get_thermal_instab_quant(obj, quant, THERMAL_INSTAB_QUANT=THERMAL_INSTAB_QUANT)
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

  output = obj.zero()    
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
    with Caching(obj, nfluid=0) as cache:
      output  = obj.get_var('tot_e')
      output += obj.get_var('tot_ke')
      output += obj.get_var('e_ef')
      output += obj.get_var('e_b')
      cache(var, output)
  
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
    with Caching(obj, nfluid=0) as cache:
      # E = - ue x B + (ne |qe|)^-1 * ( grad(pressure_e) + (ion & rec terms) + sum_j(R_e^(ej)) )
      # ----- calculate the necessary component of -ue x B (== B x ue) ----- #
      # There is a flag, "do_hall", when "false", we don't let the contribution
      ## from current to ue to enter in to the B x ue for electric field.
      if obj.match_aux() and obj.get_param('do_hall', default="false")=="false":
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
      if obj.params['do_ohm_ecol'][obj.snapInd]:
        # efx is at (0, -1/2, -1/2)
        ## rijx is at (-1/2, 0, 0)    (same as ux)
        ## --> to align with efx, we shift rijx by xup ydn zdn
        interp = x+'up'+y+'dn'+z+'dn'
        sum_rejx = obj.get_var('rijsum'+x + interp, iS=-1)
        ## sum_rejx has units [simu. momentum density units / simu. time units]
      else:
        sum_rejx = obj.zero()
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
      cache(var, output)
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
      return obj.get_var('nre')
    else:                   # not electrons
      mass = obj.get_mass(obj.mf_ispecies, units='simu') # [simu. mass units]
      return obj.get_var('r') / mass   # [simu number density units]

  elif var == 'nq':
    charge = obj.get_charge(obj.ifluid, units='simu') # [simu. charge units]
    if charge == 0:
      return obj.zero()
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


def get_electron_var(obj, var, ELECTRON_QUANT=None):
  '''variables related to electrons (requires looping over ions to calculate).'''

  if ELECTRON_QUANT is None:
    ELECTRON_QUANT = ['nel', 'nre', 're', 'eke', 'pe']
    ELECTRON_QUANT += [ue + x for ue in ['ue', 'pe', 'uej', 'uep'] for x in ['x', 'y', 'z']]

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'ELECTRON_QUANT', ELECTRON_QUANT, get_electron_var.__doc__, nfluid=0)
    docvar('nel',  'electron number density [cm^-3]')
    docvar('nre',  'electron number density [simu. number density units]')
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

  output = obj.zero()

  if var == 'nel': # number density of electrons [cm^-3]
    return obj.get_var('nre') * obj.uni.u_nr   # [cm^-3]

  elif var == 'nre': # number density of electrons [simu. units]
    with Caching(obj, nfluid=0) as cache:
      for fluid in fl.Fluids(dd=obj).ions():
        output += obj.get_var('nr', ifluid=fluid.SL) * fluid.ionization   #[simu. number density units]
      cache(var, output)
      return output

  elif var == 're': # mass density of electrons [simu. mass density units]
    return obj.get_var('nr', mf_ispecies=-1) * obj.uni.simu_m_e

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
    nqe    = obj.zero()  # charge density of electrons.
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
    with Caching(obj, nfluid=0) as cache:
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
      nqe    = obj.zero()  # charge density of electrons.
      for fluid in fl.Fluids(dd=obj).ions():
        nq   = obj.get_var('nq' + interp, ifluid=fluid.SL)  # [simu. charge density units]
        ux   = obj.get_var('u'+x, ifluid=fluid.SL)          # [simu. velocity units]
        output -= nq * ux                                   # [simu. current per area units]
        nqe    -= nq                                        # [simu. charge density units]
      output = output / nqe
      cache(var, output)
      return output  # [simu velocity units]

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


def get_momentum_quant(obj, var, MOMENTUM_QUANT=None):
  '''terms related to momentum equations of fluids.
  The units for these quantities are [simu. momentum density units / simu. time units].
  '''
  if MOMENTUM_QUANT is None:
    MOMENTUM_QUANT = []
    MQVECS = ['rij', 'rijsum', 'momflorentz', 'gradp', 'momrate']
    MOMENTUM_QUANT += [v + x for v in MQVECS for x in ['x', 'y', 'z']]

  if var == '':
    docvar = document_vars.vars_documenter(obj, 'MOMENTUM_QUANT', MOMENTUM_QUANT, get_momentum_quant.__doc__)
    for x in ['x', 'y', 'z']:
      docvar('rij'+x, ('{x:}-component of momentum density exchange between ifluid and jfluid ' +\
                       '[simu. momentum density units / simu. time units]. ' +\
                       'rij{x:} = R_i^(ij) {x:} = mi ni nu_ij * (u{x:}_j - u{x:}_i)').format(x=x), nfluid=2)
    for x in ['x', 'y', 'z']:
      docvar('rijsum'+x, x+'-component of momentum density change of ifluid ' +\
                           'due to collisions with all other fluids. = sum_j rij'+x, nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('momflorentz'+x, x+'-component of momentum density change of ifluid due to Lorentz force.' +\
                           '[simu. momentum density units / simu. time units]. = ni qi (E + ui x B).', nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('gradp'+x, x+'-component of grad(Pi), face-centered (interp. loc. aligns with momentum).', nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('momrate'+x, x+'-component of rate of change of momentum. ' +\
                          '= (-gradp + momflorentz + rijsum)_'+x, nfluid=1)
    return None

  if var not in MOMENTUM_QUANT:
    return None

  if var in ['rijx', 'rijy', 'rijz']:
    if obj.i_j_same_fluid():      # when ifluid==jfluid, u_j = u_i, so rij = 0.
      return obj.zero()           # save time by returning 0 without reading any data.
    x = var[-1]  # axis; x= 'x', 'y', or 'z'.
    # rij = mi ni nu_ij * (u_j - u_i) = ri nu_ij * (u_j - u_i)
    nu_ij = obj.get_var('nu_ij')
    ri  = obj.get_var('ri')
    uix = obj.get_var('ui'+x)
    ujx = obj.get_var('ui'+x, ifluid=obj.jfluid)
    return ri * nu_ij * (ujx - uix)

  elif var in ['rijsumx', 'rijsumy', 'rijsumz']:
    x = var[-1]
    result = obj.get_var('rij'+x, jS=-1)            # rijx for j=electrons
    for fluid in fl.Fluids(dd=obj):
      result += obj.get_var('rij'+x, jfluid=fluid)  # rijx for j=fluid
    return result

  elif var in ['momflorentz'+x for x in ['x', 'y', 'z']]:
    # momflorentz = ni qi (E + ui x B)
    qi = obj.get_charge(obj.ifluid, units='simu')
    if qi == 0:
      return obj.zero()    # no lorentz force for neutrals - save time by just returning 0 here :)
    ni = obj.get_var('nr')
    # make sure we get the interpolation correct:
    ## B and ui are face-centered vectors, and we want a face-centered result to align with p.
    ## Thus we use ui_facecrosstoface_b (which gives a face-centered result).
    ## Meanwhile, E is edge-centered, so we must shift all three coords.
    ## Ex is at (0, -0.5, -0.5), so we shift by xdn, yup, zup
    x = var[-1] # axis; x= 'x', 'y', or 'z'.
    y, z = dict(x=('y', 'z'), y=('z', 'x'), z=('x', 'y'))[x]
    Ex = obj.get_var('efx' + x+'dn' + y+'up' + z+'up', cache_with_nfluid=0)
    uxB__x = obj.get_var('ui_facecrosstoface_b'+x)
    return ni * qi * (Ex + uxB__x)

  elif var in ['gradpx', 'gradpy', 'gradpz']:
    x = var[-1]
    # px is at (-0.5, 0, 0); pressure is at (0, 0, 0), so we do dpdxdn
    return obj.get_var('dpd'+x+'dn')

  elif var in ['momratex', 'momratey', 'momratez']:
    x = var[-1]
    if obj.get_param('do_recion', default=False):
      warnings.warn('momentum contribution from ionization & recombination have not yet been added.')
    gradpx    = obj.get_var('gradp'+x)
    florentzx = obj.get_var('momflorentz'+x)
    rijsumx   = obj.get_var('rijsum'+x)
    return florentzx - gradpx + rijsumx


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
  _TGQCOL_EQUIL = ['tgqcol_equil' + x for x in ('_uj', '_tgj', '_j', '_u', '_tg', '')]
  if HEATING_QUANT is None:
    HEATING_QUANT = ['qcol_uj', 'qcol_tgj', 'qcol_coeffj', 'qcolj',
                     'qcol_u', 'qcol_tg', 'qcol',
                     'qjoulei']
    HEATING_QUANT += _TGQCOL_EQUIL

  if var=='':
    docvar = document_vars.vars_documenter(obj, 'HEATING_QUANT', HEATING_QUANT, get_heating_quant.__doc__)
    units = '[simu. energy density per time]'
    heati = 'heating of ifluid '+units
    docvar('qcol_uj',  heati + ' due to jfluid, due to collisions and velocity drifts.', nfluid=2)
    docvar('qcol_tgj', heati + ' due to jfluid, due to collisions and temperature differences.', nfluid=2)
    docvar('qcol_coeffj', 'coefficient common to qcol_uj and qcol_tj terms.' +\
                          ' == (mi / (gamma - 1) (mi + mj)) * ni * nu_ij. [simu units: length^-3 time^-1]', nfluid=2)
    docvar('qcolj',    'total '+heati+' due to jfluid.', nfluid=2)
    docvar('qcol_u',   heati + ' due to collisions and velocity drifts.', nfluid=1)
    docvar('qcol_tg',  heati + ' due to collisions and temperature differences.', nfluid=1)
    docvar('qcol',     'total '+heati+'.', nfluid=1)
    # "simple equilibrium" vars
    equili = '"simple equilibrium" temperature [K] of ifluid (setting sum_j Qcol_ij=0 and solving for Ti)'
    ## note: these all involve setting sum_j Qcol_ij = 0 and solving for Ti.
    ## Let Cij = qcol_coeffj; Uij = qcol_uj, Tj = temperature of j. Then:
    ### Ti == ( sum_{s!=i}(Cis Uis + Cis * 2 kB Ts) ) / ( 2 kB sum_{s!=i}(Cis) )
    ## so for the "components" terms, we pick out only one term in this sum (in the numerator), e.g.:
    ### tgqcol_equil_uj == Cij Uij / ( 2 kB sum_{s!=i}(Cis) )
    docvar('tgqcol_equil_uj', equili + ', due only to contribution from velocity drift with jfluid.', nfluid=2)
    docvar('tgqcol_equil_tgj', equili + ', due only to contribution from temperature of jfluid.', nfluid=2)
    docvar('tgqcol_equil_j', equili + ', due only to contribution from jfluid.', nfluid=2)
    docvar('tgqcol_equil_u', equili + ', due only to contribution from velocity drifts with fluids.', nfluid=1)
    docvar('tgqcol_equil_tg', equili + ', due only to contribution from temperature of fluids.', nfluid=1)
    docvar('tgqcol_equil', equili + '.', nfluid=1)
    # "ohmic heating" (obsolete (?) - nonphysical to include this qjoule and the qcol_u term as it appears here.)
    docvar('qjoulei',  heati + ' due to Ji dot E. (Ji = qi ni ui).', nfluid=1)
    return None

  if var not in HEATING_QUANT:
    return None

  def heating_is_off():
    '''returns whether we should treat heating as if it is turned off.'''
    if obj.match_physics():
      return False
    if obj.mf_ispecies < 0 or obj.mf_jspecies < 0:  # electrons
      return (obj.get_param('do_ohm_ecol', True) and obj.get_param('do_qohm', True))
    else: # not electrons
      return (obj.get_param('do_col', True) and obj.get_param('do_qcol', True))

  # qcol terms
  
  if var == 'qcol_coeffj':
    if heating_is_off() or obj.i_j_same_fluid():
      return obj.zero()
    ni = obj.get_var('nr')             # [simu. units]
    mi = obj.get_mass(obj.mf_ispecies) # [amu]
    mj = obj.get_mass(obj.mf_jspecies) # [amu]
    nu_ij = obj.get_var('nu_ij')       # [simu. units]
    coeff = (1 / obj.uni.gamma - 1) * (mi / (mi + mj)) * ni * nu_ij   # [simu units: length^-3 time^-1]
    return coeff

  if var in ['qcol_uj', 'qcol_tgj']:
    if heating_is_off() or obj.i_j_same_fluid():
      return obj.zero()
    coeff = obj.get_var('qcol_coeffj')
    if var == 'qcol_uj':
      mj_simu = obj.get_mass(obj.mf_jspecies, units='simu') # [simu mass]
      energy = (3/2) * mj_simu * obj.get_var('uid2')        # [simu energy]
    elif var == 'qcol_tgj':
      simu_kB = obj.uni.ksi_b * (obj.uni.usi_nr / obj.uni.usi_e)   # kB [simu energy / K]
      tgi = obj.get_var('tg')                       # [K]
      tgj = obj.get_var('tg', ifluid=obj.jfluid)    # [K]
      energy = 2 * simu_kB * (tgj - tgi)
    return coeff * energy  # [simu energy density / time]

  elif var == 'qcolj':
    if heating_is_off(): return obj.zero()
    return obj.get_var('qcol_uj') + obj.get_var('qcol_tgj')

  elif var in ['qcol_u', 'qcol_tg']:
    if heating_is_off(): return obj.zero()
    varj   = var + 'j'   # qcol_uj or qcol_tgj
    output = obj.get_var(varj, jS=-1)   # get varj for j = electrons
    for fluid in fl.Fluids(dd=obj):
      if fluid.SL != obj.ifluid:        # exclude varj for j = i  # not necessary but doesn't hurt.
        output += obj.get_var(varj, jfluid=fluid)
    return output

  elif var == 'qcol':
    if heating_is_off(): return obj.zero()
    return obj.get_var('qcol_u') + obj.get_var('qcol_tg')

  # other terms

  elif var in _TGQCOL_EQUIL:
    suffix  = var.split('_')[-1]  # uj, tgj, j, u, tg, or equil
    ## Let Cij = qcol_coeffj; Uij = qcol_uj, Tj = temperature of j. Then:
    ### Ti == ( sum_{s!=i}(Cis Uis + Cis * 2 kB Ts) ) / ( 2 kB sum_{s!=i}(Cis) )
    ## so for the "components" terms, we pick out only one term in this sum (in the numerator), e.g.:
    ### tgqcol_equil_uj == Cij Uij / ( 2 kB sum_{s!=i}(Cis) )
    if suffix == 'j':            # total contribution (u + tg) from j
      return obj.get_var('tgqcol_equil_uj') + obj.get_var('tgqcol_equil_tgj')
    elif suffix in ['u', 'tg']:  # total contribution (sum over j); total from u or total from tg
      result = obj.get_var('tgqcol_equil_'+suffix+'j', jfluid=(-1,0))
      for fluid in fl.Fluids(dd=obj):
        result += obj.get_var('tgqcol_equil_'+suffix+'j', jfluid=fluid)
      return result
    elif suffix == 'equil':      # total contribution u + tg, summed over all j.
      return obj.get_var('tgqcol_equil_u') + obj.get_var('tgqcol_equil_tg')
    else:
      # suffix == 'uj' or 'tgj'
      with obj.MaintainFluids():
        # denom = sum_{s!=i}(Cis).     [(simu length)^-3 (simu time)^-1]
        denom = obj.get_var('qcol_coeffj', jS=-1, cache_with_nfluid=2)  # coeff for j = electrons
        for fluid in fl.Fluids(dd=obj):
          denom += obj.get_var('qcol_coeffj', jfluid=fluid, cache_with_nfluid=2)
      # Based on suffix, return appropriate term.
      if suffix == 'uj':
        simu_kB = obj.uni.ksi_b * (obj.uni.usi_nr / obj.uni.usi_e)   # kB [simu energy / K]
        qcol_uj = obj.get_var('qcol_uj')
        temperature_contribution = qcol_uj / (2 * simu_kB)  # [K (simu length)^-3 (simu time)^-1]
      elif suffix == 'tgj':
        coeffj  = obj.get_var('qcol_coeffj')
        tgj     = obj.get_var('tg', ifluid=obj.jfluid)
        temperature_contribution = coeffj * tgj             # [K (simu length)^-3 (simu time)^-1]
      return temperature_contribution / denom       # [K]


  elif var == 'qjoulei':
    # qjoulei = qi * ni * \vec{ui} dot \vec{E}
    # ui is on grid cell faces while E is on grid cell edges.
    # We must interpolate to align with energy density e, which is at center of grid cells.
    # uix is at (-0.5, 0, 0) while Ex is at (0, -0.5, -0.5)
    # --> we shift uix by xup, and Ex by yup zup
    result = obj.zero()
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
    COLFRE_QUANT = ['nu_ij','nu_sj',                           # basics: frequencies
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
    # TODO: also check mf_param_file tables to see if the collision is turned off.
    if obj.match_aux():
      # return constant if constant collision frequency is turned on.
      i_elec, j_elec = (obj.mf_ispecies < 0, obj.mf_jspecies < 0)
      if i_elec or j_elec:
        const_nu_en = obj.get_param('ec_const_nu_en', default= -1.0)
        const_nu_in = obj.get_param('ec_const_nu_in', default= -1.0)
        if const_nu_en>=0 or const_nu_in>=0:  # at least one constant collision frequency is turned on.
          non_elec_fluid   = getattr(obj, '{}fluid'.format('j' if i_elec else 'i'))
          non_elec_neutral = obj.get_charge( non_elec_fluid ) == 0   # whether the non-electrons are neutral.
          def nu_ij(const_nu):
            result = obj.zero() + const_nu
            if i_elec:
              return result
            else:
              return result * obj.get_var('nu_ij_to_ji', ifluid=jfluid, jfluid=ifluid)
          if non_elec_neutral and const_nu_en >= 0:
            return nu_ij(const_nu_en)
          elif (not non_elec_neutral) and const_nu_in >= 0:
            return nu_ij(const_nu_in)
    # << if we reach this line, constant colfreq is off for this i,j; so now calculate colfreq.
    coll_type = obj.get_coll_type()   # gets 'EL', 'MX', 'CL', or None
    if coll_type is not None:
      if coll_type[0] == 'EE':     # electrons --> use "implied" coll type.
        coll_type = coll_type[1]   # TODO: add coll_keys to mf_eparams.in??
      nu_ij_varname = 'nu_ij_{}'.format(coll_type.lower())  # nu_ij_el, nu_ij_mx, or nu_ij_cl
      return obj.get_var(nu_ij_varname)
    else:
      errmsg = ("Found no valid coll_keys for ifluid={}, jfluid={}. "
        "looked for 'CL' for coulomb collisions, or 'EL' or 'MX' for other collisions. "
        "You can enter coll_keys in the COLL_KEYS section in mf_param_file='{}'.")
      mf_param_file = obj.get_param('mf_param_file', default='mf_params.in')
      raise ValueError(errmsg.format(obj.ifluid, obj.jfluid, mf_param_file))
      

  # collision frequency - elastic or coulomb
  if var in ['nu_ij_el', 'nu_ij_cl']:
    with Caching(obj, nfluid=2) as cache:
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
        result = scalars * logcul * n_j / tgij**1.5  # [ simu frequency units]
        
      # elastic collisions:
      elif var.endswith('el'):
        cross    = obj.get_var('cross', match_type=MATCH_PHYSICS)    # [cm^2]
        tg_speed = np.sqrt(8 * (obj.uni.kboltzmann/obj.uni.amu) * tgij / (np.pi * m_ij)) # [cm s^-1]
        result = 4./3. * n_j * m_jfrac * cross * tg_speed / obj.uni.u_hz  # [simu frequency units]

      # cache result, then return:
      cache(var, result)
      return result


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

  # sum of collision frequencies: sum_{i in ions} (nu_{ifluid, i})
  elif var == 'nu_si':
    ifluid = obj.ifluid
    result = obj.zero()
    for fluid in fl.Fluids(dd=obj).ions():
      if fluid.SL != ifluid:
        result += obj.get_var('nu_ij', jfluid=fluid.SL)
    return result

  # sum of collision frequencies: sum_{n in neutrals} (nu_{ifluid, n})
  elif var == 'nu_sn':
    ifluid = obj.ifluid
    result = obj.zero()
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

  if obj.match_aux():
    # return 0 if ifluid > jfluid. (comparing species, then level if species are equal)
    # we do this because mm_cross gives 0 if ifluid > jfluid (and jfluid is not electrons))
    if (obj.ifluid > obj.jfluid) and obj.mf_jspecies > 0:
      return obj.zero()

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
        return obj.zero()
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
    p = obj.zero()
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
      return obj.zero()
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
    return (2 * np.pi / dx1d) + obj.zero()


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
             'result is in units of [simu. frequency * simu. length^2].', nfluid=2)
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

def get_thermal_instab_quant(obj, quant, THERMAL_INSTAB_QUANT=None):
  '''very specific quantities which are related to the ion thermal and/or electron thermal instabilities.
  For source of formulae, see paper by Dimant & Oppenheim, 2004.

  In general, ion ifluid --> calculate for ion thermal instability; electron fluid --> for electron thermal.
  Electron thermal is not yet implemented.

  Quantities which depend on two fluids expect ifluid to be ion or electron, and jfluid to be neutral.
  '''
  if THERMAL_INSTAB_QUANT is None:
    THERMAL_INSTAB_QUANT = ['thermal_growth_rate',
                            'thermal_freq', 'thermal_tan2xopt',
                            'thermal_xopt', 'thermal_xopt_rad', 'thermal_xopt_deg']
    vecs = ['thermal_u0', 'thermal_v0']
    THERMAL_INSTAB_QUANT += [v+x for v in vecs for x in ['x', 'y', 'z']]
    # add thermal_growth_rate with combinations of terms.
    THERMAL_GROWRATE_QUANTS = ['thermal_growth_rate' + x for x in ['', '_fb', '_thermal', '_damping']]
    THERMAL_GROWRATE_QUANTS += [quant+'_max' for quant in THERMAL_GROWRATE_QUANTS]
    THERMAL_INSTAB_QUANT += THERMAL_GROWRATE_QUANTS

  if quant=='':
    docvar = document_vars.vars_documenter(obj, 'THERMAL_INSTAB_QUANT', THERMAL_INSTAB_QUANT,
                                           get_thermal_instab_quant.__doc__, nfluid=1)
    docvar('thermal_growth_rate', 'thermal instability optimal growth rate divided by wavenumber (k) squared. ' +\
             'result is in units of [simu. frequency * simu. length^2].', nfluid=1)
    docvar('thermal_growth_rate_max', 'thermal_growth_rate times (maximum resolvable wavenumber squared).', nfluid=1)
    for x in ['fb', 'thermal', 'damping']:
      docvar('thermal_growth_rate_'+x, 'thermal instability optimal growth rate divided by wavenumber (k) squared, ' +\
             'but just the '+x+' term. Result is in units of [simu. frequency * simu. length^2].', nfluid=1)
    for thermal_xopt_rad in ['thermal_xopt', 'thermal_xopt_rad']:
      docvar(thermal_xopt_rad, 'thermal instability optimal angle between k and (Ve - Vi) to maximize growth.' +\
                'result will be in radians. Result will be between -pi/4 and pi/4.', nfluid=1)
    docvar('thermal_xopt_deg', 'thermal instability optimal angle between k and (Ve - Vi) to maximize growth.' +\
                'result will be in degrees. Result will be between -45 and 45.', nfluid=1)
    docvar('thermal_tan2xopt', 'tangent of 2 times thermal_xopt', nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('thermal_u0'+x, x+'-component of (Ve - Vi). Warning: proper interpolation not yet implemented.', nfluid=1)
    for x in ['x', 'y', 'z']:
      docvar('thermal_v0'+x, x+'-component of E x B / B^2. Warning: proper interpolation not yet implemented.', nfluid=0)
    return None

  #if quant not in THERMAL_INSTAB_QUANT:
  #  return None

  def check_fluids_ok(nfluid=1):
    '''checks that ifluid is ion and jfluid is neutral. Only checks up to nfluid. raise error if bad.'''
    if nfluid >=1:
      icharge = obj.get_charge(obj.ifluid)
      if icharge == 0:
        raise ValueError('Expected ion or electron ifluid for Thermal Instability quants, but got neutral ifluid.')
      elif icharge < 0:
        raise ValueError('Electron Thermal Instability quantities not yet implemented. ispecies<0 not allowed.')
    if nfluid >=2:
      if obj.get_charge(obj.jfluid) != 0:
        raise ValueError('Expected neutral jfluid but got non-neutral jfluid.')
    return True

  if quant.startswith('thermal_growth_rate'):
    check_fluids_ok(nfluid=1)
    if '_max' in quant:
      quant = quant.replace('_max', '')
      k2 = max(obj.get_kmax())**2
    else:
      k2 = 1
    if quant=='thermal_growth_rate':
      include_terms = ['fb', 'thermal', 'damping']
    else:
      include_terms = quant.split('_')[3:]
    # prep work
    result = obj.zero()
    psi    = obj.get_var('psi0')
    U02    = obj.get_var('thermal_u02')  # U_0^2
    nu_in  = obj.get_var('nu_sn')
    front_coeff = psi * U02 / ((1 + psi) * nu_in)   # leading coefficient (applies to all terms)
    if 'fb' in include_terms or 'thermal' in include_terms:
      # if calculating fb or thermal terms, need to know these values:
      ki2  = obj.get_var('kappa')**2     # kappa_i^2
      A    = (8 + (1 - ki2)**2 + 4 * psi * ki2)**(-1/2)
    # calculating terms
    if 'fb' in include_terms:
      fbterm    = (1 - ki2) * (1 + (3 - ki2) * A) / (2 * (1 + psi)**2)
      result += fbterm
    if 'thermal' in include_terms:
      thermterm = ki2 * (1 + (4 - ki2 + psi) * A) / (3 * (1 + psi))
      result += thermterm
    if 'damping' in include_terms:
      Cs = obj.get_var('ci')
      dampterm  = -1 * Cs**2 / U02
      result += dampterm
    # multiply by leading coefficient
    result *= front_coeff
    # multiply by k^2 (1 unless '_max' in name of quant)
    result *= k2
    return result

  elif quant in ['thermal_u0'+x for x in ['x', 'y', 'z']]:
    check_fluids_ok(nfluid=1)
    # TODO: handle interpolation properly.
    x = quant[-1]
    qi    = obj.get_charge(obj.ifluid, units='simu')
    efx   = obj.get_var('ef'+x)
    mi    = obj.get_mass(obj.ifluid, units='simu')
    nu_in = obj.get_var('nu_sn')
    Vix     =    qi * efx / (mi * nu_in)
    V0x   = obj.get_var('thermal_v0'+x)
    ki2   = obj.get_var('kappa')**2
    return (V0x - Vix) / (1 + ki2)

  elif quant in ['thermal_v0'+x for x in ['x', 'y', 'z']]:
    # TODO: handle interpolation properly.
    x = quant[-1]
    ExB__x = obj.get_var('eftimesb'+x)
    B2     = obj.get_var('b2')
    return ExB__x/B2

  elif quant == 'thermal_tan2xopt':
    check_fluids_ok(nfluid=1)
    ki  = obj.get_var('kappa')
    psi = obj.get_var('psi0')
    return 2 * ki * (1 + psi) / (ki**2 - 3)

  elif quant in ['thermal_xopt', 'thermal_xopt_rad']:
    #TODO: think about which results are being dropped because np.arctan is not multi-valued.
    return 0.5 * np.arctan(obj.get_var('thermal_tan2xopt'))

  elif quant == 'thermal_xopt_deg':
    return np.rad2deg(obj.get_var('thermal_xopt_rad'))