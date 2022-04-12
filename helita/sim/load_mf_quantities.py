# import builtins
import warnings
import itertools

# import internal modules
from . import document_vars
from .file_memory import Caching   # never alters results, but caches them for better efficiency.
                                   # use sparingly on "short" calculations; apply liberally to "long" calculations.
                                   # see also cache_with_nfluid and cache kwargs of get_var.
from .load_arithmetic_quantities import do_stagger

## import the relevant things from the internal module "units"
from .units import (
  UNI, USI, UCGS, UCONST,
  Usym, Usyms, UsymD,
  U_TUPLE,
  DIMENSIONLESS, UNITS_FACTOR_1, NO_NAME,
  UNI_length, UNI_time, UNI_mass,
  UNI_speed, UNI_rho, UNI_nr, UNI_hz
)

# import external public modules
import numpy as np

# set constants
MATCH_PHYSICS = 0  # don't change this value.  # this one is the default (see ebysus.py)
MATCH_AUX     = 1  # don't change this value.
AXES          = ('x', 'y', 'z')   # the axes names.
YZ_FROM_X     = dict(x=('y', 'z'), y=('z', 'x'), z=('x', 'y'))  # right-handed coord system x,y,z given x.

# TODO:
#  adapt maxwell collisions from load_quantities.py file, to improve maxwell collisions in this file.

# construct some frequently-used units
units_e = dict(uni_f=UNI.e, usi_name=Usym('J') / Usym('m')**3)  #ucgs_name= ???



def load_mf_quantities(obj, quant, *args__None, GLOBAL_QUANT=None, EFIELD_QUANT=None,
                       ONEFLUID_QUANT=None, ELECTRON_QUANT=None,
                       CONTINUITY_QUANT=None, MOMENTUM_QUANT=None, HEATING_QUANT=None,
                       SPITZERTERM_QUANT=None,
                       COLFRE_QUANT=None, LOGCUL_QUANT=None, CROSTAB_QUANT=None, 
                       DRIFT_QUANT=None, MEAN_QUANT=None, CFL_QUANT=None, PLASMA_QUANT=None,
                       HYPERDIFFUSIVE_QUANT=None,
                       WAVE_QUANT=None, FB_INSTAB_QUANT=None, THERMAL_INSTAB_QUANT=None,
                       **kw__None):
  '''load multifluid quantity indicated by quant.
  *args__None and **kw__None go nowhere.
  '''
  __tracebackhide__ = True  # hide this func from error traceback stack.

  quant = quant.lower()

  document_vars.set_meta_quant(obj, 'mf_quantities',
      ("These are the multi-fluid quantities; only used by ebysus.\n"
       "nfluid means 'number of fluids used to read the quantity'.\n"
       "  2  -> uses obj.ifluid and obj.jfluid. (e.g. 'nu_ij')\n"
       "  1  -> uses obj.ifluid (but not jfluid). (e.g. 'ux', 'tg')\n"
       "  0  -> does not use ifluid nor jfluid. (e.g. 'bx', 'nel', 'tot_e'))\n")
                              )

   # tell which getter function is associated with each QUANT.
  ## (would put this list outside this function if the getter functions were defined there, but they are not.)
  _getter_QUANT_pairs = (
    (get_global_var, 'GLOBAL_QUANT'),
    (get_efield_var, 'EFIELD_QUANT'),
    (get_onefluid_var, 'ONEFLUID_QUANT'),
    (get_electron_var, 'ELECTRON_QUANT'),
    (get_continuity_quant, 'CONTINUITY_QUANT'),
    (get_momentum_quant, 'MOMENTUM_QUANT'),
    (get_heating_quant, 'HEATING_QUANT'),
    (get_spitzerterm, 'SPITZERTERM_QUANT'),
    (get_mf_colf, 'COLFRE_QUANT'),
    (get_mf_logcul, 'LOGCUL_QUANT'),
    (get_mf_cross, 'CROSTAB_QUANT'),
    (get_mf_driftvar, 'DRIFT_QUANT'),
    (get_mean_quant, 'MEAN_QUANT'),
    (get_cfl_quant, 'CFL_QUANT'),
    (get_mf_plasmaparam, 'PLASMA_QUANT'),
    (get_hyperdiffusive_quant, 'HYPERDIFFUSIVE_QUANT'),
    (get_mf_wavequant, 'WAVE_QUANT'),
    (get_fb_instab_quant, 'FB_INSTAB_QUANT'),
    (get_thermal_instab_quant, 'THERMAL_INSTAB_QUANT'),
  )

  val = None
  # loop through the function and QUANT pairs, running the functions as appropriate.
  for getter, QUANT_STR in _getter_QUANT_pairs:
    QUANT = locals()[QUANT_STR]   # QUANT = value of input parameter named QUANT_STR.
    #if QUANT != '':
    val = getter(obj, quant, **{QUANT_STR : QUANT})
    if val is not None:
      break
  return val


# default
_GLOBAL_QUANT = ('GLOBAL_QUANT',
                 ['totr', 'rc', 'rions', 'rneu',
                  'tot_e', 'tot_ke', 'e_ef', 'e_b', 'total_energy',
                  'tot_px', 'tot_py', 'tot_pz',
                  'grph', 'tot_part', 'mu',
                  'jx', 'jy', 'jz', 'resistivity'
                  ]
                )
# get value
@document_vars.quant_tracking_simple(_GLOBAL_QUANT[0])
def get_global_var(obj, var, GLOBAL_QUANT=None):
  '''Variables which are calculated by looping through species or levels.'''
  if GLOBAL_QUANT is None:
      GLOBAL_QUANT = _GLOBAL_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _GLOBAL_QUANT[0], GLOBAL_QUANT, get_global_var.__doc__, nfluid=0)
    docvar('totr', 'sum of mass densities of all fluids [simu. mass density units]', uni=UNI_rho)
    for rc in ['rc', 'rions']:
      docvar(rc,   'sum of mass densities of all ionized fluids [simu. mass density units]', uni=UNI_rho)
    docvar('rneu', 'sum of mass densities of all neutral species [simu. mass density units]', uni=UNI_rho)
    docvar('tot_e',  'sum of internal energy densities of all fluids [simu. energy density units]', **units_e)
    docvar('tot_ke', 'sum of kinetic  energy densities of all fluids [simu. energy density units]', **units_e)
    docvar('e_ef', 'energy density in electric field [simu. energy density units]', **units_e)
    docvar('e_b', 'energy density in magnetic field [simu. energy density units]', **units_e)
    docvar('total_energy', 'total energy density. tot_e + tot_ke + e_ef + e_b [simu units].', **units_e)
    docvar('resistivity', 'total resistivity of the plasma. sum of partial resistivity.' +\
                          '[(simu. E-field units)/(simu. current per area units)]',
                          uni_f=UNI.ef / UNI.i, usi_name=(Usym('V'))/(Usym('A')*Usym('m')))
    for axis in AXES:
      docvar('tot_p'+axis, 'sum of '+axis+'-momentum densities of all fluids [simu. mom. dens. units] ' +\
                           'NOTE: does not include "electron momentum" which is assumed to be ~= 0.',
                            uni=UNI_speed * UNI_rho)
    docvar('grph',  'grams per hydrogen atom')
    docvar('tot_part', 'total number of particles, including free electrons [cm^-3]')
    docvar('mu', 'ratio of total number of particles without free electrong / tot_part')
    for axis in AXES:
      docvar('j'+axis, 'sum of '+axis+'-component of current per unit area [simu. current per area units]',
                        uni_f=UNI.i, usi_name=Usym('A')/Usym('m')**2)  # ucgs_name= ???
    return None

  if var not in GLOBAL_QUANT:
      return None

  output = obj.zero_at_mesh_center()
  if var == 'totr':  # total density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)

    return output
  elif var in ['rc', 'rions']:  # total ionized density
    for fluid in obj.fluids.ions():
      output += obj.get_var('r', ifluid=fluid)
    return output
  elif var == 'rneu':  # total neutral density
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      for ilevel in range(1,nlevels+1):
        if (obj.att[ispecies].params.levels['stage'][ilevel-1] == 1): 
          output += obj.get_var('r', mf_ispecies=ispecies, mf_ilevel=ilevel)
    return output
  elif var == 'tot_e':
    output += obj.get_var('e', mf_ispecies= -1) # internal energy density of electrons
    for fluid in obj.fluids:
      output += obj.get_var('e', ifluid=fluid.SL) # internal energy density of fluid
    return output
  elif var == 'tot_ke':
    output = obj.get_var('eke')   # kinetic energy density of electrons
    for fluid in obj.fluids:
      output += obj.get_var('ke', ifluid=fluid.SL)  # kinetic energy density of fluid
    return output
  elif var == 'e_ef':
    ef2  = obj.get_var('ef2')   # |E|^2  [simu E-field units, squared]
    eps0 = obj.uni.permsi       # epsilon_0 [SI units]
    units = obj.uni.usi_ef**2 / obj.uni.usi_e   # convert ef2 * eps0 to [simu energy density units]
    return (0.5 * eps0 * units) * ef2
  elif var == 'resistivity': 

    ne = obj.get_var('nr', mf_ispecies=-1)   # [simu. number density units]
    neqe = ne * obj.uni.simu_qsi_e
    rhoe = obj.get_var('re') 
    nu_sum = 0.0
    for fluid in obj.fluids:
      nu_sum += obj.get_var('nu_ij',mf_ispecies=-1,jfluid=fluid)
    output = nu_sum * rhoe / (neqe)**2  
    return output

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
    return output  
  
  elif var.startswith('tot_p'):  # note: must be tot_px, tot_py, or tot_pz.
    axis = var[-1]
    for fluid in obj.fluids:
      output += obj.get_var('p'+axis, ifluid=fluid.SL)   # momentum density of fluid

    return output
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
    return output
  elif var == 'tot_part':
    for ispecies in obj.att:
      nlevels = obj.att[ispecies].params.nlevel
      weight = obj.att[ispecies].params.atomic_weight * \
            obj.uni.amu / obj.uni.u_r
      for ilevel in range(1,nlevels+1):
        output += obj.get_var('r', mf_ispecies=ispecies,
            mf_ilevel=ilevel) / weight * (obj.att[ispecies].params.levels[ilevel-1]+1)
    return output
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
  else:
    # if we reach this line, var is a global_var quant but we did not handle it.
    raise NotImplementedError(f'{repr(var)} in get_global_var')


# default
_EFIELD_QUANT = ('EFIELD_QUANT',
                   ['efx', 'efy', 'efz',
                   'uexbx', 'uexby', 'uexbz',
                   'uepxbx', 'uepxby', 'uepxbz',
                   'batx', 'baty', 'batz',
                   'emomx', 'emomy', 'emomz',
                   'efneqex', 'efneqey', 'efneqez']
                  )
# get value
@document_vars.quant_tracking_simple(_EFIELD_QUANT[0])
def get_efield_var(obj, var, EFIELD_QUANT=None):
  '''variables related to electric field.'''
  if EFIELD_QUANT is None:
    EFIELD_QUANT = _EFIELD_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _EFIELD_QUANT[0], EFIELD_QUANT, get_efield_var.__doc__, nfluid=0)
    EF_UNITS = dict(uni_f=UNI.ef, usi_name=Usym('V')/Usym('m')) # ucgs_name= ???
    for x in AXES:
      docvar('ef'+x, x+'-component of electric field [simu. E-field units] ', **EF_UNITS)          
    for x in AXES:
      docvar('uexb'+x, x+'-component of u_e cross B [simu. E-field units]. Note efx = - uexbx + ...', **EF_UNITS)
    for x in AXES:
      docvar('uepxb'+x, x+'-component of uep cross B [simu. E-field units]. Note efx = - uexbx + ... . ' +\
                        ' uep is the electron velocity assuming current = 0.', **EF_UNITS)
    for x in AXES:
      docvar('bat'+x, x+'-component of "battery term" (contribution to electric field) [simu. E-field units]. ' +\
                      '== grad(P_e) / (n_e q_e), where q_e < 0. ', **EF_UNITS)
    for x in AXES:
      docvar('emom'+x, x+'-component of collisions contribution to electric field [simu. E-field units]. ' +\
                       '== sum_j R_e^(ej) / (n_e q_e)', **EF_UNITS)
    for x in AXES:
      docvar('efneqe'+x, 'value of n_e * q_e, interpolated to align with the {}-component of E '.format(x) +\
                       '[simu. charge density units]. Note q_e < 0, so efneqe{} < 0.'.format(x),
                       uni_f=UNI.nq, usi_name=Usym('C')/Usym('m')**3)
    return None

  if var not in EFIELD_QUANT:
    return None

  x = var[-1]  # axis; 'x', 'y', or 'z'
  y, z = YZ_FROM_X[x]
  base = var[:-1]   # var without axis. E.g. 'ef', 'uexb', 'emom'.

  if base == 'ef':   # electric field    # efx
    with Caching(obj, nfluid=0) as cache:
      # E = - ue x B + (ne qe)^-1 * ( grad(pressure_e) - (ion & rec terms) - sum_j(R_e^(ej)) )
      #   (where the convention used is qe < 0.)
      # ----- -ue x B contribution ----- #
      # There is a flag, "do_hall", when "false", we don't let the contribution
      ## from current to ue to enter in to the B x ue for electric field.
      if obj.match_aux() and obj.get_param('do_hall', default="false")=="false":
        ue = 'uep'  # include only the momentum contribution in ue, in our ef calculation.
        if obj.verbose:
          warnings.warn('do_hall=="false", so we are dropping the j (current) contribution to ef (E-field)')
      else:
        ue = 'ue'   # include the full ue term, in our ef calculation.
      B_cross_ue__x = -1 * obj.get_var(ue+'xb'+x)

      # ----- grad Pe contribution ----- #
      battery_x = obj.get_var('bat'+x)
      # ----- calculate ionization & recombination effects ----- #
      if obj.get_param('do_recion', default=False):
        if obj.verbose:
          warnings.warn('E-field contribution from ionization & recombination have not yet been added.')
      # ----- calculate collisional effects ----- #
      emom_x = obj.get_var('emom'+x)
      # ----- calculate efx ----- #
      result = B_cross_ue__x + battery_x + emom_x   # [simu. E-field units] 
      cache(var, result)

  elif base in ('uexb', 'uepxb'):   # ue x B    # (aligned with efx)
    ue = 'ue' if (base == 'uexb') else 'uep'
    # interpolation:
    ## B and ue are face-centered vectors.
    ## Thus we use _facecross_ from load_arithmetic_quantities.
    result = obj.get_var(ue+'_facecross_b'+x)

  elif base == 'bat':  # grad(P_e) / (ne qe)
    if obj.match_aux() and (not obj.get_param('do_battery', default=False)):
      return obj.zero_at_mesh_edge(x)
    # interpolation:
    ## efx is at (0, -1/2, -1/2).
    ## P is at (0,0,0).
    ## dpdxup is at (1/2, 0, 0).
    ## dpdxup xdn ydn zdn is at (0, -1/2, -1/2) --> aligned with efx.
    interp   = 'xdnydnzdn'
    gradPe_x = obj.get_var('dpd'+x+'up'+interp, iS=-1) # [simu. energy density units]
    neqe     = obj.get_var('efneqe'+x)   # ne qe, aligned with efx
    result = gradPe_x / neqe

  elif base == 'emom':  # -1 * sum_j R_e^(ej) / (ne qe)     (aligned with efx)
    if obj.match_aux() and (not obj.get_param('do_ohm_ecol', default=False)):
      return obj.zero_at_mesh_edge(x)
    # interpolation:
    ## efx is at (0, -1/2, -1/2)
    ## rijx is at (-1/2, 0, 0)    (same as ux)
    ## --> to align with efx, we shift rijx by xup ydn zdn
    interp = x+'up'+y+'dn'+z+'dn'
    sum_rejx = obj.get_var('rijsum'+x + interp, iS=-1)   # [simu. momentum density units / simu. time units]
    neqe     = obj.get_var('efneqe'+x)   # ne qe, aligned with efx
    result = -1 * sum_rejx / neqe

  elif base == 'efneqe':   # ne qe   (aligned with efx)
    # interpolation:
    ## efx is at (0, -1/2, -1/2)
    ## ne is at (0, 0, 0)
    ## to align with efx, we shift ne by ydn zdn
    interp = y+'dn'+z+'dn'
    result = obj.get_var('nq'+interp, iS=-1)   # [simu. charge density units]  (Note: 'nq' < 0 for electrons)

  else:
    raise NotImplementedError(f'{repr(base)} in get_efield_var')

  return result



# default
_ONEFLUID_QUANT = ('ONEFLUID_QUANT',
                   ['nr', 'nq', 'p', 'pressure', 'tg', 'temperature', 'tgjoule', 'ke', 'vtherm', 'vtherm_simple',
                    'ri', 'uix', 'uiy', 'uiz', 'pix', 'piy', 'piz']
                  )
# get value
@document_vars.quant_tracking_simple(_ONEFLUID_QUANT[0])
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
    ONEFLUID_QUANT = _ONEFLUID_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _ONEFLUID_QUANT[0], ONEFLUID_QUANT, get_onefluid_var.__doc__, nfluid=1)
    docvar('nr', 'number density of ifluid [simu. number density units]', uni=UNI_nr)
    docvar('nq', 'charge density of ifluid [simu. charge density units]',
                  uni_f= UNI.q * UNI_nr.f, usi_name=Usym('C') / Usym('m')**3)
    for tg in ['tg', 'temperature']:
      docvar(tg, 'temperature of ifluid [K]', uni=U_TUPLE(UNITS_FACTOR_1, Usym('K')))
    docvar('tgjoule', 'temperature of ifluid [ebysus energy units]. == tg [K] * k_boltzmann [J/K]', uni=U_TUPLE(UNITS_FACTOR_1, Usym('J')))
    for p in ['p', 'pressure']:
      docvar(p, 'pressure of ifluid [simu. energy density units]', uni_f=UNI.e)
    docvar('ke', 'kinetic energy density of ifluid [simu. units]', **units_e)
    _equivstr = " Equivalent to obj.get_var('{ve:}') when obj.mf_ispecies < 0; obj.get_var('{vf:}'), otherwise."
    equivstr = lambda v: _equivstr.format(ve=v.replace('i', 'e'), vf=v.replace('i', ''))
    docvar('vtherm', 'thermal speed of ifluid [simu. velocity units]. = sqrt (8 * k_b * T_i / (pi * m_i) )', uni=UNI_speed)
    docvar('vtherm_simple', '"simple" thermal speed of ifluid [simu. velocity units]. '+\
                            '= sqrt (k_b * T_i / m_i)', uni=UNI_speed)
    docvar('ri', 'mass density of ifluid [simu. mass density units]. '+equivstr('ri'), uni=UNI_rho)
    for uix in ['uix', 'uiy', 'uiz']:
      docvar(uix, 'velocity of ifluid [simu. velocity units]. '+equivstr(uix), uni=UNI_speed)
    for pix in ['pix', 'piy', 'piz']:
      docvar(pix, 'momentum density of ifluid [simu. momentum density units]. '+equivstr(pix), uni=UNI_rho * UNI_speed)
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
      return obj.zero_at_mesh_center()
    else:
      return charge * obj.get_var('nr')

  elif var in ['p', 'pressure']:
    gamma = obj.uni.gamma
    return (gamma - 1) * obj.get_var('e')          # p = (gamma - 1) * internal energy

  elif var in ['tg', 'temperature']:
    p  = obj.get_var('p') * obj.uni.u_e    # [cgs units]
    nr = obj.get_var('nr') * obj.uni.u_nr  # [cgs units]
    return p / (nr * obj.uni.k_b)          # [K]         # p = n k T

  elif var == 'tgjoule':
    return obj.uni.ksi_b * obj('tg')

  elif var == 'ke':
    return 0.5 * obj.get_var('ri') * obj.get_var('ui2')

  elif var == 'vtherm':
    Ti     = obj.get_var('tg')                           # [K]
    mi     = obj.get_mass(obj.mf_ispecies, units='si')   # [kg]
    vtherm = np.sqrt(obj.uni.ksi_b * Ti / mi)            # [m / s]
    consts = np.sqrt(8 / np.pi)
    return consts * vtherm / obj.uni.usi_u                   # [simu. velocity units]

  elif var == 'vtherm_simple':
    Ti     = obj('tg')                                   # [K]
    mi     = obj.get_mass(obj.mf_ispecies, units='si')   # [kg]
    vtherm = np.sqrt(obj.uni.ksi_b * Ti / mi)            # [m / s]
    return vtherm / obj.uni.usi_u                   # [simu. velocity units]

  else:
    if var in ['ri', 'uix', 'uiy', 'uiz', 'pix', 'piy', 'piz']:
      if obj.mf_ispecies < 0:  # electrons
        e_var = var.replace('i', 'e')
        return obj.get_var(e_var)
      else:                    # not electrons
        f_var = var.replace('i', '')
        return obj.get_var(f_var)

    else:
      raise NotImplementedError(f'{repr(var)} in get_onefluid_var')


# default
_ELECTRON_QUANT = ['nel', 'nre', 're', 'eke', 'pe']
_ELECTRON_QUANT += [ue + x for ue in ['ue', 'pe', 'uej', 'uep'] for x in AXES]
_ELECTRON_QUANT = ('ELECTRON_QUANT', _ELECTRON_QUANT)
# get value
@document_vars.quant_tracking_simple(_ELECTRON_QUANT[0])
def get_electron_var(obj, var, ELECTRON_QUANT=None):
  '''variables related to electrons (requires looping over ions to calculate).'''

  if ELECTRON_QUANT is None:
    ELECTRON_QUANT = _ELECTRON_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _ELECTRON_QUANT[0], ELECTRON_QUANT, get_electron_var.__doc__, nfluid=0)
    docvar('nel',  'electron number density [cm^-3]')
    docvar('nre',  'electron number density [simu. number density units]', uni=UNI_nr)
    docvar('re',   'mass density of electrons [simu. mass density units]', uni=UNI_rho)
    docvar('eke',  'electron kinetic energy density [simu. energy density units]', **units_e)
    docvar('pe',   'electron pressure [simu. pressure units]', uni_f=UNI.e)
    for x in AXES:
      docvar('ue'+x, '{}-component of electron velocity [simu. velocity units]'.format(x),
                      uni=UNI_speed)
    for x in AXES:
      docvar('pe'+x, '{}-component of electron momentum density [simu. momentum density units]'.format(x),
                      uni=UNI_speed * UNI_rho)
    for x in AXES:
      docvar('uej'+x,'{}-component of current contribution to electron velocity [simu. velocity units]'.format(x),
                      uni=UNI_speed)
    for x in AXES:
      docvar('uep'+x,'{}-component of species velocities contribution to electron velocity [simu. velocity units]'.format(x),
                      uni=UNI_speed)
    return None

  if (var not in ELECTRON_QUANT):
    return None

  if var == 'nel': # number density of electrons [cm^-3]
    return obj.get_var('nre') * obj.uni.u_nr   # [cm^-3]

  elif var == 'nre': # number density of electrons [simu. units]
    with Caching(obj, nfluid=0) as cache:
      output = obj.zero_at_mesh_center()
      for fluid in obj.fluids.ions():
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
    output = obj.zero_at_mesh_face(x)
    nqe    = obj.zero_at_mesh_face(x)  # charge density of electrons.
    for fluid in obj.fluids.ions():
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
    y, z    = tuple(set(AXES) - set((x)))
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
      y, z   = tuple(set(AXES) - set((x)))
      interp = x+'dn' + y+'up' + z+'up'
      output = obj.get_var('j'+x + interp)   # [simu current per area units]
      # get component due to velocities:
      ## r is in center of cells, while u is on faces, so we need to interpolate.
      ## r is at (0, 0, 0); ux is at (-0.5, 0, 0)
      ## ---> to align with ux, we shift r by xdn
      interp = x+'dn'
      nqe    = obj.zero_at_mesh_face(x)  # charge density of electrons.
      for fluid in obj.fluids.ions():
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

  else:
    raise NotImplementedError(f'{repr(var)} in get_electron_var')


# default
_CONTINUITY_QUANT = ('CONTINUITY_QUANT',
                     ['ndivu', 'udivn', 'udotgradn', 'flux_nu', 'flux_un',
                     'gradnx', 'gradny', 'gradnz']
                     )
# get value
@document_vars.quant_tracking_simple(_CONTINUITY_QUANT[0])
def get_continuity_quant(obj, var, CONTINUITY_QUANT=None):
  '''terms related to the continuity equation.
  In the simple case (e.g. no ionization), expect dn/dt + flux_un = 0.
  '''
  if CONTINUITY_QUANT is None:
    CONTINUITY_QUANT = _CONTINUITY_QUANT[1]

  if var == '':
    docvar = document_vars.vars_documenter(obj, _CONTINUITY_QUANT[0], CONTINUITY_QUANT,
                                           get_continuity_quant.__doc__, nfluid=1, uni=UNI_nr * UNI_hz)
    docvar('ndivu', 'number density times divergence of velocity')
    for udivn in ('udivn', 'udotgradn'):
      docvar(udivn, 'velocity dotted with gradient of number density')
    for x in AXES:
      docvar('gradn'+x, x+'-component of grad(nr), face-centered.', nfluid=1, uni=UNI.qc(0)) # qc0 will be e.g. dnrdxdn.
    for flux_un in ('flux_un', 'flux_nu'):
      docvar(flux_un, 'divergence of (velocity times number density). Calculated via ndivu + udotgradn.')
    docvar('flux_p', 'divergence of momentum density')
    return None

  if var not in CONTINUITY_QUANT:
    return None

  # --- continuity equation terms --- #
  if var == 'ndivu':
    n    = obj('nr')
    divu = obj('divupui')  # divup(ui). up to align with n.
    return n * divu

  elif var in ('gradnx', 'gradny', 'gradnz'):
    return obj(f'dnrd{var[-1]}dn')

  elif var in ('udivn', 'udotgradn'):
    return obj('ui_facedot_gradn')

  elif var in ('flux_nu', 'flux_un'):
    return obj('ndivu') + obj('udivn')

  else:
    raise NotImplementedError(f'{repr(var)} in get_momentum_quant')

# default
_MOMENTUM_QUANT = []
_MQVECS = ['rij', 'rijsum', 'momflorentz','momohme', 'mombat', 'gradp', 'momrate', '_ueq_scr', 'ueq', 'ueqsimple']
_MOMENTUM_QUANT += [v + x for v in _MQVECS for x in AXES]
_MOMENTUM_QUANT = ('MOMENTUM_QUANT', _MOMENTUM_QUANT)
# get value
@document_vars.quant_tracking_simple(_MOMENTUM_QUANT[0])
def get_momentum_quant(obj, var, MOMENTUM_QUANT=None):
  '''terms related to momentum equations of fluids.
  The units for these quantities are [simu. momentum density units / simu. time units].
  '''
  if MOMENTUM_QUANT is None:
    MOMENTUM_QUANT = _MOMENTUM_QUANT[1]

  if var == '':
    docvar = document_vars.vars_documenter(obj, _MOMENTUM_QUANT[0], MOMENTUM_QUANT, get_momentum_quant.__doc__)
    units_dpdt = dict(uni_f=UNI.phz, uni_name=UNI_rho.name * UNI_speed.name / UNI_time.name)
    for x in AXES:
      docvar('rij'+x, ('{x:}-component of momentum density exchange between ifluid and jfluid ' +\
                       '[simu. momentum density units / simu. time units]. ' +\
                       'rij{x:} = R_i^(ij) {x:} = mi ni nu_ij * (u{x:}_j - u{x:}_i)').format(x=x), nfluid=2, **units_dpdt)
    for x in AXES:
      docvar('rijsum'+x, x+'-component of momentum density change of ifluid ' +\
                           'due to collisions with all other fluids. = sum_j rij'+x, nfluid=1, **units_dpdt)
    for x in AXES:
      docvar('momflorentz'+x, x+'-component of momentum density change of ifluid due to Lorentz force. ' +\
                           '[simu. momentum density units / simu. time units]. = ni qi (E + ui x B).', nfluid=1, **units_dpdt)
    for x in AXES:
      docvar('momohme'+x, x+'-component of momentum density change of ifluid due the ohmic term in the electric field. ' +\
                           '[simu. momentum density units / simu. time units]. = ni qi E = ni qi nu_es (ui-epUx) .', nfluid=1, **units_dpdt)
    for x in AXES:
      docvar('mombat'+x, x+'-component of momentum density change of ifluid due to battery term. ' +\
                           '[simu. momentum density units / simu. time units]. = ni qi grad(P_e) / (ne qe).', nfluid=1, **units_dpdt)
    for x in AXES:
      docvar('gradp'+x, x+'-component of grad(Pi), face-centered (interp. loc. aligns with momentum).', nfluid=1, uni=UNI.qc(0))
    for x in AXES:
      docvar('momrate'+x, x+'-component of rate of change of momentum density. ' +\
                          '= (-gradp + momflorentz + rijsum)_'+x, nfluid=1, **units_dpdt)
    for x in AXES:
      docvar('ueq'+x, x+'-component of equilibrium velocity of ifluid. Ignores derivatives in momentum equation. ' +\
                       '= '+x+'-component of [qs (_ueq_scr x B) + (ms) (sum_{j!=s} nu_sj) (_ueq_scr)] /' +\
                       ' [(qs^2/ms) B^2 + (ms) (sum_{j!=s} nu_sj)^2]. [simu velocity units].', nfluid=1, uni=UNI_speed)
    for x in AXES:
      docvar('ueqsimple'+x, x+'-component of "simple" equilibrium velocity of ifluid. ' +\
                       'Treats these as 0: derivatives in momentum equation, velocity of jfluid, nu_sb for b not jfluid.' +\
                       '= '+x+'-component of [(qs/(ms nu_sj))^2 (E x B) + qs/(ms nu_sj) E] /' +\
                       ' [( (qs/ms) (|B|/nu_sj) )^2 + 1]. [simu. velocity units].', nfluid=2, uni=UNI_speed)
    for x in AXES:
      docvar('_ueq_scr'+x, x+'-component of helper term which appears twice in formula for ueq. '+x+'-component of ' +\
                       ' [(qs/ms) E + (sum_{j!=s} nu_sj uj)]. face-centered. [simu velocity units].', nfluid=1, uni=UNI_speed)
    return None

  if var not in MOMENTUM_QUANT:
    return None

  # --- momentum equation terms --- #
  x = var[-1] # axis; x= 'x', 'y', or 'z'.
  if x in AXES:
    y, z = YZ_FROM_X[x]
    base = var[:-1]
  else:
    base = var

  if base == 'rij':
    if obj.i_j_same_fluid():      # when ifluid==jfluid, u_j = u_i, so rij = 0.
      return obj.zero_at_mesh_face(x)  # save time by returning 0 without reading any data.
    # rij = mi ni nu_ij * (u_j - u_i) = ri nu_ij * (u_j - u_i)
    # (Note: this does NOT equal to nu_ij * (rj u_j - ri u_i))
    ## Scalars are at (0,0,0) so we must shift by xdn to align with face-centered u at (-0.5,0,0)
    nu_ij = obj.get_var('nu_ij' + x+'dn')
    ri  = obj.get_var('ri' + x+'dn')
    uix = obj.get_var('ui'+x)
    ujx = obj.get_var('ui'+x, ifluid=obj.jfluid)
    return ri * nu_ij * (ujx - uix)

  elif base == 'rijsum':
    result = obj.get_var('rij'+x, jS=-1)            # rijx for j=electrons
    for fluid in obj.fluids:
      result += obj.get_var('rij'+x, jfluid=fluid)  # rijx for j=fluid
    return result

  elif base == 'rejsum':
    result = obj.get_var('rij'+x, jS=-1)            # rijx for j=electrons
    return result

  elif base == 'momflorentz':
    # momflorentz = ni qi (E + ui x B)
    qi = obj.get_charge(obj.ifluid, units='simu')
    if qi == 0:
      return obj.zero_at_mesh_face(x)    # no lorentz force for neutrals - save time by just returning 0 here :)
    # make sure we get the interpolation correct:
    ## B and ui are face-centered vectors, and we want a face-centered result to align with p.
    ## Thus we use ui_facecrosstoface_b (which gives a face-centered result).
    ## Meanwhile, E is edge-centered, so we must shift all three coords.
    ## And n is fully centered, so we shift by xdn.
    ni = obj.get_var('nr'+x+'dn')
    Ex = obj.get_var('ef'+x + x+'dn' + y+'up' + z+'up', cache_with_nfluid=0)
    uxB__x = obj.get_var('ui_facecrosstoface_b'+x)
    return ni * qi * (Ex + uxB__x)

  elif base == 'momohme':
    # momflorentz = ni qi (E + ui x B)
    qi = obj.get_charge(obj.ifluid, units='simu')
    if qi == 0:
      return obj.zero_at_mesh_face(x)    # no lorentz force for neutrals - save time by just returning 0 here :)
    ni = obj.get_var('nr')
    # make sure we get the interpolation correct:
    ## B and ui are face-centered vectors, and we want a face-centered result to align with p.
    ## Thus we use ui_facecrosstoface_b (which gives a face-centered result).
    ## Meanwhile, E is edge-centered, so we must shift all three coords.
    ## Ex is at (0, -0.5, -0.5), so we shift by xdn, yup, zup
    Ex = obj.get_var('emom'+x + x+'dn' + y+'up' + z+'up', cache_with_nfluid=0)

    return ni * qi * Ex

  elif base == 'mombat':
    # px is at (-0.5, 0, 0); nq is at (0, 0, 0), so we shift by xdn
    interp = x+'dn'
    niqi = obj('nq'+interp)
    with obj.MaintainFluids():
      obj.iS = -1
      neqe     = obj('nq'+interp)
      gradPe_x = obj('gradp'+x)  # gradp handles the interpolation already.
    return (niqi / neqe) * gradPe_x

  elif base == 'gradp':
    # px is at (-0.5, 0, 0); pressure is at (0, 0, 0), so we do dpdxdn
    return obj.get_var('dpd'+x+'dn')

  elif base == 'momrate':
    if obj.get_param('do_recion', default=False):
      if obj.verbose:
        warnings.warn('momentum contribution from ionization & recombination have not yet been added.')
    gradpx    = obj.get_var('gradp'+x)
    florentzx = obj.get_var('momflorentz'+x)
    rijsumx   = obj.get_var('rijsum'+x)
    return florentzx - gradpx + rijsumx

  # --- "equilibrium velocity" terms --- #
  elif base == '_ueq_scr':
    qi = obj.get_charge(obj.ifluid, units='simu')
    mi = obj.get_mass(obj.ifluid, units='simu')
    ifluid_orig = obj.ifluid
    # make sure we get the interpolation correct:
    ## We want a face-centered result to align with u.
    ## E is edge-centered, so we must shift all three coords.
    ## Ex is at (0, -0.5, -0.5), so we shift by xdn, yup, zup
    ## Meanwhile, scalars are at (0,0,0), so we shift those by xdn to align with u.
    Ex = obj.get_var('ef'+x + x+'dn' + y+'up' + z+'up', cache_with_nfluid=0)
    sum_nu_u = 0
    for jSL in obj.iter_fluid_SLs():
      if jSL != ifluid_orig:
        nu_sj = obj.get_var('nu_ij' + x+'dn', ifluid=ifluid_orig, jfluid=jSL)
        uj    = obj.get_var('ui'+x, ifluid=jSL)
        sum_nu_u += nu_sj * uj
    return (qi / mi) * Ex + sum_nu_u

  elif base == 'ueq':
    qi = obj.get_charge(obj.ifluid, units='simu')
    mi = obj.get_mass(obj.ifluid, units='simu')
    # make sure we get the interpolation correct:
    ## We want a face-centered result to align with u.
    ## B and _ueq_scr are face-centered, so we use _facecrosstoface_ to get a face-centered cross product.
    ## Meanwhile, E is edge-centered, so we must shift all three coords.
    ## Finally, scalars are at (0,0,0), so we shift those by xdn to align with u.
    B2 = obj.get_var('b2' + x+'dn')
    # begin calculations
    ueq_scr_x_B__x = obj.get_var('_ueq_scr_facecrosstoface_b'+x)
    ueq_scr__x     = obj.get_var('_ueq_scr'+x)
    sumnu = 0
    for jSL in obj.iter_fluid_SLs():
      if jSL != obj.ifluid:
        sumnu += obj.get_var('nu_ij' + x+'dn', jfluid=jSL)
    numer = qi * ueq_scr_x_B__x + mi * sumnu * ueq_scr__x
    denom = (qi**2/mi) * B2 + mi * sumnu**2
    return numer / denom

  elif base == 'ueqsimple':
    qi = obj.get_charge(obj.ifluid, units='simu')
    mi = obj.get_mass(obj.ifluid, units='simu')
    # make sure we get the interpolation correct:
    ## B and ui are face-centered vectors, and we want a face-centered result to align with u.
    ## Thus we use ui_facecrosstoface_b (which gives a face-centered result).
    ## Meanwhile, E is edge-centered, so we must shift all three coords.
    ## Ex is at (0, -0.5, -0.5), so we shift by xdn, yup, zup
    ## Finally, scalars are at (0,0,0), so we shift those by xdn to align with u.
    Ex = obj.get_var('ef'+x + x+'dn' + y+'up' + z+'up', cache_with_nfluid=0)
    B2 = obj.get_var('b2' + x+'dn')
    ExB__x = obj.get_var('ef_edgefacecross_b'+x)
    nu_ij = obj.get_var('nu_ij' + x+'dn')
    # begin calculations
    q_over_m_nu = (qi/mi) / nu_ij
    q_over_m_nu__squared = q_over_m_nu**2
    numer = q_over_m_nu__squared * ExB__x + q_over_m_nu * Ex
    denom = q_over_m_nu__squared * B2 + 1
    return numer / denom

  else:
    raise NotImplementedError(f'{repr(base)} in get_momentum_quant')


# default
_HEATING_QUANT = ['qcol_uj', 'qcol_tgj', 'qcol_coeffj', 'qcolj', 'qcol_j',
                 'qcol_u', 'qcol_tg', 'qcol',
                 'e_to_tg',
                 'tg_qcol',  # TODO: add tg_qcol_... for as many of the qcol terms as applicable.
                 'tg_qcol_uj', 'tg_qcol_u', 'tg_qcol_tgj', 'tg_qcol_tg', 'tg_qcol_j', 'tg_qcolj',
                 'qjoulei',
                 'tgdu',
                 'tg_rate',
                 'qcol_u_noe', 'qcol_tg_noe',
                 ]
_TGQCOL_EQUIL  = ['tgqcol_equil' + x for x in ('_uj', '_tgj', '_j', '_u', '_tg', '')]
_HEATING_QUANT += _TGQCOL_EQUIL
_HEATING_QUANT = ('HEATING_QUANT', _HEATING_QUANT)
# get value
@document_vars.quant_tracking_simple(_HEATING_QUANT[0])
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
    HEATING_QUANT = _HEATING_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _HEATING_QUANT[0], HEATING_QUANT, get_heating_quant.__doc__)
    units_qcol    = dict(uni_f=UNI.e / UNI.t, usi_name=Usym('J')/(Usym('m')**3 * Usym('s')))
    units_e_to_tg = dict(uni_f=UNITS_FACTOR_1 / UNI.e, usi_name=Usym('K') / (Usym('J') / Usym('m')**3))
    units_tg      = dict(uni_f=UNITS_FACTOR_1, uni_name=Usym('K'))
    units_dtgdt   = dict(uni_f=UNI.hz, uni_name=Usym('K')/Usym('s'))

    # docs for qcol and tg_qcol terms.
    qcol_docdict = {
      'qcol_uj'    : ('{heati} due to collisions with jfluid, and velocity drifts.', dict(nfluid=2)),
      'qcol_u'     :              ('{heati} due to collisions and velocity drifts.', dict(nfluid=1)),
      'qcol_u_noe' :              ('{heati} due to collisions and velocity drifts without electrons.', dict(nfluid=1)),
      'qcol_tgj'   : ('{heati} due to collisions with jfluid, and temperature differences.', dict(nfluid=2)),
      'qcol_tg'    :              ('{heati} due to collisions and temperature differences.', dict(nfluid=1)),
      'qcol_tg_noe':              ('{heati} due to collisions and temperature differences without electrons.', dict(nfluid=1)),
      'qcolj'      : ('total {heati} due to collisions with jfluid.', dict(nfluid=2)),
      'qcol'       : ('total {heati} due to collisions.', dict(nfluid=1)),
    }
    qcol_docdict['qcol_j'] = qcol_docdict['qcolj']  # alias
    
    # qcol: heating due to collisions in addition to velocity and/or temperature differences
    ## qcol tells the energy density change per unit time.
    ## tg_qcol tells the temperature change per unit time.
    for vname, (vdoc, kw_nfluid) in qcol_docdict.items():
      docvar(vname, vdoc.format(heati='heating of ifluid [simu. energy density per time]'), **kw_nfluid, **units_qcol)
      docvar('tg_'+vname, vdoc.format(heati='heating of ifluid [Kelvin per simu. time]'), **kw_nfluid, **units_tg)
    docvar('qcol_coeffj', 'coefficient common to qcol_uj and qcol_tj terms.' +\
                          ' == (mi / (gamma - 1) (mi + mj)) * ni * nu_ij. [simu units: length^-3 time^-1]',
                          nfluid=2, **units_qcol)
    docvar('e_to_tg',  'conversion factor from energy density to temperature for ifluid. '+\
                       'e_ifluid * e_to_tg = tg_ifluid', nfluid=1, **units_e_to_tg)
    # the other heating in the heating equation
    docvar('tgdu', 'rate of change of Ti due to -2/3 * T * div(u).', **units_dtgdt)
    docvar('tg_rate', 'predicted total rate of change of Ti, including all contributions', **units_dtgdt)

    # "simple equilibrium" vars
    equili = '"simple equilibrium" temperature [K] of ifluid (setting sum_j Qcol_ij=0 and solving for Ti)'
    ## note: these all involve setting sum_j Qcol_ij = 0 and solving for Ti.
    ## Let Cij = qcol_coeffj; Uij = qcol_uj, Tj = temperature of j. Then:
    ### Ti == ( sum_{s!=i}(Cis Uis + Cis * 2 kB Ts) ) / ( 2 kB sum_{s!=i}(Cis) )
    ## so for the "components" terms, we pick out only one term in this sum (in the numerator), e.g.:
    ### tgqcol_equil_uj == Cij Uij / ( 2 kB sum_{s!=i}(Cis) )
    docvar('tgqcol_equil_uj', equili + ', due only to contribution from velocity drift with jfluid.', nfluid=2, **units_tg)
    docvar('tgqcol_equil_tgj', equili + ', due only to contribution from temperature of jfluid.', nfluid=2, **units_tg)
    docvar('tgqcol_equil_j', equili + ', due only to contribution from jfluid.', nfluid=2, **units_tg)
    docvar('tgqcol_equil_u', equili + ', due only to contribution from velocity drifts with fluids.', nfluid=1, **units_tg)
    docvar('tgqcol_equil_tg', equili + ', due only to contribution from temperature of fluids.', nfluid=1, **units_tg)
    docvar('tgqcol_equil', equili + '.', nfluid=1, **units_tg)
    # "ohmic heating" (obsolete (?) - nonphysical to include this qjoule and the qcol_u term as it appears here.)
    docvar('qjoulei',  'qi ni ui dot E. (obsolete, nonphysical to include this term and the qcol_u term)', nfluid=1, **units_qcol)
    return None

  if var not in HEATING_QUANT:
    return None

  def heating_is_off():
    '''returns whether we should treat heating as if it is turned off.'''
    if obj.match_physics():
      return False
    if obj.mf_ispecies < 0 or obj.mf_jspecies < 0:  # electrons
      return not (obj.get_param('do_ohm_ecol', True) and obj.get_param('do_qohm', True))
    else: # not electrons
      return not (obj.get_param('do_col', True) and obj.get_param('do_qcol', True))

  # qcol terms
  if var == 'qcol_coeffj':
    if heating_is_off() or obj.i_j_same_fluid():
      return obj.zero_at_mesh_center()
    ni = obj.get_var('nr')             # [simu. units]
    mi = obj.get_mass(obj.mf_ispecies) # [amu]
    mj = obj.get_mass(obj.mf_jspecies) # [amu]
    nu_ij = obj.get_var('nu_ij')       # [simu. units]
    coeff = (mi / (mi + mj)) * ni * nu_ij   # [simu units: length^-3 time^-1]
    return coeff

  if var in ['qcol_uj', 'qcol_tgj']:
    if heating_is_off() or obj.i_j_same_fluid():
      return obj.zero_at_mesh_center()
    coeff = obj.get_var('qcol_coeffj')
    if var == 'qcol_uj':
      mj_simu = obj.get_mass(obj.mf_jspecies, units='simu') # [simu mass]
      energy = mj_simu * obj.get_var('uid2')        # [simu energy]
    elif var == 'qcol_tgj':
      simu_kB = obj.uni.ksi_b * (obj.uni.usi_nr / obj.uni.usi_e)   # kB [simu energy / K]
      tgi = obj.get_var('tg')                       # [K]
      tgj = obj.get_var('tg', ifluid=obj.jfluid)    # [K]
      energy = 3. * simu_kB * (tgj - tgi)
    return coeff * energy  # [simu energy density / time]

  elif var in ['qcolj', 'qcol_j']:
    if heating_is_off(): return obj.zero_at_mesh_center()
    return obj.get_var('qcol_uj') + obj.get_var('qcol_tgj')

  elif var in ['qcol_u', 'qcol_tg']:
    if heating_is_off(): return obj.zero_at_mesh_center()
    varj   = var + 'j'   # qcol_uj or qcol_tgj
    output = obj.get_var(varj, jS=-1)   # get varj for j = electrons
    for fluid in obj.fluids:
      if fluid.SL != obj.ifluid:        # exclude varj for j = i  # not necessary but doesn't hurt.
        output += obj.get_var(varj, jfluid=fluid)
    return output

  elif var in ['qcol_u_noe', 'qcol_tg_noe']:
    output = obj.zero_at_mesh_center()
    if heating_is_off(): return obj.zero_at_mesh_center()
    varj   = var[:-4] + 'j'   # qcol_uj or qcol_tgj
    for fluid in obj.fluids:
      if fluid.SL != obj.ifluid:        # exclude varj for j = i  # not necessary but doesn't hurt.
        output += obj.get_var(varj, jfluid=fluid)
    return output

  elif var == 'qcol':
    if heating_is_off(): return obj.zero_at_mesh_center()
    return obj.get_var('qcol_u') + obj.get_var('qcol_tg')

  # rate of change of T, terms
  elif var == 'tgdu':
    tg = obj('tg')
    divu = obj('divup'+'ui')
    return -2/3 * tg * divu

  elif var == 'tg_rate':
    tgqcol = obj('tg_qcol')
    tgdu = obj('tgdu')
    return tgqcol + tgdu

  # converting to temperature (from energy density) terms
  elif var == 'e_to_tg':
    simu_kB = obj.uni.ksi_b * (obj.uni.usi_nr / obj.uni.usi_e)   # kB [simu energy / K]
    return (obj.uni.gamma - 1) / (obj.get_var('nr') * simu_kB)

  elif var.startswith('tg_'):
    qcol = var[len('tg_') : ]  # var looks like tg_qcol
    assert qcol in HEATING_QUANT, "qcol must be in heating quant to get tg_qcol. qcol={}".format(repr(qcol))
    qcol_value = obj.get_var(qcol)         # [simu energy density / time]
    e_to_tg    = obj.get_var('e_to_tg')    # [K / simu energy density (of ifluid)]
    return qcol_value * e_to_tg            # [K]

  # "simple equilibrium temperature" terms
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
      for fluid in obj.fluids:
        result += obj.get_var('tgqcol_equil_'+suffix+'j', jfluid=fluid)
      return result
    elif suffix == 'equil':      # total contribution u + tg, summed over all j.
      return obj.get_var('tgqcol_equil_u') + obj.get_var('tgqcol_equil_tg')
    else:
      # suffix == 'uj' or 'tgj'
      with obj.MaintainFluids():
        # denom = sum_{s!=i}(Cis).     [(simu length)^-3 (simu time)^-1]
        denom = obj.get_var('qcol_coeffj', jS=-1, cache_with_nfluid=2)  # coeff for j = electrons
        for fluid in obj.fluids:
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
    result = obj.zero_at_mesh_center()
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

  else:
    raise NotImplementedError(f'{repr(var)} in get_heating_quant')


# default
_SPITZTERM_QUANT = ('SPITZTERM_QUANT', ['kappaq','dxTe','dyTe','dzTe','rhs'])
# get value
@document_vars.quant_tracking_simple(_SPITZTERM_QUANT[0])
def get_spitzerterm(obj, var, SPITZERTERM_QUANT=None):
  '''spitzer conductivies'''
  if SPITZERTERM_QUANT is None:
    SPITZERTERM_QUANT = _SPITZTERM_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _SPITZTERM_QUANT[0], SPITZERTERM_QUANT, get_spitzerterm.__doc__, nfluid='???')
    docvar('kappaq', 'Electron thermal diffusivity coefficient [Ebysus units], in SI: W.m-1.K-1')
    docvar('dxTe',   'Gradient of electron temperature in the x direction [simu.u_te/simu.u_l] in SI: K.m-1', uni=UNI.quant_child(0))
    docvar('dyTe',   'Gradient of electron temperature in the y direction [simu.u_te/simu.u_l] in SI: K.m-1', uni=UNI.quant_child(0))
    docvar('dzTe',   'Gradient of electron temperature in the z direction [simu.u_te/simu.u_l] in SI: K.m-1', uni=UNI.quant_child(0))
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

  elif (var == 'dxTe'):     
    gradx_Te = obj.get_var('dtgdxup', iS=-1)
    result = gradx_Te

  elif (var == 'dyTe'):
    grady_Te = obj.get_var('dtgdyup', iS=-1)
    result = grady_Te
  
  elif (var == 'dzTe'):
    gradz_Te = obj.get_var('dtgdzup', iS=-1)
    result = gradz_Te

  elif (var == 'rhs'):  
    bx =   obj.get_var('bx')
    by =   obj.get_var('by')
    bz =   obj.get_var('bz')
    gradx_Te = obj.get_var('dtgdxup', iS=-1)
    grady_Te = obj.get_var('dtgdyup', iS=-1)
    gradz_Te = obj.get_var('dtgdzup', iS=-1)

    bmin = 1E-5 

    normb = np.sqrt(bx**2+by**2+bz**2)
    norm2bmin = bx**2+by**2+bz**2+bmin**2

    bbx = bx/normb
    bby = by/normb
    bbz = bz/normb

    bm = (bmin**2)/norm2bmin

    rhs = bbx*gradx_Te + bby*grady_Te + bbz*gradz_Te
    result = rhs

  else:
    raise NotImplementedError(f'{repr(var)} in get_spitzterm')

  return result


# default
_COLFRE_QUANT = ('COLFRE_QUANT',
                 ['nu_ij','nu_sj',                                 # basics: frequencies
                  'nu_si','nu_sn','nu_ei','nu_en',                 # sum of frequencies
                  'nu_ij_el', 'nu_ij_mx', 'nu_ij_cl',              # colfreq by type
                  'nu_ij_res', 'nu_se_spitzcoul', 'nu_ij_capcoul', # alternative colfreq formulae
                  'nu_ij_to_ji', 'nu_sj_to_js',                    # conversion factor nu_ij --> nu_ji
                  'c_tot_per_vol', '1dcolslope',                   # misc.
                 ]           
                )
# get value
@document_vars.quant_tracking_simple(_COLFRE_QUANT[0])
def get_mf_colf(obj, var, COLFRE_QUANT=None):
  '''quantities related to collision frequency.

  Note the collision frequencies here are the momentum transer collision frequencies.
  These obey the identity m_a  n_a  nu_ab  =  m_b  n_b  nu_ba.
  This identity ensures total momentum (sum over all species) does not change due to collisions.
  '''

  if COLFRE_QUANT is None:
    COLFRE_QUANT = _COLFRE_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _COLFRE_QUANT[0], COLFRE_QUANT, get_mf_colf.__doc__, uni=UNI_hz)
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
    docvar('nu_ij_to_ji', 'nu_ij_to_ji * nu_ij = nu_ji.  nu_ij_to_ji = m_i * n_i / (m_j * n_j) = r_i / r_j',
                          nfluid=2, uni=DIMENSIONLESS)
    docvar('nu_sj_to_js', 'nu_sj_to_js * nu_sj = nu_js.  nu_sj_to_js = m_s * n_s / (m_j * n_j) = r_s / r_j',
                          nfluid=2, uni=DIMENSIONLESS)
    docvar('1dcolslope', '-(nu_ij + nu_ji)', nfluid=2)
    docvar('c_tot_per_vol', 'number density of collisions per volume per time '
                            '[simu. number density * simu. frequency] between ifluid and jfluid.', nfluid=2,
                            uni=UNI_nr * UNI_hz)
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
        const_nu_ei = obj.get_param('ec_const_nu_ei', default= -1.0)
        if const_nu_en>=0 or const_nu_ei>=0:  # at least one constant collision frequency is turned on.
          non_elec_fluid   = getattr(obj, '{}fluid'.format('j' if i_elec else 'i'))
          non_elec_neutral = obj.get_charge( non_elec_fluid ) == 0   # whether the non-electrons are neutral.
          def nu_ij(const_nu):
            result = obj.zero_at_mesh_center() + const_nu
            if i_elec:
              return result
            else:
              return result * obj.get_var('nu_ij_to_ji', ifluid=obj.jfluid, jfluid=obj.ifluid)
          if non_elec_neutral and const_nu_en >= 0:
            return nu_ij(const_nu_en)
          elif (not non_elec_neutral) and const_nu_ei >= 0:
            return nu_ij(const_nu_ei)
    # << if we reach this line, we don't have to worry about constant electron colfreq.
    coll_type = obj.get_coll_type()   # gets 'EL', 'MX', 'CL', or None
    if coll_type is not None:
      if coll_type[0] == 'EE':     # electrons --> use "implied" coll type.
        coll_type = coll_type[1]   # TODO: add coll_keys to mf_eparams.in??
      nu_ij_varname = 'nu_ij_{}'.format(coll_type.lower())  # nu_ij_el, nu_ij_mx, or nu_ij_cl
      return obj.get_var(nu_ij_varname)
    elif obj.match_aux() and (obj.get_charge(obj.ifluid) > 0) and (obj.get_charge(obj.jfluid) > 0):
      # here, we want to match aux, i and j are ions, and coulomb collisions are turned off.
      return obj.zero_at_mesh_center()   ## so we return zero (instead of making a crash)
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
      cache(var, result)#/ 1.0233)
      return result#/ 1.0233


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
    result = obj.zero_at_mesh_center()
    for fluid in obj.fluids.ions():
      if fluid.SL != ifluid:
        result += obj.get_var('nu_ij', jfluid=fluid.SL)
    return result

  # sum of collision frequencies: sum_{n in neutrals} (nu_{ifluid, n})
  elif var == 'nu_sn':
    ifluid = obj.ifluid
    result = obj.zero_at_mesh_center()
    for fluid in obj.fluids.neutrals():
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
    cross    = np.pi*2.0*(b_0**2)*(np.log(2.0*obj.get_var('ldebye')*obj.uni.usi_l/b_0)-2.0*euler_constant) # [m2]
    # Before we had np.log(2.0*obj.get_var('ldebye')*obj.uni.usi_l/b_0)-0.5-2.0*euler_constant
    # Not sure about the coefficient 0.5 from Capitelli et al. (2000). Should be only euler constant according to Liboff (1959, eq. 4.28)

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
    if obj.verbose:
      warnings.warn(DeprecationWarning('1dcolslope will be removed at some point in the future.'))
    return -1 * obj.get_var("nu_ij") * (1 + obj.get_var('nu_ij_to_ji'))

  else:
    raise NotImplementedError(f'{repr(var)} in get_mf_colf')


# default
_LOGCUL_QUANT = ('LOGCUL_QUANT', ['logcul'])
# get value
@document_vars.quant_tracking_simple(_LOGCUL_QUANT[0])
def get_mf_logcul(obj, var, LOGCUL_QUANT=None):
  '''coulomb logarithm'''
  if LOGCUL_QUANT is None:
    LOGCUL_QUANT = _LOGCUL_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _LOGCUL_QUANT[0], LOGCUL_QUANT, get_mf_logcul.__doc__)
    docvar('logcul', 'Coulomb Logarithmic used for Coulomb collisions.', nfluid=0, uni=DIMENSIONLESS)
    return None

  if var not in LOGCUL_QUANT:
    return None
  
  if var == "logcul":
    etg = obj.get_var('tg', mf_ispecies=-1)
    nel = obj.get_var('nel')
    return 23. + 1.5 * np.log(etg / 1.e6) - \
          0.5 * np.log(nel / 1e6)

  else:
    raise NotImplementedError(f'{repr(var)} in get_logcul')


# default
_CROSTAB_QUANT = ('CROSTAB_QUANT', ['cross','tgij'])
# get value
@document_vars.quant_tracking_simple(_CROSTAB_QUANT[0])
def get_mf_cross(obj, var, CROSTAB_QUANT=None):
  '''cross section between species.'''
  if CROSTAB_QUANT is None:
    CROSTAB_QUANT = _CROSTAB_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _CROSTAB_QUANT[0], CROSTAB_QUANT, get_mf_cross.__doc__, nfluid=2)
    docvar('cross', 'cross section between ifluid and jfluid [cm^2]. Use species < 0 for electrons.',
                    uni_name=UNI_length.name**2, ucgs_f=UNITS_FACTOR_1, usi_f=UCONST.cm_to_m**2)
    return None

  if var not in CROSTAB_QUANT:
    return None

  if obj.match_aux():
    # return 0 if ifluid > jfluid. (comparing species, then level if species are equal)
    # we do this because mm_cross gives 0 if ifluid > jfluid (and jfluid is not electrons))
    if (obj.ifluid > obj.jfluid) and obj.mf_jspecies > 0:
      return obj.zero_at_mesh_center()

  # get masses & temperatures, then restore original obj.ifluid and obj.jfluid values.
  with obj.MaintainFluids():
    m_i = obj.get_mass(obj.mf_ispecies)
    m_j = obj.get_mass(obj.mf_jspecies)
    tgi = obj.get_var('tg', ifluid = obj.ifluid)
    tgj = obj.get_var('tg', ifluid = obj.jfluid)

  # temperature, weighted by mass of species
  tg = (tgi*m_j + tgj*m_i)/(m_i + m_j)
  if var == 'tgij' : 
    return tg
  else: 
    # look up cross table and get cross section
    #crossunits = 2.8e-17  
    crossobj = obj.get_cross_sect(ifluid=obj.ifluid, jfluid=obj.jfluid)
    crossunits = crossobj.cross_tab[0]['crossunits']
    cross = crossunits * crossobj.tab_interp(tg)

    return cross


# default
_DRIFT_QUANT  = ['ed', 'rd', 'tgd']
_DRIFT_QUANT += [dq + x for dq in ('ud', 'pd', 'uid') for x in AXES]
_DRIFT_QUANT  = ('DRIFT_QUANT', _DRIFT_QUANT)
# get value
@document_vars.quant_tracking_simple(_DRIFT_QUANT[0])
def get_mf_driftvar(obj, var, DRIFT_QUANT=None):
  '''var drift between fluids. I.e. var_ifluid - var_jfluid.'''
  if DRIFT_QUANT is None:
    DRIFT_QUANT = _DRIFT_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _DRIFT_QUANT[0], DRIFT_QUANT, get_mf_driftvar.__doc__,
                                           nfluid=2, uni=UNI.quant_child(0))
    def doc_start(var):
      return '"drift" for quantity "{var}". I.e. ({var} for ifluid) - ({var} for jfluid). '.format(var=var)
    for x in AXES:
      docvar('ud'+x, doc_start(var='u'+x) + 'u = velocity [simu. units].')
    for x in AXES:
      docvar('uid'+x, doc_start(var='ui'+x) + 'ui = velocity [simu. units].')
    for x in AXES:
      docvar('pd'+x, doc_start(var='p'+x) + 'p = momentum density [simu. units].')
    docvar('ed', doc_start(var='ed') + 'e = energy (density??) [simu. units].')
    docvar('rd', doc_start(var='rd') + 'r = mass density [simu. units].')
    docvar('tgd', doc_start(var='tgd') + 'tg = temperature [K].')
    return None

  if var not in DRIFT_QUANT:
    return None

  else:
    if var[-1] == 'd':    # scalar drift quant                 e.g. tgd
      quant = var[:-1] # "base quant"; without d.              e.g. tg
    elif var[-2] == 'd':  # vector drift quant                 e.g. uidx
      quant = var[:-2] + var[-1]  # "base quant"; without d    e.g. uix

    q_i = obj.get_var(quant, ifluid=obj.ifluid)
    q_j = obj.get_var(quant, ifluid=obj.jfluid)
    return q_i - q_j


# default
_MEAN_QUANT = ('MEAN_QUANT',
               ['neu_meannr_mass', 'ion_meannr_mass',
                ]
              )
# get value
@document_vars.quant_tracking_simple(_MEAN_QUANT[0])
def get_mean_quant(obj, var, MEAN_QUANT=None):
  '''weighted means of quantities.'''
  if MEAN_QUANT is None:
    MEAN_QUANT = _MEAN_QUANT[1]

  if var=='':
    docvar = document_vars.vars_documenter(obj, _MEAN_QUANT[0], MEAN_QUANT, get_mean_quant.__doc__)
    docvar('neu_meannr_mass', 'number density weighted mean mass of neutrals.'
                              ' == sum_n(mass_n * nr_n) / sum_n(nr_n). [simu mass units]',
                              nfluid=0, uni_name=UsymD(usi='kg', ucgs='g'), uni_f=UNI.m)
    docvar('ion_meannr_mass', 'number density weighted mean mass of ions.'
                              ' == sum_i(mass_i * nr_i) / sum_i(nr_i). [simu mass units]',
                              nfluid=0, uni_name=UsymD(usi='kg', ucgs='g'), uni_f=UNI.m)
    return None

  if var not in MEAN_QUANT:
    return None

  if var.endswith('_meannr_mass'):
    neu = var[:-len('_meannr_mass')]
    fluids = obj.fluids
    if neu == 'neu':
      fluids = fluids.neutrals()
    elif neu == 'ion':
      fluids = fluids.ions()
    else:
      raise NotImplementedError('only know _meannr_mass for neu or ion but got {}'.format(neu))
    numer = obj.zero_at_mesh_center()
    denom = obj.zero_at_mesh_center()
    for fluid in fluids:
      r = obj.get_var('r', ifluid=fluid)
      m = obj.get_mass(fluid, units='simu')
      numer += r
      denom += r / m
    return numer / denom

  else:
    raise NotImplementedError(f'{repr(var)} in get_mean_quant')


# default
_CFL_QUANTS = ['ohm']
_CFL_QUANT = ['cfl_' + q for q in _CFL_QUANTS]
_CFL_QUANT = ('CFL_QUANT', _CFL_QUANT)
# get value
@document_vars.quant_tracking_simple(_CFL_QUANT[0])
def get_cfl_quant(obj, quant, CFL_QUANT=None):
  '''CFL quantities. All are in simu. frequency units.'''
  if CFL_QUANT is None:
    CFL_QUANT = _CFL_QUANT[1]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, _CFL_QUANT[0], CFL_QUANT, get_cfl_quant.__doc__)
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

  else:
    raise NotImplementedError(f'{repr(quant)} in get_cfl_quant')


# default
_PLASMA_QUANT = ('PLASMA_QUANT',
                 ['beta', 'beta_ions', 'va', 'va_ions', 'cs', 's', 'ke', 'mn', 'man', 'hp',
                  'vax', 'vay', 'vaz', 'hx', 'hy', 'hz', 'kx', 'ky', 'kz',
                  'sgyrof', 'gyrof', 'skappa', 'kappa', 'ldebye', 'ldebyei',
                 ]
                )
# get value
@document_vars.quant_tracking_simple(_PLASMA_QUANT[0])
def get_mf_plasmaparam(obj, quant, PLASMA_QUANT=None):
  '''plasma parameters, e.g. plasma beta, sound speed, pressure scale height'''
  if PLASMA_QUANT is None:
    PLASMA_QUANT = _PLASMA_QUANT[1]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, _PLASMA_QUANT[0], PLASMA_QUANT, get_mf_plasmaparam.__doc__)
    docvar('beta', "plasma beta", nfluid='???', uni=DIMENSIONLESS) #nfluid= 1 if mfe_p = p_ifluid; 0 if mfe_p = sum of pressures.
    docvar('beta_ions', "plasma beta using sum of ion pressures. P / (B^2 / (2 mu0)).", nfluid=0, uni=DIMENSIONLESS)
    docvar('va', "alfven speed [simu. units]", nfluid=0, uni=UNI_speed)
    docvar('va_ions', "alfven speed [simu. units], using density := density of ions.", nfluid=0, uni=UNI_speed)
    docvar('vai', "alfven speed [simu. units] of ifluid. Vai = sqrt(B^2 / (mu0 * rho_i))", nfluid=1, uni=UNI_speed)
    docvar('cs', "sound speed [simu. units]", nfluid='???', uni=UNI_speed)
    docvar('csi', "sound speed [simu. units] of ifluid. Csi = sqrt(gamma * pressure_i / rho_i)", nfluid=1, uni=UNI_speed)
    docvar('cfast', "Cfast for ifluid. == (Csi**2 + Vai**2 + Cse**2)?? NEEDS UPDATING.", nfluid=1, uni=UNI_speed)
    docvar('s', "entropy [log of quantities in simu. units]", nfluid='???', uni=DIMENSIONLESS)
    docvar('mn', "mach number (using sound speed)", nfluid=1, uni=DIMENSIONLESS)
    docvar('man', "mach number (using alfven speed)", nfluid=1, uni=DIMENSIONLESS)
    docvar('hp', "Pressure scale height", nfluid='???')
    for x in AXES:
      docvar('va'+x, x+"-component of alfven velocity [simu. units]", nfluid=0, uni=UNI_speed)
    for x in AXES:
      docvar('k'+x, ("{axis} component of kinetic energy density of ifluid [simu. units]."+\
                  "(0.5 * rho * (get_var(u{axis})**2)").format(axis=x), nfluid=1, **units_e)
    docvar('sgyrof', "signed gryofrequency for ifluid. I.e. qi * |B| / mi. [1 / (simu. time units)]", nfluid=1, uni=UNI_hz)
    docvar('gyrof', "gryofrequency for ifluid. I.e. abs(qi * |B| / mi). [1 / (simu. time units)]", nfluid=1, uni=UNI_hz)
    kappanote = ' "Highly magnetized" when kappa^2 >> 1.'
    docvar('skappa', "signed magnetization for ifluid. I.e. sgryof/nu_sn." + kappanote, nfluid=1, uni=DIMENSIONLESS)
    docvar('kappa', "magnetization for ifluid. I.e. gyrof/nu_sn." + kappanote, nfluid=1, uni=DIMENSIONLESS)
    docvar('ldebyei', "debye length of ifluid [simu. length units]. sqrt(kB eps0 q^-2 Ti / ni)", nfluid=1, uni=UNI_length)
    docvar('ldebye', "debye length of plasma [simu. length units]. " +\
                     "sqrt(kB eps0 e^-2 / (ne/Te + sum_j(Zj^2 * nj / Tj)) ); Zj = qj/e"+\
                     "1/sum_j( (1/ldebye_j) for j in fluids and electrons)", nfluid=0, uni=UNI_length)
    return None

  if quant not in PLASMA_QUANT:
    return None

  if quant in ['hp', 's', 'cs', 'beta']:
    var = obj.get_var('mfe_p')  # is mfe_p pressure for ifluid, or sum of all fluid pressures? - SE Apr 19 2021
    if quant == 'hp':
      if getattr(obj, 'nx') < 5:
        return obj.zero()
      else:
        return 1. / (do_stagger(var, 'ddzup',obj=obj) + 1e-12)
    elif quant == 'cs':
      return np.sqrt(obj.params['gamma'][obj.snapInd] *
                     var / obj.get_var('totr'))
    elif quant == 's':
      return (np.log(var) - obj.params['gamma'][obj.snapInd] *
              np.log(obj.get_var('totr')))
    else: # quant == 'beta':
      return 2 * var / obj.get_var('b2')

  elif quant == 'csi':
    p = obj('p')
    r = obj('r')
    return np.sqrt(obj.uni.gamma * p / r)

  elif quant == 'cfast':
    warnings.warn('cfast implementation may be using the wrong formula.')
    speeds = [obj('csi')]   # sound speed
    i_charged = obj.get_charge(obj.ifluid) != 0
    if i_charged:
      speeds.append(obj('vai'))   # alfven speed
      if not obj.fluids_equal(obj.ifluid, (-1,0)):   # if ifluid is not electrons
        speeds.append(obj('csi', ifluid=(-1,0)))  # sound speed of electrons
    result = sum(speed**2 for speed in speeds)
    return result

  elif quant == 'beta_ions':
    p = obj.zero()
    for fluid in obj.fluids.ions():
      p += obj.get_var('p', ifluid=fluid)
    bp = obj.get_var('b2') / 2    # (dd.uni.usi_b**2 / dd.uni.mu0si) == 1 by def'n of b in ebysus.
    return p / bp

  elif quant in ['mn', 'man']:
    var = obj.get_var('modu')
    if quant == 'mn':
      return var / (obj.get_var('cs') + 1e-12)
    else:
      return var / (obj.get_var('va') + 1e-12)

  elif quant in ['va', 'vax', 'vay', 'vaz']:
    var = obj.get_var('totr')
    if len(quant) == 2:
      return obj.get_var('modb') / np.sqrt(var)
    else:
      axis = quant[-1]
      return np.sqrt(obj.get_var('b' + axis + 'c') ** 2 / var)

  elif quant in ['va_ions']:
    r = obj.get_var('rions')
    return obj.get_var('modb') / np.sqrt(r)

  elif quant == 'vai':
    r = obj('r')
    b = obj('modb')
    return b / np.sqrt(r)   # [simu speed units]. note: mu0 = 1 in simu units.

  elif quant in ['hx', 'hy', 'hz', 'kx', 'ky', 'kz']:
    axis = quant[-1]
    var = obj.get_var('p' + axis + 'c')
    if quant[0] == 'h':
      # anyone can delete this warning once you have confirmed that get_var('hx') does what you think it should:
      warnmsg = ('get_var(hx) (or hy or hz) uses get_var(p), and used it since before get_var(p) was implemented. '
                 'Maybe should be using get_var(mfe_p) instead? '
                 'You should not trust results until you check this.  - SE Apr 19 2021.')
      if obj.verbose:
        warnings.warn(warnmsg)
      return ((obj.get_var('e') + obj.get_var('p')) /
              obj.get_var('r') * var)
    else:
      return obj.get_var('u2') * var * 0.5

  elif quant == 'sgyrof':
    B = obj.get_var('modb')                       # magnitude of B [simu. B-field units]
    q = obj.get_charge(obj.ifluid, units='simu')     #[simu. charge units]
    m = obj.get_mass(obj.mf_ispecies, units='simu')  #[simu. mass units]
    return q * B / m

  elif quant == 'gyrof':
    return np.abs(obj.get_var('sgyrof'))

  elif quant == 'skappa':
    gyrof = obj.get_var('sgyrof') #[simu. freq.]
    nu_sn = obj.get_var('nu_sn')  #[simu. freq.]
    return gyrof / nu_sn 

  elif quant == 'kappa':
    return np.abs(obj.get_var('skappa'))

  elif quant == 'ldebyei':
    Zi2 = obj.get_charge(obj.ifluid)**2
    if Zi2 == 0:
      return obj.zero_at_mesh_center()
    const = obj.uni.permsi * obj.uni.ksi_b / obj.uni.qsi_electron**2
    tg = obj.get_var('tg')                     # [K]
    nr = obj.get_var('nr') * obj.uni.usi_nr    # [m^-3]
    ldebsi = np.sqrt(const * tg / (nr * Zi2))  # [m]
    return ldebsi / obj.uni.usi_l  # [simu. length units]

  elif quant == 'ldebye':
    # ldebye = 1/sum_j( (1/ldebye_j) for j in fluids and electrons)
    ldeb_inv_sum = 1/obj.get_var('ldebyei', mf_ispecies=-1)
    for fluid in obj.fluids.ions():
      ldeb_inv_sum += 1/obj.get_var('ldebyei', ifluid=fluid.SL)
    return 1/ldeb_inv_sum

  else:
    raise NotImplementedError(f'{repr(quant)} in get_mf_plasmaparam')


# default
_FUNDAMENTALS = ('r', 'px', 'py', 'pz', 'e', 'bx', 'by', 'bz')
_HD_Fs = ('part',   # part --> only get the internal part. e.g. nu1 * Cfast.
          *_FUNDAMENTALS)
_HD_QUANTS  = ['hd1_part', 'hd2_part']       # << without the factor of nu1, nu2
_HD_QUANTS += ['hd1_partnu', 'hd2_partnu']   # << include the factor of nu1, nu2
_HD_QUANTS += [f'hd3{x}_part' for x in AXES] + [f'hd3{x}_bpart' for x in AXES]       # << without the factor of nu3
_HD_QUANTS += [f'hd3{x}_partnu' for x in AXES] + [f'hd3{x}_bpartnu' for x in AXES]   # << include the factor of nu3
_HD_QUANTS += [f'hd{x}quench_{f}' for x in AXES for f in _FUNDAMENTALS]  # Q(∂f/∂x)
_HD_QUANTS += [f'hd{x}coeff_{f}' for x in AXES for f in _FUNDAMENTALS]   # nu dx (∂f/∂x) * Q(∂f/∂x)
_HD_QUANTS += [f'{d}hd{n}{x}_{f}' for d in ('', 'd')       # E.g. hd1x_r == hd1_part * nu dx (∂f/∂x) * Q(∂f/∂x)
                                  for n in (1,2,3)         # and dhd1x_r == ∂[hd1_part * nu dx (∂f/∂x) * Q(∂f/∂x)]/∂x
                                  for x in AXES
                                  for f in _FUNDAMENTALS]
_HYPERDIFFUSIVE_QUANT = ('HYPERDIFFUSIVE_QUANT', _HD_QUANTS)
# get value
@document_vars.quant_tracking_simple(_HYPERDIFFUSIVE_QUANT[0])
def get_hyperdiffusive_quant(obj, quant, HYPERDIFFUSIVE_QUANT=None):
  '''hyperdiffusive terms. All in simu units.'''
  if HYPERDIFFUSIVE_QUANT is None:
    HYPERDIFFUSIVE_QUANT = _HYPERDIFFUSIVE_QUANT[1]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, _HYPERDIFFUSIVE_QUANT[0], HYPERDIFFUSIVE_QUANT,
                                           get_hyperdiffusive_quant.__doc__, nfluid=1)
    docvar('hd1_part'  ,       'Cfast_i', uni=UNI_speed)
    docvar('hd1_partnu', 'nu1 * Cfast_i', uni=UNI_speed)
    docvar('hd2_part'  ,       '|ui|'   , uni=UNI_speed)
    docvar('hd2_partnu', 'nu2 * |ui|'   , uni=UNI_speed)
    for x in AXES:
      docvar(f'hd3{x}_part'   ,       f'd{x} * grad1{x}(ui{x})'         , uni=UNI_speed)
      docvar(f'hd3{x}_partnu' , f'nu3 * d{x} * grad1{x}(ui{x})'         , uni=UNI_speed)
      docvar(f'hd3{x}_bpart'  ,       f'd{x} * |grad1_perp_to_b(ui{x})|', uni=UNI_speed)
      docvar(f'hd3{x}_bpartnu', f'nu3 * d{x} * |grad1_perp_to_b(ui{x})|', uni=UNI_speed)
    for x in AXES:
      for f in _FUNDAMENTALS:
        docvar(f'hd{x}quench_{f}', f'Q(∂{f}/∂{x})', uni=DIMENSIONLESS)
        docvar(f'hd{x}coeff_{f}',  f'nu d{x} (∂{f}/∂{x}) * Q(∂{f}/∂{x})', uni=UNI.qc(0))
    for x in AXES:
      for n in (1,2,3):
        for f in _FUNDAMENTALS:
          if n==3 and f.startswith('b'):
            docvar(f'hd{n}{x}_{f}', f'nu{n} * hd{n}_bpart * hd{x}coeff_{f}', ) #uni=[TODO]
          else:
            docvar(f'hd{n}{x}_{f}', f'nu{n} * hd{n}_part * hd{x}coeff_{f}', ) #uni=[TODO]
          docvar(f'dhd{n}{x}_{f}', f'∂[hd{n}{x}_{f}]/∂{x}', )
    return None

  if quant not in HYPERDIFFUSIVE_QUANT:
    return None

  # nu1 term
  if quant == 'hd1_part':
    return obj('cfast')
  elif quant == 'hd1_partnu':
    return obj('hd1_part') * obj.get_param('nu1')

  # nu2 term
  elif quant == 'hd2_part':
    return obj('ui_mod')
  elif quant == 'hd2_partnu':
    return obj('hd2_part') * obj.get_param('nu2')

  # nu3 term
  elif quant.startswith('hd3') and quant in (f'hd3{x}_part' for x in AXES):
    x = quant[len('hd3')+0]  # 'x', 'y', or 'z'
    # dx * grad1x (uix)
    raise NotImplementedError(f'hd3{x}_part')

  elif quant.startswith('hd3') and quant in (f'hd3{x}_bpart' for x in AXES):
    x = quant[len('hd3')+0]  # 'x', 'y', or 'z'
    # dx * |grad1_perp_to_b(ui{x})|
    raise NotImplementedError(f'hd3{x}_bpart')

  elif quant.starstwith('hd3') and quant in (f'hd3{x}_{b}partnu' for x in AXES for b in ('', 'b')):
    part_without_nu = quant[:-len('nu')]
    return obj(part_without_nu) * obj.get_param('nu3')

  # quench term
  elif quant.startswith('hd') and quant.partition('_')[0] in (f'hd{x}quench' for x in AXES):
    base, _, f = quant.partition('_')
    x = base[len('hd')]  # 'x', 'y', or 'z'
    fval = obj(f)    # value of f, e.g. value of 'r' or 'bx'
    # Q(∂f/∂x)
    raise NotImplementedError(f'hd{x}quench_{f}')

  # coeff term
  elif quant.startswith('hd') and quant.partition('_')[0] in (f'hd{x}coeff' for x in AXES):
    base, _, f = quant.partition('_')
    x = base[len('hd')]  # 'x', 'y', or 'z'
    nu = NotImplemented  # << TODO
    dx = obj.dx          # << TODO (allow to vary in space)
    quench = obj(f'hd{x}quench_{f}')
    dfdx   = obj(f'd{f}dxdn')
    return nu * dx * dfdx * quench

  # full hd term
  elif quant.startswith('hd') and quant.partition('_')[0] in (f'hd{n}{x}' for x in AXES for n in (1,2,3)):
    base, _, f = quant.partition('_')
    n, x = base[2:4]
    nu = obj.get_param(f'nu{n}')
    if n==3 and f.startswith('b'):
      hd_part = obj('hd3_bpart')
    else:
      hd_part = obj(f'hd{n}_part')
    coeff = obj(f'hd{x}coeff_{f}')
    return nu * hd_part * coeff

  # full hd term, with derivative
  elif quant.startswith('dhd') and quant.partition('_')[0] in (f'dhd{n}{x}' for x in AXES for n in (1,2,3)):
    quant_str = quant[1:]
    return obj('d'+quant_str+'dxdn')


# default
_WAVE_QUANT = ('WAVE_QUANT',
               ['ci', 'fplasma', 'kmaxx', 'kmaxy', 'kmaxz']
              )
# get value
@document_vars.quant_tracking_simple(_WAVE_QUANT[0])
def get_mf_wavequant(obj, quant, WAVE_QUANT=None):
  '''quantities related most directly to waves in plasmas.'''
  if WAVE_QUANT is None:
    WAVE_QUANT = _WAVE_QUANT[1]

  if quant == '':
    docvar = document_vars.vars_documenter(obj, _WAVE_QUANT[0], WAVE_QUANT, get_mf_wavequant.__doc__)
    docvar('ci', "ion acoustic speed for ifluid (must be ionized) [simu. velocity units]",
                 nfluid=1, uni=UNI_speed)
    docvar('fplasma', "('angular') plasma frequency for ifluid (must be charged) [simu. frequency units]. " +\
                      "== sqrt(n_i q_i**2 / (epsilon_0 m_i))", nfluid=1, uni=UNI_hz)
    for x in AXES:
      docvar('kmax'+x, "maximum resolvable wavevector in "+x+" direction. Determined via 2*pi/obj.d"+x+"1d",
                       nfluid=0, uni=UNI_length)
    return None

  if quant not in _WAVE_QUANT[1]:
    return None

  if quant == 'ci':
    assert obj.mf_ispecies != -1, "ifluid {} must be ion to get ci, but got electron.".format(obj.ifluid)
    fluids = obj.fluids
    ion = fluids[obj.ifluid]
    assert ion.ionization >= 1, "ifluid {} is not ionized; cannot get ci (==ion acoustic speed).".format(obj.ifluid)
    # (we only want to get ion acoustic speed for ions; it doesn't make sense to get it for neutrals.)
    tg_i   = obj.get_var('tg')                  # [K] temperature of fluid
    tg_e   = obj.get_var('tg', mf_ispecies=-1)  # [K] temperature of electrons
    igamma = obj.uni.gamma        # gamma (ratio of specific heats) of fluid
    egamma = obj.uni.gamma        # gamma (ratio of specific heats) of electrons
    m_i    = obj.get_mass(ion, units='si')   # [kg] mass of ions
    ci     = np.sqrt(obj.uni.ksi_b * (ion.ionization * igamma * tg_i + egamma * tg_e) / m_i)
    ci_sim = ci / obj.uni.usi_u
    return ci_sim

  elif quant == 'fplasma':
    q    = obj.get_charge(obj.ifluid, units='si')
    assert q != 0, "ifluid {} must be charged to get fplasma.".format(obj.ifluid)
    m    = obj.get_mass(obj.ifluid, units='si')
    eps0 = obj.uni.permsi
    n    = obj('nr')
    unit = 1 / obj.uni.usi_hz   # convert from si frequency to ebysus frequency.
    consts = np.sqrt(q**2 / (eps0 * m)) * unit
    return consts * np.sqrt(n)    # [ebysus frequency units]

  elif quant in ['kmaxx', 'kmaxy', 'kmaxz']:
    x = quant[-1] # axis; 'x', 'y', or 'z'.
    xidx = dict(x=0, y=1, z=2)[x]  # axis; 0, 1, or 2.
    dx1d = getattr(obj, 'd'+x+'1d')  # 1D; needs dims to be added. add dims below.
    dx1d = np.expand_dims(dx1d, axis=tuple(set((0,1,2)) - set([xidx])))
    return (2 * np.pi / dx1d) + obj.zero()

  else:
    raise NotImplementedError(f'{repr(quant)} in get_mf_wavequant')


# default
_FB_INSTAB_QUANT = ['psi0', 'psii', 'vde', 'fb_ssi_vdtrigger', 'fb_ssi_possible',
                    'fb_ssi_freq', 'fb_ssi_growth_rate']
_FB_INSTAB_VECS  = ['fb_ssi_freq_max', 'fb_ssi_growth_rate_max', 'fb_ssi_growth_time_min']
_FB_INSTAB_QUANT += [v+x for v in _FB_INSTAB_VECS for x in AXES]
_FB_INSTAB_QUANT = ('FB_INSTAB_QUANT', _FB_INSTAB_QUANT)
# get value
@document_vars.quant_tracking_simple(_FB_INSTAB_QUANT[0])
def get_fb_instab_quant(obj, quant, FB_INSTAB_QUANT=None):
  '''very specific quantities which are related to the Farley-Buneman instability.'''
  if FB_INSTAB_QUANT is None:
    FB_INSTAB_QUANT = _FB_INSTAB_QUANT[1]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, _FB_INSTAB_QUANT[0], FB_INSTAB_QUANT, get_fb_instab_quant.__doc__)
    for psi in ['psi0', 'psii']:
      docvar(psi, 'psi_i when k_parallel==0. equals to: (kappa_e * kappa_i)^-1.', nfluid=1, uni=DIMENSIONLESS)
    docvar('vde', 'electron drift velocity. equals to: |E|/|B|. [simu. velocity units]', nfluid=0, uni=UNI_speed)
    docvar('fb_ssi_vdtrigger', 'minimum vde [in simu. velocity units] above which the FB instability can grow, ' +\
             'in the case of SSI (single-species-ion). We assume ifluid is the single ion species.', nfluid=1, uni=UNI_speed)
    docvar('fb_ssi_possible', 'whether SSI Farley Buneman instability can occur (vde > fb_ssi_vdtrigger). ' +\
             'returns an array of booleans, with "True" meaning "can occur at this point".', nfluid=1, uni=DIMENSIONLESS)
    docvar('fb_ssi_freq', 'SSI FB instability wave frequency (real part) divided by wavenumber (k). ' +\
             'assumes wavevector in E x B direction. == Vd / (1 + psi0). ' +\
             'result is in units of [simu. frequency * simu. length].', nfluid=2, uni=UNI_speed)
    docvar('fb_ssi_growth_rate', 'SSI FB instability growth rate divided by wavenumber (k) squared. ' +\
             'assumes wavevector in E x B direction. == (Vd^2/(1+psi0)^2 - Ci^2)/(nu_in*(1+1/psi0)). ' +\
             'result is in units of [simu. frequency * simu. length^2].', nfluid=2, uni=UNI_hz * UNI_length**2)
    for x in AXES:
      docvar('fb_ssi_freq_max'+x, 'SSI FB instability max frequency in '+x+' direction ' +\
               '[simu. frequency units]. calculated using fb_ssi_freq * kmax'+x, nfluid=2, uni=UNI_hz)
    for x in AXES:
      docvar('fb_ssi_growth_rate_max'+x, 'SSI FB instability max growth rate in '+x+' direction ' +\
               '[simu. frequency units]. calculated using fb_ssi_growth_rate * kmax'+x, nfluid=2, uni=UNI_hz)
    for x in AXES:
      docvar('fb_ssi_growth_time_min'+x, 'SSI FB instability min growth time in '+x+' direction ' +\
               '[simu. time units]. This is the amount of time it takes for the wave amplitude for the wave ' +\
               'with the largest wave vector to grow by a factor of e. == 1/fb_ssi_growth_rate_max'+x, nfluid=2, uni=UNI_time)

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

  elif quant in ['fb_ssi_freq_max'+x for x in AXES]:
    x = quant[-1]
    freq        = obj.get_var('fb_ssi_freq')
    kmaxx       = obj.get_var('kmax'+x)
    return kmaxx**2 * freq

  elif quant in ['fb_ssi_growth_rate_max'+x for x in AXES]:
    x = quant[-1]
    growth_rate = obj.get_var('fb_ssi_growth_rate')
    kmaxx       = obj.get_var('kmax'+x)
    return kmaxx**2 * growth_rate

  elif quant in ['fb_ssi_growth_time_min'+x for x in AXES]:
    x = quant[-1]
    return 1/obj.get_var('fb_ssi_growth_rate_max'+x)

  else:
    raise NotImplementedError(f'{repr(quant)} in get_fb_instab_quant')


# default
_THERMAL_INSTAB_QUANT  = ['thermal_growth_rate',
                          'thermal_freq', 'thermal_tan2xopt',
                          'thermal_xopt', 'thermal_xopt_rad', 'thermal_xopt_deg',
                          'ethermal_s0', 'ethermal_tan2xopt',
                          'ethermal_xopt', 'ethermal_xopt_rad', 'ethermal_xopt_deg',
                          ]
_THERMAL_INSTAB_VECS   = ['thermal_u0', 'thermal_v0']
_THERMAL_INSTAB_QUANT += [v+x for v in _THERMAL_INSTAB_VECS for x in AXES]
## add thermal_growth_rate with combinations of terms.
### LEGACY: assumes we are using optimal angle for ion thermal plus FB effects;
### the code implements a formula where the optimal angle is plugged in already.
### NON-LEGACY: allows to plug in the optimal angle.
_LEGACY_THERMAL_GROWRATE_QUANTS  = ['legacy_thermal_growth_rate' + x for x in ['', '_fb', '_thermal', '_damping']]
_LEGACY_THERMAL_GROWRATE_QUANTS += [quant+'_max' for quant in _LEGACY_THERMAL_GROWRATE_QUANTS]
_THERMAL_INSTAB_QUANT           += _LEGACY_THERMAL_GROWRATE_QUANTS

_THERMAL_GROWRATE_QUANTS  = ['thermal_growth_rate' + x for x in ['', '_fb', '_thermal', '_damping']]
_THERMAL_GROWRATE_QUANTS += [quant+'_max' for quant in _THERMAL_GROWRATE_QUANTS]
_THERMAL_INSTAB_QUANT    += _THERMAL_GROWRATE_QUANTS
_ETHERMAL_GROWRATE_QUANTS   = ['ethermal_growth_rate' + x for x in ['', '_fb', '_it', '_et', '_damping']]
_ETHERMAL_GROWRATE_QUANTS  += [quant+'_max' for quant in _ETHERMAL_GROWRATE_QUANTS]
_THERMAL_INSTAB_QUANT      += _ETHERMAL_GROWRATE_QUANTS

_THERMAL_INSTAB_QUANT = ('THERMAL_INSTAB_QUANT', _THERMAL_INSTAB_QUANT)
# get_value
@document_vars.quant_tracking_simple(_THERMAL_INSTAB_QUANT[0])
def get_thermal_instab_quant(obj, quant, THERMAL_INSTAB_QUANT=None):
  '''very specific quantities which are related to the ion thermal and/or electron thermal instabilities.
  For source of formulae, see paper by Dimant & Oppenheim, 2004.

  In general, ion ifluid --> calculate for ion thermal instability; electron fluid --> for electron thermal.
  Electron thermal is not yet implemented.

  Quantities which depend on two fluids expect ifluid to be ion or electron, and jfluid to be neutral.
  '''
  if THERMAL_INSTAB_QUANT is None:
    THERMAL_INSTAB_QUANT = _THERMAL_INSTAB_QUANT[1]

  if quant=='':
    docvar = document_vars.vars_documenter(obj, _THERMAL_INSTAB_QUANT[0], THERMAL_INSTAB_QUANT,
                                           get_thermal_instab_quant.__doc__, nfluid=1)
    # document ion thermal stuff. (Intrinsically coupled to FB and diffusion effects.)
    for thermal_xopt_rad in ['thermal_xopt', 'thermal_xopt_rad']:
      docvar(thermal_xopt_rad, 'thermal instability optimal angle between k and (Ve - Vi) to maximize growth.' +\
                'result will be in radians. Result will be between -pi/4 and pi/4.', nfluid=1,
                uni_f=UNITS_FACTOR_1, uni_name='radians')
    docvar('thermal_xopt_deg', 'thermal instability optimal angle between k and (Ve - Vi) to maximize growth.' +\
                'result will be in degrees. Result will be between -45 and 45.', nfluid=1,
                uni_f=UNITS_FACTOR_1, uni_name='degrees')
    docvar('thermal_tan2xopt', 'tangent of 2 times thermal_xopt', nfluid=1, uni=DIMENSIONLESS)
    for x in AXES:
      docvar('thermal_u0'+x, x+'-component of (Ve - Vi). Warning: proper interpolation not yet implemented.',
                            nfluid=1, uni=UNI_speed)
    for x in AXES:
      docvar('thermal_v0'+x, x+'-component of E x B / B^2. Warning: proper interpolation not yet implemented.',
                            nfluid=0, uni=UNI_speed)
    # document electron thermal stuff. (Intrinsically coupled to ion thermal, FB, and diffusion effects.)
    docvar('ethermal_s0', 'S0 = S / sin(2 * ethermal_xopt). (Used in calculated ethermal effects.)' +\
                          'ifluid must be ion; jfluid must be neutral.', nfluid=2, uni=DIMENSIONLESS)
    docvar('ethermal_tan2xopt', 'tangent of 2 times ethermal_xopt', nfluid=2, uni=DIMENSIONLESS)
    docvar('ethermal_xopt', 'ethermal instability optimal angle between k and (Ve - Vi) to maximize growth, ' +\
                            'when accounting for ion thermal, electron thermal, and Farley-Buneman effects. ' +\
                            'result will be in radians, and between -pi/4 and pi/4.', nfluid=2,
                            uni_f=UNITS_FACTOR_1, uni_name='radians')
    # document ion thermal growrate terms
    for growquant in _LEGACY_THERMAL_GROWRATE_QUANTS + _THERMAL_GROWRATE_QUANTS + _ETHERMAL_GROWRATE_QUANTS:
      # build docstring depending on growquant. final will be sGR + sMAX + sINC + sLEG.
      # determine if MAX.
      q, ismax, m = growquant.partition('_max')
      if m != '':
        continue # looks like '..._max_moretext'. Unrecognized format. Don't document.
      if ismax:
        sMAX = 'using wavenumber = 2*pi/(pixel width). result units are [simu. frequency].'
        units = UNI_hz
      else:
        sMAX = 'divided by wavenumber squared. result units are [(simu. frequency) * (simu. length)^2].'
        units = UNI_hz * UNI_length**2
      # split (take away the 'thermal_growth_rate' part).
      e, _, terms = q.partition('thermal_growth_rate')
      # determine if electron thermal is included.
      if e == 'e':
        sGR = 'Optimal growth rate for [electron thermal plus ion thermal plus Farley-Buneman] instability '
        nfluid = 2
      else:
        sGR = 'Optimal growth rate for [ion thermal plus Farley-Buneman] instability '
        nfluid = 1
      # determine which term or terms we are INCluding
      if terms == '':
        sINC = ''
      elif terms == '_fb':
        sINC = ' Includes only the FB term. (No thermal, no damping.)'
      elif terms == '_damping':
        sINC = ' Includes only the damping term. (No FB, no thermal.)'
      elif terms == '_thermal':  # (only available if electron thermal is NOT included)
        sINC = ' Includes only the ion thermal term. (No FB, no damping.)'
      elif terms == '_it':       # (only available if electron thermal IS included)
        sINC = ' Includes only the ion thermal term. (No FB, no damping, no electron thermal.)'
      elif terms == '_et':       # (only available if electron thermal IS included)
        sINC = ' Includes only the electron thermal term. (No FB, no damping, no ion thermal.)'
      # determine if LEGacy.
      if q.startswith('legacy'):
        sLEG = ' Calculated using formula where optimal angle has already been entered.'
      else:
        sLEG = ''
      # actually document.
      docvar(growquant, sGR + sMAX + sINC + sLEG, nfluid=nfluid, uni=units)
    return None

  #if quant not in THERMAL_INSTAB_QUANT:
  #  return None

  def check_fluids_ok(nfluid=1):
    '''checks that ifluid is ion and jfluid is neutral. Only checks up to nfluid. raise error if bad.'''
    if nfluid >=1:
      icharge = obj.get_charge(obj.ifluid)
      if icharge <= 0:
        raise ValueError('Expected ion ifluid for Thermal Instability quants, but got charge(ifluid)={}.'.format(icharge))
    if nfluid >=2:
      jcharge = obj.get_charge(obj.jfluid)
      if jcharge != 0:
        raise ValueError('Expected neutral jfluid but got charge(jfluid)={}.'.format(jcharge))
    return True

  if '_max' in quant:
    if 'thermal_growth_rate' in quant:
      q_without_max = quant.replace('_max', '')
      k2 = max(obj.get_kmax())**2   # units [simu. length^-2]. (cancels with the [simu. length^2] from non-"_max" quant.)
      result = obj.get_var(q_without_max)
      result *= k2
      return result

  elif quant.startswith('thermal_growth_rate') or quant.startswith('ethermal_growth_rate'):
    # determine included terms.
    if quant=='ethermal_growth_rate':
        include_terms = ['fb', 'it', 'et', 'damping']
    elif quant=='thermal_growth_rate':
      include_terms = ['fb', 'thermal', 'damping']
    else:
      include_terms = quant.split('_')[3:]
    # do prep work; calculate coefficient which is in front of all terms.
    psi    = obj.get_var('psi0')
    U02    = obj.get_var('thermal_u02')  # U_0^2
    nu_in  = obj.get_var('nu_sn')        # sum_{j for j in neutrals and j!=i} (nu_ij) 
    front_coeff = psi * U02 / ((1 + psi) * nu_in)   # leading coefficient (applies to all terms)
    # do prep work; calculate components which appear in multiple terms.
    def any_included(*check):
      return any((x in include_terms for x in check))
    if any_included('fb', 'thermal', 'it'):
      kappai = obj.get_var('kappa')
    if any_included('et', 'damping'):
      Cs     = obj.get_var('ci')
      Cs2_over_U02 = Cs**2 / U02
    if any_included('fb', 'thermal', 'it', 'et'):
      s_getX = 'ethermal_xopt' if quant.startswith('e') else 'thermal_xopt'
      X      = obj.get_var(s_getX)
      cosX   = np.cos(X)
      sinX   = np.sin(X)
    if any_included('et'):
      sin2X  = np.sin(2*X)
      S0     = obj.get_var('ethermal_s0')
    # calculating terms and returning result.
    result = obj.zero()
    if any_included('fb'):
      term_fb = (1 - kappai**2) * cosX**2 / (1 + psi)**2
      result += term_fb
    if any_included('thermal', 'it'):
      term_it = (2 / 3) * (kappai**2 * cosX**2 - kappai * cosX * sinX) / (1 + psi)
      result += term_it
    if any_included('et'):
      term_et = (2 / 3) * Cs2_over_U02 * (S0**2 * sin2X**2 - (17/5) * S0 * sin2X)
      result += term_et
    if any_included('damping'):
      term_dp = -1 * Cs2_over_U02
      result += term_dp
    result *= front_coeff
    return result

  elif quant.startswith('legacy_thermal_growth_rate'):
    if quant=='legacy_thermal_growth_rate':
      include_terms = ['fb', 'thermal', 'damping']
    else:
      include_terms = quant.split('_')[3:]
    # prep work
    psi    = obj.get_var('psi0')
    U02    = obj.get_var('thermal_u02')  # U_0^2
    nu_in  = obj.get_var('nu_sn')        # sum_{j for j in neutrals and j!=i} (nu_ij) 
    front_coeff = psi * U02 / ((1 + psi) * nu_in)   # leading coefficient (applies to all terms)
    if 'fb' in include_terms or 'thermal' in include_terms:
      # if calculating fb or thermal terms, need to know these values:
      ki2  = obj.get_var('kappa')**2     # kappa_i^2
      A    = (8 + (1 - ki2)**2 + 4 * psi * ki2)**(-1/2)
    # calculating terms
    result = obj.zero()
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
    return result

  elif quant in ['thermal_u0'+x for x in AXES]:
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

  elif quant in ['thermal_v0'+x for x in AXES]:
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

  # begin electron thermal quants
  elif quant == 'ethermal_s0':
    # S = S0 sin(2 * xopt).
    # S0 = (psi / (1+psi)) (gyro_i / (delta_en nu_en)) (V0^2 / C_s^2)
    check_fluids_ok(nfluid=2)
    psi      = obj.get_var('psi0')
    gyroi    = obj.get_var('gyrof')
    with obj.MaintainFluids():
      nu_en  = obj.get_var('nu_sn', iS=-1)
    m_n = obj.get_mass(obj.mf_jspecies, units='amu')
    m_e = obj.get_mass(-1, units='amu')
    delta_en = 2 * m_e / (m_e + m_n)
    V02      = obj.get_var('thermal_v02')  # V0^2
    Cs2      = obj.get_var('ci')**2        # Cs^2
    factor1 = psi / (1 + psi)
    factor2 = gyroi / (delta_en * nu_en)
    factor3 = V02 / Cs2
    return factor1 * factor2 * factor3

  elif quant == 'ethermal_tan2xopt':
    # optimal tan(2\chi)... see Sam's derivation (available upon request).
    # (Assumes |4 * thermal_s0 * sin(2 xopt)| << 34 / 5)
    check_fluids_ok(nfluid=1)
    S0     = obj.get_var('ethermal_s0')
    kappai = obj.get_var('kappa')
    psi    = obj.get_var('psi0')
    U02    = obj.get_var('thermal_u02')  # V0^2
    Cs2    = obj.get_var('ci')**2        # Cs^2

    c1 = - (1 - kappai**2)/(1 + psi)**2  -  2 * kappai**2 / (3 * (1 + psi))
    c2 = - 2 * kappai / (3 * (1 + psi))  -  (68 / 15) * (Cs2 / U02) * S0
    return - c2 / c1
  
  elif quant in ['ethermal_xopt', 'ethermal_xopt_rad']:
    #TODO: think about which results are being dropped because np.arctan is not multi-valued.
    return 0.5 * np.arctan(obj.get_var('ethermal_tan2xopt'))
