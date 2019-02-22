import numpy as np
"""
physical constant definitions
"""

ee = 1.602189e-12  # electron volt [erg]
hh = 6.626176e-27  # Planck's constant [erg s]
cc = 2.99792458e10 # Speed of light [cm/s]
em = 9.109534e-28  # Electron mass [g] 
uu = 1.6605655e-24 # Atomic mass unit [g]
bk = 1.380662e-16  # Boltzman's cst. [erg/K]
pi = 3.14159265359 # Pi
ec = 4.80325e-10   # electron charge [statcoulomb]

hc2     = hh * cc**2
h_e     = hh / ee
hc_e    = hh * cc / ee
twoh_c2 = 2.0 * hh / cc**2
c2_2h   = cc**2 / 2.0 / hh
twohc   = 2.0 * hh * cc
hc_k    = hh * cc / bk 
e_k      = ee/bk
h_k     = hh / bk
h_4pi    = hh / 4.0 / pi

f_to_Aji = 8.0 * pi**2 * ec**2 / em / cc
KM_TO_CM =1e5
CM_TO_KM =1e-5
AA_TO_CM =1e-8
CM_TO_AA =1e8
grph = 2.380491e-24   

# bunch of utility function

####################################################################

def planck(tg,l):
    """
    Planck function, tg in K, l in Angstrom, Bnu in erg/(s cm2 Hz ster)
    """ 
    nu = cc/(l*AA_TO_CM)
    return  2.0 * hh * nu * (nu/cc)**2 / (np.exp((hh/bk)*(nu/tg))-1.0)

####################################################################

def trad(inu,l):
    """
    inverse Planck function, Inu in  erg/(s cm2 Hz ster), l in Angstrom
    output in K
    """
    ll = l*AA_TO_CM
    return (hh/bk) * (cc/ll) / np.log(2.0 * cc * hh / ll**3 / inu +1.0) 
