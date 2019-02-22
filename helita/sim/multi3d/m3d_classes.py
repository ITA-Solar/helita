class Geometry:
    """
    class def for geometry
    """

    def __init__(self):
        self.nx = -1
        self.ny = -1
        self.nz = -1
        self.nmu = -1
        self.x = None
        self.y = None
        self.z = None
        self.mux = None
        self.muy = None
        self.muz = None
        self.wmu= None

####################################################################

class Atom:
    """
    class def for atom
    """
    
    def __init__(self):
        self.nrad = -1
        self.nrfix = -1
        self.ncont = -1
        self.nline = -1
        self.nlevel = -1
        self.id = None
        self.crout = None
        self.label = None
        self.ion = None
        self.ilin = None
        self.icon = None
        self.abnd = -1e10
        self.awgt = -1e10
        self.ev = None
        self.g = None
        self.n = None
        self.nstar = None
        self.totn = None
        self.dopfac = None

class Atmos:
    """
    class def for atmos
    """

    def __init__(self):
        self.ne = None
        self.tg = None
        self.vx = None
        self.vy = None
        self.vz = None
        self.r = None
        self.nh = None
        self.vturb = None
        self.x500 = None

class Spectrum:
    """
    class def for spectrum
    """

    def __init__(self):
        self.nnu = -1
        self.maxal = -1
        self.maxac = -1
        self.nu = None
        self.wnu = None 
        self.ac = None 
        self.al = None 
        self.nac = None 
        self.nal = None

class Cont:
    """
    class def for continuum
    """
    def __init__(self):
        self.f_type = None
        self.j = -1
        self.i = -1
        self.nnu = -1
        self.ntrans = -1
        self.ired = -1
        self.iblue = -1
        self.nu0 = -1.0
        self.numax = -1.0
        self.alpha0 = -1.0
        self.alpha = None
        self.nu = None
        self.wnu = None

class Line:
    """
    class def for spectral line
    """

    def __init__(self):

        self.profile_type = None
        self.ga = -1.0
        self.gw = -1.0
        self.gq = -1.0
        self.lambda0 = -1.0
        self.nu0 = -1.0
        self.Aji = -1.0
        self.Bji = -1.0
        self.Bij = -1.0
        self.f = -1.0
        self.qmax = -1.0
        self.Grat = -1.0
        self.ntrans = -1
        self.j = -1
        self.i = -1
        self.nnu = -1
        self.ired = -1
        self.iblue = -1
        self.nu = None
        self.q = None
        self.wnu =None
        self.wq = None
        self.adamp = None       

class Default:
    """
    class to hold default transition info for IO
    """

    def __init__(self):
        self.i = -1
        self.j = -1
        self.isline = False
        self.iscont = False
        self.kr = -1
        self.nnu = -1
        self.nu = None
        self.l = None
        self.dl = None
        self.ired = -1
        self.ff = -1
        self.ang = -1
