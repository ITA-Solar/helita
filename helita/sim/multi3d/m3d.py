import  m3d_classes as mc
import numpy as np
class m3d:

    """
    class for reading and handling multi3d output
    """

    ####################################################################

    def __init__(self,inputfile=None,directory=None,printinfo=True):
        """
        initializes object, default directory to look for files is ./
        default input options file name is multi3d.input
        """

        self.inputfile = None
        self.directory = None
        self.printinfo = None
        self.theinput  = None
        self.outnnu    = -1
        self.outff     = -1

        self.sp       = mc.Spectrum()
        self.geometry = mc.Geometry()
        self.atom     = mc.Atom()
        self.atmos    = mc.Atmos()
        self.d        = mc.Default()

        # set default values
        if inputfile == None:
            self.inputfile = "multi3d.input"
        else:
            self.inputfile = inputfile
        if directory == None:
            self.directory = "../run/output/"
        else:
            self.directory = directory
            if self.directory[-1] is not "/":
                self.directory+="/"
        self.printinfo = printinfo

    ####################################################################
    def readall(self):
        """
        reads multi3d.input file and all the out_* files
        """
        self.readinput()
        self.readpar()
        self.readnu()
        self.readn()
        self.readatmos()
        self.readrtq()


    ####################################################################

    def readinput(self):

        """
        reads input from self.inputfile into a dict.
        """

        fname = self.directory + self.inputfile

        try:
            lines = [line.strip() for line in open(fname)]
            if self.printinfo: print "reading " + fname
        except Exception as e:
            print e
            return

        list = []
        #Remove IDL comments ;
        for line in lines:
            head, sep, tail = line.partition(';')
            list.append(head)
        #Remove blank lines
        list = filter(None, list)

        #Create a dic with keys and values from the input file
        self.theinput = dict()

        for line in list:
            head,sep,tail = line.partition("=")
            tail = tail.strip()
            head = head.strip()

            #Checks which type the values are
            try:
                # integer
                int(tail)
                tail = int(tail)
            except:
                try:
                    # float
                    float(tail)
                    tail = float(tail)
                except:
                    # string, remove first and last token, which are quotes
                    tail=tail[1:-1]

            # special items, multiple float values in a string
            if(head == "muxout" or head == "muyout" or head == "muzout"):
                temp=[]
                for item in tail.split():
                    temp.append(float(item))
                    self.theinput[head] = temp
            else:
                # simple str
                self.theinput[head] = tail

        # set xn,ny,nz here as they are now known
        self.geometry.nx = self.theinput["nx"]
        self.geometry.ny = self.theinput["ny"]
        self.geometry.nz = self.theinput["nz"]

    ####################################################################

    def readpar(self):
        """
        reads the out_par file
        """

        import scipy.io as scio
        import numpy as np
        from collections import namedtuple

        fname = self.directory + "out_par"
        f = scio.FortranFile(fname, 'r')
        if self.printinfo: print "reading " + fname

        # geometry struct
        self.geometry.nmu = int(f.read_ints( dtype = np.int32))
        self.geometry.nx = int(f.read_ints( dtype = np.int32))
        self.geometry.ny = int(f.read_ints( dtype = np.int32))
        self.geometry.nz = int(f.read_ints( dtype = np.int32))

        self.geometry.x = f.read_reals( dtype = np.float64)
        self.geometry.y = f.read_reals( dtype = np.float64)
        self.geometry.z = f.read_reals( dtype = np.float64)

        self.geometry.mux = f.read_reals( dtype = np.float64)
        self.geometry.muy = f.read_reals( dtype = np.float64)
        self.geometry.muz = f.read_reals( dtype = np.float64)
        self.geometry.wmu = f.read_reals( dtype = np.float64)

        self.sp.nnu = int(f.read_ints( dtype = np.int32))
        self.sp.maxac = int(f.read_ints( dtype = np.int32))
        self.sp.maxal = int(f.read_ints( dtype = np.int32))

        self.sp.nu = f.read_reals( dtype = np.float64)
        self.sp.wnu = f.read_reals( dtype = np.float64)

        # next two need reform
        self.sp.ac = f.read_ints( dtype = np.int32)
        self.sp.al = f.read_ints( dtype = np.int32)
        self.sp.nac = f.read_ints( dtype = np.int32)
        self.sp.nal = f.read_ints( dtype = np.int32)

        # atom struct
        self.atom.nrad = int( f.read_ints( dtype = np.int32))
        self.atom.nrfix = int( f.read_ints( dtype = np.int32))
        self.atom.ncont = int( f.read_ints( dtype = np.int32))
        self.atom.nline = int( f.read_ints( dtype = np.int32))
        self.atom.nlevel = int( f.read_ints( dtype = np.int32))
        ss=[self.atom.nlevel, self.atom.nlevel]
        self.atom.id = (f.read_record( dtype='S20'))[0].strip()
        self.atom.crout = (f.read_record( dtype='S20'))[0].strip()
        self.atom.label = f.read_record( dtype='S20').tolist()
        self.atom.ion = f.read_ints( dtype = np.int32)
        self.atom.ilin = f.read_ints( dtype = np.int32).reshape(ss)
        self.atom.icon = f.read_ints( dtype = np.int32).reshape(ss)
        self.atom.abnd = f.read_reals( dtype = np.float64)[0]
        self.atom.awgt = f.read_reals( dtype = np.float64)[0]
        self.atom.ev = f.read_reals( dtype = np.float64)
        self.atom.g = f.read_reals( dtype = np.float64)

        self.sp.ac.resize( [self.sp.nnu, self.atom.ncont])
        self.sp.al.resize( [self.sp.nnu, self.atom.nline])

        # cont info
        self.cont = [ mc.Cont() for i in xrange(self.atom.ncont)]

        for c in self.cont:

            c.bf_type = f.read_record( dtype='S20')[0].strip()

            c.i      = int( f.read_ints( dtype = np.int32))
            c.j      = int( f.read_ints( dtype = np.int32))
            c.ntrans = int( f.read_ints( dtype = np.int32))
            c.nnu    = int( f.read_ints( dtype = np.int32))
            c.ired   = int( f.read_ints( dtype = np.int32))
            c.iblue  = int( f.read_ints( dtype = np.int32))

            c.nu0    = f.read_reals( dtype = np.float64)[0]
            c.numax  = f.read_reals( dtype = np.float64)[0]
            c.alpha0 = f.read_reals( dtype = np.float64)[0]
            c.alpha  = f.read_reals( dtype = np.float64)[0]

            c.nu     = f.read_reals( dtype = np.float64)
            c.wnu    = f.read_reals( dtype = np.float64)


        #line info
        self.line = [ mc.Line() for i in xrange(self.atom.nline)]
        for l in self.line:

            l.profile_type = f.read_record( dtype='S72')[0].strip()

            l.ga      = f.read_reals( dtype = np.float64)[0]
            l.gw      = f.read_reals( dtype = np.float64)[0]
            l.gq      = f.read_reals( dtype = np.float64)[0]
            l.lambda0 = f.read_reals( dtype = np.float64)[0]


            l.nu0     = f.read_reals( dtype = np.float64)[0]
            l.Aji     = f.read_reals( dtype = np.float64)[0]
            l.Bji     = f.read_reals( dtype = np.float64)[0]
            l.Bij     = f.read_reals( dtype = np.float64)[0]
            l.f       = f.read_reals( dtype = np.float64)[0]
            l.qmax    = f.read_reals( dtype = np.float64)[0]
            l.Grat    = f.read_reals( dtype = np.float64)[0]

            l.ntrans = int( f.read_ints( dtype = np.int32))
            l.j      = int( f.read_ints( dtype = np.int32))
            l.i      = int( f.read_ints( dtype = np.int32))
            l.nnu    = int( f.read_ints( dtype = np.int32))
            l.ired   = int( f.read_ints( dtype = np.int32))
            l.iblue  = int( f.read_ints( dtype = np.int32))

            l.nu  = f.read_reals( dtype = np.float64)
            l.q   = f.read_reals( dtype = np.float64)
            l.wnu = f.read_reals( dtype = np.float64)
            l.wq  = f.read_reals( dtype = np.float64)

        f.close()

    ####################################################################

    def readn(self):
        """
        reads populations as numpy memmap
        """

        import numpy as np

        if self.theinput is None:
            self.readinput()


        fname = self.directory + "out_pop"
        nlevel=self.atom.nlevel
        nx,ny,nz = self.geometry.nx, self.geometry.ny, self.geometry.nz

        gs=nx*ny*nz*nlevel*4

        self.atom.n = np.memmap(fname, dtype='float32', mode='r',
                      shape=(nx,ny,nz,nlevel),order='F')
        self.atom.nstar = np.memmap(fname, dtype='float32', mode='r',
                          shape=(nx,ny,nz,nlevel), offset=gs, order='F')
        
        self.atom.ntot = np.memmap(fname, dtype='float32', mode='r',
                         shape=(nx,ny,nz) ,offset=gs*2, order='F' )
        if self.printinfo: print "reading " + fname

    ####################################################################

    def readatmos(self):
        """
        reads atmosphere as numpy memmap
        """

        import numpy as np

        if self.theinput is None:
            self.readinput()

        fname = self.directory + "out_atm"
        nhl = 6
        nx,ny,nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        s=(nx,ny,nz)
        gs=nx*ny*nz*4

        self.atmos.ne  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s, order='F')
        self.atmos.tg  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s, offset=gs, order='F')
        self.atmos.vx  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s ,offset=gs*2, order='F' )
        self.atmos.vy  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s ,offset=gs*3, order='F' )
        self.atmos.vz  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s ,offset=gs*4, order='F' )
        self.atmos.rho = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s ,offset=gs*5, order='F' )
        self.atmos.nh  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=(nx,ny,nz,nhl) ,offset=gs*6, order='F' )
        #self.atmos.vturb = np.memmap(fname, dtype='float32', mode='r',
        #                             shape=s ,offset=gs*12, order='F' )

        if self.printinfo: print "reading " + fname

   ####################################################################

    def readrtq(self):
        """
        reads out_rtq as numpy memmap
        """

        import numpy as np

        if self.theinput is None:
            self.readinput()

        if self.sp is None:
            self.readpar()


        fname = self.directory + "out_rtq"
        nx,ny,nz = self.geometry.nx, self.geometry.ny, self.geometry.nz
        s=(nx,ny,nz)
        gs=nx*ny*nz*4

        self.atmos.x500  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s, order='F')
        self.atom.dopfac  = np.memmap(fname, dtype='float32', mode='r',
                                   shape=s, offset=gs, order='F')
        i=2
        for l in self.line:
            l.adamp = np.memmap(fname, dtype='float32', mode='r',
                                shape=s ,offset=gs*i, order='F' )
            i+=1

    ####################################################################

    def readnu(self):
        """
        reads the out_nu file
        """

        import scipy.io as scio
        import numpy as np

        fname = self.directory + "out_nu"
        f = scio.FortranFile(fname, 'r')
        if self.printinfo: print "reading " + fname

        self.outnnu = int(f.read_ints( dtype = np.int32))
        self.outff = f.read_ints( dtype = np.int32)

    ####################################################################

    def setdefault(self, i,j, fr=-1,ang=0):

        from m3d_util import cc, CM_TO_AA

        self.d.i = i-1
        self.d.j = j-1
        self.d.isline = self.atom.ilin[self.d.i, self.d.j] != 0
        self.d.iscont = self.atom.icon[self.d.i, self.d.j] != 0

        if self.d.isline:
            self.d.kr = self.atom.ilin[self.d.i, self.d.j] - 1
            self.d.nnu = self.line[self.d.kr].nnu
            self.d.nu = np.copy(self.line[self.d.kr].nu)
            self.d.l = cc / self.d.nu * CM_TO_AA
            self.d.dl = cc * CM_TO_AA * (1.0/self.d.nu - 1.0/self.line[self.d.kr].nu0)
            self.d.ired = self.line[self.d.kr].ired
        elif self.d.iscont:
            self.d.kr = self.atom.icon[self.d.i, self.d.j] - 1
            self.d.nnu = self.cont[self.d.kr].nnu
            self.d.nu = np.copy(self.cont[self.d.kr].nu)
            self.d.l = cc / self.d.nu * CM_TO_AA
            self.d.dl = None
            self.d.ired = self.cont[self.d.kr].ired
        else:
            print ('upper and lower level '+str(i)+','+str(j)+
            ' are not connected with a radiative transition.')
            return

        if fr == -1:
            self.d.ff=-1
        else:
            self.d.ff = self.d.ired+fr

        self.d.ang = ang

        print 'default values set to:'
        print ' i   =', self.d.i
        print ' j   =', self.d.j
        print ' kr  =', self.d.kr
        print ' ff  =', self.d.ff
        print ' ang =', self.d.ang

    ####################################################################

    def readvar(self, var, all=False):

        import os

        allowed_names=['chi','ie','jnu','zt1','st','xt','cf','snu',
                       'chi_c','scatt','therm']
        if not var in allowed_names:
            print "'"+var+"' is not an valid variable name."
            return -1

        mx = "{:+.2f}".format(self.theinput['muxout'][self.d.ang])
        my = "{:+.2f}".format(self.theinput['muyout'][self.d.ang])
        mz = "{:+.2f}".format(self.theinput['muzout'][self.d.ang])
        mus= '_' + mx + '_' + my + '_' + mz
        fname = self.directory + var + mus + '_allnu'
        if os.path.exists(fname):
            print 'reading from ' + fname
        else:
            print fname + ' does not exist'
            return -1

        sg=self.geometry

        if var in ('ie','zt1'):
            if all:
                shape = (sg.nx, sg.ny, self.outnnu)
                offset = 0
            elif self.d.ff == -1:
                shape = (sg.nx, sg.ny, self.d.nnu)
                offset = ( 4 * sg.nx * sg.ny
                           * np.where(self.outff == self.d.ired)[0] )
            else:
                shape = (sg.nx, sg.ny)
                offset = ( 4 * sg.nx * sg.ny
                           * np.where(self.outff == self.d.ff)[0] )

        else:

            if all:
                shape = (sg.nx, sg.ny, sg.nz, self.outnnu)
                offset = 0
            elif self.d.ff == -1:
                shape = (sg.nx, sg.ny, sg.nz, self.d.nnu)
                offset = ( 4 * sg.nx * sg.ny * sg.nz
                           * np.where(self.outff == self.d.ired)[0] )
            else:
                shape = (sg.nx, sg.ny, sg.nz)
                offset = ( 4 * sg.nx * sg.ny * sg.nz
                           * np.where(self.outff == self.d.ff)[0] )

        thedata = np.memmap(fname,dtype='float32', mode='r',
                        shape=shape,order='F',offset=offset)

        return thedata

    ####################################################################
