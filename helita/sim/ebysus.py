#
# Set of programs to read and interact with output from Bifrost
#

import numpy as N
import os
import re
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii, subs2grph

class EbysusData(BifrostData):

    """
    Class to hold data from Multifluid/multispecies simulations
    in native format.
    """

    def __init__(self, file_root='qsmag-by00_t', meshfile=None, fdir='.',
                 verbose=True, dtype='f4', big_endian=False,*args, **kwargs):
        super(EbysusData, self).__init__(file_root, meshfile, fdir,verbose, dtype, big_endian)

        if kwargs.has_key('no_aux'):
            print "[no_aux argument is deprecated, you can skip it]"

        '''
        if not os.path.isfile(self.template +'.aux'):
            self.no_aux = False
        else:
            self.no_aux = True
        '''

        '''
        # endianness and data type
        if big_endian:
            self.dtype = '>' + dtype
        else:
            self.dtype = '<' + dtype
        '''

        # variables: lists and initialisation
        #self.compvars = ['rc','ux', 'uy', 'uz', 's', 'bxc', 'byc', 'bzc', 'rup', 'dxdbup', 'dxdbdn', 'dydbup', 'dydbdn', 'dzdbup', 'dzdbdn', 'modb', 'modp'] # composite variables

        '''
        if self.multifluid:
           if (self.params['mf_multifluid'] == 1):
              self.multifluid = True
           else:
              self.multifluid = False
        '''
        # Multifluid specific:
        #if self.multifluid:
        #  if self.verbose: print "[warning] : switching to multispiecies format!"

        self.variables = {}
        #else:
        #  self.init_vars()
        return

#------------------------------------------------------------------------
    def init_mf_load(self, snap, meshfile=None,
                 verbose=True,dtype='f4', **kwargs):
        ''' Main object for extracting Bifrost datacubes. '''

        self.snap     = snap
        self.snap_str = '_%03i' % snap

        if (snap >=0):
            self.templatesnap = self.file_root + self.snap_str
        else:
            self.templatesnap = self.file_root

        # read idl file
        self._BifrostData__read_params()
        # super(StaticEmRenderer, self).__init__(SERCUDACODE, snaprange, acont_filenames,
                                               #name_template, data_dir, snap)
        self.read_mf_params()
        # read mesh file
        if meshfile is None:
            if self.fdir.strip() == '':
                meshfile = self.params['meshfile'].strip()
            else:
                meshfile = self.fdir + '/' + self.params['meshfile'].strip()

        if not os.path.isfile(meshfile):
            if self.fdir.strip() == '':
                meshfile = 'mesh.dat'
            else:
                meshfile = self.fdir + '/' + 'mesh.dat'

        if not os.path.isfile(meshfile):
            print('[Warning] Mesh file %s does not exist' % meshfile)
        self._BifrostData__read_mesh(meshfile)

        self.nspecies_max = 28
        self.nlevels_max = 28

        self.mf_template = {} # empty dictionary
        self.mm_template = {} # empty dictionary
        self.mfe_template = {} # empty dictionary
        self.mfc_template = {} # empty dictionary
        tmp_basis, sep, tmp_ext = self.file_root.partition('%')
        self.mf_b_template = (tmp_basis + '_mf_common' + sep + tmp_ext) #% snap
        for mf_ilevel in range(self.nlevels_max):
            for mf_ispecies in range(self.nspecies_max):
                self.mf_template [(mf_ispecies,mf_ilevel)] = (tmp_basis + '_mf_%02i_%02i')  % (mf_ispecies+1, mf_ilevel+1)
                self.mm_template [(mf_ispecies,mf_ilevel)] = (tmp_basis + '_mm_%02i_%02i')  % (mf_ispecies+1, mf_ilevel+1)
                self.mfe_template[(mf_ispecies,mf_ilevel)] = (tmp_basis + '_mfe_%02i_%02i') % (mf_ispecies+1, mf_ilevel+1)
                self.mfc_template[(mf_ispecies,mf_ilevel)] = (tmp_basis + '_mfc_%02i_%02i') % (mf_ispecies+1, mf_ilevel+1)

        if (self.mf_epf==1):
            self.snapvars = ['r', 'px', 'py', 'pz', 'e']
        else:
            self.snapvars = ['r', 'px', 'py', 'pz']

        #self.multifluid = self.params.has_key('mf_multifluid')
        if self.do_mhd == 1:
            if (self.mf_epf==1):
                self.commvars = ['bx', 'by', 'bz']
            else:
                self.commvars = ['e','bx', 'by', 'bz']
        else:
            if (self.mf_epf==1):
                self.commvars = []
            else:
                self.commvars = ['e']
        self.compvars = ['ux', 'uy', 'uz', 's', 'bxc', 'byc', 'bzc', 'rup',
                     'dxdbup', 'dxdbdn', 'dydbup', 'dydbdn', 'dzdbup',
                     'dzdbdn', 'modb', 'modp']   # composite variables

        self.with_electrons = self.params.has_key('mf_electrons')
        tmp_basis, sep, tmp_ext = self.file_root.partition('%')
        self.snapevars = []
        if self.with_electrons:
            if (self.params['mf_electrons'] == 1):
              self.with_electrons = True
              self.mf_e_template = (tmp_basis + '_mf_e' + sep + tmp_ext)
              if (snap >= 0):
                self.mf_e_templatesnap = self.mf_e_template +'_%03i' % snap
              else:
                  self.mf_e_templatesnap = self.mf_e_template
              self.snapevars = ['r', 'px', 'py', 'pz','e']
            elif (self.mf_epf == 1):
                self.snapevars = ['e']
                self.mf_e_template = (tmp_basis + '_mf_e' + sep + tmp_ext)
                if (snap >= 0):
                    self.mf_e_templatesnap = self.mf_e_template +'_%03i' % snap
                else:
                    self.mf_e_templatesnap = self.mf_e_template
                self.with_electrons = False
        elif (self.mf_epf == 1):
            self.mf_e_template = (tmp_basis + '_mf_e' + sep + tmp_ext)
            if (snap >= 0):
                self.mf_e_templatesnap = self.mf_e_template +'_%03i' % snap
            else:
                self.mf_e_templatesnap = self.mf_e_template
            self.snapevars = ['e']

        if (snap >= 0):
            self.mf_b_templatesnap = self.mf_b_template +'_%03i' % snap
        else:
            self.mf_b_templatesnap = self.mf_b_template

        self.mf_templatesnap = {} # empty dictionary
        self.mm_templatesnap = {} # empty dictionary
        self.mfe_templatesnap = {} # empty dictionary
        self.mfc_templatesnap = {} # empty dictionary
        for mf_ilevel in range(self.nlevels_max):
            for mf_ispecies in range(self.nspecies_max):
                if (snap >= 0):
                    self.mf_templatesnap[(mf_ispecies,mf_ilevel)]  = self.mf_template[(mf_ispecies,mf_ilevel)] +'_%03i' % snap
                    self.mm_templatesnap[(mf_ispecies,mf_ilevel)]  = self.mm_template[(mf_ispecies,mf_ilevel)] +'_%03i' % snap
                    self.mfc_templatesnap[(mf_ispecies,mf_ilevel)] = self.mfc_template[(mf_ispecies,mf_ilevel)] +'_%03i' % snap
                    self.mfe_templatesnap[(mf_ispecies,mf_ilevel)] = self.mfe_template[(mf_ispecies,mf_ilevel)] +'_%03i' % snap
                else:
                    self.mf_templatesnap[(mf_ispecies,mf_ilevel)]  = self.mf_template[(mf_ispecies,mf_ilevel)]
                    self.mm_templatesnap[(mf_ispecies,mf_ilevel)]  = self.mm_template[(mf_ispecies,mf_ilevel)]
                    self.mfc_templatesnap[(mf_ispecies,mf_ilevel)] = self.mfc_template[(mf_ispecies,mf_ilevel)]
                    self.mfe_templatesnap[(mf_ispecies,mf_ilevel)] = self.mfe_template[(mf_ispecies,mf_ilevel)]

        return

    #------------------------------------------------------------------------

    def read_mf_params(self):
        ''' Reads parameter file (.idl) '''

        if (self.snap >=0):
            filename = self.templatesnap + '.idl'
        else:
            filename = self.templatesnap + '.idl.scr'
        self.params = read_idl_ascii(filename)

        # assign some parameters to root object


        try:
            self.mf_epf = self.params['mf_epf']
        except KeyError:
            raise KeyError('read_params: could not find mf_epf in idl file!')

        try:
            self.mf_total_nlevel = self.params['mf_total_nlevel']
        except KeyError:
            print('warning, this idl file does not include mf_total_nlevel')
            #raise KeyError('read_params: could not find mf_total_nlevel in idl file!')

        try:
            self.auxvars  = self.params['aux'].split()
            if self.verbose: print 'auxvars =',self.auxvars
             # special case for the ixy1 variable, lives in a separate file
            if 'ixy1' in self.auxvars:
                self.auxvars.remove('ixy1')
                self.auxxyvars.append('ixy1')
            # Remove for the var list the 2D aux vars, and stores these 2D varnames in vars2d.
            self.vars2d = []
            for var in self.auxvars:
                if any(i in var for i in ('xy','yz','xz')):
                    self.vars2d.append(var)
            for var in self.vars2d:
                self.auxvars.remove(var)
            # Remove for the var list the mfe aux vars, and stores these mfe varnames.
            self.varsmfe = []
            if (self.mf_epf == 1):
                for var in self.auxvars:
                    if any(i in var for i in ('mfe_','empty')):
                        self.varsmfe.append(var)
                for var in self.varsmfe:
                    self.auxvars.remove(var)
            # Remove for the var list the mfc aux vars, and stores these mfc varnames.
            self.varsmfc = []
            if (self.mf_epf == 1):
                for var in self.auxvars:
                    if any(i in var for i in ('mfc_','empty')):
                        self.varsmfc.append(var)
                for var in self.varsmfc:
                    self.auxvars.remove(var)
            # Remove for the var list the mf aux vars, and stores these mf varnames.
            self.varsmf = []
            if (self.mf_epf == 1):
                for var in self.auxvars:
                    if any(i in var for i in ('mf_','empty')):
                        self.varsmf.append(var)
                for var in self.varsmf:
                    self.auxvars.remove(var)
            self.varsmm = []
            if (self.mf_epf == 1):
                for var in self.auxvars:
                    if any(i in var for i in ('mm_','empty')):
                        self.varsmm.append(var)
                for var in self.varsmm:
                    self.auxvars.remove(var)

        except KeyError:
            self.auxvars = {}
            raise KeyError('read_params: could not find aux idl file!')

        return

    def clearattr(self):
        "cleans storage variables"
        vars = [self.compvars + self.auxvars + self.snapvars + self.commvars + self.varsmf + self.varsmm + self.varsmfc + self.varsmfe]
        for name in vars:
            if (hasattr(self,name)):
                delattr(self,name)

    #------------------------------------------------------------------------
    def getvar(self, var, snap, slice=None, order='F', mf_ilevel=0, mf_ispecies=0, mf_electrons=False):
        ''' Reads a given variable from the relevant files. '''
        import os

        if (hasattr(self,'isnap')) and (hasattr(self,'mf_ilevel')) and (hasattr(self,'mf_ispecies')):
            if (self.isnap != snap) or (self.mf_ilevel != mf_ilevel) or (self.mf_ispecies != mf_ispecies):
                self.clearattr()
                self.init_mf_load(snap)
        else:
            self.init_mf_load(snap)

        self.mf_ilevel=mf_ilevel
        self.mf_ispecies=mf_ispecies

        readingmm = False
        if var == 'x':
            return self.x
        elif var == 'y':
            return self.y
        elif var == 'z':
            return self.z

        if mf_electrons:
            if (not self.with_electrons): print '[Error] Run without electrons'
            template = self.mf_e_templatesnap
        else:
            b_test = (var in self.commvars)
            e_test = (var in self.snapevars and mf_ispecies == -1 and (self.mf_epf or self.with_electrons))
            if (b_test):
               template = self.mf_b_templatesnap
            elif (e_test):
               template = self.mf_e_templatesnap
            else:
                if (mf_ispecies >=0):
                    template = self.mf_templatesnap[mf_ispecies,mf_ilevel]
                    tmp_basis, sep, tmp_ext = self.file_root.partition('%')
                    if (snap >= 0):
                        auxtemplate = tmp_basis+'_%03i' % snap
                    else:
                        auxtemplate = tmp_basis

                    mmtemplate = self.mm_templatesnap[mf_ispecies,mf_ilevel]
                    mfetemplate = self.mfe_templatesnap[mf_ispecies,mf_ilevel]
                    mfctemplate = self.mfc_templatesnap[mf_ispecies,mf_ilevel]
                else: # this is for common or electron aux variables
                    tmp_basis, sep, tmp_ext = self.file_root.partition('%')
                    if (snap >= 0):
                        auxtemplate = tmp_basis+'_%03i' % snap
                    else:
                        auxtemplate = tmp_basis
            if self.verbose: print '[Looking for %s in %s]' % (var, template)

        # find in which file the variable is
        if var in self.compvars:
            # if variable is composite, use getcompvar
            return self._get_compvar(var,snap,slice,mf_ispecies=mf_ispecies, mf_ilevel=mf_ilevel)
        elif var in self.snapvars:
            if (snap >= 0):
                fsuffix = '.snap'
            else:
                fsuffix = '.snap.scr'
            if (e_test):
                if var in self.snapevars:
                    idx = self.snapevars.index(var)
                else:
                    raise ValueError('getvar: variable %s not available in electron snap file. Available vars:'
                          % (var) + '\n' + repr(self.snapevars))
            else:
                idx = self.snapvars.index(var)
            filename = template + fsuffix
        elif var in self.commvars:
            if (snap >= 0):
                fsuffix = '.snap'
            else:
                fsuffix = '.snap.scr'
            idx = self.commvars.index(var)
            filename = template + fsuffix
        elif var in self.auxvars:
            if (snap >= 0):
                fsuffix = '.aux'
            else:
                fsuffix = '.aux.scr'
            idx = self.auxvars.index(var)
            filename = auxtemplate + fsuffix
        elif var in self.varsmf:
            if (snap >= 0):
                fsuffix = '.aux'
            else:
                fsuffix = '.aux.scr'
            idx = self.varsmf.index(var)
            filename = template + fsuffix
        elif var in self.varsmm:
            if (snap >= 0):
                fsuffix = '.aux'
            else:
                fsuffix = '.aux.scr'
            idx = self.varsmm.index(var)
            filename = mmtemplate + fsuffix
            readingmm = True
        elif var in self.varsmfe:
            if (snap >= 0):
                fsuffix = '.aux'
            else:
                fsuffix = '.aux.scr'
            idx = self.varsmfe.index(var)
            filename = mfetemplate + fsuffix
        elif var in self.varsmfc:
            #print self.varsmfc,self.varsmf,self.varsmfe,self.varsmm
            if (snap >= 0):
                fsuffix = '.aux'
            else:
                fsuffix = '.aux.scr'
            idx = self.varsmfc.index(var)
            filename = mfctemplate + fsuffix
        else:
            raise ValueError('getvar: variable %s not available. Available vars:'
                  % (var) + '\n' + repr(self.snapvars + self.commvars + self.auxvars + self.varsmf + self.varsmm + self.varsmfe + self.varsmfc + self.compvars))

        # Now memmap the variable
        if not os.path.isfile(filename):
            raise IOError('getvar: variable %s should be in %s file, not found!' %
                            (var, filename))

        # size of the data type
        if self.dtype[1:] == 'f4':
            dsize = 4
        else:
            raise ValueError('getvar: datatype %s not supported' % self.dtype)


        if readingmm:
            offset = self.nx*self.ny*self.nz*idx*dsize*self.mf_total_nlevel
            if self.verbose: print('filename,offset',filename,offset)
            output = N.memmap(filename, dtype=self.dtype,order=order, offset=offset, mode='r', shape=(self.nx,self.ny,self.nz,self.mf_total_nlevel))
        else:
            offset = self.nx*self.ny*self.nz*idx*dsize
            if self.verbose: print('filename,offset',filename,offset)
            output = N.memmap(filename, dtype=self.dtype,order=order, offset=offset, mode='r', shape=(self.nx,self.ny,self.nz))

        setattr(self,str(var),output)
        return output

    #-----------------------------------------------------------------------

    def init_vars(self):
        ''' Memmaps aux and snap variables, and maps them to methods. '''

        self.variables = {}

        '''
        if self.no_aux:
            avaible_variables = self.snapvars
        else:
            # remove ixy1 if in var
            if 'ixy1' in self.auxvars:
                self.auxvars.remove('ixy1')
            avaible_variables = self.snapvars + self.auxvars
        '''
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
        avaible_variables = self.snapvars + self.auxvars +  self.commvars

        # snap variables
        for var in avaible_variables:
            self.variables[var] = self.getvar(var,(int(self.snap))
            setattr(self,var,self.variables[var])

        return
