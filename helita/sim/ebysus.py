"""
Set of programs to read and interact with output from Multifluid/multispecies
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, Bifrost_units, Cross_sect
from .bifrost import read_idl_ascii, subs2grph
from . import cstagger
from at_tools import atom_tools as at
from .load_mf_quantities import *
from .load_quantities import *
from .load_arithmetic_quantities import *

class EbysusData(BifrostData):

    """
    Class to hold data from Multifluid/multispecies simulations
    in native format.
    """

    def __init__(self, *args,  **kwargs):

        super(EbysusData, self).__init__(*args, **kwargs)

        
        self.att = {}
        for ispecies in range(1,self.mf_nspecies+1):
            if (self.mf_nspecies == 1):
                self.att[ispecies]=at.Atom_tools(atom_file=self.mf_tabparam['SPECIES'][2],fdir=self.fdir)
            else:
                self.att[ispecies]=at.Atom_tools(atom_file=self.mf_tabparam['SPECIES'][ispecies-1][2],fdir=self.fdir)

    def _set_snapvars(self,firstime=False):

        if os.path.exists('%s.io' % self.file_root):
            self.snaprvars = ['r']
            self.snappvars = ['px', 'py', 'pz']
        else:
            self.snapvars = ['r', 'px', 'py', 'pz']

        self.snapevars = ['e']
        self.mhdvars = []
        if (self.do_mhd):
            self.mhdvars = ['bx', 'by', 'bz']
        self.auxvars = self.params['aux'][self.snapInd].split()

        self.compvars = ['ux', 'uy', 'uz', 's', 'ee']

        self.varsmfc = [v for v in self.auxvars if v.startswith('mfc_')]
        self.varsmf = [v for v in self.auxvars if v.startswith('mf_')]
        self.varsmm = [v for v in self.auxvars if v.startswith('mm_')]
        self.varsmfe = [v for v in self.auxvars if v.startswith('mfe_')]

        if (self.mf_epf):
            # add internal energy to basic snaps
            #self.snapvars.append('e')
            # make distiction between different aux variable
            self.mf_e_file = self.file_root + '_mf_e'
        else:  # one energy for all fluid
            self.mhdvars.insert(0, 'e')
            self.snapevars = []

        if hasattr(self, 'with_electrons'):
            if self.with_electrons:
                self.mf_e_file = self.file_root + '_mf_e'
                # JMS This must be implemented
                self.snapelvars=['r', 'px', 'py', 'pz', 'e']

        for var in (
                self.varsmfe +
                self.varsmfc +
                self.varsmf +
                self.varsmm):
            self.auxvars.remove(var)

        #if hasattr(self, 'mf_total_nlevel'):
        #    if self.mf_total_nlevel == 1:
        #        self.snapvars.append('e')

        if os.path.exists('%s.io' % self.file_root):
            self.simple_vars = self.snaprvars + self.snappvars + \
                self.snapevars + self.mhdvars + self.auxvars + \
                self.varsmf + self.varsmfe + self.varsmfc + self.varsmm
        else:
            self.simple_vars = self.snapvars + self.snapevars + \
                self.mhdvars + self.auxvars + self.varsmf + self.varsmfe + \
                self.varsmfc + self.varsmm

        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.auxxyvars.append('ixy1')

        for var in self.auxvars:
            if any(i in var for i in ('xy', 'yz', 'xz')):
                self.auxvars.remove(var)
                self.vars2d.append(var)

        '''self.compvars = ['ux', 'uy', 'uz', 's', 'rup', 'dxdbup', 'dxdbdn',
                            'dydbup', 'dydbdn', 'dzdbup', 'dzdbdn', 'modp']
        if (self.do_mhd):
            self.compvars = self.compvars + ['bxc', 'byc', 'bzc', 'modb']'''

    # def set_snap(self,snap):
    #     super(EbysusData, self).set_snap(snap)

    def _read_params(self,firstime=False):
        ''' Reads parameter file specific for Multi Fluid Bifrost '''
        super(EbysusData, self)._read_params()

        self.nspecies_max = 28
        self.nlevels_max = 28
        try:
            self.mf_epf = self.params['mf_epf'][self.snapInd]
        except KeyError:
            raise KeyError('read_params: could not find mf_epf in idl file!')
        try:
            self.mf_nspecies = self.params['mf_nspecies'][self.snapInd]
        except KeyError:
            raise KeyError('read_params: could not find mf_nspecies in idl file!')
        try:
            self.with_electrons = self.params['mf_electrons'][self.snapInd]
        except KeyError:
            raise KeyError(
                'read_params: could not find with_electrons in idl file!')
        try:
            self.mf_total_nlevel = self.params['mf_total_nlevel'][self.snapInd]
        except KeyError:
            print('warning, this idl file does not include mf_total_nlevel')
        try:
            filename = os.path.join(
                self.fdir, self.params['mf_param_file'][self.snapInd].strip())
            self.mf_tabparam = read_mftab_ascii(filename)
        except KeyError:
            print('warning, this idl file does not include mf_param_file')

    def _init_vars(self, firstime=False, fast=None, *args, **kwargs):
        """
        Initialises variables (common for all fluid)
        
        fast: None, True, or False.
            whether to only read density (and not all the other variables).
            if None, use self.fast instead.
        """
        fast = fast if fast is not None else self.fast
        if self._fast_skip_flag is True:
            return
        elif self._fast_skip_flag is False:
            self._fast_skip_flag = True #swaps flag to True, then runs the rest of the code (this time around).
        #else, fast_skip_flag is None, so the code should never be skipped.
        #as long as fast is False, fast_skip_flag should be None.

        self.mf_common_file = (self.root_name + '_mf_common')
        if os.path.exists('%s.io' % self.file_root):
            self.mfr_file = (self.root_name + '_mfr_%02i_%02i')
            self.mfp_file = (self.root_name + '_mfp_%02i_%02i')
        else:
            self.mf_file = (self.root_name + '_mf_%02i_%02i')
        self.mfe_file = (self.root_name + '_mfe_%02i_%02i')
        self.mfc_file = (self.root_name + '_mfc_%02i_%02i')
        self.mm_file = (self.root_name + '_mm_%02i_%02i')
        self.mf_e_file = (self.root_name + '_mf_e')
        self.aux_file = (self.root_name)

        self.variables = {}

        self.set_mfi(None, None)
        self.set_mfj(None, None)

        
        varlist = ['r'] if fast else self.simple_vars
        for var in varlist:
            try:
                self.variables[var] = self._get_simple_var(
                    var, self.mf_ispecies, self.mf_ilevel, *args, **kwargs)
                setattr(self, var, self.variables[var])
            except BaseException:
                if self.verbose:
                    if not (self.mf_ilevel == 1 and var in self.varsmfc):
                        if (firstime):
                            print(('(WWW) init_vars: could not read '
                                'variable %s' % var))

        rdt = self.r.dtype
        if (self.nz>1):
            cstagger.init_stagger(self.nz, self.dx, self.dy, self.z.astype(rdt),
                              self.zdn.astype(rdt), self.dzidzup.astype(rdt),
                              self.dzidzdn.astype(rdt))

    def set_mfi(self, mf_ispecies=None, mf_ilevel=None):
        """
        adds mf_ispecies and mf_ilevel attributes if they don't exist and
        changes mf_ispecies and mf_ilevel if needed. It will set defaults to 1
        """

        if (mf_ispecies is not None):
            if (mf_ispecies != self.mf_ispecies):
                self.mf_ispecies = mf_ispecies
            elif not hasattr(self, 'mf_ispecies'):
                self.mf_ispecies = 1
        elif not hasattr(self, 'mf_ispecies'):
            self.mf_ispecies = 1

        if (mf_ilevel is not None):
            if (mf_ilevel != self.mf_ilevel):
                self.mf_ilevel = mf_ilevel
            elif not hasattr(self, 'mf_ilevel'):
                self.mf_ilevel = 1
        elif not hasattr(self, 'mf_ilevel'):
            self.mf_ilevel = 1

    def set_mfj(self, mf_jspecies=None, mf_jlevel=None):
        """
        adds mf_ispecies and mf_ilevel attributes if they don't exist and
        changes mf_ispecies and mf_ilevel if needed. It will set defaults to 1
        """

        if (mf_jspecies is not None):
            if (mf_jspecies != self.mf_jspecies):
                self.mf_jspecies = mf_jspecies
            elif not hasattr(self, 'mf_jspecies'):
                self.mf_ispecies = 1
        elif not hasattr(self, 'mf_jspecies'):
            self.mf_jspecies = 1

        if (mf_jlevel is not None):
            if (mf_jlevel != self.mf_jlevel):
                self.mf_jlevel = mf_jlevel
            elif not hasattr(self, 'mf_jlevel'):
                self.mf_jlevel = 1
        elif not hasattr(self, 'mf_jlevel'):
            self.mf_jlevel = 1

    def get_var(self, var, snap=None, iix=slice(None), iiy=slice(None),
                iiz=slice(None), mf_ispecies=None, mf_ilevel=None,
                mf_jspecies=None, mf_jlevel=None, panic=False, *args, **kwargs):
        """
        Reads a given variable from the relevant files.

        Parameters
        ----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        mf_ispecies - integer [1, 28]
            Species ID
        mf_ilevel - integer
            Ionization level
        snap - integer, optional
            Snapshot number to read. By default reads the loaded snapshot;
            if a different number is requested, will load that snapshot
            by running self.set_snap(snap).
        """
        if var == '':
            print(help(self.get_var))

        if var in ['x', 'y', 'z']:
            return getattr(self, var)

        if var in self.varsmfc:
            if mf_ilevel is None and self.mf_ilevel == 1:
                mf_ilevel = 2
                print("Warning: mfc is only for ionized species,"
                      "Level changed to 2")
            if mf_ilevel == 1:
                mf_ilevel = 2
                print("Warning: mfc is only for ionized species."
                      " Level changed to 2")

        #if var not in self.snapevars:
        #    if (mf_ispecies is None):
        #        if self.mf_ispecies < 1:
        #            mf_ispecies = 1
        #            print("Warning: variable is only for electrons, "
        #                  "iSpecie changed to 1")
        #    elif (mf_ispecies < 1):
        #        mf_ispecies = 1
        #        print("Warning: variable is only for electrons, "
        #              "iSpecie changed to 1")

        if not hasattr(self, 'iix'):
            self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            self.set_domain_iiaxis(iinum=iiz, iiaxis='z')
            self.variables={}
        else:
            if (iix != slice(None)) and np.any(iix != self.iix):
                if self.verbose:
                    print('(get_var): iix ', iix, self.iix)
                self.set_domain_iiaxis(iinum=iix, iiaxis='x')
                self.variables={}
            if (iiy != slice(None)) and np.any(iiy != self.iiy):
                if self.verbose:
                    print('(get_var): iiy ', iiy, self.iiy)
                self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
                self.variables={}
            if (iiz != slice(None)) and np.any(iiz != self.iiz):
                if self.verbose:
                    print('(get_var): iiz ', iiz, self.iiz)
                self.set_domain_iiaxis(iinum=iiz, iiaxis='z')
                self.variables={}
        if self.cstagop and ((self.iix != slice(None)) or
                             (self.iiy != slice(None)) or
                             (self.iiz != slice(None))):
            self.cstagop = False
            print(
                'WARNING: cstagger use has been turned off,',
                'turn it back on with "dd.cstagop = True"')

        if ((snap is not None) and np.any(snap != self.snap)):
            self.set_snap(snap)
            self.variables={}

        if ((mf_ispecies is not None) and (mf_ispecies != self.mf_ispecies)):
            self.set_mfi(mf_ispecies, mf_ilevel)
            self.variables={}
        elif (( mf_ilevel is not None) and (mf_ilevel != self.mf_ilevel)):
            self.set_mfi(mf_ispecies, mf_ilevel)
            self.variables={}
        if ((mf_jspecies is not None) and (mf_jspecies != self.mf_jspecies)):
            self.set_mfj(mf_jspecies, mf_jlevel)
            self.variables={}
        elif (( mf_ilevel is not None) and (mf_jlevel != self.mf_jlevel)):
            self.set_mfj(mf_jspecies, mf_jlevel)
            self.variables={}

        # This should not be here because mf_ispecies < 0 is for electrons.
        #assert (self.mf_ispecies > 0 and self.mf_ispecies <= 28)

        # # check if already in memmory
        if var in self.variables:
            return self.variables[var]
        elif var in self.simple_vars:  # is variable already loaded?
            val = self._get_simple_var(var, self.mf_ispecies, self.mf_ilevel,
                                self.mf_jspecies, self.mf_jlevel,panic=panic)
        elif var in self.auxxyvars:
            val =  super(EbysusData, self)._get_simple_var_xy(var)
        elif var in self.compvars:
            val =  super(EbysusData, self)._get_composite_var(var)
        else:
            # Loading quantities
            val = load_quantities(self,var,PLASMA_QUANT='',
                        CYCL_RES='', COLFRE_QUANT='', COLFRI_QUANT='',
                        IONP_QUANT='', EOSTAB_QUANT='', TAU_QUANT='',
                        DEBYE_LN_QUANT='', CROSTAB_QUANT='',
                        COULOMB_COL_QUANT='', AMB_QUANT='')
            # Loading arithmetic quantities
            if np.shape(val) is ():
                val = load_arithmetic_quantities(self,var)
            if np.shape(val) is ():
                val =  load_mf_quantities(self,var)

        if var == '':
            print(help(self.get_var))
            print(self.description['ALL'])
            print(self.mf_description['ALL'])
            return None

        if np.shape(val) is ():
            raise ValueError(('get_var: do not know (yet) how to '
                          'calculate quantity %s. Note that simple_var '
                          'available variables are: %s.\nIn addition, '
                          'get_quantity can read others computed variables '
                          'see e.g. help(self.get_var) or get_var('')) for guidance'
                          '.' % (var, repr(self.simple_vars))))

        if np.shape(val) != (self.xLength, self.yLength, self.zLength):
            # at least one slice has more than one value
            if np.size(self.iix) + np.size(self.iiy) + np.size(self.iiz) > 3:
                # x axis may be squeezed out, axes for take()
                axes = [0, -2, -1]

                for counter, dim in enumerate(['iix', 'iiy', 'iiz']):
                    if (np.size(getattr(self, dim)) > 1 or
                            getattr(self, dim) != slice(None)):
                        # slicing each dimension in turn
                        val = val.take(getattr(self, dim), axis=axes[counter])
            else:
                # all of the slices are only one int or slice(None)
                val = val[self.iix, self.iiy, self.iiz]

            # ensuring that dimensions of size 1 are retained
            val = np.reshape(val, (self.xLength, self.yLength, self.zLength))

        return val

    def _get_simple_var(
            self,
            var,
            mf_ispecies=None,
            mf_ilevel=None,
            mf_jspecies=None,
            mf_jlevel=None,
            order='F',
            mode='r',
            panic=False, 
            *args,
            **kwargs):
        """
        Gets "simple" variable (ie, only memmap, not load into memory).

        Overloads super class to make a distinction between different
        filenames for different variables

        Parameters:
        -----------
        var - string
            Name of the variable to read. Must be Bifrost internal names.
        order - string, optional
            Must be either 'C' (C order) or 'F' (Fortran order, default).
        mode - string, optional
            numpy.memmap read mode. By default is read only ('r'), but
            you can use 'r+' to read and write. DO NOT USE 'w+'.

        Returns
        -------
        result - numpy.memmap array
            Requested variable.
        """
        if var == '':
            print(help(self._get_simple_var))

        if (np.size(self.snap) > 1):
            currSnap = self.snap[self.snapInd]
            currStr = self.snap_str[self.snapInd]
        else:
            currSnap = self.snap
            currStr = self.snap_str
        if currSnap < 0:
            filename = self.file_root
            fsuffix_b = '.scr'
            currStr = ''
        elif currSnap == 0:
            filename = self.file_root
            fsuffix_b = ''
            currStr = ''
        else:
            filename = self.file_root
            fsuffix_b = ''

        self.mf_arr_size = 1
        if os.path.exists('%s.io' % self.file_root):
            if (var in self.mhdvars and self.mf_ispecies > 0) or (
                    var in ['bx', 'by', 'bz']):
                idx = self.mhdvars.index(var)
                fsuffix_a = '.snap'
                dirvars = '%s.io/mf_common/' % self.file_root
                filename = self.mf_common_file
            elif var in self.snaprvars and self.mf_ispecies > 0:
                idx = self.snaprvars.index(var)
                fsuffix_a = '.snap'
                dirvars = '%s.io/mf_%02i_%02i/mfr/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mfr_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.snappvars and self.mf_ispecies > 0:
                idx = self.snappvars.index(var)
                fsuffix_a = '.snap'
                dirvars = '%s.io/mf_%02i_%02i/mfp/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mfp_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.snapevars and self.mf_ispecies > 0:
                idx = self.snapevars.index(var)
                fsuffix_a = '.snap'
                dirvars = '%s.io/mf_%02i_%02i/mfe/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mfe_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.snapevars and self.mf_ispecies < 0:
                idx = self.snapevars.index(var)
                filename = self.mf_e_file
                dirvars = '%s.io/mf_e/'% self.file_root
                fsuffix_a = '.snap'
            elif var in self.auxvars:
                idx = self.auxvars.index(var)
                fsuffix_a = '.aux'
                dirvars = '%s.io/mf_common/' % self.file_root
                filename = self.aux_file
            elif var in self.varsmf:
                idx = self.varsmf.index(var)
                fsuffix_a = '.aux'
                dirvars = '%s.io/mf_%02i_%02i/mfa/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mf_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.varsmm:
                idx = self.varsmm.index(var)
                fsuffix_a = '.aux'
                dirvars = '%s.io/mf_%02i_%02i/mm/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mm_file % (self.mf_ispecies, self.mf_ilevel)
                self.mf_arr_size = self.mf_total_nlevel
                jdx=0
                for ispecies in range(1,self.mf_nspecies+1):
                    nlevels=self.att[ispecies].params.nlevel
                    for ilevel in range(1,nlevels+1):
                        if (ispecies < self.mf_jspecies):
                            jdx += 1
                        elif ((ispecies == self.mf_jspecies) and (ilevel < self.mf_jlevel)):
                            jdx += 1
            elif var in self.varsmfe:
                idx = self.varsmfe.index(var)
                fsuffix_a = '.aux'
                dirvars = '%s.io/mf_%02i_%02i/mfe/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mfe_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.varsmfc:
                idx = self.varsmfc.index(var)
                fsuffix_a = '.aux'
                dirvars = '%s.io/mf_%02i_%02i/mfc/' % (self.file_root,
                        self.mf_ispecies, self.mf_ilevel)
                filename = self.mfc_file % (self.mf_ispecies, self.mf_ilevel)
        else:
            dirvars = ''
            if (var in self.mhdvars and self.mf_ispecies > 0) or (
                    var in ['bx', 'by', 'bz']):
                idx = self.mhdvars.index(var)
                fsuffix_a = '.snap'
                filename = self.mf_common_file
            elif var in self.snapvars and self.mf_ispecies > 0:
                idx = self.snapvars.index(var)
                fsuffix_a = '.snap'
                filename = self.mf_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.snapevars and self.mf_ispecies > 0:
                idx = self.snapevars.index(var)
                fsuffix_a = '.snap'
                filename = self.mfe_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.snapevars and self.mf_ispecies < 0:
                idx = self.snapevars.index(var)
                filename = self.mf_e_file
                fsuffix_a = '.snap'
            elif var in self.auxvars:
                idx = self.auxvars.index(var)
                fsuffix_a = '.aux'
                filename = self.aux_file
            elif var in self.varsmf:
                idx = self.varsmf.index(var)
                fsuffix_a = '.aux'
                filename = self.mf_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.varsmm:
                idx = self.varsmm.index(var)
                fsuffix_a = '.aux'
                filename = self.mm_file % (self.mf_ispecies, self.mf_ilevel)
                self.mf_arr_size = self.mf_total_nlevel
                jdx=0
                for ispecies in range(1,self.mf_nspecies+1):
                    nlevels=self.att[ispecies].params.nlevel
                    for ilevel in range(1,nlevels+1):
                        if (ispecies < self.mf_jspecies):
                            jdx += 1
                        elif ((ispecies == self.mf_jspecies) and (ilevel < self.mf_jlevel)):
                            jdx += 1

            elif var in self.varsmfe:
                idx = self.varsmfe.index(var)
                fsuffix_a = '.aux'
                filename = self.mfe_file % (self.mf_ispecies, self.mf_ilevel)
            elif var in self.varsmfc:
                idx = self.varsmfc.index(var)
                fsuffix_a = '.aux'
                filename = self.mfc_file % (self.mf_ispecies, self.mf_ilevel)

        if panic: 
            if fsuffix_a == '.aux': 
                filename = dirvars + filename  + fsuffix_a + '.panic' 
            else: 
                filename = dirvars + filename + '.panic'

        else: 
            filename = dirvars + filename + currStr + fsuffix_a + fsuffix_b

        '''if var not in self.mhdvars and not (var in self.snapevars and
            self.mf_ispecies < 0) and var not in self.auxvars :
            filename = filename % (self.mf_ispecies, self.mf_ilevel)'''

        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * self.nzb * idx * dsize * self.mf_arr_size
        if (self.mf_arr_size == 1):
            return np.memmap(
                filename,
                dtype=self.dtype,
                order=order,
                offset=offset,
                mode=mode,
                shape=(self.nx, self.ny, self.nzb))
        else:
            if var in  self.varsmm:
                offset += self.nx * self.ny * self.nzb * jdx * dsize
                return np.memmap(
                    filename,
                    dtype=self.dtype,
                    order=order,
                    offset=offset,
                    mode=mode,
                    shape=(self.nx, self.ny, self.nzb))
            else:
                return np.memmap(
                    filename,
                    dtype=self.dtype,
                    order=order,
                    offset=offset,
                    mode=mode,
                    shape=(self.nx, self.ny, self.nzb, self.mf_arr_size))

    def get_varTime(self, var, snap=None, iix=None, iiy=None, iiz=None,
                    mf_ispecies=None, mf_ilevel=None, mf_jspecies=None,
                    mf_jlevel=None,order='F',
                    mode='r', *args, **kwargs):
        """ Gets and returns the value of var over multiple snaps.

        snap should be a np.array of the snapshot numbers to read.

        return array will have dimensions of [*var_dimensions, len(snap)].
            (e.g. var.shape==[32,10,20], snap=[4,5,6], --> result is shape [32,10,20,3])
        """
        self.iix = iix
        self.iiy = iiy
        self.iiz = iiz

        try:
            if (snap is not None):
                if (np.size(snap) == np.size(self.snap)):
                    if (any(snap != self.snap)):
                        self.set_snap(snap)
                        self.variables={}
                else:
                    self.set_snap(snap)
                    self.variables={}
        except ValueError:
            print('WWW: snap has to be a numpy.arrange parameter')

        if var in self.varsmfc:
            if mf_ilevel is None and self.mf_ilevel == 1:
                mf_ilevel = 2
                self.variables={}
                print("Warning: mfc is only for ionized species,"
                      "Level changed to 2")
            if mf_ilevel == 1:
                mf_ilevel = 2
                self.variables={}
                print("Warning: mfc is only for ionized species."
                      "Level changed to 2")

        #if var not in self.snapevars:
        #    if (mf_ispecies is None):
        #        if self.mf_ispecies < 1:
        #            mf_ispecies = 1
        #            print("Warning: variable is only for electrons,"
        #                  "iSpecie changed to 1")
        #    elif (mf_ispecies < 1):
        #        mf_ispecies = 1
        #        print("Warning: variable is only for electrons,"
        #              "iSpecie changed to 1")

        if (((mf_ispecies is not None) and (
                mf_ispecies != self.mf_ispecies)) or ((
                mf_ilevel is not None) and (mf_ilevel != self.mf_ilevel))):
            self.set_mfi(mf_ispecies, mf_ilevel)
            self.variables={}
        if (((mf_jspecies is not None) and (
                mf_jspecies != self.mf_jspecies)) or ((
                mf_jlevel is not None) and (mf_jlevel != self.mf_jlevel))):
            self.set_mfj(mf_jspecies, mf_jlevel)
            self.variables={}
        # lengths for dimensions of return array
        self.xLength = 0
        self.yLength = 0
        self.zLength = 0

        for dim in ('iix', 'iiy', 'iiz'):
            if getattr(self, dim) is None:
                if dim[2] == 'z':
                    setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]+'b'))
                    setattr(self, dim, np.arange(0,getattr(self, 'n' + dim[2]+'b')))
                else:
                    setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]))
                    setattr(self, dim, np.arange(0,getattr(self, 'n' + dim[2])))
                setattr(self, dim, slice(None))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)

        snapLen = np.size(self.snap)
        value = np.empty([self.xLength, self.yLength, self.zLength, snapLen])

        for it in range(0, snapLen):
            self.snapInd = 0
            try:
                value[..., it] = self.get_var(var, snap=snap[it],
                    iix=self.iix, iiy=self.iiy, iiz=self.iiz,
                    mf_ispecies = self.mf_ispecies, mf_ilevel=self.mf_ilevel,
                    mf_jspecies = self.mf_jspecies, mf_jlevel=self.mf_jlevel)
            except:
                print("Error at not-fully-tested spot in get_varTime (~line 680 of helita/sim/ebysus.py).")
                print("Consider whether error may be related to not doing _init_vars & _set_snapvars.")
                raise

        return value

    def get_nspecies(self):
        return len(self.mf_tabparam['SPECIES'])

###########
#  TOOLS  #
###########


def write_mfr(rootname,inputdata,mf_ispecies,mf_ilevel):
    if mf_ispecies < 1:
        print('(WWW) species should start with 1')
    if mf_ilevel < 1:
        print('(WWW) levels should start with 1')
    directory = '%s.io/mf_%02i_%02i/mfr' % (rootname,mf_ispecies,mf_ilevel)
    nx, ny, nz = inputdata.shape
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.memmap(directory+'/%s_mfr_%02i_%02i.snap' % (rootname,mf_ispecies,mf_ilevel), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,1))
    data[...,0] = inputdata
    data.flush()

def write_mfp(rootname,inputdatax,inputdatay,inputdataz,mf_ispecies,mf_ilevel):
    if mf_ispecies < 1:
        print('(WWW) species should start with 1')
    if mf_ilevel < 1:
        print('(WWW) levels should start with 1')
    directory = '%s.io/mf_%02i_%02i/mfp' % (rootname,mf_ispecies,mf_ilevel)
    nx, ny, nz = inputdatax.shape
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.memmap(directory+'/%s_mfp_%02i_%02i.snap' % (rootname,mf_ispecies,mf_ilevel), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,3))
    data[...,0] = inputdatax
    data[...,1] = inputdatay
    data[...,2] = inputdataz
    data.flush()

def write_mfe(rootname,inputdata,mf_ispecies,mf_ilevel):
    if mf_ispecies < 1:
        print('(WWW) species should start with 1')
    if mf_ilevel < 1:
        print('(WWW) levels should start with 1')
    directory = '%s.io/mf_%02i_%02i/mfe' % (rootname,mf_ispecies,mf_ilevel)
    nx, ny, nz = inputdata.shape
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.memmap(directory+'/%s_mfe_%02i_%02i.snap' % (rootname,mf_ispecies,mf_ilevel), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,1))
    data[...,0] = inputdata
    data.flush()

def write_mf_common(rootname,inputdatax,inputdatay,inputdataz,inputdatae=None):
    directory = '%s.io/mf_common' % (rootname)
    nx, ny, nz = inputdatax.shape
    if not os.path.exists(directory):
        os.makedirs(directory)
    if np.any(inputdatae) == None:
        data = np.memmap(directory+'/%s_mf_common.snap' % (rootname), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,3))
        data[...,0] = inputdatax
        data[...,1] = inputdatay
        data[...,2] = inputdataz
    else:
        data = np.memmap(directory+'/%s_mf_common.snap' % (rootname), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,4))
        data[...,0] = inputdatae
        data[...,1] = inputdatax
        data[...,2] = inputdatay
        data[...,3] = inputdataz
    data.flush()

def write_mf_e(rootname,inputdata):
    directory = '%s.io/mf_e/' % (rootname)
    nx, ny, nz = inputdata.shape
    if not os.path.exists(directory):
        os.makedirs(directory)
    data = np.memmap(directory+'/%s_mf_e.snap' % (rootname), dtype='float32', mode='w+', order='f',shape=(nx,ny,nz,1))
    data[...,0] = inputdata
    data.flush()

def printi(fdir='./',rootname='',it=1):
    dd=EbysusData(rootname,fdir=fdir,verbose=False)
    nspecies=len(dd.mf_tabparam['SPECIES'])
    for ispecies in range(0,nspecies):
        aa=at.Atom_tools(atom_file=dd.mf_tabparam['SPECIES'][ispecies][2],fdir=fdir)
        nlevels=aa.params.nlevel
        print('reading %s'%dd.mf_tabparam['SPECIES'][ispecies][2])
        for ilevel in range(1,nlevels+1):
            print('ilv = %i'%ilevel)
            r=dd.get_var('r',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) * dd.params['u_r']
            print('dens=%6.2E,%6.2E g/cm3'%(np.min(r),np.max(r)))
            r=dd.get_var('nr',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) 
            print('ndens=%6.2E,%6.2E 1/cm3'%(np.min(r),np.max(r)))
            ux=dd.get_var('ux',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) * dd.params['u_u'] / 1e5
            print('ux=%6.2E,%6.2E km/s'%(np.min(ux),np.max(ux)))
            uy=dd.get_var('uy',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) * dd.params['u_u'] / 1e5
            print('uy=%6.2E,%6.2E km/s'%(np.min(uy),np.max(uy)))
            uz=dd.get_var('uz',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) * dd.params['u_u'] / 1e5
            print('uz=%6.2E,%6.2E km/s'%(np.min(uz),np.max(uz)))
            tg=dd.get_var('mfe_tg',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1)
            print('tg=%6.2E,%6.2E K'%(np.min(tg),np.max(tg)))
            ener=dd.get_var('e',it,mf_ilevel=ilevel,mf_ispecies=ispecies+1) * dd.params['u_e']
            print('e=%6.2E,%6.2E erg'%(np.min(ener),np.max(ener)))

    bx=dd.get_var('bx',it) * dd.params['u_b']
    print('bx=%6.2E,%6.2E G'%(np.min(bx),np.max(bx)))
    by=dd.get_var('by',it) * dd.params['u_b']
    print('by=%6.2E,%6.2E G'%(np.min(by),np.max(by)))
    bz=dd.get_var('bz',it) * dd.params['u_b']
    print('bz=%6.2E,%6.2E G'%(np.min(bz),np.max(bz)))
    va=dd.get_var('va',it) * dd.params['u_u'] / 1e5
    print('va=%6.2E,%6.2E km/s'%(np.min(va),np.max(va)))


def read_mftab_ascii(filename):
    '''
    Reads mf_tabparam.in-formatted (command style) ascii file into dictionary
    '''
    li = 0
    params = {}
    # go through the file, add stuff to dictionary
    with open(filename) as fp:
        for line in fp:
            # ignore empty lines and comments
            line = line.strip()
            if len(line) < 1:
                li += 1
                continue
            if line[0] == '#':
                li += 1
                continue
            line, sep, tail = line.partition('#')
            line = line.strip()
            line = line.split(';')[0].split('\t')
            # if (len(line) > 2):
            #  print(('(WWW) read_params: line %i is invalid, skipping' % li))
            #    li += 1
            #    continue
            if (np.size(line) == 1):
                key = line
                ii = 0
            # force lowercase because IDL is case-insensitive
            if (np.size(line) == 2):
                value = line[0].strip()
                text = line[1].strip().lower()
                try:
                    value = int(value)
                except BaseException:
                    print('(WWW) read_mftab_ascii: could not find datatype in'
                          'line %i, skipping' % li)
                    li += 1
                    continue
                if not (key[0] in params):
                    params[key[0]] = [value, text]
                else:
                    params[key[0]] = np.vstack((params[key[0]], [value, text]))
            if (np.size(line) == 3):
                value = line[0].strip()
                value2 = line[1].strip()
                text = line[2].strip()
                if key != 'species':
                    try:
                        value = int(value)
                    except BaseException:
                        print(
                            '(WWW) read_mftab_ascii: could not find datatype'
                            'in line %i, skipping' % li)
                else:
                    try:
                        value = int(value)
                        value2 = int(value2)
                    except BaseException:
                        print(
                            '(WWW) read_mftab_ascii: could not find datatype'
                            'in line %i, skipping' % li)
                    li += 1
                    continue
                if not (key[0] in params):
                    params[key[0]] = [value, value2, text]
                else:
                    params[key[0]] = np.vstack(
                        (params[key[0]], [value, value2, text]))
            if (np.size(line) > 3):
                # int type
                try:
                    arr = [int(numeric_string) for numeric_string in line]
                except BaseException:
                    print('(WWW) read_mftab_ascii: could not find datatype in'
                          'line %i, skipping' % li)
                    li += 1
                    continue
                if not (key[0] in params):
                    params[key[0]] = [arr]
                else:
                    params[key[0]] = np.vstack((params[key[0]], [arr]))
            li += 1
    return params

def write_idlparamsfile(snapname,mx=1,my=1,mz=1):
    '''Write default .idl file'''
    default_idl=[
     '; ************************* From   params ************************* \n',
     '             mx =         {}                                        \n'.format(mx),
     '             my =         {}                                        \n'.format(my),
     '             mz =         {}                                        \n'.format(mz),
     '             mb =         5                                         \n',
     '          nstep =        10                                         \n',
     '     nstepstart =         0                                         \n',
     '          debug =         0                                         \n',
     '       time_lim = -1.000E+00                                        \n',
     '          tstop = -1.00000000E+00                                   \n',
     'mf_total_nlevel =         5                                         \n',
     '   mf_electrons =    0                                              \n',
     '        mf_epf =    1                                               \n',
     '   mf_nspecies =         2                                          \n',
     ' mf_param_file = "mf_params.in"                                     \n',
     '; ************************* From parallel ************************* \n',
     '    periodic_x =    1                                               \n',
     '    periodic_y =    1                                               \n',
     '    periodic_z =    0                                               \n',
     '          ndim =    3                                               \n',
     '       reorder =    1                                               \n',
     '; ************************* From    units ************************* \n',
     '           u_l =  1.000E+08                                         \n',
     '           u_t =  1.000E+02                                         \n',
     '           u_r =  1.000E-07                                         \n',
     '           u_p =  1.000E+05                                         \n',
     '           u_u =  1.000E+06                                         \n',
     '          u_kr =  1.000E-01                                         \n',
     '          u_ee =  1.000E+12                                         \n',
     '           u_e =  1.000E+05                                         \n',
     '          u_te =  1.000E+11                                         \n',
     '          u_tg =  1.212E+04                                         \n',
     '           u_B =  1.121E+03                                         \n',
     '; ************************* From  stagger ************************* \n,'
     '      meshfile =             "{}.mesh"                     \n'.format(snapname),
     '            dx =  1.000E+00                                         \n',
     '            dy =  1.000E+00                                         \n',
     '            dz =  2.993E-02                                         \n',
     '; ************************* From timestep ************************* \n',
     '           Cdt =  0.030                                             \n',
     '            dt =  1.e-11                                            \n',
     '             t =  0.0                                               \n',
     ' timestepdebug =    0                                               \n',
     '; ************************* From      mhd ************************* \n',
     '           nu1 =  0.100                                             \n',
     '           nu2 =  0.300                                             \n',
     '           nu3 =  0.800                                             \n',
     '          nu_r =  0.100                                             \n',
     '        nu_r_z =  9.990E+02                                         \n',
     '       nu_r_mz =  0.100                                             \n',
     '         nu_ee =  0.100                                             \n',
     '       nu_ee_z =  9.990E+02                                         \n',
     '      nu_ee_mz =  0.100                                             \n',
     '       nu_e_ee =  0.000                                             \n',
     '     nu_e_ee_z =  9.990E+02                                         \n',
     '    nu_e_ee_mz =  0.000                                             \n',
     '   symmetric_e =    0                                               \n',
     '   symmetric_b =    0                                               \n',
     '          grav = -2.740                                             \n',
     '          eta3 =  3.000E-01                                         \n',
     '        ca_max =  0.000E+00                                         \n',
     '      mhddebug =    0                                               \n',
     '        do_mhd =    1                                               \n',
     '      mhdclean =        -1                                          \n',
     '   mhdclean_ub =    0                                               \n',
     '   mhdclean_lb =    0                                               \n',
     '  mhdclean_ubx =    0                                               \n',
     '  mhdclean_lbx =    0                                               \n',
     '  mhdclean_uby =    0                                               \n',
     '  mhdclean_lby =    0                                               \n',
     '    do_e_joule =    1                                               \n',
     '  do_ion_joule =    1                                               \n',
     '          nue1 =  0.050                                             \n',
     '          nue2 =  0.100                                             \n',
     '          nue3 =  0.050                                             \n',
     '          nue4 =  0.000                                             \n',
     '; ************************* From       io ************************* \n',
     '      one_file =    0                                               \n',
     '      snapname =                  "{}"                     \n'.format(snapname),
     '         isnap =         0                                          \n',
     '  large_memory =    1                                               \n',
     '         nsnap = 100000000                                          \n',
     '          nscr =       250                                          \n',
     '           aux = " nel mfe_tg etg "                                 \n',
     '        dtsnap =  5.000E-09                                         \n',
     '        newaux =    0                                               \n',
     '    rereadpars =   1000000                                          \n',
     '         dtscr =  1.000E+04                                         \n',
     '         tsnap =  0.0                                               \n',
     '          tscr =  0.00000000E+00                                    \n',
     '   boundarychk =    0                                               \n',
     '   print_stats =    0                                               \n',
     '; ************************* From     math ************************* \n',
     '         max_r =    5                                               \n',
     '      smooth_r =    3                                               \n',
     '   divhc_niter = 1000                                               \n',
     '     divhc_cfl =  0.400                                             \n',
     '       divhc_r =  0.180                                             \n',
     '     divhc_vxr =  0.000                                             \n',
     '     divhc_vyr =  0.000                                             \n',
     '     divhc_vzr =  0.950                                             \n',
     '     divhc_tol =  1.000E-05                                         \n',
     '; ************************* From   quench ************************* \n',
     '          qmax =  8.000                                             \n',
     '; ************************* From      eos ************************* \n',
     '         gamma =  1.667                                             \n',
     '      eosdebug =    0                                               \n',
     '; ************************* From     collisions utils ************* \n',
     '        do_col =    0                                               \n',
     '     col_debug =    0                                               \n',
     '       do_qcol =    1                                               \n',
     '       do_ecol =    0                                               \n',
     'col_calc_nu_in =    1                                               \n',
     'col_const_nu_in = -1.000E+03                                        \n',
     '   col_cnu_max =  1.000E+03                                         \n',
     '     col_utiny = -1.000E-05                                         \n',
     'col_trans_tim0 =  0.000E+00                                         \n',
     '  col_trans_dt =  1.000E+00                                         \n',
     'col_trans_ampl =  1.000E-10                                         \n',
     '     col_tabin = "mf_coltab.in"                                     \n',
     '; ************************* From          collisions  ************* \n',
     '    qcol_method = "expl"                                            \n',
     'col_matrix_norm =    0                                              \n',
     '; ************************* From              ionrec  ************* \n',
      '   qri_method = "impl"                                             \n',
     '; ************************* From   mf_recion (utils)  ************* \n',
     '     do_recion =    0                                               \n',
     '  recion_debug =    0                                               \n',
     '     calc_freq =    1                                               \n',
     '     three_bdy =    1                                               \n',
     '    const_fion = -1.000E+00                                         \n',
     '    const_frec = -1.000E+00                                         \n',
     '  recion_tabin = "mf_reciontab.in"                                  \n',
     'recion_modname = "atomic"                                           \n',
     '; ************************* From     hall ************************* \n',
     '       do_hall = "false"                                            \n',
     '    tstep_hall = "ntsv"                                             \n',
     '     eta_hallo =  1.000E+00                                         \n',
     '     eta4_hall = [ 0.100,  0.100,  0.100 ]                          \n',
     'mts_max_n_hall =   10                                               \n',
     '; ************************* From Bierman  ************************* \n',
     '    do_battery =    0                                               \n',
     '       bb_bato =  1.000E+00                                         \n',
     'bb_extdyn_time = -1.000E+00                                         \n',
     '     bb_ext_bb =  0.000E+00                                         \n',
     'bb_debug_battery =    0                                             \n',
     '       do_qbat =    0                                               \n',
     '; ************************* From            ohm_ecol  ************* \n',
     '   do_ohm_ecol =    0                                               \n',
     '       do_qohm =    1                                               \n',
     'ec_ohm_ecoll_debug =    0                                           \n',
     ' ec_calc_nu_en =    1                                               \n',
     ' ec_calc_nu_ei =    1                                               \n',
     'ec_const_nu_en = -1.000E+00                                         \n',
     'ec_const_nu_ei = -1.000E+00                                         \n',
     '      ec_tabin = "mf_ecoltab.in"                                    \n',
     'mf_eparam_file = "mf_eparams.in"                                    \n',
     '; ************************* From  spitzer ************************* \n',
     '       spitzer = "impl"                                             \n',
     ' debug_spitzer =    0                                               \n',
     '  info_spitzer =    0                                               \n',
     '   spitzer_amp =  0.000                                             \n',
     '      theta_mg =  0.900                                             \n',
     '        dtgerr =  1.000E-05                                         \n',
     '      ntest_mg =         1                                          \n',
     '          tgb0 =  0.000E+00                                         \n',
     '          tgb1 =  0.000E+00                                         \n',
     '        tau_tg =  1.000E+00                                         \n',
     '   fix_grad_tg =    1                                               \n',
     '   niter_mg = [   2,    5,    5,    5,   30 ]                       \n',
     '          bmin =  1.000E-04                                         \n',
     '       kappaq0 =  0.000E+00                                         \n',
     '; ************************* From   genrad ************************* \n',
     '     do_genrad =    1                                               \n',
     '    genradfile =                  "qthresh.dat"                     \n',
     '  debug_genrad =    0                                               \n',
     ' incrad_detail =    0                                               \n',
     '   incrad_quad =    3                                               \n',
     '      dtincrad =  1.000E-03                                         \n',
     '  dtincrad_lya =  1.000E-04                                         \n',
     '  debug_incrad =    0                                               \n',
     '; ************************* From         ue_electric  ************* \n',
     'do_ue_electric =    1                                               \n',
     'ue_electric_debug =    0                                            \n',
     'ue_fudge_mass =  1.000E+00                                          \n',
     '       ue_incr =  0.000                                             \n',
     '     ue_dt_inc = -1.000E+00                                         \n',
     '         ue_nu = [ 0.000,  0.000,  0.000,  0.000,  0.000 ]          \n',
     '      eionsfrz =    1                                               \n',
     '; ************************* From   bc_lowerx_magnetic ************* \n',
     '  bctypelowerx = "mcccc"                                            \n',
     '     bcldebugx =    0                                               \n',
     '  nextrap_bclx =         1                                          \n',
     '  nsmooth_bclx =         0                                          \n',
     'nsmoothbyz_bcl =         0                                          \n',
     '; ************************* From   bc_upperx_magnetic ************* \n',
     ' bctypeupperx = "mcccc"                                             \n',
     '     bcudebugx =    0                                               \n',
     '  nextrap_bcux =         1                                          \n',
     '  nsmooth_bcux =         0                                          \n',
     'nsmoothbyz_bcu =         0                                          \n',
     '; ************************* From   bc_lowery_magnetic ************* \n',
     ' bctypelowery = "mcccc"                                             \n',
     '     bcldebugy =    0                                               \n',
     '  nextrap_bcly =         1                                          \n',
     '  nsmooth_bcly =         0                                          \n',
     'nsmoothbxz_bcl =         0                                          \n',
     '; ************************* From   bc_uppery_magnetic ************* \n',
     ' bctypeuppery = "mcccc"                                             \n',
     '     bcudebugy =    0                                               \n',
     '  nextrap_bcuy =         1                                          \n',
     '  nsmooth_bcuy =         0                                          \n',
     'nsmoothbxz_bcu =         0                                          \n',
     '; ************************* From   bc_lowerz_magnetic ************* \n',
     '  bctypelowerz = "mesec"                                            \n',
     '     bcldebugz =    0                                               \n',
     '  nextrap_bclz =         1                                          \n',
     '  nsmooth_bclz =         0                                          \n',
     'nsmoothbxy_bcl =         0                                          \n',
     '; ************************* From   bc_upperz_magnetic ************* \n',
     '  bctypeupperz = "mesec"                                            \n',
     '     bcudebugz =    0                                               \n',
     '  nextrap_bcuz =         1                                          \n',
     '  nsmooth_bcuz =         0                                          \n',
     'nsmoothbxy_bcu =         0                                          \n'
          ]
    out=open('{}.idl'.format(snapname),'w')
    out.writelines(default_idl)
    return
      
def keyword_update(inoutfile,new_values):
   ''' Updates a given number of fields with values on a snapname.idl file.
       These are given in a dictionary: fvalues = {field: value}.
       Reads from snapname.idl and writes back into the same file.'''
   lines = list()
   with open(inoutfile) as f:
     for line in f.readlines():
       if line[0] == '#' or line[0] == ';':
         continue
       elif line.find('=') < 0:
         continue
       else:
         ss = line.split('=')[0]
         ssv = ss.strip().lower()
         if ssv in list(new_values.keys()):
           line = '{} = {} \n'.format(ss,str(new_values[ssv]))
       lines.append(line)
       
   with open(inoutfile,"w") as f:
     f.writelines(lines)
      

def write_mftab_ascii(filename, NSPECIES_MAX=28,
                      SPECIES=None, EOS_TABLES=None, REC_TABLES=None,
                      ION_TABLES=None, CROSS_SECTIONS_TABLES=None,
                      CROSS_SECTIONS_TABLES_I=None,
                      CROSS_SECTIONS_TABLES_N=None,
                      collist=np.linspace(1,
                                          28,
                                          28)):
    '''
    Writes mf_tabparam.in

        Parameters
        ----------
        filename - string
            Name of the file to write.
        NSPECIES_MAX - integer [28], maximum # of species
        SPECIES - list of strings containing the name of the atom files
        EOS_TABLES - list of strings containing the name of the eos
                    tables (no use)
        REC_TABLES - list of strings containing the name of the rec
                    tables (no use)
        ION_TABLES - list of strings containing the name of the ion
                    tables (no use)
        CROSS_SECTIONS_TABLES - list of strings containing the name of the
                    cross section files from VK between ion and neutrals
        CROSS_SECTIONS_TABLES_I - list of strings containing the name of the
                    cross section files from VK between ions
        CROSS_SECTIONS_TABLES_N - list of strings containing the name of the
                    cross section files from VK  between ions
        collist - integer vector of the species used.
                e.g., collist = [1,2,3] will include the H, He and Li

    '''

    if SPECIES is None:
        SPECIES=['H_2.atom', 'He_2.atom']
    if EOS_TABLES is None:
        EOS_TABLES=['H_EOS.dat', 'He_EOS.dat']
    if REC_TABLES is None:
        REC_TABLES=['h_rec.dat', 'he_rec.dat']
    if ION_TABLES is None:
        ION_TABLES=['h_ion.dat', 'he_ion.dat']
    if CROSS_SECTIONS_TABLES is None:
        CROSS_SECTIONS_TABLES=[[1, 1, 'p-H-elast.txt'],
                               [1, 2, 'p-He.txt'],
                               [2, 2, 'He-He.txt']]
    if CROSS_SECTIONS_TABLES_I is None:
        CROSS_SECTIONS_TABLES_I=[]
    if CROSS_SECTIONS_TABLES_N is None:
        CROSS_SECTIONS_TABLES_N=[]

    params = [
        'NSPECIES_MAX',
        'SPECIES',
        'EOS_TABLES',
        'REC_TABLES',
        'ION_TABLES',
        'COLISIONS_TABLES',
        'CROSS_SECTIONS_TABLES',
        'COLISIONS_MAP',
        'COLISIONS_TABLES_N',
        'CROSS_SECTIONS_TABLES_N',
        'COLISIONS_MAP_N',
        'COLISIONS_TABLES_I',
        'CROSS_SECTIONS_TABLES_I',
        'COLISIONS_MAP_I',
        'EMASK']
    coll_vars_i = [
        'p',
        'hei',
        'lii',
        'bei',
        'bi',
        'ci',
        'n_i',
        'oi',
        'fi',
        'nai',
        'mgi',
        'ali',
        'sii',
        'pi',
        's_i',
        'cli',
        'ari',
        'ki',
        'cai',
        'sci',
        'tii',
        'vi',
        'cri',
        'mni',
        'fei',
        'coi',
        'nii',
        'cui']
    coll_vars_n = [
        'h',
        'he',
        'li',
        'be',
        'b',
        'c',
        'n',
        'o',
        'f',
        'na',
        'mg',
        'al',
        'si',
        'p',
        's',
        'cl',
        'ar',
        'k',
        'ca',
        'sc',
        'ti',
        'v',
        'cr',
        'mn',
        'fe',
        'co',
        'ni',
        'cu']

    coll_tabs_in = []
    coll_tabs_n = []
    coll_tabs_i = []
    coll_vars_list = []

    for i in range(0, NSPECIES_MAX):
        for j in range(0, NSPECIES_MAX):
            coll_tabs_in.append(
                'momex_vk_' +
                coll_vars_i[i] +
                '_' +
                coll_vars_n[j] +
                '.dat')
            coll_tabs_i.append(
                'momex_vk_' +
                coll_vars_i[i] +
                '_' +
                coll_vars_i[j] +
                '.dat')
            coll_tabs_n.append(
                'momex_vk_' +
                coll_vars_n[i] +
                '_' +
                coll_vars_n[j] +
                '.dat')

    if (np.shape(collist) != np.shape(SPECIES)):
        print('write_mftab_ascii: WARNING the list of atom files is \n '
              'different than the selected list of species in collist')

    CROSS_SECTIONS_TABLES_I = []
    CROSS_SECTIONS_TABLES_N = []
    COLISIONS_MAP = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
    COLISIONS_MAP_I = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
    COLISIONS_MAP_N = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
    EMASK_MAP = np.zeros((NSPECIES_MAX))

    for j in range(1, NSPECIES_MAX + 1):
        for i in range(1, j + 1):
            COLISIONS_MAP_I[j - 1, i - 1] = -1
            COLISIONS_MAP_N[j - 1, i - 1] = -1
            if (i in collist) and (j in collist):
                COLISIONS_MAP[i - 1, j - 1] = (i - 1) * NSPECIES_MAX + j
                coll_vars_list.append(coll_vars_n[i - 1])
                coll_vars_list.append(coll_vars_n[j - 1])
                if (i < j):
                    COLISIONS_MAP_I[i - 1, j - 1] = (i - 1) * NSPECIES_MAX + j
                    COLISIONS_MAP_N[i - 1, j - 1] = (i - 1) * NSPECIES_MAX + j

    for j in range(0, NSPECIES_MAX):
        EMASK_MAP[j] = 99

    for symb in SPECIES:
        symb = symb.split('_')[0]
        if not(symb.lower() in coll_vars_list):
            print('write_mftab_ascii: WARNING there may be a mismatch between'
                  'the atom files and selected species.\n'
                  'Check for species', symb.lower())

    f = open(filename, 'w')
    for head in params:
        f.write(head + "\n")
        if head == 'NSPECIES_MAX':
            f.write("\t" + str(NSPECIES_MAX) + "\n")
            f.write("\n")
        if head == 'SPECIES':
            li = 0
            for spc in SPECIES:
                symb = spc.split('_')[0]
                li += 1
                f.write(
                    "\t" +
                    str(li).zfill(2) +
                    "\t" +
                    symb +
                    "\t" +
                    spc +
                    "\n")
            f.write("\n")
        if head == 'EOS_TABLES':
            li = 0
            for eos in EOS_TABLES:
                f.write("\t" + str(li).zfill(2) + "\t" + eos + "\n")
                li += 1
            f.write("\n")
        if head == 'REC_TABLES':
            li = 0
            for rec in REC_TABLES:
                li += 1
                f.write("\t" + str(li).zfill(2) + "\t" + rec + "\n")
            f.write("\n")
        if head == 'ION_TABLES':
            li = 0
            for ion in ION_TABLES:
                li += 1
                f.write("\t" + str(li).zfill(2) + "\t" + ion + "\n")
            f.write("\n")
        if head == 'COLISIONS_TABLES':
            li = 0
            for coll in coll_tabs_in:
                li += 1
                if (li in COLISIONS_MAP):
                    f.write("\t" + str(li).zfill(2) + "\t" + str(coll) + "\n")
            f.write("\n")
        if head == 'COLISIONS_TABLES_I':
            li = 0
            for coll in coll_tabs_i:
                li += 1
                if (li in COLISIONS_MAP_I):
                    f.write("\t" + str(li).zfill(2) + "\t" + str(coll) + "\n")
            f.write("\n")
        if head == 'COLISIONS_TABLES_N':
            li = 0
            for coll in coll_tabs_n:
                li += 1
                if (li in COLISIONS_MAP_N):
                    f.write("\t" + str(li).zfill(2) + "\t" + str(coll) + "\n")
            f.write("\n")
        if head == 'CROSS_SECTIONS_TABLES':
            num_cs_tab = np.shape(CROSS_SECTIONS_TABLES)[:][0]
            for crs in range(0, num_cs_tab):
                f.write("\t" +
                        str(int(CROSS_SECTIONS_TABLES[crs][0])).zfill(2) +
                        "\t" +
                        str(int(CROSS_SECTIONS_TABLES[crs][1])).zfill(2) +
                        "\t" +
                        CROSS_SECTIONS_TABLES[crs][2] +
                        "\n")
            f.write("\n")
        if head == 'CROSS_SECTIONS_TABLES_N':
            num_cs_tab = np.shape(CROSS_SECTIONS_TABLES_N)[:][0]
            for crs in range(0, num_cs_tab):
                f.write("\t" +
                        str(int(CROSS_SECTIONS_TABLES_N[crs][0])).zfill(2) +
                        "\t" +
                        str(int(CROSS_SECTIONS_TABLES_N[crs][1])).zfill(2) +
                        "\t" +
                        CROSS_SECTIONS_TABLES_N[crs][2] +
                        "\n")
            f.write("\n")
        if head == 'CROSS_SECTIONS_TABLES_I':
            num_cs_tab = np.shape(CROSS_SECTIONS_TABLES_I)[:][0]
            for crs in range(0, num_cs_tab):
                f.write("\t" +
                        str(int(CROSS_SECTIONS_TABLES_I[crs][0])).zfill(2) +
                        "\t" +
                        str(int(CROSS_SECTIONS_TABLES_I[crs][1])).zfill(2) +
                        "\t" +
                        CROSS_SECTIONS_TABLES_I[crs][2] +
                        "\n")
            f.write("\n")
        if head == 'COLISIONS_MAP':
            f.write("#\t" + "\t".join(
                    [coll_vars_n[v].upper().ljust(2) for v in range(
                            0, NSPECIES_MAX)]) + "\n")
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join(
                        [str(int(
                            COLISIONS_MAP[crs][v])).zfill(2) for v in range(
                                    0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
        if head == 'COLISIONS_MAP_I':
            f.write("#\t" + "\t".join(
                    [coll_vars_n[v].upper().ljust(2) for v in range(
                            0, NSPECIES_MAX)]) + "\n")
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join([str(int(
                        COLISIONS_MAP_I[crs][v])).zfill(2) for v in range(
                                0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
        if head == 'COLISIONS_MAP_N':
            f.write("#\t" + "\t".join(
                    [coll_vars_n[v].upper().ljust(2) for v in range(
                            0, NSPECIES_MAX)]) + "\n")
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join([str(int(
                        COLISIONS_MAP_N[crs][v])).zfill(2) for v in range(
                                0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
        if head == 'EMASK':
            f.write("#\t" + "\t".join(
                    [coll_vars_n[v].upper().ljust(2) for v in range(
                            0, NSPECIES_MAX)]) + "\n")
            f.write("\t" + "\t".join([str(
                    int(EMASK_MAP[v])).zfill(2) for v in range(
                            0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
    f.close()
