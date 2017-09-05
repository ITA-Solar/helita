"""
Set of programs to read and interact with output from Multifluid/multispecies
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii, subs2grph


class EbysusData(BifrostData):

    """
    Class to hold data from Multifluid/multispecies simulations
    in native format.
    """

    def __init__(self, *args, **kwargs):
        super(EbysusData, self).__init__(*args, **kwargs)


    def _set_snapvars(self):
        self.snapvars = ['r', 'px', 'py', 'pz']
        self.snapevars = ['e']
        self.mhdvars = []
        if (self.do_mhd):
            self.mhdvars = ['bx', 'by', 'bz']
        self.auxvars = self.params['aux'].split()

        if (self.mf_epf):
            # add internal energy to basic snaps
            self.snapvars.append('e')
            # make distiction between different aux variable
            self.varsmfe = [v for v in self.auxvars if v.startswith('mfe_')]
            self.varsmfc = [v for v in self.auxvars if v.startswith('mfc_')]
            self.varsmf = [v for v in self.auxvars if v.startswith('mf_')]
            self.varsmm = [v for v in self.auxvars if v.startswith('mm_')]
            for var in (self.varsmfe + self.varsmfc + self.varsmf +self.varsmm):
                self.auxvars.remove(var)
        else:  # one energy for all fluid
            self.mhdvars = 'e' + self.mhdvars
            if self.with_electrons:
                self.snapevars.remove('ee')
        if hasattr(self, 'with_electrons'):
            if self.with_electrons:
                self.mf_e_file = self.file_root + '_mf_e'

        self.auxxyvars = []
        # special case for the ixy1 variable, lives in a separate file
        if 'ixy1' in self.auxvars:
            self.auxvars.remove('ixy1')
            self.auxxyvars.append('ixy1')

        self.simple_vars = self.snapvars + self.mhdvars + \
            self.auxvars + self.varsmf + self.varsmfe + self.varsmfc + self.varsmm
        '''self.compvars = ['ux', 'uy', 'uz', 's', 'rup', 'dxdbup',
                         'dxdbdn', 'dydbup', 'dydbdn', 'dzdbup', 'dzdbdn', 'modp']
        if (self.do_mhd):
            self.compvars = self.compvars + ['bxc', 'byc', 'bzc', 'modb']'''

    # def set_snap(self,snap):
    #     super(EbysusData, self).set_snap(snap)

    def _read_params(self):
        ''' Reads parameter file specific for Multi Fluid Bifrost '''
        super(EbysusData, self)._read_params()

        self.nspecies_max = 28
        self.nlevels_max = 28

        try:
            self.mf_epf = self.params['mf_epf']
        except KeyError:
            raise KeyError('read_params: could not find mf_epf in idl file!')
        try:
            self.with_electrons = self.params['mf_electrons']
        except KeyError:
            raise KeyError(
                'read_params: could not find with_electrons in idl file!')
        try:
            self.mf_total_nlevel = self.params['mf_total_nlevel']
        except KeyError:
            print('warning, this idl file does not include mf_total_nlevel')
            #raise KeyError('read_params: could not find mf_total_nlevel in idl file!')

    def _init_vars(self, *args, **kwargs):
        """
        Initialises variable (common for all fluid)
        """

        self.mf_common_file = (self.file_root + '_mf_common')
        self.mf_file = (self.file_root + '_mf_%02i_%02i')
        self.mm_file = (self.file_root + '_mm_%02i_%02i')
        self.mfe_file = (self.file_root + '_mfe_%02i_%02i')
        self.mfc_file = (self.file_root + '_mfc_%02i_%02i')
        self.mf_e_file = (self.file_root + '_mf_e')

        self.variables = {}

        self.set_mfi(None, None)

        for var in self.simple_vars:
            try:
                self.variables[var] = self._get_simple_var(
                    var, self.mf_ispecies, self.mf_ilevel, *args, **kwargs)
                setattr(self, var,  self.variables[var])
            except:
                if self.verbose:
                    print(('(WWW) init_vars: could not read variable %s' % var))

    def set_mfi(self, mf_ispecies=None, mf_ilevel=None):
        """
        adds mf_ispecies and mf_ilevel attributes if they don't exist and
        changes mf_ispecies and mf_ilevel if needed. It will set defaults to 1
        """

        if (mf_ispecies is not None):
            if (mf_ispecies != self.mf_ispecies):
                self.mf_ispecies=mf_ispecies
            elif not hasattr(self,'mf_ispecies'):
                self.mf_ispecies=1
        elif not hasattr(self,'mf_ispecies'):
            self.mf_ispecies=1

        if (mf_ilevel is not None):
            if (mf_ilevel != self.mf_ilevel):
                self.mf_ilevel=mf_ilevel
            elif not hasattr(self,'mf_ilevel'):
                self.mf_ilevel=1
        elif not hasattr(self,'mf_ilevel'):
            self.mf_ilevel=1

    def get_var(self, var, snap=None, mf_ispecies=None, mf_ilevel=None, *args, **kwargs):
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
        '''assert (mf_ispecies > 0 and mf_ispecies <= 28)'''
        if var == 'x':
            return self.x
        elif var == 'y':
            return self.y
        elif var == 'z':
            return self.z

        if (((snap is not None) and (snap != self.snap)) or
            ((mf_ispecies is not None) and (mf_ispecies != self.mf_ispecies)) or
            ((mf_ilevel is not None) and (mf_ilevel != self.mf_ilevel))):
            self.set_mfi(mf_ispecies, mf_ilevel)
            self.set_snap(snap)

        assert (self.mf_ispecies <= 28)

        # # check if already in memmory
        # if var in self.variables:
        #     return self.variables[var]

        if var in self.simple_vars:  # is variable already loaded?
            return self._get_simple_var(var, self.mf_ispecies, self.mf_ilevel)
        elif var in self.auxxyvars:
            return super(EbysusData, self)._get_simple_var_xy(var)
        else:
            return self._get_composite_mf_var(var)
        '''else:
            raise ValueError(("get_var: could not read variable"
                              "%s. Must be one of %s" % (var, str(self.simple_vars + self.compvars + self.auxxyvars))))'''

    def _get_simple_var(self, var, mf_ispecies=None, mf_ilevel=None, order='F', mode='r', *args, **kwargs):
        """
        Gets "simple" variable (ie, only memmap, not load into memory).

        Overloads super class to make a distinction between different filenames for different variables

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

        if self.snap < 0:
            snapstr = ''
            fsuffix_b = '.scr'
        elif self.snap == 0:
            snapstr = ''
            fsuffix_b = ''
        else:
            snapstr = self.snap_str
            fsuffix_b = ''

        mf_arr_size = 1
        if (var in self.mhdvars and self.mf_ispecies > 0) or (var in ['bx','by','bz']):
            idx = self.mhdvars.index(var)
            fsuffix_a = '.snap'
            filename = self.mf_common_file
        elif var in self.snapvars and self.mf_ispecies > 0:
            idx = self.snapvars.index(var)
            fsuffix_a = '.snap'
            filename = self.mf_file % (self.mf_ispecies, self.mf_ilevel)
        elif var in self.snapevars and self.mf_ispecies < 0:
            idx = self.snapevars.index(var)
            filename = self.mf_e_file
            fsuffix_a = '.snap'
        elif var in self.auxvars:
            idx = self.auxvars.index(var)
            fsuffix_a = '.aux'
            filename = self.file_root
        elif var in self.varsmf:
            idx = self.varsmf.index(var)
            fsuffix_a = '.aux'
            filename = self.mf_file % (self.mf_ispecies, self.mf_ilevel)
        elif var in self.varsmm:
            idx = self.varsmm.index(var)
            fsuffix_a = '.aux'
            filename = self.mm_file % (self.mf_ispecies, self.mf_ilevel)
            mf_arr_size = self.mf_total_nlevel
        elif var in self.varsmfe:
            idx = self.varsmfe.index(var)
            fsuffix_a = '.aux'
            filename = self.mfe_file % (self.mf_ispecies, self.mf_ilevel)
        elif var in self.varsmfc:
            idx = self.varsmfc.index(var)
            fsuffix_a = '.aux'
            filename = self.mfc_file % (self.mf_ispecies, self.mf_ilevel)

        filename = filename + snapstr + fsuffix_a + fsuffix_b

        '''if var not in self.mhdvars and not (var in self.snapevars and self.mf_ispecies < 0) and var not in self.auxvars :
            filename = filename % (self.mf_ispecies, self.mf_ilevel)'''

        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * self.nzb * idx * dsize  * mf_arr_size

        if (mf_arr_size == 1):
            return np.memmap(filename, dtype=self.dtype, order=order, offset=offset,
                         mode=mode, shape=(self.nx, self.ny, self.nzb))
        else:
            return np.memmap(filename, dtype=self.dtype, order=order, offset=offset,
                         mode=mode, shape=(self.nx, self.ny, self.nzb, mf_arr_size))

    def _get_composite_mf_var(self, var, order='F', mode='r', *args, **kwargs):
        """
        Gets composite variables for multi species fluid.
        """
        from . import cstagger as cs

        if var == 'totr':  # velocities
            for mf_ispecies in range(28):
                for mf_ispecies in range(28):
                    r = self._get_simple_var(
                        'e', mf_ispecies=self.mf_ispecies, mf_ilevel=self.mf_ilevel, order=order, mode=mode)
            return r
        else:
            return super(EbysusData, self)._get_composite_var(var)
