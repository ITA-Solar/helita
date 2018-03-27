"""
Set of programs to read and interact with output from Multifluid/multispecies
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii, subs2grph

class MFData(BifrostData):

    """
    Class to hold data from Multifluid/multispecies simulations
    in native format.
    """

    def __init__(self, *args, **kwargs):
        super(MFData, self).__init__(*args, **kwargs)
        self.mf_params = [
               v for k, v in self.params.items() if k.startswith('mf_')]
        self.__read_mf_params()
        if (self.mf_epf == 1):
            self.varsmfe = [
                v for v in self.auxvars if v.startswith('mfe_')]
            self.varsmfc = [
                v for v in self.auxvars if v.startswith('mfc_')]
            self.varsmf = [v for v in self.auxvars if v.startswith('mf_')]
            # Remove for the var list the mf aux vars, and stores these mf
            # varnames.
            for var in (self.varsmfe + self.varsmfc + self.varsmf):
                self.auxvars.remove(var)
        else:  # add sinle fluid energy (same for all fluids)
            self.mhdvars = 'e' + self.mhdvars
            self.snapvars.remove('e')
            if self.with_electrons:
                self.snapevars.remove('ee')
        if hasattr(self, 'with_electrons'):
            if self.with_electrons:
                self.mf_e_file = self.file_root + '_mf_e'


    def __read_mf_params(self):
        ''' Reads parameter file specific for Multi Fluid Bifrost '''
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

    def getvar(self, var, mf_ispecies=0, mf_ilevel=0, order='F', mode='r'):
        """
        Reads a given variable from the relevant files.
        """
        if var in ['x', 'y', 'z']:
            return getattr(self, var)
        # find filename template
        if self.snap < 0:
            snapstr = ''
            fsuffix_b = '.scr'
        elif self.snap == 0:
            snapstr = ''
            fsuffix_b = ''
        else:
            snapstr = self.snap_str
            fsuffix_b = ''
        # mhd variables
        if var in self.compvars:
            return self._get_compvar(var, mf_ispecies=mf_ispecies,
                                     mf_ilevel=mf_ilevel)
        elif var in self.mhdvars:  # could also have energy
            idx = self.mhdvars.index(var)
            fsuffix_a = '.snap'
            filename = self.mf_common_file
        elif var in self.snapvars:
            idx = self.snapvars.index(var)
            fsuffix_a = '.snap'
            filename = self.mf_file
        elif var in self.snapevars:
            idx = self.snapevars.index(var)
            filename = self.mf_e_file
        elif var in self.auxvars:
            idx = self.auxvars.index(var)
            fsuffix_a = '.aux'
            filename = self.mf_file
        elif var in self.varsmf:
            idx = self.varsmf.index(var)
            fsuffix_a = '.aux'
            filename = self.mf_file
        elif var in self.varsmfe:
            idx = self.varsmfe.index(var)
            fsuffix_a = '.aux'
            filename = self.mfe_file
        elif var in self.varsmfc:
            idx = self.varsmfc.index(var)
            fsuffix_a = '.aux'
            filename = self.mfc_file
        else:
            all_vars = (self.mhdvars + self.snapvars + self.auxvars +
                      self.varsmf + self.varsmfe + self.varsmfc + self.compvars)
            raise ValueError('getvar: variable %s not available. Available vars:'
                             % (var) + '\n' +  repr(all_vars))

        filename = filename + snapstr + fsuffix_a + fsuffix_b
        if var not in self.mhdvars:
            filename = filename % (mf_ispecies, mf_ilevel)

        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * self.nzb * idx * dsize
        return np.memmap(filename, dtype=self.dtype, order=order, offset=offset,
                         mode=mode, shape=(self.nx, self.ny, self.nzb))

    def _get_compvar(self, var, mf_ispecies=0, mf_ilevel=0):
        ''' Gets composite variables for single fluid. '''
        from . import cstagger

        var_sufix = '_s%dl%d' % (mf_ispecies, mf_ilevel)
        # if rho is not loaded, do it (essential for composite variables)
        if not hasattr(self, 'r' + var_sufix):
            setattr(self, 'r' + var_sufix,
                    self.getvar('r', mf_ispecies=mf_ispecies,
                                mf_ilevel=mf_ilevel))
            self.variables['r' + var_sufix] = getattr(self, 'r' + var_sufix)
        # rc is the same as r, but in C order (so that cstagger works)
        if not hasattr(self, 'rc' + var_sufix):
            setattr(self, 'rc' + var_sufix,
                    self.getvar('r', mf_ispecies=mf_ispecies,
                                mf_ilevel=mf_ilevel, order='C'))
            self.variables['rc' + var_sufix] = getattr(self, 'rc' + var_sufix)
            # initialise cstagger
            rdt = self.variables['rc' + var_sufix].dtype
            cstagger.init_stagger(self.nzb, self.z.astype(rdt),
                                  self.zdn.astype(rdt))

        if var == 'ux':  # x velocity
            if not hasattr(self, 'px' + var_sufix):
                setattr(self, 'px' + var_sufix,
                        self.getvar('px', mf_ispecies=mf_ispecies,
                                    mf_ilevel=mf_ilevel))
                self.variables[ 'px' + var_sufix] = getattr(self,
                                                            'px' + var_sufix)
            if self.nx < 5:  # do not recentre for 2D cases (or close)
                return (getattr(self, 'px' + var_sufix) /
                            getattr(self, 'rc' + var_sufix))
            else:
                return (getattr(self, 'px' + var_sufix) /
                            cstagger.xdn(getattr(self, 'rc' + var_sufix)))
        elif var == 'uy':  # y velocity
            if not hasattr(self, 'py' + var_sufix):
                setattr(self, 'py' + var_sufix,
                       self.getvar('py',mf_ispecies=mf_ispecies,
                                   mf_ilevel=mf_ilevel))
                self.variables[
                    'py' + var_sufix] = getattr(self, 'py' + var_sufix)
            if self.ny < 5:  # do not recentre for 2D cases (or close)
                return (getattr(self, 'py' + var_sufix) /
                     getattr(self, 'rc' + var_sufix))
            else:
                return (getattr(self, 'py' + var_sufix) /
                     cstagger.ydn(getattr(self, 'rc' + var_sufix)))
        elif var == 'uz':  # z velocity
            if not hasattr(self, 'pz' + var_sufix):
                setattr(self, 'pz' + var_sufix,
                        self.getvar('pz', mf_ispecies=mf_ispecies,
                                    mf_ilevel=mf_ilevel))
                self.variables['pz' + var_sufix] = getattr(self,
                                                           'pz' + var_sufix)
            return (getattr(self, 'pz' + var_sufix) /
                cstagger.zdn(getattr(self, 'rc' + var_sufix)))
        elif var == 'ee':   # internal energy?
            if not hasattr(self, 'e' + var_sufix):
                setattr(self, 'e' + var_sufix,
                        self.getvar('e', mf_ispecies=mf_ispecies,
                                    mf_ilevel=mf_ilevel))
                self.variables['e' + var_sufix] = getattr(self, 'e' + var_sufix)
            return (getattr(self, 'e' + var_sufix) /
                 getattr(self, 'r' + var_sufix))
        elif var == 'bxc':  # x field (cell center)
            if (not hasattr(self, 'bxc')):
                self.bxc = self.variables['bxc'] = self.getvar('bx', order='C')
            return cstagger.xup(self.bxc)
        elif var == 'byc':  # y field (cell center)
            if (not hasattr(self, 'byc')):
                self.byc = self.variables['byc'] = self.getvar('by', order='C')
            return cstagger.yup(self.byc)
        elif var == 'bzc':  # z field (cell center)
            if (not hasattr(self, 'bzc')):
                self.bzc = self.variables['bzc'] = self.getvar('bz', order='C')
            return cstagger.zup(self.bzc)
        elif var == 'dxdbup':  # x field ddxup
            if (not hasattr(self, 'bxc')):
                self.bxc = self.variables['bxc'] = self.getvar('bx', order='C')
            return cstagger.ddxup(self.bxc)
        elif var == 'dydbup':  # y field ddyup
            if (not hasattr(self, 'byc')):
                self.byc = self.variables['byc'] = self.getvar('by', order='C')
            return cstagger.ddyup(self.byc)
        elif var == 'dzdbup':  # z field ddzup
            if (not hasattr(self, 'bzc')):
                self.bzc = self.variables['bzc'] = self.getvar('bz', order='C')
            return cstagger.ddzup(self.bzc)
        elif var == 'dxdbdn':  # z field ddxdn
            if (not hasattr(self, 'bxc')):
                self.bxc = self.variables['bxc'] = self.getvar('bx', order='C')
            return cstagger.ddxdn(self.bxc)
        elif var == 'dydbdn':  # z field
            if (not hasattr(self, 'byc')):
                self.byc = self.variables['byc'] = self.getvar('by', order='C')
            return cstagger.ddydn(self.byc)
        elif var == 'dzdbdn':  # z field
            if (not hasattr(self, 'bzc')):
                self.bzc = self.variables['bzc'] = self.getvar('bz', order='C')
            return cstagger.ddzdn(self.bzc)
        elif var == 'pxc':  # x momentum
            if not hasattr(self, 'pxc' + var_sufix):
                setattr(self, 'pxc'+var_sufix,
                        self.getvar('px', order='C',mf_ispecies=mf_ispecies,
                                    mf_ilevel=mf_ilevel))
                self.variables['pxc' + var_sufix] = getattr(self,
                                                            'pxc' + var_sufix)
            return cstagger.xup(getattr(self, 'pxc' + var_sufix))
        elif var == 'pyc':  # y momentum
            if not hasattr(self, 'pyc' + var_sufix):
                setattr(self, 'pyc'+var_sufix,
                        self.getvar('py', order='C',mf_ispecies=mf_ispecies,
                                    mf_ilevel=mf_ilevel))
                self.variables[
                    'pyc' + var_sufix] = getattr(self, 'pyc' + var_sufix)
            return cstagger.yup(getattr(self, 'pyc' + var_sufix))
        elif var == 'pzc':  # z momentum
            if not hasattr(self, 'pzc' + var_sufix):
                setattr(self, 'pzc'+var_sufix,
                       self.getvar('pz', order='C', mf_ispecies=mf_ispecies,
                                   mf_ilevel=mf_ilevel))
                self.variables['pzc' + var_sufix] = getattr(self,
                                                            'pzc' + var_sufix)
            return cstagger.xup(getattr(self, 'pzc' + var_sufix))
        else:
            raise ValueError('getcompvar: composite var %s not found. Available:\n %s'
                             % (var, repr(self.compvars)))

    def _init_vars(self):
        """
        Initialises variable files
        """
        self.mf_common_file = (self.file_root + '_mf_common')
        self.mf_file = (self.file_root + '_mf_%02i_%02i')
        self.mfe_file = (self.file_root + '_mfe_%02i_%02i')
        self.mfc_file = (self.file_root + '_mfc_%02i_%02i')
