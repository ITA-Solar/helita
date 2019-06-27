"""
Set of programs to read and interact with output from Multifluid/multispecies
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, Bifrost_units
from .bifrost import read_idl_ascii, subs2grph 
from . import cstagger

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

        self.simple_vars = self.snapvars + self.snapevars + self.mhdvars + self.auxvars + \
            self.varsmf + self.varsmfe + self.varsmfc + self.varsmm

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

    def _read_params(self):
        ''' Reads parameter file specific for Multi Fluid Bifrost '''
        super(EbysusData, self)._read_params()

        self.nspecies_max = 28
        self.nlevels_max = 28
        try:
            self.mf_epf = self.params['mf_epf'][self.snapInd]
        except KeyError:
            raise KeyError('read_params: could not find mf_epf in idl file!')
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
                setattr(self, var, self.variables[var])
            except BaseException:
                if self.verbose:
                    if not (self.mf_ilevel == 1 and var in self.varsmfc):
                        print(('(WWW) init_vars: could not read '
                               'variable %s' % var))

        rdt = self.r.dtype
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

    def get_var(self, var, snap=None, iix=slice(None), iiy=slice(None),
                iiz=slice(None), mf_ispecies=None, mf_ilevel=None, *args,
                **kwargs):
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

        if var not in self.snapevars:
            if (mf_ispecies is None):
                if self.mf_ispecies < 1:
                    mf_ispecies = 1
                    print("Warning: variable is only for electrons, "
                          "iSpecie changed to 1")
            elif (mf_ispecies < 1):
                mf_ispecies = 1
                print("Warning: variable is only for electrons, "
                      "iSpecie changed to 1")

        if not hasattr(self, 'iix'):
            self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            self.set_domain_iiaxis(iinum=iiz, iiaxis='z')
        else:
            if (iix != slice(None)) and np.any(iix != self.iix):
                if self.verbose:
                    print('(get_var): iix ', iix, self.iix)
                self.set_domain_iiaxis(iinum=iix, iiaxis='x')
            if (iiy != slice(None)) and np.any(iiy != self.iiy):
                if self.verbose:
                    print('(get_var): iiy ', iiy, self.iiy)
                self.set_domain_iiaxis(iinum=iiy, iiaxis='y')
            if (iiz != slice(None)) and np.any(iiz != self.iiz):
                if self.verbose:
                    print('(get_var): iiz ', iiz, self.iiz)
                self.set_domain_iiaxis(iinum=iiz, iiaxis='z')

        if self.cstagop and ((self.iix != slice(None)) or
                             (self.iiy != slice(None)) or
                             (self.iiz != slice(None))):
            self.cstagop = False
            print(
                'WARNING: cstagger use has been turned off,',
                'turn it back on with "dd.cstagop = True"')

        if ((snap is not None) and np.any(snap != self.snap)):
            self.set_snap(snap)

        if ((mf_ispecies is not None) and (mf_ispecies != self.mf_ispecies)):
            self.set_mfi(mf_ispecies, mf_ilevel)
        elif (( mf_ilevel is not None) and (mf_ilevel != self.mf_ilevel)):
            self.set_mfi(mf_ispecies, mf_ilevel)

        assert (self.mf_ispecies > 0 and self.mf_ispecies <= 28)

        # # check if already in memmory
        # if var in self.variables:
        #     return self.variables[var]
        if var in self.simple_vars:  # is variable already loaded?
            val = self._get_simple_var(var, self.mf_ispecies, self.mf_ilevel)
        elif var in self.auxxyvars:
            val =  super(EbysusData, self)._get_simple_var_xy(var)
        else:
            val =  self._get_composite_mf_var(var)

        if np.shape(val) != (self.xLength, self.yLength, self.zLength):

            if np.size(self.iix)+np.size(self.iiy)+np.size(self.iiz) > 3:
                # at least one slice has more than one value

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
            order='F',
            mode='r',
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

        mf_arr_size = 1
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

        filename = filename + currStr + fsuffix_a + fsuffix_b

        '''if var not in self.mhdvars and not (var in self.snapevars and
            self.mf_ispecies < 0) and var not in self.auxvars :
            filename = filename % (self.mf_ispecies, self.mf_ilevel)'''

        dsize = np.dtype(self.dtype).itemsize
        offset = self.nx * self.ny * self.nzb * idx * dsize * mf_arr_size

        if (mf_arr_size == 1):
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
                shape=(self.nx, self.ny, self.nzb, mf_arr_size))

    def _get_composite_mf_var(self, var, order='F', mode='r', *args, **kwargs):
        """
        Gets composite variables for multi species fluid.
        """
        if var == 'totr':  # velocities
            for mf_ispecies in range(28):
                for mf_ispecies in range(28):
                    r = self._get_simple_var(
                        'e',
                        mf_ispecies=self.mf_ispecies,
                        mf_ilevel=self.mf_ilevel,
                        order=order,
                        mode=mode)
            return r
        elif var in self.compvars:
            return super(EbysusData, self)._get_composite_var(var)
        else:
            return super(EbysusData, self).get_quantity(var)

    def get_varTime(self, var, snap=None, iix=None, iiy=None, iiz=None,
                    mf_ispecies=None, mf_ilevel=None, order='F',
                    mode='r', *args, **kwargs):

        self.iix = iix
        self.iiy = iiy
        self.iiz = iiz

        try:
            if ((snap is not None) and  any(snap != self.snap)):
                self.set_snap(snap)

        except ValueError:
            if ((snap is not None) and any(snap != self.snap)):
                self.set_snap(snap)

        if var in self.varsmfc:
            if mf_ilevel is None and self.mf_ilevel == 1:
                mf_ilevel = 2
                print("Warning: mfc is only for ionized species,"
                      "Level changed to 2")
            if mf_ilevel == 1:
                mf_ilevel = 2
                print("Warning: mfc is only for ionized species."
                      "Level changed to 2")

        if var not in self.snapevars:
            if (mf_ispecies is None):
                if self.mf_ispecies < 1:
                    mf_ispecies = 1
                    print("Warning: variable is only for electrons,"
                          "iSpecie changed to 1")
            elif (mf_ispecies < 1):
                mf_ispecies = 1
                print("Warning: variable is only for electrons,"
                      "iSpecie changed to 1")

        if (((mf_ispecies is not None) and (
                mf_ispecies != self.mf_ispecies)) or ((
                mf_ilevel is not None) and (mf_ilevel != self.mf_ilevel))):
            self.set_mfi(mf_ispecies, mf_ilevel)

        # lengths for dimensions of return array
        self.xLength = 0
        self.yLength = 0
        self.zLength = 0

        for dim in ('iix', 'iiy', 'iiz'):
            if getattr(self, dim) is None:
                if dim[2] == 'z':
                    setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]+'b'))
                else:
                    setattr(self, dim[2] + 'Length', getattr(self, 'n' + dim[2]))
                setattr(self, dim, slice(None))
            else:
                indSize = np.size(getattr(self, dim))
                setattr(self, dim[2] + 'Length', indSize)
        print(self.zLength,self.nzb)
        snapLen = np.size(self.snap)
        value = np.empty([self.xLength, self.yLength, self.zLength, snapLen])

        for i in range(0, snapLen):
            self.snapInd = 0
            self._set_snapvars()
            self._init_vars()
            value[:, :, :, i] = self.get_var(
                var, snap=snap[i], iix=self.iix, iiy=self.iiy, iiz=self.iiz,
                mf_ispecies = self.mf_ispecies, mf_ilevel=self.mf_ilevel)

        try:
            if ((snap is not None) and (snap != self.snap)):
                self.set_snap(snap)

        except ValueError:
            if ((snap is not None) and any(snap != self.snap)):
                self.set_snap(snap)

        return value


###########
#  TOOLS  #
###########


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
