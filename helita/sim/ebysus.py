"""
Set of programs to read and interact with output from Multifluid/multispecies
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii, subs2grph, bifrost_units
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
        self.auxvars = self.params['aux'].split()

        if (self.mf_epf):
            # add internal energy to basic snaps
            self.snapvars.append('e')
            # make distiction between different aux variable
            self.varsmfe = [v for v in self.auxvars if v.startswith('mfe_')]
            self.varsmfc = [v for v in self.auxvars if v.startswith('mfc_')]
            self.varsmf = [v for v in self.auxvars if v.startswith('mf_')]
            self.varsmm = [v for v in self.auxvars if v.startswith('mm_')]
            for var in (
                    self.varsmfe +
                    self.varsmfc +
                    self.varsmf +
                    self.varsmm):
                self.auxvars.remove(var)
        else:  # one energy for all fluid
            self.mhdvars = 'e' + self.mhdvars
            if self.with_electrons:
                self.snapevars.remove('ee')
        if hasattr(self, 'with_electrons'):
            if self.with_electrons:
                self.mf_e_file = self.file_root + '_mf_e'

        self.simple_vars = self.snapvars + self.mhdvars + self.auxvars + \
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
        try:
            filename = os.path.join(
                self.fdir, self.params['mf_param_file'].strip())
            self.mf_tabparam = read_mftab_ascii(filename)
        except KeyError:
            print('warning, this idl file does not include mf_param_file')
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
                setattr(self, var, self.variables[var])
            except BaseException:
                if self.verbose:
                    print(('(WWW) init_vars: could not read variable %s' % var))

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

    def get_var(
            self,
            var,
            snap=None,
            mf_ispecies=None,
            mf_ilevel=None,
            *args,
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
        if (var in self.mhdvars and self.mf_ispecies > 0) or (
                var in ['bx', 'by', 'bz']):
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
        offset = self.nx * self.ny * self.nzb * idx * dsize * mf_arr_size

        if (mf_arr_size == 1):
            return np.memmap(
                filename,
                dtype=self.dtype,
                order=order,
                offset=offset,
                mode=mode,
                shape=(
                    self.nx,
                    self.ny,
                    self.nzb))
        else:
            return np.memmap(
                filename,
                dtype=self.dtype,
                order=order,
                offset=offset,
                mode=mode,
                shape=(
                    self.nx,
                    self.ny,
                    self.nzb,
                    mf_arr_size))

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
        else:
            return super(EbysusData, self)._get_composite_var(var)


###########
#  TOOLS  #
###########
def read_mftab_ascii(filename):
    ''' Reads mf_tabparam.in-formatted (command style) ascii file into dictionary '''
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
            line = line.split(';')[0].split('\t')
            # if (len(line) > 2):
            #    print(('(WWW) read_params: line %i is invalid, skipping' % li))
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
                    print('(WWW) read_mftab_ascii: could not find datatype in '
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
                            '(WWW) read_mftab_ascii: could not find datatype in '
                            'line %i, skipping' %
                            li)
                else:
                    try:
                        value = int(value)
                        value2 = int(value2)
                    except BaseException:
                        print(
                            '(WWW) read_mftab_ascii: could not find datatype in '
                            'line %i, skipping' %
                            li)
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
                    print('(WWW) read_mftab_ascii: could not find datatype in '
                          'line %i, skipping' % li)
                    li += 1
                    continue
                if not (key[0] in params):
                    params[key[0]] = [arr]
                else:
                    params[key[0]] = np.vstack((params[key[0]], [arr]))
            li += 1
    return params


def write_mftab_ascii(filename,
                      NSPECIES_MAX=28,
                      SPECIES=['H_2.atom',
                               'He_3.atom'],
                      EOS_TABLES=['default.dat'],
                      REC_TABLES=['hrec.dat'],
                      ION_TABLES=['hion.dat'],
                      CROSS_SECTIONS_TABLES=[[1,
                                              1,
                                              'p-H-elast.txt'],
                                             [1,
                                              2,
                                              'p-He.txt'],
                                             [2,
                                              2,
                                              'He-He.txt']],
                      CROSS_SECTIONS_TABLES_I=[],
                      CROSS_SECTIONS_TABLES_N=[],
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
        EOS_TABLES - list of strings containing the name of the eos tables (no use)
        REC_TABLES - list of strings containing the name of the rec tables (no use)
        ION_TABLES - list of strings containing the name of the ion tables (no use)
        CROSS_SECTIONS_TABLES - list of strings containing the name of the cross section files from VK between ion and neutrals
        CROSS_SECTIONS_TABLES_I - list of strings containing the name of the cross section files from VK between ions
        CROSS_SECTIONS_TABLES_N - list of strings containing the name of the cross section files from VK  between ions
        collist - integer vector of the species used.
                e.g., collist = [1,2,3] will include the H, He and Li

    '''

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
        'COLISIONS_MAP_I']
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
        print('write_mftab_ascii: WARNING the list of atom files is different \n than the selected list of species in collist')

    CROSS_SECTIONS_TABLES_I = []
    CROSS_SECTIONS_TABLES_N = []
    COLISIONS_MAP = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
    COLISIONS_MAP_I = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
    COLISIONS_MAP_N = np.zeros((NSPECIES_MAX, NSPECIES_MAX))
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

    for symb in SPECIES:
        symb = symb.split('_')[0]
        if not(symb.lower() in coll_vars_list):
            print('write_mftab_ascii: WARNING there may be a mismatch between the atom files and selected species.')
            print('                   Check for species', symb.lower())

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
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join([str(int(COLISIONS_MAP[crs][v])).zfill(2)
                                          for v in range(0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
        if head == 'COLISIONS_MAP_I':
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join([str(int(COLISIONS_MAP_I[crs][v])).zfill(2)
                                          for v in range(0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
        if head == 'COLISIONS_MAP_N':
            for crs in range(0, NSPECIES_MAX):
                f.write("\t" + "\t".join([str(int(COLISIONS_MAP_I[crs][v])).zfill(2)
                                          for v in range(0, NSPECIES_MAX)]) + "\n")
            f.write("\n")
    f.close()


def read_voro_ascii(
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat'):
    ''' Reads the miscelaneous Vofonov & abundances table formatted (command style) ascii file into dictionary '''
    li = 0
    params = {}
    headers = ['NSPECIES_MAX', 'NLVLS_MAX', 'SPECIES']
    # go through the file, add stuff to dictionary
    ii = 0
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
            line = line.split(';')[0].split('\t')

            if (np.size(line) == 1):
                if (str(line[0]) in headers):
                    key = line
                    ii = 0
                else:
                    value = line[0].strip()
                    try:
                        value = int(value)
                        exec('params["' + key[0] + '"] = [value]')
                        ii = 1
                    except BaseException:
                        print(
                            '(WWW) read_voro_ascii: could not find datatype in '
                            'line %i, skipping' %
                            li)
                        li += 1
                        continue
            elif (np.size(line) > 8):
                val_arr = []
                for iv in range(0, 9):
                    if (iv == 0) or (iv == 4):
                        try:
                            value = int(line[iv].strip())
                            val_arr.append(value)
                        except BaseException:
                            print(
                                '(WWW) read_voro_ascii: could not find datatype in '
                                'line %i, skipping' %
                                li)
                            li += 1
                            continue
                    elif (iv == 1):
                        val_arr.append(line[iv].strip().lower())
                    else:
                        try:
                            value = float(line[iv].strip())
                            val_arr.append(value)

                        except BaseException:
                            print(
                                '(WWW) read_voro_ascii: could not find datatype in '
                                'line %i, skipping' %
                                li)
                            li += 1
                            continue

                if not key[0] in params:
                    params[key[0]] = [val_arr]
                else:
                    params[key[0]].append(val_arr)

            else:
                print('(WWW) read_voro_ascii: could not find datatype in '
                      'line %i, skipping' % li)
                li += 1
                continue

        params['SPECIES'] = np.array(params['SPECIES'])
    return params


def get_abund(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'/INPUT/MISC/voronov.dat',
        Chianti=False):
    '''
        Returns abundances from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all abundances in the file.
            In this case, one will access to it by, e.g., var['h']
        filename - is the abundance file, default is voronov.dat withuot Chianti data base or sun_photospheric_1998_grevesse for
            Chianti data base.
    '''
    if Chianti:
        import ChiantiPy.core as ch
        if ('.dat' in abundance):
            abundance = 'sun_photospheric_1998_grevesse'
        ion = ch.ion(atom, abundance)

        return ion.Abundance
    else:
        atom = atom.replace("_", "")
        if len(''.join(x for x in atom if x.isdigit())) == 1:
            atom = atom.replace("1", "")
        if (len(params) == 0):
            params = read_voro_ascii(filename)

        if (len(atom) > 0):
            return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
                0]], 8].astype(np.float)[0][0]
        else:
            for ii in range(0, params['NLVLS_MAX'][0]):
                if not(any(i.isdigit() for i in params['SPECIES'][ii, 1])):
                    try:
                        abund_dic[params['SPECIES'][ii, 1]
                                  ] = params['SPECIES'][ii, 8].astype(np.float)
                    except BaseException:
                        abund_dic = {
                            params['SPECIES'][ii, 1]: params['SPECIES'][ii, 8].astype(np.float)}

            return abund_dic


def get_atomweight(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'/INPUT/MISC/voronov.dat'):
    '''
        Returns atomic weights from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all atomic weights in the file.
            In this case, one will access to it by, e.g., var['h']
    '''
    atom = atom.replace("_", "")
    if len(''.join(x for x in atom if x.isdigit())) == 1:
        atom = atom.replace("1", "")
    if (len(params) == 0):
        params = read_voro_ascii(filename)
    if (len(atom) > 0):
        return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
            0]], 2].astype(np.float)[0][0]
    else:
        for ii in range(0, params['NLVLS_MAX'][0]):
            if not(any(i.isdigit() for i in params['SPECIES'][ii, 1])):
                try:
                    weight_dic[params['SPECIES'][ii, 1]
                               ] = params['SPECIES'][ii, 2].astype(np.float)
                except BaseException:
                    weight_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 2].astype(np.float)}

        return weight_dic


def get_atomde(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat',
        Chianti=True,
        cm1=False):
    '''
        Returns ionization energy dE from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all ionization energy dE in the file.
            In this case, one will access to it by, e.g., var['he2']
        cm1 - boolean and if it is true converts from eV to cm-1
    '''
    if cm1:
        units = 1.0 / (8.621738e-5 / 0.695)
    else:
        units = 1.0
    if Chianti and atom != '':
        import ChiantiPy.core as ch
        ion = ch.ion(atom.lower())
        return ion.Ip * units
    else:
        atom = atom.replace("_", "")
        if len(''.join(x for x in atom if x.isdigit())) == 1:
            atom = atom.replace("1", "")
        if (len(params) == 0):
            params = read_voro_ascii(filename)

            if (len(atom) > 0):
                return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
                    0]], 3].astype(np.float)[0][0] * units
            else:
                for ii in range(0, params['NLVLS_MAX'][0]):
                    try:
                        de_dic[params['SPECIES'][ii, 1]] = params['SPECIES'][ii, 3].astype(
                            np.float) * units
                    except BaseException:
                        de_dic = {params['SPECIES'][ii,
                                                    1]: params['SPECIES'][ii,
                                                                          3].astype(np.float) * units}

                return de_dic


def get_atomZ(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat',
        Chianti=True):
    '''
        Returns atomic number Z from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all iatomic number Z in the file.
            In this case, one will access to it by, e.g., var['he2']
    '''

    if Chianti:
        import ChiantiPy.core as ch
        if len(''.join(x for x in atom if x.isdigit())) == 0:
            atom = atom + '_2'
        ion = ch.ion(atom.lower())
        return ion.Z
    else:
        atom = atom.replace("_", "")
        if len(''.join(x for x in atom if x.isdigit())) == 1:
            atom = atom.replace("1", "")
        if (len(params) == 0):
            params = read_voro_ascii(filename)

        if (len(atom) > 0):
            return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
                0]], 0].astype(np.int)[0][0]
        else:
            for ii in range(0, params['NLVLS_MAX'][0]):
                if not(any(i.isdigit() for i in params['SPECIES'][ii, 1])):
                    try:
                        z_dic[params['SPECIES'][ii, 1]
                              ] = params['SPECIES'][ii, 0].astype(np.int)
                    except BaseException:
                        z_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 0].astype(np.int)}

            return z_dic


def get_atomP(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat'):
    '''
        Returns P parameter for Voronov rate fitting term from the voronov.dat file.
            The parameter P was included to better fit the particular cross-section
            behavior for certain ions near threshold; it only takes on the value 0 or 1

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all P parameter in the file.
            In this case, one will access to it by, e.g., var['he2']
    '''
    atom = atom.replace("_", "")
    if len(''.join(x for x in atom if x.isdigit())) == 1:
        atom = atom.replace("1", "")
    if (len(params) == 0):
        params = read_voro_ascii(filename)

    if (len(atom) > 0):
        return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
            0]], 4].astype(np.int)[0][0]
    else:
        for ii in range(0, params['NLVLS_MAX'][0]):
            try:
                p_dic[params['SPECIES'][ii, 1]
                      ] = params['SPECIES'][ii, 4].astype(np.int)
            except BaseException:
                p_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 4].astype(np.int)}

        return p_dic


def get_atomA(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat'):
    '''
        Returns A parameter for Voronov rate fitting term  from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all A parameter in the file.
            In this case, one will access to it by, e.g., var['he2']
     '''
    atom = atom.replace("_", "")
    if len(''.join(x for x in atom if x.isdigit())) == 1:
        atom = atom.replace("1", "")
    if (len(params) == 0):
        params = read_voro_ascii(filename)

    if (len(atom) > 0):
        return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
            0]], 5].astype(np.float)[0][0]
    else:
        for ii in range(0, params['NLVLS_MAX'][0]):
            try:
                a_dic[params['SPECIES'][ii, 1]
                      ] = params['SPECIES'][ii, 5].astype(np.float)
            except BaseException:
                a_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 5].astype(np.float)}

        return a_dic


def get_atomX(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat'):
    '''
        Returns X parameter for Voronov rate fitting term from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all X parameter in the file.
            In this case, one will access to it by, e.g., var['he2']
    '''
    atom = atom.replace("_", "")
    if len(''.join(x for x in atom if x.isdigit())) == 1:
        atom = atom.replace("1", "")
    if (len(params) == 0):
        params = read_voro_ascii(filename)

    if (len(atom) > 0):
        return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
            0]], 6].astype(np.float)[0][0]
    else:
        for ii in range(0, params['NLVLS_MAX'][0]):
            try:
                x_dic[params['SPECIES'][ii, 1]
                      ] = params['SPECIES'][ii, 6].astype(np.float)
            except BaseException:
                x_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 6].astype(np.float)}

        return x_dic


def get_atomK(
        atom='',
        params=[],
        filename=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat'):
    '''
        Returns K parameter for Voronov rate fitting term  from the voronov.dat file.

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter of the atom of interest.
            If atom is not defined then it returns a list of all K parameter in the file.
            In this case, one will access to it by, e.g., var['he2']
    '''
    atom = atom.replace("_", "")
    if len(''.join(x for x in atom if x.isdigit())) == 1:
        atom = atom.replace("1", "")

    if (len(params) == 0):
        params = read_voro_ascii(filename=filename)

    if (len(atom) > 0):
        return params['SPECIES'][[np.where(params['SPECIES'][:, 1] == atom)[
            0]], 7].astype(np.float)[0][0]
    else:
        for ii in range(0, params['NLVLS_MAX'][0]):
            try:
                k_dic[params['SPECIES'][ii, 1]
                      ] = params['SPECIES'][ii, 7].astype(np.float)
            except BaseException:
                k_dic = {params['SPECIES'][ii, 1]: params['SPECIES'][ii, 7].astype(np.float)}

        return k_dic


def info_atom(atom=''):
    '''
    provides general information about specific atom, e.g., ionization and excitation levels etc

    Parameters
    ----------
    atom - lower case letter of the atom of interest, e.g., 'he'
    '''
    import ChiantiPy.core as ch

    ion = ch.ion(atom + '_1')
    Z = ion.Z
    wgt = get_atomweight(atom)
    print('Atom', 'Z', 'Weight  ', 'FIP  ', 'Abnd')
    print(atom, ' ', Z, wgt, ion.FIP, ion.Abundance)
    for ilvl in range(1, Z + 1):
        ion = ch.ion(atom + '_' + str(ilvl))
        print('    ', 'ion', 'Level', 'dE (eV)', 'g')
        print('    ', ion.Spectroscopic, ion.Ion, ' ', ion.Ip, 'g')
        if hasattr(ion, 'Elvlc'):
            nl = len(ion.Elvlc['lvl'])
            print('        ', 'Level', 'str    ', 'dE (cm-1)', 'g')
            for elvl in range(0, nl):
                print(
                    '          ',
                    ion.Elvlc['lvl'][elvl],
                    ion.Elvlc['pretty'][elvl],
                    ion.Elvlc['ecmth'][elvl],
                    'g')


def get_excidE(atom='', lvl=1, params=[], Chianti=True, cm1=False):
    '''
        Returns ionization energy dE for excited levels

        Parameters
        ----------
        params - list containing the information of voronov.dat (following the format of read_voro_ascii)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all ionization energy dE in the file.
            In this case, one will access to it by, e.g., var['he2']
        cm1 - boolean and if it is true converts from eV to cm-1
    '''
    import ChiantiPy.core as ch

    unitscm1 = 1.0 / (8.621738e-5 / 0.695)
    if cm1:
        units = 1.0
    else:
        units = 1. / unitscm1
    import ChiantiPy.core as ch
    ion = ch.ion(atom)
    if hasattr(ion, 'Elvlc'):
        return (ion.Ip * unitscm1 + ion.Elvlc['ecmth'][lvl]) * units
    else:
        print('No Elvlc in the Chianti Data base')


def rrec(Te, atom='h'):
    ''' gives the recombination rate per particle McWhirter (1965) '''
    units = bifrost_units()
    vfac = 2.6e-19
    Z = get_atomZ(atom)
    TeV = Te * units.K_TO_EV
    return vfac * np.sqrt(1.0 / TeV) * Z**2


def vrec(nel, Te, atom='h'):
    ''' gives the recombination frequency McWhirter (1965) '''
    return nel * rrec(Te, atom=atom)

def rion(Te, atom='h'):
    ''' gives the ionization rate per particle using Voronov 1997 fitting formula'''
    units = bifrost_units()
    A = get_atomA(atom) * 1.0e6  # converted to SI 2.91e-14
    X = get_atomX(atom)  # 0.232
    K = get_atomK(atom)  # 0.39
    P = get_atomP(atom)  # 1
    phion = get_atomde(atom, Chianti=False)  # 13.6
    TeV = Te * units.K_TO_EV
    return A * (1 + np.sqrt(phion / TeV) * P) / (X + phion /
                                                 TeV) * (phion / TeV)**K * np.exp(-phion / TeV)

def vion(nel, Te, atom='h'):
    ''' gives the ionization frequency using Voronov 1997 fitting formula'''
    return nel * rion(Te, atom=atom)


def ionfraction(Te, atom='h'):
    ''' gives the ionization fraction using vrec and vion'''
    return rion(Te, atom=atom) / (rrec(Te, atom=atom) + 2.0 * rion(Te, atom=atom))


def ionse(ntot, Te, atom='h'):
    ''' gives electron or ion number density using vrec and vion'''
    units = bifrost_units()
    return ntot * ionfraction(Te, atom=atom)


def neuse(ntot, Te, atom='h'):
    ''' gives neutral number density using vrec and vion'''
    return ntot - 2.0 * ionse(ntot, Te, atom=atom)

def vionr3body(nel,Te, atom='h',lowlevel = 0,atomfile='H_2.atom'):
    ''' three body recombination '''
    units = bifrost_units()
    param = read_atom_ascii(atomfile)
    gst_hi=float(param['lvl'][lowlevel,1]) #2.0
    gst_lo=float(param['lvl'][lowlevel+1,1]) #1.0
    if lowlevel == 0:
        ionlevel = atom
    else:
        ionlevel = atom + '_' + str(lowlevel+1)
    phion = get_atomde(ionlevel, Chianti=False)
    dekt=phion/units.K_TO_EV/Te
    saha=2.07e-16*nel*gst_lo/gst_hi*Te**(-1.5)*np.exp(dekt)  # Assuming nel in cgs. (For SI units would be 2.07e-22)
    return saha*vion(nel,Te) # vion is collisional ionization rate

def inv_pop(ntot,Te,atom='h',nlevels=2,niter=100,nel=None,atomfile='H_2.atom',threebody=True):
    ''' Inverts the Matrix for Statistical Equilibrum'''
    # nel starting guess is:
    if nel is None: nel=ntot*1.0
    shape=np.shape(ntot)
    nelf=np.ravel(nel)
    ntotf=np.ravel(ntot)
    tef=np.ravel(Te)
    npoints=len(tef)
    n_isp=np.zeros((npoints,nlevels))
    for ipoint in range(0,npoints):
        if (ipoint*100/(1.0*npoints) in np.linspace(0,99,100)): print('Done %s grid points of %s' %(str(ipoint),str(npoints)))
        for iel in range(1,niter):
            B=np.zeros((nlevels))
            A=np.zeros((nlevels,nlevels))
            for ilev in range(0,nlevels-1):
                if ilev > 0:
                    ionm=atom+'_'+str(ilev+1)
                else:
                    ionm=atom
                Rip=vrec(nelf[ipoint],tef[ipoint],atom=ionm)
                if (threebody):
                    Ri3d=vionr3body(nelf[ipoint],tef[ipoint],atom=atom,lowlevel=ilev,atomfile=atomfile)
                else:
                    Ri3d = 0.0
                Cip=vion(nelf[ipoint],tef[ipoint],atom=ionm)
                A[ilev,ilev] += - Cip
                A[ilev,ilev+1] = Rip + Ri3d
                if ilev < nlevels-2:
                    A[ilev+1,ilev+1] = - Rip - Ri3d
                    A[ilev+1,ilev] = Cip
            A[ilev+1,:] = 1.0
            B[ilev+1] = ntotf[ipoint]
            n_isp[ipoint,:] = np.linalg.solve(A,B)
            nelpos = 0.0
            for ilev in range(1,nlevels):
                nelpos += n_isp[ipoint,ilev]*ilev
            if (nelf[ipoint] - nelpos)/(nelf[ipoint] + nelpos) < 1e-4:
                #print("Jump iter with iter = ",iel)
                nelf[ipoint] =nelpos
                break
            if (iel == niter-1):
                if (nelf[ipoint] - nelpos)/(nelf[ipoint] + nelpos) > 1e-4:
                    print("Warning, No stationary solution was found",(nelf[ipoint] - nelpos)/(nelf[ipoint] + nelpos),nelpos,nelf[ipoint])
            nelf[ipoint] =nelpos

    n_isp=n_isp.reshape(np.append(shape,nlevels))
    return n_isp

def pop_over_species(ntot,Te,atom=['h','he'],nlevels=[2,3],atomfile=['H_2.atom','He_3.atom'],threebody=True):
    ''' this will do the SE for many species taking into account their abundances'''
    units = bifrost_units()
    totabund = 0.0
    for isp in range(0,len(atom)): totabund += 10.0**get_abund(atom[isp])

    all_pop_species={}
    for isp in range(0,len(atom)):
        abund = np.array(10.0**get_abund(atom[isp]))
        atomweight = get_atomweight(atom[isp])*units.AMU
        n_species = np.zeros((np.shape(ntot)))
        n_species = ntot*(np.array(abund/totabund))
        pop_species = inv_pop(n_species,Te,atom=atom[isp],nlevels=nlevels[isp],niter=100,atomfile=atomfile[isp],threebody=threebody)

        all_pop_species[atom[isp]] = pop_species
        print('Done with atom', atom[isp])
    return all_pop_species

def add_voro_atom(
        inputfile,
        outputfile,
        atom='',
        vorofile=os.environ.get('EBYSUS')+'INPUT/MISC/voronov.dat',
        nk='100'):
    '''
        Add voronov information at the end of the atom file.

        Parameters
        ----------
        inputfile - name of the input atom file
        outputfile - name of the output atom file which will include the VORONOV information
        atom - lower case letters of the atom of interest. Make sure that it matches with the atom file.
        vorofile - voronot table file.
    '''

    import shutil

    shutil.copy(inputfile, outputfile)
    atom = atom.lower()
    params = read_voro_ascii(vorofile)
    f = open(inputfile, "r")
    data = f.readlines()
    f.close()
    infile = open(inputfile)
    f = open(outputfile, "w")
    for line in infile:
        if not ('END' in line):
            f.write(line)
    infile.close()

    f.write("\n")
    f.write("VORONOV\n")

    f.write(
        "# from Voronov fit formula for ionization rates by electron impact \n" +
        "# by G. S. Voronov: \n" +
        "# ATOMIC DATA AND NUCLEAR DATA TABLES 65, 1-35 (1997) ARTICLE NO. DT970732\n" +
        "# <cross> = A (1+P*U^(1/2))/(X + U)*U^K e-U (cm3/s) with U = dE/Te\n")

    strat = ''
    if len(''.join(x for x in atom if x.isdigit())) == 0:
        strat = '_1'

    # where I need to add a check if atom is the same as the one in the atom
    # file or even better use directly the atom file info for this.
    Z = get_atomZ(atom=atom + strat)

    f.write(str(Z) + "\n")
    jj = 1
    f.write('#   i    j    dE(eV)     P  A(cm3/s)   X      K  \n')
    for ii in range(0, params['NLVLS_MAX'][0]):
        if (Z == int(params['SPECIES'][ii, 0])) and jj < nk:
            strat = ''
            if len(
                    ''.join(x for x in params['SPECIES'][ii, 1] if x.isdigit())) == 0:
                strat = '_1'
            f.write('  {0:3d}'.format(jj) +
                    '  {0:3d}'.format(jj +
                                      1) +
                    '  {0:9.3f}'.format(get_atomde(atom=params['SPECIES'][ii, 1] +
                                                   strat, Chianti=False)) +
                    '  {0:3}'.format(get_atomP(atom=params['SPECIES'][ii, 1])) +
                    '  {0:7.3e}'.format(get_atomA(atom=params['SPECIES'][ii, 1])) +
                    ' {0:.3f}'.format(get_atomX(atom=params['SPECIES'][ii, 1])) +
                    ' {0:.3f}'.format(get_atomK(atom=params['SPECIES'][ii, 1])) +
                    '\n')
            jj += 1
    f.write("END")
    f.close()


def read_atom_ascii(atomfile):
    ''' Reads the atom (command style) ascii file into dictionary '''
    def readnextline(lines, lp):
        line = lines[lp]
        while line == '\n':
            lp += 1
            line = lines[lp]
        while len(line) < 1 or line[0] == '#' or line[0] == '*':
            lp += 1
            line=lines[lp]
        line = line.split(';')[0].split(' ')
        while '\n' in line:
            line.remove('\n')
        while '' in line:
            line.remove('')
        return line, lp + 1
    li = 0
    params = {}
    # go through the file, add stuff to dictionary
    ii = 1
    kk = 0
    bins = 0
    ncon = 0
    nlin = 0
    nk = 0
    #nl = sum(1 for line in open(atomfile))
    f = open(atomfile)
    start = True
    key = ''
    headers = [
        'GENCOL',
        'CEXC',
        'AR85-CDI',
        'AR85-CEA',
        'AR85-CH',
        'AR85-CHE',
        'CI',
        'CE',
        'CP',
        'OHM',
        'BURGESS',
        'SPLUPS',
        'SHULL82',
        'TEMP',
        'RECO',
        'VORONOV',
        'EMASK']  # Missing AR85-RR, RADRAT, SPLUPS5, I think AR85-CHE is not used in OOE
    headerslow = [
        'gencol',
        'cexc',
        'ar85-cdi',
        'ar85-cea',
        'ar85-ch',
        'ar85-che',
        'ci',
        'ce',
        'cp',
        'ohm',
        'burgess',
        'slups',
        'shull82',
        'temp',
        'reco',
        'voronov',
        'emask']
    lines=f.readlines()
    f.close()
    for il in range(0,len(lines)): # for line in iter(f):
        # ignore empty lines and comments
        line=lines[il]
        line = line.strip()
        if len(line) < 1:
            li += 1
            continue
        if line[0] == '#' or line[0] == '*':
            li += 1
            continue

        line = line.split(';')[0].split(' ')

        if line[0].strip().lower() in headerslow:
            break
        while '' in line:
            line.remove('')

        if (np.size(line) == 1) and (ii == 1):
            params = {'atom': line[0].strip()}
            ii = 2
            li += 1
            continue
        elif (ii == 2):
            if line[0] == '#':
                li += 1
                continue
            elif (np.size(line) == 2):
                params['abund'] = float(line[0].strip())
                params['weight'] = float(line[1].strip())
                ii += 1
                li += 1
                continue
        elif (ii == 3):
            li += 1
            if (np.size(line) == 4):
                params['nk'] = int(line[0].strip())
                params['nlin'] = int(line[1].strip())
                params['ncnt'] = int(line[2].strip())
                params['nfix'] = int(line[3].strip())
                continue
            elif(np.size(line) > 4):
                if nk < int(params['nk']):
                    string = [" ".join(line[v].strip()
                                       for v in range(3, np.size(line) - 3))]
                    nk += 1
                    if 'lvl' in params:
                        params['lvl'] = np.vstack((params['lvl'], [float(line[0].strip()), float(
                            line[1].strip()), string[0], int(line[-2].strip()), int(line[-1].strip())]))
                    else:
                        params['lvl'] = [float(line[0].strip()), float(
                            line[1].strip()), string[0], int(line[-2].strip()), int(line[-1].strip())]
                    continue
                elif nlin < int(params['nlin']):
                    nlin += 1
                    if len(line) > 6:  # this is for OOE standards
                        if 'line' in params:
                            params['line'] = np.vstack(
                                (params['line'], [
                                    int(
                                        line[0].strip()), int(
                                        line[1].strip()), float(
                                        line[2].strip()), int(
                                        line[3].strip()), float(
                                        line[4].strip()), float(
                                        line[5].strip()), int(
                                        line[6].strip()), float(
                                        line[7].strip()), float(
                                            line[8].strip()), float(
                                                line[9].strip())]))
                        else:
                            params['line'] = [
                                int(
                                    line[0].strip()), int(
                                    line[1].strip()), float(
                                    line[2].strip()), int(
                                    line[3].strip()), float(
                                    line[4].strip()), float(
                                    line[5].strip()), int(
                                    line[6].strip()), float(
                                    line[7].strip()), float(
                                        line[8].strip()), float(
                                            line[9].strip())]
                    else:  # this is for HION, HELIUM or MF standards
                        if 'line' in params:
                            params['line'] = np.vstack(
                                (params['line'], [
                                    int(
                                        line[0].strip()), int(
                                        line[1].strip()), float(
                                        line[2].strip()), int(
                                        line[3].strip()), float(
                                        line[4].strip()), line[5].strip()]))
                        else:
                            params['line'] = [int(line[0].strip()), int(line[1].strip()), float(
                                line[2].strip()), int(line[3].strip()), float(line[4].strip()), line[5].strip()]
                    continue
                elif ncon < int(params['ncnt']):
                    ncon += 1
                    if len(line) > 2:  # this is for HION standards
                        if 'cont' in params:
                            params['cont'] = np.vstack(
                                (params['cont'], [
                                    int(
                                        line[0].strip()), int(
                                        line[1].strip()), float(
                                        line[2].strip()), int(
                                        line[3].strip()), float(
                                        line[4].strip()), line[5].strip()]))
                        else:
                            params['cont'] = [int(line[0].strip()), int(line[1].strip()), float(
                                line[2].strip()), int(line[3].strip()), float(line[4].strip()), line[5].strip()]
                    else:
                        ii = 4  # this is for Helium format
                    continue
                if nk == int(params['nk']) - 1 and nlin == int(params['nlin']
                                                               ) - 1 and ncnt == int(params['ncon']) - 1:
                    ii = 4
                continue
        elif(ii == 4):
            li += 1
            if (np.size(line) == 1):
                if kk == 0:
                    nbin = int(line[0].strip())
                    bin_euv = np.zeros(nbin)
                else:
                    bin_euv[kk - 1] = float(line[0].strip)
                kk += 1
                params['bin_euv'] = [nbin, [bin_euv]]
                if kk == 7:
                    kk = 0
            if (np.size(line) == 2):
                if kk == 7:
                    kk = 0
                if kk == 0:
                    kk += 1
                    tr = line[0].strip() + line[1].strip()
                    continue
                kk += 1

                if not('photioncross' in params):
                    params['photioncross'] = {}
                try:
                    params['photioncross'][tr] = np.vstack(
                        (params['photioncross'][tr], [int(line[0].strip()), float(line[1].strip())]))
                except BaseException:
                    params['photioncross'][tr] = [
                        int(line[0].strip()), float(line[1].strip())]

    if 'bin_euv' not in params:
        # JMS default from HION, however, this should be check from HION.
        params['bin_euv'] = [
            6, [911.7, 753.143, 504.0, 227.800, 193.919, 147.540, 20.0]]

    lp=li
    while True:
        line, lp = readnextline(lines, lp)
        if(line[0].strip().lower() == 'end'):
            break
        #if line == "":
        #    break

        if line[0].strip().lower() in headerslow:
            if(line[0].strip().lower() == 'gencol'):
                key = 'gencol'
                continue
            # JMS we should add wavelength bins here.
            elif(line[0].strip().lower() == 'cexc'):
                key = 'cexc'
                niter = 0
                line, lp = readnextline(lines, lp)
                niter = int(line[0].strip())
                for itercexc in range(0, niter):
                    line, lp = readnextline(lines, lp)
                    lp += 1
                    if (itercexc == 0):
                        params['cexc'] = float(line[0].strip())
                    else:
                        params['cexc'] = np.vstack(
                            (params['cexc'], float(line[0].strip())))

            elif(line[0].strip().lower() == 'ar85-cdi'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp0 = [int(line[0].strip()), int(line[1].strip())]
                line, lp = readnextline(lines, lp)
                niter = int(line[0].strip())
                for iterar in range(0, niter):
                    line, lp = readnextline(lines, lp)
                    if iterar == 0:
                        temp = [float(line[v].strip()) for v in range(0, 5)]
                    else:
                        temp = [temp, [
                                float(line[v].strip()) for v in range(0, 5)]]
                temp = [temp0, niter, temp]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() == 'ar85-cea' or line[0].strip().lower() == 'burgess'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp0 = [int(line[0].strip()), int(line[1].strip())]
                line, lp = readnextline(lines, lp)
                temp = float(line[0].strip())
                temp = [[temp0], temp]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() == 'ar85-ch') or (line[0].strip().lower() == 'ar85-che'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp0 = [int(line[0].strip()), int(line[1].strip())]
                line, lp = readnextline(lines, lp)
                temp = [float(line[v].strip()) for v in range(0, 6)]
                temp = [[temp0], [temp]]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() == 'splups9'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp = [[int(line[v].strip()) for v in range(0, 3)], [
                    float(line[v].strip()) for v in range(3, 15)]]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]
            elif(line[0].strip().lower() == 'splups'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp = [[int(line[v].strip()) for v in range(0, 3)], [
                    float(line[v].strip()) for v in range(3, 11)]]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() == 'shull82'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                temp = [[int(line[v].strip()) for v in range(0, 2)], [
                    float(line[v].strip()) for v in range(2, 9)]]
                line, lp = readnextline(lines, lp)
                temp = [temp, [float(line[0].strip())]]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() == 'voronov'):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                z = int(line[0].strip())
                line, lp = readnextline(lines, lp)
                vorpar = np.zeros((z, 7))
                for iterv in range(0,z):
                    vorpar[iterv,0] = int(line[0].strip())
                    vorpar[iterv,1] = int(line[1].strip())
                    vorpar[iterv,2] = float(line[2].strip())
                    vorpar[iterv,3] = int(line[3].strip())
                    vorpar[iterv,4] = float(line[4].strip())
                    vorpar[iterv,5] = float(line[5].strip())
                    vorpar[iterv,6] = float(line[6].strip())
                if key in params:
                    params[key] = np.vstack((params[key], [vorpar]))
                else:
                    params[key] = [vorpar]

            elif(line[0].strip().lower() == 'temp'):
                key = 'temp'
                line, lp = readnextline(lines, lp)
                nitert = int(line[0].strip())
                temp = np.zeros((nitert))
                itertemp = 0
                while itertemp < nitert:
                    line, lp = readnextline(lines, lp)
                    for v in range(0, np.size(line)):
                        temp[itertemp] = float(line[v].strip())
                        itertemp += 1
                temp = [[''], [nitert], [temp]]
                if key in params:
                    params[key] = np.vstack((params[key], [temp]))
                else:
                    params[key] = [temp]

            elif(line[0].strip().lower() in ['reco', 'ci', 'ohm', 'ce', 'cp']):
                key = line[0].strip().lower()
                line, lp = readnextline(lines, lp)
                params['temp'][-1][0] = key
                ij = [int(line[0].strip()), int(line[1].strip())]
                itertemp = 0
                reco = np.zeros((nitert))
                for v in range(2, np.size(line)):
                    reco[itertemp] = float(line[v].strip())
                    itertemp += 1
                while itertemp < nitert:
                    line, lp = readnextline(lines, lp)
                    for v in range(0, np.size(line)):
                        reco[itertemp] = float(line[v].strip())
                        itertemp += 1
                reco = [ij, reco]
                if key in params:
                    params[key] = np.vstack((params[key], [reco]))
                else:
                    params[key] = [reco]

    return params


def write_atom_ascii(atomfile, atom):
    ''' Writes the atom (command style) ascii file into dictionary '''
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
               (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

    def num2roman(num):
        ''' converts integer to roman number '''
        roman = ''
        while num > 0:
            for i, r in num_map:
                while num >= i:
                    roman += r
                    num -= i
        return roman

    import datetime
    datelist = []
    today = datetime.date.today()
    datelist.append(today)
    text = ['# Created on ' + str(datelist[0]) + ' \n' +
            '## ADD HERE NECESSARY INFORMATION \n' +
            '# \n ']
    z = get_atomZ(atom=atom + '_1')
    abund = get_abund(atom=atom)
    awgt = get_atomweight(atom=atom)
    nk = z + 1
    ncont = z
    nlin = 0  # Numero de lineas?
    nfix = 0
    neuv_bins = 6
    euv_bound = [911.7, 753.143, 504.0, 227.800, 193.919, 147.540, 20.0]
    enerlvl = np.zeros(nk)
    for iv in range(1, nk):
        alvl = str(iv)
        # JMS eventually we could use get_excidE
        enerlvl[iv] = get_atomde(atom=atom + '_' + alvl, cm1=True)
    g = np.zeros(nk)  # No clue where to get those...
    levelname = ['noclue' for v in range(0, nk)]
    # No clue where to get those.
    phcross = [
        0.00000000000,
        0.00000000000,
        4.9089501e-18,
        1.6242972e-18,
        1.1120017e-18,
        9.3738273e-19]
    nbin = len(phcross)
    f = open(atomfile, "w")
    f.write(str(text[0]))
    f.write(atom.upper() + "\n")
    text = ['# nk is number of levels, continuum included \n' +
            '# nlin is number of spectral lines in detail \n' +
            '# ncont is number of continua in detail \n' +
            '# nfix is number of fixed transitions \n' +
            '#   ABUND     AWGT \n']
    f.write(str(text[0]))
    f.write(
        '    {0:5.2f}'.format(
            float(abund)) +
        '   {0:5.2f}'.format(
            float(awgt)) +
        '\n')
    f.write('#  NK NLIN NCNT NFIX \n')
    f.write(
        '    {0:3d}'.format(nk) +
        '  {0:3d}'.format(nlin) +
        '  {0:3d}'.format(ncont) +
        '  {0:3d}'.format(nfix) +
        '\n')

    text = [
        "# E[cm-1]    g                  label[20]         stage   levelNo \n" +
        "#                     '----|----|----|----|----'\n"]
    f.write(str(text[0]))
    for iv in range(0, nk):
        f.write(
            '    {0:10.3f}'.format(
                enerlvl[iv]) +
            '  {0:4.2}'.format(
                g[iv]) +
            ' {:2}'.format(
                atom.upper()))
        f.write(' {:5}'.format(num2roman(iv + 1)))
        f.write(
            ' {:12}'.format(
                levelname[iv]) +
            '  {0:3d}'.format(
                iv +
                1) +
            '  {0:3d}'.format(
                iv +
                1) +
            '\n')  # the two iv are wrong at the end...

    text = ['# \n' +
            '# photoionization cross sections for continua  \n' +
            '# number of neuv_bins, \n']
    f.write(str(text[0]))
    f.write('  ' + str(neuv_bins) + ' \n')
    f.write('# bin boundaries\n')
    for iv in range(0, 7):
        f.write('   {0:7.3f}'.format(euv_bound[iv]) + '\n')
    for iv in range(0, nk - 1):
        f.write('# i j \n')
        f.write('  {0:2d}'.format(iv + 1) + '  {0:2d}'.format(iv + 2) + ' \n')
        f.write('# bin       sigma \n')
        for it in range(0, nbin):
            f.write(
                '  {0:2d}'.format(it) +
                '    {0:9.7e}'.format(
                    phcross[it]) +
                ' \n')
    f.write(' GENCOL \n')
    text = [
        '# Collisional excitation 304  \n' +
        '# data from Chianti. Maximum error about 3%. \n' +
        '# upsilon as function of (natural) logarithmic temp. \n' +
        '# rate coefficient: c = ne * upsilon * 8.63e-6 / sqrt(T) * exp(-E/kt) \n']
    f.write(str(text[0]))
    f.write(' CEXC \n')
    ncexc = nbin  # No clue how to get this.
    cexc = [5.9381e+00, -2.8455e+00, 5.4103e-01, -
            4.9805e-02, 2.1959e-03, -3.6167e-05]
    f.write('   ' + str(ncexc) + '\n')
    for iv in range(0, ncexc):
        f.write('   {0:10.4e}'.format(cexc[iv]) + '\n')
    f.write('# \n')
    AR85 = [24.60, 17.80, -11.00, 7.00, -23.20]  # No clue how to get this.
    for iv in range(0, nk - 1):
        f.write(' AR85-CDI \n')
        f.write('  {0:2d}'.format(iv + 1) + '  {0:2d}'.format(iv + 2) + ' \n')
        f.write('  {0:2d}'.format(1) + ' \n')  # No clue how to get this.
        for it in range(0, 5):
            f.write('   {0:6.2f}'.format(AR85[it]))
        f.write('\n')
    f.write(' TEMP \n')
    temp = [1000.0000, 1063.7600, 1131.6000, 1203.7500, 128000.50]
    f.write('   {0:2d}'.format(len(temp)) + '\n')
    it = 0
    while it < len(temp):
        for iv in range(0, 5):
            f.write('  {0:10g}'.format(temp[it]))
            it += 1
            if it == len(temp):
                continue
        f.write('\n')
    reco = [
        1.2960009e-13,
        1.2917243e-13,
        1.2871904e-13,
        1.2823854e-13,
        1.2772931e-13]
    for iv in range(0, nk - 1):
        f.write(' RECO \n')
        f.write('  ' + str(iv + 1) + '  ' + str(iv + 2) + ' \n')
        it = 0
        while it < len(reco):
            for iv in range(0, 5):
                f.write('  {0:9.7e}'.format(reco[it]))
                it += 1
                if it == len(reco):
                    continue
            f.write('\n')
    f.write('END')
    f.close()

    import shutil
    shutil.copy(atomfile, 'temp.atom')

    add_voro_atom('temp.atom', atomfile, atom=atom)


def diper2eb_atom_ascii(atomfile, output):
    ''' Writes the atom (command style) ascii file into dictionary '''
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
               (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

    def num2roman(num):
        ''' converts integer to roman number '''
        roman = ''
        while num > 0:
            for i, r in num_map:
                while num >= i:
                    roman += r
                    num -= i
        return roman

    def copyfile(scr, dest):
        import shutil
        try:
            shutil.copy(scr, dest)
        except shutil.Error as e:  # scr and dest same
            print('Error: %s' % e)
        except IOError as e:  # scr or dest does not exist
            print('Error: %s' % e.strerror)

    import datetime

    datelist = []
    today = datetime.date.today()
    datelist.append(today)
    ''' Writes the atom (command style) ascii file into dictionary '''
    text0 = ['# Created on ' +
             str(datelist[0]) +
             ' \n' +
             '# with diper2eb_atom_ascii only for ground ionized levels \n' +
             '# the atom file has been created using diper 1.1, REGIME=1, APPROX=1 \n']

    neuv_bins = 6
    euv_bound = [911.7, 753.143, 504.0, 227.800, 193.919, 147.540, 20.0]
    # No clue where to get those.
    phcross = [
        0.00000000000,
        0.00000000000,
        4.9089501e-18,
        1.6242972e-18,
        1.1120017e-18,
        9.3738273e-19]
    nbin = len(phcross)
    copyfile(atomfile, output)
    f = open(output, "r")
    data = f.readlines()
    f.close()
    for v in range(0, len(data)):
        data[v] = data[v].replace("*", "#")
    data = data[0:2] + [str(data[2]).upper()] + [data[3]] + \
        [str(data[4]).upper()] + data[5:]
    data = text0 + data
    text = ['# nk is number of levels, continuum included \n' +
            '# nlin is number of spectral lines in detail \n' +
            '# ncont is number of continua in detail \n' +
            '# nfix is number of fixed transitions \n' +
            '#   ABUND   AWGT \n']
    data = data[0:3] + text + data[4:]
    line = data[2]
    line = line.split(';')[0].split(' ')
    while '' in line:
        line.remove('')
    atom = str(line[0])
    atom = atom.replace("\n", "")
    data[2] = ' ' + atom + '\n'
    data = data[:5] + ['#    NK NLIN NCNT NFIX \n'] + data[6:]
    line = data[6]
    line = line.split(';')[0].split(' ')
    while '' in line:
        line.remove('')

    nk = int(line[0])
    nlin = int(line[1])
    ncont = int(line[2])
    nfix = int(line[3])

    data[6] = '    {0:3d}'.format(nk) + '  {0:3d}'.format(nlin) + \
        '  {0:3d}'.format(ncont) + '  {0:3d}'.format(nfix) + '\n'

    text = [
        "#        E[cm-1]     g              label[35]                   stg  lvlN \n" +
        "#                        '----|----|----|----|----|----|----|'\n"]
    data = data[0:7] + text + data[7:]

    for iv in range(8, 8 + nk):
        line = data[iv]
        line = line.split(';')[0].split(' ')
        while '' in line:
            line.remove('')
        while "'" in line:
            line.remove("'")
        line[2] = line[2].replace("'", "")
        strlvl = [" ".join(line[v].strip()
                           for v in range(2, np.size(line) - 1))]
        data[iv] = ('    {0:13.3f}'.format(float(line[0])) + '  {0:5.2f}'.format(float(line[1])) + " ' {0:2}".format(atom.upper()) + ' {0:5}'.format(num2roman(
            int(line[-1]))) + ' {0:26}'.format(strlvl[0]) + "'  {0:3d}".format(int(line[-1])) + '   {0:3d}'.format(iv - 7) + '\n')  # the two iv are wrong at the end...

    headers = [
        'GENCOL',
        'CEXC',
        'AR85-CDI',
        'AR85-CEA',
        'AR85-CH',
        'AR85-CHE',
        'CI',
        'CE',
        'CP',
        'OHM',
        'BURGESS',
        'SPLUPS',
        'SHULL82',
        'TEMP',
        'RECO',
        'VORONOV',
        'EMASK']  # Missing AR85-RR, RADRAT, SPLUPS5, I think AR85-CHE is not used in OOE
    DONE = 'AR85-CDI', 'AR85-CH', 'AR85-CHE', 'SHULL82'

    textar85cdi = [
        '# Data for electron impact ionization Arnaud and Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n' +
        '# 1/(u I^2) (A (1 - 1/u) + B (1 - 1/u)^2) + C ln(u) + D ln(u)/u) (cm^2)  \n' +
        '#   i   j \n']

    textar85cdishell = ['# Numbers of shells \n']
    textar85cdiparam = ['# dE(eV)  A   B   C   D \n']

    textar85ct = [
        '# Data for charge transfer rate of ionization and recombination Arnaud and Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n']
    textar85cea = [
        '# Data authoionization following excitation Arnaud and Rothenflug (1985) \n' +
        '# (this is a bit of caos... uses different expression for different species) See appendix A.  \n' +
        '#   i   j \n']

    textshull82 = [
        '# Recombination rate coefficients Shull and Steenberg (1982) \n' +
        '# provides direct collisional ionization with the following fitting: \n' +
        '# Ci  = Acol T^(0.5) (1 + Ai T / Tcol)^(-1) exp(-Tcol/T), with Ai ~ 0.1 \n' +
        '# for the recombination rate combines the sum of radiative and dielectronic recombination rate \n' +
        '# alpha_r = Arad (T_4)^(-Xrad) ; and alpha_d = Adi T^(-3/2) exp(-T0/T) (1+Bdi exp(-T1/T))\n' +
        '#   i  j   Acol     Tcol     Arad     Xrad      Adi      Bdi       T0       T1 \n']

    textar85ch = [
        '# charge transfer recombination with neutral hydrogen Arnaud and Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n' +
        '# alpha = a (T_4)^b (1 + c exp(d T_4)) \n' +
        '#   i   j \n']
    textar85chparam = [
        '#   Temperature range (K)   a(1e-9cm3/s)    b      c    d \n']
    textar85chem = [
        '# charge transfer recombination with ionized hydrogen Arnaud and Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n' +
        '# alpha = a (T_4)^b (1 + c exp(d T_4)) \n'
        '#   i   j \n']

    # if 'SHULL82\n' in data:
    try:
        iloc = data.index('SHULL82\n')
        if iloc > 1:
            data = data[0:iloc + 1] + textshull82 + data[iloc + 1:]
    except BaseException:
        print('no key')

    # if 'AR85-CHE+\n' in data:
    try:
        iloc = data.index('AR85-CHE+\n')
        if iloc > 1:
            data = data[0:iloc + 1] + textar85chem + data[iloc + 1:]
            data = data[0:iloc + 3] + textar85chparam + data[iloc + 3:]
    except BaseException:
        print('no key')
    # if 'AR85-CH\n' in data:
    try:
        iloc = data.index('AR85-CH\n')
        if iloc > 1:
            data = data[0:iloc + 1] + textar85ch + data[iloc + 1:]
            data = data[0:iloc + 3] + textar85chparam + data[iloc + 3:]
    except BaseException:
        print('no key')

    # if 'AR85-CDI\n' in data:
    try:
        iloc = data.index('AR85-CDI\n')
        if iloc > 1:
            data = data[0:iloc + 1] + textar85cdi + data[iloc + 1:]
            data = data[0:iloc + 3] + textar85cdishell + data[iloc + 3:]
            data = data[0:iloc + 5] + textar85cdiparam + data[iloc + 5:]
    except BaseException:
        print('no key')

    # if 'AR85-CEA\n' in data:
    try:
        iloc = data.index('AR85-CEA\n')
        if iloc > 1:
            data = data[0:iloc + 1] + textar85cea + data[iloc + 1:]
    except BaseException:
        print('no key')

    '''
    text=['# \n' +
        '# photoionization cross sections for continua  \n' +
        '# number of neuv_bins, \n']
    f.write(str(text[0]))
    f.write('  ' + str(neuv_bins) + ' \n')
    f.write('# bin boundaries\n')
    for iv in range(0,7):
        f.write('   {0:7.3f}'.format(euv_bound[iv]) + '\n')
    for iv in range(0,nk-1):
        f.write('# i j \n')
        f.write('  {0:2d}'.format(iv+1) + '  {0:2d}'.format(iv + 2) + ' \n')
        f.write('# bin       sigma \n')
        for it in range(0,nbin):
            f.write('  {0:2d}'.format(it) + '    {0:9.7e}'.format(phcross[it]) + ' \n')
    f.write(' GENCOL \n')
    text = ['# Collisional excitation 304  \n' +
        '# data from Chianti. Maximum error about 3%. \n' +
        '# upsilon as function of (natural) logarithmic temp. \n' +
        '# rate coefficient: c = ne * upsilon * 8.63e-6 / sqrt(T) * exp(-E/kt) \n']
    f.write(str(text[0]))
    f.write(' CEXC \n')
    ncexc=nbin # No clue how to get this.
    cexc=[5.9381e+00, -2.8455e+00,5.4103e-01,-4.9805e-02,2.1959e-03,-3.6167e-05]
    f.write('   ' + str(ncexc) + '\n')
    for iv in range(0,ncexc):
        f.write('   {0:10.4e}'.format(cexc[iv]) + '\n')
    f.write('# \n')
    AR85 = [24.60, 17.80, -11.00, 7.00, -23.20] # No clue how to get this.
    for iv in range(0,nk-1):
        f.write(' AR85-CDI \n')
        f.write('  {0:2d}'.format(iv+1) + '  {0:2d}'.format(iv + 2) + ' \n')
        f.write('  {0:2d}'.format(1) + ' \n') # No clue how to get this.
        for it in range(0,5):
            f.write('   {0:6.2f}'.format(AR85[it]))
        f.write('\n')
    f.write(' TEMP \n')
    temp=[1000.0000,1063.7600,1131.6000,1203.7500, 128000.50]
    f.write('   {0:2d}'.format(len(temp)) + '\n')
    it=0
    while it < len(temp):
        for iv in range(0,5):
            f.write('  {0:10g}'.format(temp[it]))
            it += 1
            if it == len(temp): continue
        f.write('\n')
    reco=[1.2960009e-13,1.2917243e-13,1.2871904e-13,1.2823854e-13,1.2772931e-13]
    for iv in range(0,nk-1):
        f.write(' RECO \n')
        f.write('  ' + str(iv+1) + '  ' + str(iv + 2) + ' \n')
        it=0
        while it < len(reco):
            for iv in range(0,5):
                f.write('  {0:9.7e}'.format(reco[it]))
                it += 1
                if it == len(reco): continue
            f.write('\n')
    f.write('END')
    f.close()
    '''

    f = open('temp.atom', "w")
    for i in range(0, len(data)):
        f.write(data[i])

    if 'GENCOL\n' not in data:
        f.write('GENCOL\n')
    else:
        print('gencol is in data')
    f.close()
    add_voro_atom('temp.atom', output, atom=atom.lower(), nk=nk)
