"""
Set of programs to read and interact with atom files focus.
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii
from .bifrost import subs2grph, bifrost_units
from .ebysus import EbysusData
from . import cstagger
import re
import pickle
import shutil
import ChiantiPy.core as ch
import datetime
import periodictable as pt

class atom_tools(object):
    """
    Reads data from atom files in Ebysus format.
    """

    def __init__(self, atom_name='', stage='', atom_file='',
                 voronov_file=os.environ.get(
                     'EBYSUS') + 'INPUT/MISC/voronov.dat'):
        '''
            Init file

            Parameters (INPUT)
            ----------
            atom_file - atom file of interest (in this case, atom_name
                        and stage will be ignore).
            atom_name - lower case letter if the atom of interest.
                        This is mostly just for using some CHIANTI read out.
                        If atom is not defined then it returns a list of all
                        abundances in the file.
                In this case, one will access to it by, e.g., var['h'],
            stage - This is for the CHIANTI read out options.
            voronov_file - path of the Ebysus file that contains all the
                        information about Voronov. For CHIANTI stuff, it is
                        the abundance file, default is voronov.dat without
                        Chianti data base or sun_photospheric_1998_grevesse for
                        Chianti data base.

            Parameters (DATA IN OBJECT)
            ----------
            params - list containing the information of voronov.dat
                    (following the format of read_voro_file)

        '''
        self.voronov_file = voronov_file
        self.read_voro_file()
        self.keyword_atomic = [
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
            'reco']
        if (atom_file != ''):
            self.atom_file = atom_file
            self.read_atom_file()
            self.atom = self.params['atom']
            self.atom = self.atom.lower()
        elif (atom_name != ''):
            self.atom = atom_name.replace("_", "")

            if len(''.join(x for x in atom_name if x.isdigit())) == 1:
                self.stage = int(re.findall('\d+', atom_name)[0])
                self.atom = self.atom.replace("1", "")

            if stage != '':
                self.stage = stage

    def read_voro_file(self):
        '''
        Reads the miscelaneous Vofonov & abundances table formatted
        (command style) ascii file into dictionary
        '''
        li = 0
        self.vor_params = {}
        headers = ['NSPECIES_MAX', 'NLVLS_MAX', 'SPECIES']
        # go through the file, add stuff to dictionary
        ii = 0
        with open(self.voronov_file) as fp:
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
                                '(WWW) read_voro_file: could not find datatype'
                                'in line %i, skipping' %
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
                                    '(WWW) read_voro_file: could not find'
                                    'datatype in line %i, skipping' %
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
                                    '(WWW) read_voro_file: could not find'
                                    'datatype in line %i, skipping' %
                                    li)
                                li += 1
                                continue

                    if not key[0] in self.vor_params:
                        self.vor_params[key[0]] = [val_arr]
                    else:
                        self.vor_params[key[0]].append(val_arr)

                else:
                    print('(WWW) read_voro_file: could not find datatype in '
                          'line %i, skipping' % li)
                    li += 1
                    continue

            self.vor_params['SPECIES'] = np.array(self.vor_params['SPECIES'])

    def get_abund(self,
                  Chianti=False, abundance='sun_photospheric_1998_grevesse'):
        '''
            Returns abundances from the voronov.dat file.

            Parameters
            ----------
            Chianti - if true uses chianti data base, otherwise uses atom files
                    information
        '''

        try:
            return self.params['abund']

        except BaseException:
            if not hasattr(self, 'stage'):
                stage = 1
            else:
                stage = self.stage

            if Chianti:
                if ('.dat' in abundance):
                    abundance = 'sun_photospheric_1998_grevesse'
                self.ion = ch.ion(self.atom + '_' + str(stage),
                                  1e5, abundance=abundance)
                return self.ion.Abundance
            else:

                if (hasattr(self, 'atom_file') and (len(self.atom) > 0)):
                    self.abund_dic = self.vor_params['SPECIES'][[np.where(
                        self.vor_params['SPECIES'][:, 1] == self.atom + str(
                            stage))[0]], 8].astype(np.float)[0][0]
                else:
                    for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                        if not(any(
                                i.isdigit() for i in self.vor_params[
                                        'SPECIES'][ii, 1])):
                            try:
                                abund_dic[self.vor_params['SPECIES'][ii, 1]
                                          ] = self.vor_params['SPECIES'][
                                            ii, 8].astype(np.float)
                            except BaseException:
                                abund_dic = {
                                    self.vor_params['SPECIES'][
                                            ii, 1]: self.vor_params[
                                                'SPECIES'][ii, 8].astype(
                                                        np.float)}

                    self.abund_dic = abund_dic
                return self.abund_dic

    def get_atomweight(self):
        '''
        Returns atomic weights from the voronov.dat file.

        Parameters
        ----------
        '''
        try:
            return self.params['weight']
        except BaseException:
            if (hasattr(self, 'atom_file') and len(self.atom) > 0):
                self.weight_dic = self.vor_params['SPECIES'][[np.where(
                    self.vor_params['SPECIES'][:, 1] == self.atom + str(
                        self.stage))[0]], 2].astype(np.float)[0][0]
            else:
                for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                    if not(any(i.isdigit()
                               for i in self.vor_params['SPECIES'][ii, 1])):
                        try:
                            weight_dic[self.vor_params['SPECIES'][ii, 1]
                                       ] = self.vor_params['SPECIES'][
                                            ii, 2].astype(np.float)
                        except BaseException:
                            weight_dic = {
                                self.vor_params['SPECIES'][
                                    ii, 1]: self.vor_params['SPECIES'][
                                            ii, 2].astype(np.float)}

                self.weight_dic = weight_dic
            return self.weight_dic

    def get_atomde(self, Chianti=True, cm1=False):
        '''
        Returns ionization energy dE from the voronov.dat file.

        Parameters
        ----------
        Chianti - if true uses chianti data base, otherwise uses atom files
                information
        cm1 - boolean and if it is true converts from eV to cm-1
        '''
        if not hasattr(self, 'cm1'):
            self.cm1 = cm1
        else:
            if self.cm1 != cm1:
                self.cm1 = cm1

        if not hasattr(self, 'Chianti'):
            self.Chianti = Chianti
        else:
            if self.Chianti != Chianti:
                self.Chianti = Chianti

        if self.cm1:
            units = 1.0 / (8.621738e-5 / 0.695)
        else:
            units = 1.0

        if Chianti and self.atom != '':
            ion = ch.ion(self.atom + '_' + str(self.stage))
            self.de = ion.Ip * units
            return self.de
        else:

            if (self.atom_file != '') or (len(self.atom) > 0):
                print('get_De', self.atom + str(self.stage))
                self.de = self.vor_params['SPECIES'][[np.where(
                    self.vor_params['SPECIES'][:, 1] == self.atom + str(
                        self.stage))[0]], 3].astype(np.float)[0][0] * units
                return self.de
            else:
                for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                    try:
                        de_dic[self.vor_params['SPECIES'][
                            ii, 1]] = self.vor_params['SPECIES'][ii, 3].astype(
                                np.float) * units
                    except BaseException:
                        de_dic = {self.vor_params['SPECIES'][
                                ii, 1]: self.vor_params['SPECIES'][
                                    ii, 3].astype(np.float) * units}

                self.de_dic = de_dic
                return self.de_dic

    def get_atomZ(self,
                  Chianti=True):
        '''
            Returns atomic number Z from the voronov.dat file.

            Parameters
            ----------
        '''

        if Chianti:
            ion = ch.ion(self.atom + '_' + str(self.stage))
            self.ion = ion
            self.Z = ion.Z
        else:

            if ((self.atom_file == '') > 0) and (len(self.atom) > 0):
                self.z = self.vor_params['SPECIES'][[np.where(self.vor_params[
                    'SPECIES'][:, 1] == self.atom + str(self.stage))[
                        0]], 0].astype(np.int)[0][0]
            else:
                for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                    if not(any(i.isdigit()
                               for i in self.vor_params['SPECIES'][ii, 1])):
                        try:
                            z_dic[self.vor_params['SPECIES'][ii, 1]
                                  ] = self.vor_params['SPECIES'][
                                    ii, 0].astype(np.int)
                        except BaseException:
                            z_dic = {
                                self.vor_params['SPECIES'][
                                    ii, 1]: self.vor_params[
                                        'SPECIES'][ii, 0].astype(np.int)}

                self.z_dic = z_dic

    def get_atomP(self):
        '''
        Returns P parameter for Voronov rate fitting term from the voronov.dat
        file. The parameter P was included to better fit the particular
        cross-section behavior for certain ions near threshold; it only takes
        on the value 0 or 1

        Parameters
        ----------
        atom - lower case letter if the atom of interest.
                If atom is not defined then it returns a list of all P
                parameter in the file. In this case, one will access to it by,
                e.g., var['he2']
        '''

        if ((self.atom_file == '') > 0) and (len(self.atom) > 0):
            self.p = self.vor_params['SPECIES'][[np.where(self.vor_params[
                'SPECIES'][:, 1] == self.atom + str(self.stage))[
                0]], 4].astype(np.int)[0][0]
        else:
            for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                try:
                    p_dic[self.vor_params['SPECIES'][ii, 1]
                          ] = self.vor_params['SPECIES'][ii, 4].astype(np.int)
                except BaseException:
                    p_dic = {
                        self.vor_params['SPECIES'][ii, 1]: self.vor_params[
                            'SPECIES'][ii, 4].astype(np.int)}

            self.p_dic = p_dic

    def get_atomA(self):
        '''
        Returns A parameter for Voronov rate fitting term from the voronov.dat
        file.

        Parameters
        ----------
         '''

        if ((self.atom_file == '') > 0) and (len(self.atom) > 0):
            self.a = self.vor_params['SPECIES'][[np.where(
                self.vor_params['SPECIES'][:, 1] == self.atom + str(
                    self.stage))[0]], 5].astype(np.float)[0][0]
        else:
            for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                try:
                    a_dic[self.vor_params['SPECIES'][ii, 1]
                          ] = self.vor_params['SPECIES'][ii, 5].astype(
                            np.float)
                except BaseException:
                    a_dic = {
                        self.vor_params['SPECIES'][ii, 1]: self.vor_params[
                            'SPECIES'][ii, 5].astype(np.float)}

            self.a_dic = a_dic

    def get_atomX(self):
        '''
        Returns X parameter for Voronov rate fitting term from the voronov.dat
        file.

        Parameters
        ----------

        '''

        if ((self.atom_file == '') > 0) and (len(self.atom) > 0):
            self.x = self.vor_params['SPECIES'][[np.where(self.vor_params[
                'SPECIES'][:, 1] == self.atom + str(self.stage))[
                    0]], 6].astype(np.float)[0][0]
        else:
            for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                try:
                    x_dic[self.vor_params['SPECIES'][ii, 1]
                          ] = self.vor_params['SPECIES'][
                            ii, 6].astype(np.float)
                except BaseException:
                    x_dic = {
                        self.vor_params['SPECIES'][ii, 1]: self.vor_params[
                            'SPECIES'][ii, 6].astype(np.float)}

            self.x_dic = x_dic

    def get_atomK():
        '''
        Returns K parameter for Voronov rate fitting term  from the voronov.dat
        file.

        Parameters
        ----------
        params - list containing the information of voronov.dat
                (following the format of read_voro_file)
        atom - lower case letter of the atom of interest.
                If atom is not defined then it returns a list of all K
                parameter in the file.
                In this case, one will access to it by, e.g., var['he2']
        '''

        if ((self.atom_file == '') > 0) and (len(self.atom) > 0):
            self.k = self.vor_params['SPECIES'][[np.where(self.vor_params[
                'SPECIES'][:, 1] == self.atom + str(self.stage))[
                    0]], 7].astype(np.float)[0][0]
        else:
            for ii in range(0, self.vor_params['NLVLS_MAX'][0]):
                try:
                    k_dic[self.vor_params['SPECIES'][
                        ii, 1]] = self.vor_params['SPECIES'][ii, 7].astype(
                                np.float)
                except BaseException:
                    k_dic = {
                        self.vor_params['SPECIES'][ii, 1]: self.vor_params[
                            'SPECIES'][ii, 7].astype(np.float)}

            self.k_dic = k_dic

    def info_atom(self):
        '''
        provides general information about specific atom, e.g., ionization
        and excitation levels etc

        Parameters
        ----------
        atom - lower case letter of the atom of interest, e.g., 'he'
        '''

        ion = ch.ion(self.atom + '_' + str(self.stage))
        Z = ion.Z
        wgt = self.get_atomweight(atom)
        print('Atom', 'Z', 'Weight  ', 'FIP  ', 'Abnd')
        print(self.atom, ' ', Z, wgt, ion.FIP, ion.Abundance)
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

    def get_excidE(self, Chianti=True, cm1=False):
        '''
        Returns ionization energy dE for excited levels

        Parameters
        ----------
        params - list containing the information of voronov.dat (following
            the format of read_voro_file)
        atom - lower case letter if the atom of interest.
            If atom is not defined then it returns a list of all ionization
            energy dE in the file.
            In this case, one will access to it by, e.g., var['he2']
        cm1 - boolean and if it is true converts from eV to cm-1
        '''

        if not hasattr(self, cm1):
            self.cm1 = cm1
        else:
            if self.cm1 != cm1:
                self.cm1 = cm1

        if not hasattr(self, Chianti):
            self.Chianti = Chianti
        else:
            if self.Chianti != Chianti:
                self.Chianti = Chianti

        unitscm1 = 1.0 / (8.621738e-5 / 0.695)
        if self.cm1:
            units = 1.0
        else:
            units = 1. / unitscm1

        ion = ch.ion(self.atom + '_' + str(self.stage))
        if hasattr(ion, 'Elvlc'):
            if self.stage == '':
                self.de = (ion.Ip * unitscm1 + ion.Elvlc['ecmth'][0]) * units
            else:
                self.de = (ion.Ip * unitscm1 +
                           ion.Elvlc['ecmth'][self.stage]) * units
        else:
            print('No Elvlc in the Chianti Data base')

    def rrec(self, ntot, Te, lo_lvl=1, hi_lvl=2,
             GENCOL_KEY='voronov', threebody=False):
        '''
        gives the recombination rate per particle
        Parameters:
        ------
        threebody  - False or the gencol_key for ionization.
        '''

        units = bifrost_units()
        TeV = Te * units.K_TO_EV

        g_ilv = float(self.params['lvl'][lo_lvl][1])
        g_jlv = float(self.params['lvl'][hi_lvl][1])
        dE = float(self.params['lvl'][hi_lvl][0]) - \
            float(self.params['lvl'][lo_lvl][0])
        dE = dE * units.CLIGHT.value * units.HPLANCK.value
        scr1 = dE / Te / units.KBOLTZMANN.value

        if ((threebody) is not False):
            if not hasattr(self, 'frec3bd'):
                self.r3body(ntot, Te, lo_lvl=lo_lvl,
                            hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
            cdn = self.frec3bd
        else:
            cdn = 0

        if (GENCOL_KEY.lower() == 'atomic'):
            keylist = self.keyword_atomic
        else:
            keylist = GENCOL_KEY.lower()
        if np.size(keylist) == 1:
            keylist = [keylist]

        for keyword in keylist:
            if keyword == 'shull82':
                summrs = 1.0
                sqrtte = np.sqrt(Te)
                arrec = Arad * (Te / 1.e4)**(-Xrad)
                adrec = Adi / Te / sqrtte * \
                    exp(-T0 / Te) * (1.e0 + Bdi * np.exp(-T1 / Te))
                cdn = arrec + summrs * adrec + cdn

            elif keyword == 'voronov':  # mcwhirter65
                vfac = 2.6e-19
                self.stage = 1
                self.get_atomZ()
                cdn = vfac * np.sqrt(1.0 / TeV) * self.Z**2 + cdn

        self.cdn = cdn

    def vrec(self, nel, Te, lo_lvl=1, hi_lvl=2,
             GENCOL_KEY='voronov', threebody=False):
        '''
        gives the recombination frequency
        '''
        if not hasattr(self, 'cdn'):
            self.rrec(nel, Te, lo_lvl=lo_lvl, hi_lvl=hi_lvl,
                      GENCOL_KEY=GENCOL_KEY, threebody=threebody)
        self.frec = nel * self.cdn

    def rion(self, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        '''
        gives the ionization rate per particle using Voronov 1997 fitting
        formula
        '''
        units = bifrost_units()
        TeV = Te * units.K_TO_EV

        if (GENCOL_KEY.lower() == 'atomic'):
            keylist = self.keyword_atomic
        else:
            keylist = [GENCOL_KEY.lower()]

        for keyword in keylist:
            if keyword == 'voronov':
                tr_line = np.where(
                    self.params['voronov'][0][:, 0] == lo_lvl + 1)
                # get_atomde(atom, Chianti=False)  # 13.6
                phion = self.params['voronov'][0][tr_line[0], 2]
                # get_atomA(atom) * 1.0e6  # converted to SI 2.91e-14
                A = self.params['voronov'][0][tr_line[0], 4] * 1e6
                # get_atomX(atom)  # 0.232
                X = self.params['voronov'][0][tr_line[0], 5]
                # get_atomK(atom)  # 0.39
                K = self.params['voronov'][0][tr_line[0], 6]
                # get_atomP(atom)  # 0
                P = self.params['voronov'][0][tr_line[0], 3]

                self.cup = A * (1 + np.sqrt(phion / TeV) * P) / (
                        X + phion / TeV) * (phion / TeV)**K * np.exp(
                            -phion / TeV)

            elif keyword == 'ar85-cdi':
                '''  Data for electron impact ionization Arnaud and
                    Rothenflug (1985)
                   1/(u I^2) (A (1 - 1/u) + B (1 - 1/u)^2) + C ln(u) +
                                                            D ln(u)/u) (cm^2)
                        #   i   j
                        # Numbers of shells
                    # dE(eV)  A   B   C   D
                '''
                tr_list = self.params['ar85-cdi'][:, 0][:]
                tr_line = [v for v in tr_list[:][0] if tr_list[v][0] == lo_lvl]
                nshells = var['ar85-cdi'][tr_line][1]
                phion = np.zeros(nshells)
                A = np.zeros(nshells)
                B = np.zeros(nshells)
                C = np.zeros(nshells)
                D = np.zeros(nshells)
                for ishell in range(0, nshells):
                    phion[ishell] = self.params['ar85-cdi'][tr_line][2][
                            ishell][0]
                    A[ishell] = self.params['ar85-cdi'][tr_line][2][ishell][1]
                    B[ishell] = self.params['ar85-cdi'][tr_line][2][ishell][2]
                    C[ishell] = self.params['ar85-cdi'][tr_line][2][ishell][3]
                    D[ishell] = self.params['ar85-cdi'][tr_line][2][ishell][2]

                g_ilv = float(self.params['lvl'][lo_lvl][1])
                g_jlv = float(self.params['lvl'][hi_lvl][1])

                for ishell in range(0, nshells):
                    xj = phion[ishell] * units.EV_TO_ERG / \
                        units.KBOLTZMANN.value / Te
                    fac = np.exp(-xj) * sqrt(xj)
                    fxj = fac * (A[ishell] + B[ishell] * (1. + xj) + (
                        C[ishell] - xj * (A[ishell] + B[ishell] * (
                            2. + xj))) * fone(
                        xj, 0) + D[ishell] * xj * ftwo(xj, 0))
                    fac = 6.69e-07 / TeV / np.sqrt(TeV)
                    cup = cup + fac * fxj

                self.cup = cup

            elif keyword == 'ar85-cea':
                '''
                '''
                tr_list = self.params['ar85-cea'][:, 0][:]
                tr_line = [v for v in tr_list[:][0]
                           [0] if tr_list[v][0][0] == lo_lvl]
                coeff = self.params['ar85-cea'][lo_lvl][1]

            elif keyword == 'burgess':
                '''
                '''
                tr_list = self.params['burgess'][:, 0][:]
                tr_line = [v for v in tr_list[:][0]
                           [0] if tr_list[v][0][0] == lo_lvl]
                coeff = self.params['burgess'][lo_lvl][1]

            elif keyword == 'shull82':
                '''
                    Recombination rate coefficients Shull and Steenberg (1982)
                    provides direct collisional ionization with the following
                    fitting:
                    Ci  = Acol T^(0.5) (1 + Ai T / Tcol)^(-1) exp(-Tcol/T),
                        with Ai ~ 0.1
                    for the recombination rate combines the sum of radiative
                    and dielectronic recombination rate
                    alpha_r = Arad (T_4)^(-Xrad) ; and alpha_d = Adi T^(-3/2)
                    i  j   Acol     Tcol     Arad     Xrad      Adi      Bdi
                        exp(-T0/T) (1+Bdi exp(-T1/T))
                            T0       T1
                '''
                tr_line = [v for v in range(
                    0, len(self.params['shull82'])) if self.params['shull82'][
                            v, 0][0] == lo_lvl]
                Acol = self.params['shull82'][lo_lvl][1][0]
                Tcol = self.params['shull82'][lo_lvl][1][2]
                Arad = self.params['shull82'][lo_lvl][1][3]
                Xrad = self.params['shull82'][lo_lvl][1][4]
                Adi = self.params['shull82'][lo_lvl][1][5]
                Bdi = self.params['shull82'][lo_lvl][1][6]
                T0 = self.params['shull82'][lo_lvl][1][7]
                T1 = self.params['shull82'][lo_lvl][2][0]
                if (T_4 < T0) or (T_4 > T1):
                    print('Warning[ar85-che]: Temperature out of bounds')

                g_ilv = float(self.params['lvl'][lo_lvl][1])
                g_jlv = float(self.params['lvl'][hi_lvl][1])

                dE = float(self.params['lvl'][hi_lvl][0]) - \
                    float(self.params['lvl'][lo_lvl][0])
                dE = dE * units.CLIGHT * units.HPLANCK
                scr1 = dE / Te / units.KBOLTZMANN.value

                sqrtte = np.sqrt(Te)
                cup = 0.
                if (Acol != 0.):
                    cup = Acol * sqrtte * \
                        np.exp(-Tcol / Te) / (1.e0 + 0.1 * Te / Tcol)

                self.cup = cup

            elif keyword == 'ar85-ch':
                '''
                charge transfer recombination with neutral hydrogen Arnaud
                and Rothenflug (1985) updated for Fe ions by Arnaud and
                Rothenflug (1992)
                alpha = a (T_4)^b (1 + c exp(d T_4))
                i   j
                Temperature range (K)   a(1e-9cm3/s)    b      c    d
                '''
                tr_line = [v for v in range(
                    0, len(self.params['ar85-ch'])) if self.params['ar85-ch'][
                            v][0][1] == lo_lvl]
                T0 = self.params['ar85-ch'][lo_lvl][1][0]
                T1 = self.params['ar85-ch'][lo_lvl][1][1]
                a = self.params['ar85-ch'][lo_lvl][1][2]
                b = self.params['ar85-ch'][lo_lvl][1][3]
                c = self.params['ar85-ch'][lo_lvl][1][4]
                d = self.params['ar85-ch'][lo_lvl][1][5]
                if (T_4 < T0) or (T_4 > T1):
                    print('Warning[ar85-che]: Temperature out of bounds')
                # self.cup = a (T_4)**b * (1 + c * np.exp(d * T_4))

            elif keyword == 'ar85-che':
                ''' same as above '''
                tr_line = [v for v in range(
                    0, len(self.params['ar85-che'])) if self.params[
                            'ar85-che'][v][0][1] == lo_lvl]
                T0 = self.params['ar85-che'][lo_lvl][1][0]
                T1 = self.params['ar85-che'][lo_lvl][1][1]
                a = self.params['ar85-che'][lo_lvl][1][2]
                b = self.params['ar85-che'][lo_lvl][1][3]
                c = self.params['ar85-che'][lo_lvl][1][4]
                d = self.params['ar85-che'][lo_lvl][1][5]
                if (T_4 < T0) or (T_4 > T1):
                    print('Warning[ar85-che]: Temperature out of bounds')
                # self.cup = a (T_4)**b * (1 + c * np.exp(d * T_4))

    def vion(self, nel, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        '''
        gives the ionization frequency using Voronov 1997 fitting formula
        '''
        if not hasattr(self, 'cup'):
            self.rion(Te, lo_lvl=lo_lvl, hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
        self.fion = nel * self.cup

    def ionfraction(self, ntot, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        ''' gives the ionization fraction using vrec and vion'''
        if not hasattr(self, 'cup'):
            self.rion(Te, lo_lvl=lo_lvl, hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
        if not hasattr(self, 'cdn'):
            self.rrec(ntot, Te, lo_lvl=lo_lvl,
                      hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)

        self.ionfrac = self.fion / (self.rec + 2.0 * self.fion)

    def ionse(self, ntot, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        ''' gives electron or ion number density using vrec and vion'''
        if not hasattr(self, ionfrac):
            self.ionfraction(ntot, Te, lo_lvl=lo_lvl,
                             hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
        self.ion_ndens = ntot * self.ionfrac

    def neuse(self, ntot, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        ''' gives neutral number density using vrec and vion'''
        if not hasattr(self, ionfrac):
            self.ionse(Te, lo_lvl=lo_lvl, hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
        self.neu_ndens = ntot - 2.0 * self.ion_ndens

    def r3body(self, nel, Te, lo_lvl=1, hi_lvl=2, GENCOL_KEY='voronov'):
        ''' three body recombination '''
        units = bifrost_units()
        gst_hi = float(self.params['lvl'][lo_lvl][1])  # 2.0
        gst_lo = float(self.params['lvl'][hi_lvl][1])  # 1.0

        if not hasattr(self, 'cup'):
            self.vion(nel, Te, lo_lvl=lo_lvl,
                      hi_lvl=hi_lvl, GENCOL_KEY=GENCOL_KEY)
        '''
        if lo_lvl > 0:
            self.stage = lo_lvl + 1
        else:
            self.stage = ''
        dekt = self.get_atomde(Chianti=False)
        dekt = float(self.get_atomde(Chianti=False))/ units.K_TO_EV / Te
        '''
        dekt = float(self.params['lvl'][hi_lvl][0]) - \
            float(self.params['lvl'][lo_lvl][0])
        dekt = dekt * units.CLIGHT.value * \
            units.HPLANCK.value / Te / units.KBOLTZMANN.value

        # Assuming nel in cgs. (For SI units would be 2.07e-22)
        saha = 2.07e-16 * nel * gst_lo / gst_hi * Te**(-1.5) * np.exp(dekt)
        self.frec3bd = saha * self.cup  # vion is collisional ionization rate

    def inv_pop_atomf(self, ntot, Te, niter=100, nel=None,
                      threebody=True, GENCOL_KEY='voronov'):
        ''' Inverts the Matrix for Statistical Equilibrum'''
        # nel starting guess is:
        if nel is None:
            nel = ntot * 1.0
        shape = np.shape(ntot)
        nelf = np.ravel(nel)
        ntotf = np.ravel(ntot)
        tef = np.ravel(Te)
        npoints = len(tef)
        nlevels = len(self.params['lvl'])
        n_isp = np.zeros((npoints, nlevels))
        for ipoint in range(0, npoints):
            if (ipoint * 100 / (1.0 * npoints) in np.linspace(0, 99, 100)):
                print('Done %s grid points of %s' %
                      (str(ipoint), str(npoints)))
            for iel in range(1, niter):
                B = np.zeros((nlevels))
                A = np.zeros((nlevels, nlevels))
                igen = 0
                for ilev in range(0, nlevels - 1):
                    if hasattr(self, 'cdn'):
                        delattr(self, 'cdn')
                    if hasattr(self, 'cup'):
                        delattr(self, 'cup')
                    self.vrec(nelf[ipoint], tef[
                        ipoint], lo_lvl=ilev, hi_lvl=ilev + 1,
                            threebody=threebody, GENCOL_KEY=GENCOL_KEY)
                    self.vion(nelf[ipoint], tef[
                        ipoint], lo_lvl=ilev, hi_lvl=ilev + 1,
                        GENCOL_KEY=GENCOL_KEY)
                    # print(nelf[ipoint],tef[ipoint],self.cdn,self.cup)
                    if igen == 0:  # JMS Not sure what is that for....
                        Rip = self.frec
                    else:
                        Rip += self.frec
                    Cip = self.fion
                    A[ilev, ilev] += - Cip
                    A[ilev, ilev + 1] = Rip  # + Ri3d #Ri3d!?
                    if ilev < nlevels - 2:
                        A[ilev + 1, ilev + 1] = - Rip  # - Ri3d
                        A[ilev + 1, ilev] = Cip
                    igen = 1
                A[ilev + 1, :] = 1.0
                B[ilev + 1] = ntotf[ipoint]
                n_isp[ipoint, :] = np.linalg.solve(A, B)
                nelpos = 0.0
                for ilev in range(1, nlevels):
                    nelpos += n_isp[ipoint, ilev] * ilev
                if (nelf[ipoint] - nelpos) / (nelf[ipoint] + nelpos) < 1e-4:
                    # print("Jump iter with iter = ",iel)
                    nelf[ipoint] = nelpos
                    break
                if (iel == niter - 1):
                    if (nelf[ipoint] - nelpos) / \
                            (nelf[ipoint] + nelpos) > 1e-4:
                        print("Warning, No stationary solution was found",
                              (nelf[ipoint] - nelpos) / (
                                nelf[ipoint] + nelpos), nelpos, nelf[ipoint])
                nelf[ipoint] = nelpos
        self.n_el = np.reshape(nelf, (shape))
        self.n_isp = np.reshape(n_isp, (np.append(shape, nlevels)))

    def read_atom_file(self):
        '''
        Reads the atom (command style) ascii file into dictionary
        '''
        def readnextline(lines, lp):
            line = lines[lp]
            while line == '\n':
                lp += 1
                line = lines[lp]
            while len(line) < 1 or line[0] == '#' or line[0] == '*':
                lp += 1
                line = lines[lp]
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
        # nl = sum(1 for line in open(atomfile))
        f = open(self.atom_file)
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
            'EMASK']  # Missing AR85-RR, RADRAT, SPLUPS5. AR85-CHE not used
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
        lines = f.readlines()
        f.close()
        for il in range(0, len(lines)):  # for line in iter(f):
            # ignore empty lines and comments
            line = lines[il]
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
                        string = [" ".join(
                            line[v].strip() for v in range(
                                3, np.size(line) - 3))]
                        nk += 1
                        if 'lvl' in params:
                            params['lvl'] = np.vstack((params['lvl'], [float(
                                line[0].strip()), float(
                                line[1].strip()), string[0], int(
                                line[-2].strip()), int(line[-1].strip())]))
                        else:
                            params['lvl'] = [float(line[0].strip()), float(
                                line[1].strip()), string[0], int(
                                line[-2].strip()), int(line[-1].strip())]
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
                                    (params['line'], [int(
                                        line[0].strip()), int(
                                        line[1].strip()), float(
                                        line[2].strip()), int(
                                        line[3].strip()), float(
                                        line[4].strip()), line[5].strip()]))
                            else:
                                params['line'] = [int(line[0].strip()), int(
                                    line[1].strip()), float(
                                    line[2].strip()), int(
                                    line[3].strip()), float(
                                    line[4].strip()), line[5].strip()]
                        continue
                    elif ncon < int(params['ncnt']):
                        ncon += 1
                        if len(line) > 2:  # this is for HION standards
                            if 'cont' in params:
                                params['cont'] = np.vstack(
                                    (params['cont'], [int(
                                        line[0].strip()), int(
                                        line[1].strip()), float(
                                        line[2].strip()), int(
                                        line[3].strip()), float(
                                        line[4].strip()), line[5].strip()]))
                            else:
                                params['cont'] = [int(line[0].strip()), int(
                                    line[1].strip()), float(
                                    line[2].strip()), int(
                                    line[3].strip()), float(
                                    line[4].strip()), line[5].strip()]
                        else:
                            ii = 4  # this is for Helium format
                        continue
                    if nk == int(params['nk']) - 1 and nlin == int(
                            params['nlin']) - 1 and ncnt == int(
                            params['ncon']) - 1:
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
                            (params['photioncross'][tr], [int(
                                line[0].strip()), float(line[1].strip())]))
                    except BaseException:
                        params['photioncross'][tr] = [
                            int(line[0].strip()), float(line[1].strip())]

        if 'bin_euv' not in params:
            # JMS default from HION, however, this should be check from HION.
            params['bin_euv'] = [
                6, [911.7, 753.143, 504.0, 227.800, 193.919, 147.540, 20.0]]

        lp = li
        while True:
            line, lp = readnextline(lines, lp)
            if(line[0].strip().lower() == 'end'):
                break
            # if line == "":
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
                            temp = [float(line[v].strip())
                                    for v in range(0, 5)]
                        else:
                            temp = [temp, [float(
                                    line[v].strip()) for v in range(0, 5)]]
                    temp = [temp0, niter, temp]
                    if key in params:
                        params[key] = np.vstack((params[key], [temp]))
                    else:
                        params[key] = [temp]

                elif((line[0].strip().lower() == 'ar85-cea') or (
                        line[0].strip().lower() == 'burgess')):
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

                elif(line[0].strip().lower() == 'ar85-ch') or (
                        line[0].strip().lower() == 'ar85-che'):
                    key = line[0].strip().lower()
                    line, lp = readnextline(lines, lp)
                    temp0 = [int(line[0].strip()), int(line[1].strip())]
                    line, lp = readnextline(lines, lp)
                    temp = [float(line[v].strip()) for v in range(0, 6)]
                    temp = [temp0, temp]
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
                    trans = [int(line[v].strip()) for v in range(0, 2)]
                    temp = [float(line[v].strip()) for v in range(2, 9)]
                    line, lp = readnextline(lines, lp)
                    temp = [trans, temp, [float(line[0].strip())]]
                    if key in params:
                        params[key] = np.vstack((params[key], [temp]))
                    else:
                        params[key] = [temp]

                elif(line[0].strip().lower() == 'voronov'):
                    key = line[0].strip().lower()
                    line, lp = readnextline(lines, lp)
                    z = int(line[0].strip())
                    vorpar = np.zeros((z, 7))
                    for iterv in range(0, z):
                        line, lp = readnextline(lines, lp)
                        vorpar[iterv, 0] = int(line[0].strip())
                        vorpar[iterv, 1] = int(line[1].strip())
                        vorpar[iterv, 2] = float(line[2].strip())
                        vorpar[iterv, 3] = int(line[3].strip())
                        vorpar[iterv, 4] = float(line[4].strip())
                        vorpar[iterv, 5] = float(line[5].strip())
                        vorpar[iterv, 6] = float(line[6].strip())
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

                elif (line[0].strip().lower() in [
                        'ci', 'ohm', 'ce', 'cp', 'reco']):

                    key = line[0].strip().lower()
                    line, lp = readnextline(lines, lp)
                    if not hasattr(params, key):
                        params[key] = {}
                    ij = line[0].strip() + line[1].strip()
                    params[key][ij] = {}
                    params[key][ij]['tr'] = [
                        int(line[0].strip()), int(line[1].strip())]
                    params[key][ij]['temp'] = temp
                    params[key][ij]['ntemp'] = nitert

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
                    params[key][ij]['cros'] = reco

        self.params = params

'''
TOOLS
'''


def pop_over_species_atomf(ntot, Te, atomfiles=['H_2.atom', 'He_3.atom'],
                           threebody=True, GENCOL_KEY='voronov'):
    '''
    this will do the SE for many species taking into account their abundances
    '''
    units = bifrost_units()
    totabund = 0.0
    for isp in range(0, len(atomfiles)):
        atominfo = atom_tools(atom_file=atomfiles[isp])
        totabund += 10.0**atominfo.get_abund(Chianti=True)

    all_pop_species = {}
    nel = 0
    for isp in range(0, len(atomfiles)):
        print('Starting with atom', atomfiles[isp])
        atominfo = atom_tools(atom_file=atomfiles[isp])
        abund = np.array(10.0**atominfo.get_abund(Chianti=True))
        atomweight = atominfo.get_atomweight() * units.AMU
        n_species = np.zeros((np.shape(ntot)))
        n_species = ntot * (np.array(abund / totabund))
        atominfo.inv_pop_atomf(n_species, Te, niter=100,
                               threebody=threebody, GENCOL_KEY=GENCOL_KEY)
        all_pop_species[atominfo.atom] = atominfo.n_isp
        nel += atominfo.n_el
        print('Done with atom', atomfiles[isp])
    all_pop_species['nel'] = nel
    return all_pop_species


def diper2eb_atom_ascii(atomfile, output):
    '''
    Writes the atom (command style) ascii file into dictionary
    '''
    num_map = [
            (1000, 'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I')]

    def num2roman(num):
        '''
        converts integer to roman number
        '''
        roman = ''
        while num > 0:
            for i, r in num_map:
                while num >= i:
                    roman += r
                    num -= i
        return roman

    def copyfile(scr, dest):
        try:
            shutil.copy(scr, dest)
        except shutil.Error as e:  # scr and dest same
            print('Error: %s' % e)
        except IOError as e:  # scr or dest does not exist
            print('Error: %s' % e.strerror)

    datelist = []
    today = datetime.date.today()
    datelist.append(today)
    ''' Writes the atom (command style) ascii file into dictionary '''
    text0 = ['# Created on ' +
             str(datelist[0]) +
             ' \n' +
             '# with diper2eb_atom_ascii only for ground ionized levels \n' +
             '# the atom file has been created using diper 1.1, ' +
             '# REGIME=1, APPROX=1 \n']

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
    data = data[0: 2] + [str(data[2]).upper()] + [
            data[3]] + [str(data[4]).upper()] + data[5:]
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
    data = data[: 5] + ['#    NK NLIN NCNT NFIX \n'] + data[6:]
    line = data[6]
    line = line.split(';')[0].split(' ')
    while '' in line:
        line.remove('')

    nk = int(line[0])
    nlin = int(line[1])
    ncont = int(line[2])
    nfix = int(line[3])

    data[6] = '    {0:3d}'.format(nk) + '  {0:3d}'.format(
            nlin) + '  {0:3d}'.format(ncont) + '  {0:3d}'.format(nfix) + '\n'

    text = [
        "#        E[cm-1]     g              label[35]                   " +
        "stg  lvlN \n" +
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
        # the two iv are wrong at the end...
        data[iv] = ('    {0:13.3f}'.format(float(
            line[0])) + '  {0:5.2f}'.format(float(
                line[1])) + " ' {0:2}".format(atom.upper()) + ' {0:5}'.format(
                    num2roman(int(line[-1]))) + ' {0:26}'.format(
                        strlvl[0]) + "'  {0:3d}".format(int(
                            line[-1])) + '   {0:3d}'.format(iv - 7) + '\n')

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
        'EMASK']  # Missing AR85-RR, RADRAT, SPLUPS5. AR85-CHE is not used
    DONE = 'AR85-CDI', 'AR85-CH', 'AR85-CHE', 'SHULL82'

    textar85cdi = [
        '# Data for electron impact ionization Arnaud and Rothenflug  \n' +
        '# (1985) updated for Fe ions by Arnaud and Rothenflug (1992) \n' +
        '# 1/(u I^2) (A (1 - 1/u) + B (1 - 1/u)^2) + C ln(u) + ' +
        ' D ln(u)/u) (cm^2)  \n' +
        '#   i   j \n']

    textar85cdishell = ['# Numbers of shells \n']
    textar85cdiparam = ['# dE(eV)  A   B   C   D \n']

    textar85ct = [
        '# Data for charge transfer rate of ionization and recombination' +
        '# Arnaud and Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n']
    textar85cea = [
        '# Data authoionization following excitation Arnaud and ' +
        'Rothenflug (1985) \n' +
        '# (this is a bit of caos... uses different expression for different' +
        '# species) See appendix A.  \n' +
        '#   i   j \n']

    textshull82 = [
        '# Recombination rate coefficients Shull and Steenberg (1982) \n' +
        '# provides direct collisional ionization with the following \n' +
        '# fitting: \n' +
        '# Ci  = Acol T^(0.5) (1 + Ai T / Tcol)^(-1) exp(-Tcol/T), with\n' +
        '# Ai ~ 0.1 for the recombination rate combines the sum of radiative\n'
        '# + and dielectronic recombination rate \n' +
        '# alpha_r = Arad (T_4)^(-Xrad) ; and alpha_d = Adi T^(-3/2)' +
        ' exp(-T0/T) (1+Bdi exp(-T1/T))\n' +
        '#   i  j   Acol     Tcol     Arad     Xrad      Adi      Bdi      ' +
        ' T0       T1 \n']

    textar85ch = [
        '# charge transfer recombination with neutral hydrogen Arnaud\n' +
        '#  and Rothenflug (1985) updated for Fe ions by Arnaud and \n' +
        '# Rothenflug (1992) \n' +
        '# alpha = a (T_4)^b (1 + c exp(d T_4)) \n' +
        '#   i   j \n']
    textar85chparam = [
        '#   Temperature range (K)   a(1e-9cm3/s)    b      c    d \n']
    textar85chem = [
        '# charge transfer recombination with ionized hydrogen Arnaud and \n' +
        '# Rothenflug (1985) \n' +
        '# updated for Fe ions by Arnaud and Rothenflug (1992) \n' +
        '# alpha = a (T_4)^b (1 + c exp(d T_4)) \n' +
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

    f = open('temp.atom', "w")
    for i in range(0, len(data)):
        f.write(data[i])

    if 'GENCOL\n' not in data:
        f.write('GENCOL\n')
    else:
        print('gencol is in data')
    f.close()
    add_voro_atom('temp.atom', output, atom=atom.lower(), nk=nk)


def add_voro_atom(
        inputfile,
        outputfile,
        atom='',
        vorofile=os.environ.get('EBYSUS') + 'INPUT/MISC/voronov.dat',
        nk='100'):
    '''
    Add voronov information at the end of the atom file.

    Parameters
    ----------
    inputfile - name of the input atom file
    outputfile - name of the output atom file which will include the VORONOV
            information
    atom - lower case letters of the atom of interest. Make sure that it
            matches with the atom file.
    vorofile - voronot table file.
    '''

    shutil.copy(inputfile, outputfile)
    atom = atom.lower()
    params = read_voro_file(vorofile)
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
        "# from Voronov fit formula for ionization rates by \n" +
        "# electron impact by G. S. Voronov: \n" +
        "# ATOMIC DATA AND NUCLEAR DATA TABLES 65, 1-35 (1997) \n" +
        "# ARTICLE NO. DT970732\n" +
        "# <cross> = A (1+P*U^(1/2))/(X + U)*U^K e-U (cm3/s) with U = dE/Te\n")

    strat = ''
    if len(''.join(x for x in atom if x.isdigit())) == 0:
        strat = '_1'

    # where I need to add a check if atom is the same as the one in the atom
    # file or even better use directly the atom file info for this.
    atominfo = atom_tools(atom_file=inputfile)
    atominfo.get_atomZ(atom=atom + strat)
    # tr_line=np.where(atominfo.params['voronov'][0][:,0] == lo_lvl+1)
    # phion = self.params['voronov'][0][tr_line[0],2] # get_atomde(atom,
    # Chianti=False)  # 13.6
    # A = self.params['voronov'][0][tr_line[0],4] * 1e6 # get_atomA(atom) *
    # 1.0e6  # converted to SI 2.91e-14
    # X = self.params['voronov'][0][tr_line[0],5] # get_atomX(atom)  # 0.232
    # K = self.params['voronov'][0][tr_line[0],6] # get_atomK(atom)  # 0.39
    # P = self.params['voronov'][0][tr_line[0],3] # get_atomP(atom)  # 0

    f.write(str(atominfo.Z) + "\n")
    jj = 1
    f.write('#   i    j    dE(eV)     P  A(cm3/s)   X      K  \n')
    for ii in range(0, params['NLVLS_MAX'][0]):
        if (atominfo.Z == int(params['SPECIES'][ii, 0])) and jj < nk:
            strat = ''
            if len(''.join(
                    x for x in params['SPECIES'][ii, 1] if x.isdigit())) == 0:
                strat = '_1'
            f.write(' {0:3d}'.format(jj) +
                    ' {0:3d}'.format(jj + 1) +
                    ' {0:9.3f}'.format(get_atomde(
                            atom=params['SPECIES'][
                                ii, 1] + strat, Chianti=False)) +
                    ' {0:3}'.format(get_atomP(atom=params['SPECIES'][ii, 1])) +
                    ' {0:7.3e}'.format(get_atomA(
                        atom=params['SPECIES'][ii, 1])) +
                    ' {0:.3f}'.format(get_atomX(
                        atom=params['SPECIES'][ii, 1])) +
                    ' {0:.3f}'.format(get_atomK(
                        atom=params['SPECIES'][ii, 1])) +
                    '\n')
            jj += 1
    f.write("END")
    f.close()


def create_goftne_tab(ionstr='fe_14',
                      wvlr=[98, 1600],
                      abundance='sun_photospheric_1998_grevesse'):
    '''
    This allows to calculate GOFT tables in a similar fashion as we do in IDL.
    '''

    ntemp = 501
    neden = 71
    temp = 10.**(4. + 0.01 * np.arange(ntemp))
    press = 10**(np.arange(neden) * 0.1 + 12)
    gofnt = np.zeros((ntemp, neden))
    for iden in range(0, neden):
        ion = ch.ion(ionstr, temperature=temp,
                     eDensity=press[iden] / temp, abundance=abundance)
        ion.populate()
        ion.intensity()
        ion.gofnt(wvlRange=wvlr, top=1, plot=False)
        if iden == 0:
            print('Doing wvl=', ion.Gofnt['wvl'])
        gofnt[:, iden] = ion.Gofnt['gofnt']

    ion.Gofnt['gofnt'] = gofnt
    ion.Gofnt['press'] = press
    try:
        ion.mass = pt.elements[ion.Z].ion[ion.Ion].mass
    except BaseException:
        ion.mass = pt.elements[ion.Z].mass
    path = os.environ['BIFROST'] + '/PYTHON/br_int/br_ioni/data/'
    name = getattr(ion, 'IonStr') + '_' + str(ion.Gofnt['wvl']) + '.opy'
    filehandler = open(path + name, 'wb')
    pickle.dump(ion, filehandler)


def add_goftab(filenames=None):

    nf = np.size(filenames)
    for ifile in range(0, nf):
        fileh = open(filenames[ifile], 'rb')
        table = pickle.load(fileh)
        if ifile == 0:
            goftab = table.Gofnt['gofnt']
        else:
            goftab += table.Gofnt['gofnt']
    return goftab


def norm_ne_goftab(table=None):

    nne = np.shape(table)[1]
    tablenorm = table * 1.0
    for i in range(0, nne):
        tablenorm[:, i] = (table[:, i] + 1e-40) / (table[:, 30] + 1e-40)

    return tablenorm
