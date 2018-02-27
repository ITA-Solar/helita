"""
Set of programs to read and interact with output from BifrostData simulations focus on magnetic field topology
"""

import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii, subs2grph, bifrost_units
from . import cstagger
from glob import glob
import scipy as sp

import imp
try:
    imp.find_module('pycuda')
    found = True
except ImportError:
    found = False

from q import qCalculatornopars

class TopologyData(BifrostData):

    """
    Class that operates radiative transfer form BifrostData simulations
    in native format.
    """

    def __init__(self, *args, **kwargs):
        super(TopologyData, self).__init__(*args, **kwargs)

    def get_topology(self, quant, *args, **kwargs):
        """
        Calculates a quantity from the simulation quantiables.

        Parameters
        ----------
        quant - string
            Name of the quantity to calculate (see below for some categories).

        Returns
        -------
        array - ndarray
            Array with the dimensions of the simulation.

        Notes
        -----
        Not all possibilities for quantities are shown here. But there are
        a few main categories:

        """
        TOPO_QUANT = ['qfac', 'alt', 'integrate', 'conn']

        if quant in TOPO_QUANT:

            if found:

                if os.environ.get('CUDA_LIB','null') == 'null':
                    os.environ['CUDA_LIB'] = os.environ['BIFROST'] + '/CUDA/q_factor/'

                #Calculation settings
                qdef = False
                adef = False
                intdef = False
                cdef = False

                if quant == 'qfac':
                    qdef = True
                elif quant == 'alt':
                    adef = True
                elif quant == 'integrate':
                    intdef = True
                else:
                    cdef = True

                opts = q_options()
                opts.temp = self.file_root
                opts.snap = self.snap
                opts.q = qdef
                opts.alt = adef
                opts.integrate = intdef
                opts.connectivity = cdef
                opts.dir = self.fdir

                var = np.empty([self.nx, self.ny, self.nz])
                q=qCalculatornopars(opts)
                for iz in range(0, self.nz):
                    print('iz=',iz)
                    opts.plane = iz
                    opts.slice = str(iz)
                    opts.rcalc = True
                    q.opts = opts
                    q.plane = iz
                    q.trace_snapshot()
                    q.init_q()
                    var[:,:,iz]=q.calculate_q(iz)
                return var
            else:
                raise ValueError(('This machine does not have cuda.'))

        else:
            raise ValueError(('get_topology: do not know (yet) how to '
                              'calculate quantity %s. Note that get_topology '
                              'available variables are: %s.\n'
                              'see e.g. self.get_topology? for guidance'
                              '.' % (quant, repr(TOPO_QUANT))))

class q_options:
     def __init__(self):
         self.slice = False
         self.rcalc = False
         self.save = False
         self.file = None
         self.dir =  ''
         self.temp = ''
         self.snap = 1
         self.plane = 0
         self.q=False
         self.alt=False
         self.integrate=False
         self.connectivity=False
