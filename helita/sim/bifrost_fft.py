"""
Created by Carmel Baharav  (Juan Martinez-Sykora)
Set of programs to read and interact with output from BifrostData
simulations focusing on Fourier transforms
"""
import time
import numpy as np
import os
from .bifrost import BifrostData, Rhoeetab, read_idl_ascii
import imp
from multiprocessing.dummy import Pool as ThreadPool
import scipy as sp

try:
    imp.find_module('pycuda')
    found = True
except ImportError:
    found = False

if found:
    from reikna import cluda
    from reikna.fft import fft


class FFTData(BifrostData):
    """
    Class that operates radiative transfer from BifrostData simulations
    in native format.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        if verbose:
            print(kwargs)
        # print(verbose)

        super(FFTData, self).__init__(*args, **kwargs)
        """
        sets all stored vars to None
        """
        # these vars are set in singleCudaRun
        # pre-compiled fft func (so only one compilation needed per fft size)
        self.preCompFunc = None
        self.preCompShape = None  # shape that preCompFunc can take as input
        self.transformed_dev = None  # array on device, filled with fft results
        self.api = None  # set to pycuda-based API module, only done once
        self.thr = None  # thread on device, also only created once
        self.found = found  # whether program has access to pycuda
        self.verbose = verbose  # for testing, True --> printouts
        self.run_gpu()  # when pycuda available, uses it by default

    def run_gpu(self, choice=True):
        '''
        activates the module that uses CUDA
        '''
        if self.found:
            self.use_gpu = choice
        else:
            self.use_gpu = False

    def singleCudaRun(self, arr):
        """
        uses pycuda fft (reikna) on given array
        - sends whole array to device at once, may be too much for device
          memory if arr is too large
        """
        shape = np.shape(arr)
        if self.verbose:
            print(shape)
        preTransform = np.complex128(arr)

        t2 = time.time()
        # sets api & creates thread if not preloaded
        if self.api is None:
            if self.verbose:
                print('api is None')
            # combination of pycuda & pyopencl
            self.api = cluda.cuda_api()
            # new thread on device
            self.thr = self.api.Thread.create()

        # creates new computation (preCompFunc- pre-compiled function) & re-
        # compiles if new input has different shape than old (preCompShape);
        # if new shape equals previous input shape, uses the stored function
        if not self.preCompShape == shape:
            if self.verbose:
                print('shape is new')
            self.preCompShape = shape
            lastAxis = len(shape) - 1
            fft1 = fft.FFT(preTransform, axes=(lastAxis, ))
            self.preCompFunc = fft1.compile(self.thr)
            # new empty array with same shape as preTransform on device
            self.transformed_piece_dev = self.thr.empty_like(preTransform)

        # sends preTransform array to device
        t3 = time.time()
        pre_dev = self.thr.to_device(preTransform)
        t4 = time.time()

        # runs compiled function with output arr and input arr
        self.preCompFunc(self.transformed_piece_dev, pre_dev)
        t5 = time.time()

        # retrieves and shifts transformed array
        transformed_piece = self.transformed_piece_dev.get()
        t6 = time.time()

        if self.verbose:
            print('compile time: ', t3 - t2)
            print('sending pre to dev time: ', t4 - t3)
            print('function time: ', t5 - t4)
            print('getting transformed from dev time: ', t6 - t5)

        transformed_piece = np.abs(np.fft.fftshift(
            transformed_piece, axes=-1), dtype=np.float128)

        return transformed_piece

    def linearTimeInterp(self, quantity, snap, iix=None, iiy=None, iiz=None):
        """
        - loads quantity
        - if gaps between snaps are inconsistent, interpolates the data

        Parameters
        ----------
        quantity - string
        snap - array or list
        iix, iiy, and iiz - ints, lists, arrays, or Nones
            slices data cube

        Returns
        -------
        nothing, but defines self.preTransform, self.evenDt, and self.dt

        Notes
        -----
            uses reikna (cuda & openCL) if available
        """

        self.preTransform = self.get_varTime(quantity, snap, iix, iiy, iiz)
        if self.verbose:
            print('done with loading vars')

        # gets rid of array dimensions of 1
        self.preTransform = np.squeeze(self.preTransform)
        self.dt = self.params['dt']
        t = self.params['t']

        # checking to see if time gap between snaps is consistent
        uneven = False
        for i in range(1, np.size(self.dt) - 1):
            if abs((t[i] - t[i - 1]) - (t[i+1] - t[i])) > 0.02:
                uneven = True
                break

        # interpolates data if time gaps are uneven
        if uneven:
            if self.verbose:
                print('uneven dt')
            evenTimes = np.linspace(t[0], t[-1], np.size(self.dt))
            interp = sp.interpolate.interp1d(t, self.preTransform)
            self.preTransform = interp(evenTimes)
            self.evenDt = evenTimes[1] - evenTimes[0]
        else:
            self.evenDt = self.dt[0]

    def get_fft(self, quantity, snap, numThreads=1, numBlocks=1,
                iix=None, iiy=None, iiz=None):
        """
        Calculates FFT (by calling fftHelper) based on time

        Parameters
        ----------
        quantity - string
        snap - array or list
        numThreads - number of threads, not using PyCuda
        numBlocks - for use with PyCuda when GPU memory limited
        iix, iiy, and iiz - ints, lists, arrays, or Nones
            slices data cube

        Returns
        -------
        dictionary -
        {'freq': 1d array of frequencies,
        'ftCube': array of results (dimensions vary based on slicing)}

        Notes
        -----
            uses reikna (cuda & openCL) if available and run_gpu has been called
        """

        # gets data cube, already sliced with iix/iiy/iiz
        if type(snap) is not str:
            self.linearTimeInterp(quantity, snap, iix, iiy, iiz)
            # finds frequency with evenly spaced times
            self.freq = np.fft.fftshift(np.fft.fftfreq(
                np.size(self.dt), self.evenDt * 100))

        t0 = time.time()

        if self.use_gpu:
            # splitting up calculations for memory limitations
            if numBlocks > 1:
                splitList = np.array_split(
                    self.preTransform, numBlocks, axis=0)
                result = list(self.singleCudaRun(splitList[0]))
                if self.verbose:
                    print(len(result))

                for arr in splitList[1:]:
                    addOn = self.singleCudaRun(arr)
                    addLen = addOn.shape[0]
                    result = np.concatenate((result, addOn), axis=0)

                transformed = result

            # calculates fft using reikna if pyCuda is found but numBlocks = 1
            else:
                transformed = self.singleCudaRun(self.preTransform)

        # no pycuda found
        else:
            # threading
            if numThreads > 1:
                transformed = threadTask(
                    singleRun, numThreads, self.preTransform)

            # single thread
            else:
                transformed = singleRun(self.preTransform)

        # returns dictionary of frequency & output cube
        output = {'freq': self.freq, 'ftCube': transformed}
        t1 = time.time()
        if self.verbose:
            print('total time: ', t1-t0)
        return output


def singleRun(arr):
    """
    uses numpy fft on given array
        - could be entire array, or a piece of a larger array
    runs linearly
    """
    transformed_piece = np.abs(np.fft.fftshift((np.fft.fft(arr)), axes=-1))
    return transformed_piece


def threadTask(task, numThreads, arr):
    """
    Threads a given method using python multiprocessing

    Parameters
    ----------
    task - method to be parallelized
    numThreads - number of threads to be run in parallel
    arr - the preTransformed data cube

    Returns
    -------
    array - same results as if done linearly, just calculated in parallel

    Notes
    -----
        does not use cuda, uses python multiprocessing
    """
    args = np.array_split(arr, numThreads)

    # make threadpool, task = task given
    pool = ThreadPool(processes=numThreads)
    result = np.concatenate(pool.map(task, args))
    return result
