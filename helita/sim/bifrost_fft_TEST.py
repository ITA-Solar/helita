import numpy as np
import helita.sim.cstagger
from helita.sim.bifrost import BifrostData, Rhoeetab, read_idl_ascii
from helita.sim.bifrost_fft import FFTData
import matplotlib.pyplot as plt

# note: this calls bifrost_fft from user, not /sanhome

dd = FFTData(file_root = 'cb10f', fdir = '/net/opal/Volumes/Amnesia/mpi3drun/Granflux')
x = np.linspace(-np.pi, np.pi, 201)
dd.preTransform = np.sin(8 *x)
dd.freq = np.fft.fftshift(np.fft.fftfreq(np.size(x)))
dd.run_gpu(False)
tester = dd.get_fft('not a real snap', snap = 0)
fig = plt.figure()

numC = 3
numR = 2

ax0 = fig.add_subplot(numC, numR, 1)
ax0.plot(x, dd.preTransform)
ax0.set_title('original signal' + '\n\nsine wave')

ax1 = fig.add_subplot(numC, numR, 2)
ax1.plot(tester['freq'], tester['ftCube'])
ax1.set_title('bifrost_fft get_fft() of signal' + '\n\n ft of sine wave')
ax1.set_xlim(-.2, .2)

n = 30000 # Number of data points
dx = .01 # Sampling period (in meters)
x = dx*np.linspace(-n/2 , n/2, n) # x coordinates

stanD = 2 # standard deviation
dd.preTransform = np.exp(-0.5 * (x/stanD)**2)

ax2 = fig.add_subplot(numC, numR, 3)
ax2.plot(x, dd.preTransform)
ax2.set_xlim(-25, 25)
ax2.set_title('gaussian curve')

dd.freq = np.fft.fftshift(np.fft.fftfreq(np.size(x)))
ft = dd.get_fft('fake snap', snap = 0)
ax3 = fig.add_subplot(numC, numR, 4)
ax3.plot(ft['freq'], ft['ftCube'])
ax3.set_xlim(-.03, .03)
ax3.set_title('ft of gaussian curve')

x = np.linspace(-20, 20, 50)
dd.preTransform = [0] * 50
ax4 = fig.add_subplot(numC, numR, 5)
ax4.plot(x, dd.preTransform)
ax4.set_title('y = 0')

dd.freq = np.fft.fftshift(np.fft.fftfreq(np.size(x)))
ft = dd.get_fft('fake snap', snap = 0)
ax5 = fig.add_subplot(numC, numR, 6)
# print(ft)
# print(np.max(ft['ftCube']))
# print(np.where(ft['ftCube'] == np.max(ft['ftCube'])))
ax5.plot(ft['freq'], ft['ftCube'])
ax5.set_title('ft of y = 0')

# # testing
# x = np.linspace(-2 * np.pi, 2 * np.pi, 201)
# dd.preTransform = np.cos(x) + np.sin(3*x)
# ax6 = fig.add_subplot(numC, numR, 7)
# ax6.plot(x, dd.preTransform)
# ax6.set_title('y = cos(x) + 2 * sin(x)')

# dd.freq = np.fft.fftshift(np.fft.fftfreq(np.size(x)))
# ft = dd.get_fft('fake snap', snap = 0)
# ax7 = fig.add_subplot(numC, numR, 8)
# print(np.max(ft['ftCube']))
# print(np.where(ft['ftCube'] == np.max(ft['ftCube'])))
# ax7.plot(ft['freq'], ft['ftCube'])
# ax7.set_xlim(-.2, .2)
# ax7.set_title('ft y = cos(x) + sin(x)')

plt.tight_layout()
plt.show()