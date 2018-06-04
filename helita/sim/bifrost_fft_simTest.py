import numpy as np
import helita.sim.cstagger
from helita.sim.bifrost import BifrostData, Rhoeetab, read_idl_ascii
from helita.sim.bifrost_fft import FFTData
import matplotlib.pyplot as plt

snaps = np.arange(430, 435)
v = 'uz'
# threads = 10

dd = FFTData(file_root = 'cb10f', fdir = '/net/opal/Volumes/Amnesia/mpi3drun/Granflux', verbose = True)

transformed = dd.get_fft(v, snaps)
print('got fft')
ft = transformed['ftCube']
freq = transformed['freq']
zaxis = dd.z

zstack = np.empty([np.size(freq), np.shape(ft)[2]])

for k in range(0, np.shape(ft)[2]):
	avg = np.average(ft[:, :, k], axis = (0, 1))
	zstack[:, k] = avg

# print(np.shape(zstack))
# print(zstack)

fig = plt.figure()
numC = 2
numR = 1

plt.subplot(numC, numR, 1)
plt.plot(freq, zstack)
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Average Amplitude of FT Frequencies at Different Z Positions (1)')

plt.subplot(numC, numR, 2)
plt.imshow(zstack.transpose(), extent = [freq[0], freq[-1], zaxis[0], zaxis[-1]])
plt.xlabel('Frequency')
plt.ylabel('Z Position')
plt.title('Average Amplitude of FT Frequencies at Different Z Positions (2)')
plt.show()