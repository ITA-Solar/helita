
import m3d
import m3d_util as m3dc #load in constants
import numpy as np
import matplotlib.pyplot as plt
"""
Read multi3d output

sample data is located in ./output, 
which was run for a FALC atmosphere (5x5x82) with a four-level hydrogen model atom

"""

# read data
a = m3d.m3d(directory="./output")
a.readall() 
a.setdefault(3,2) #set the transition (in this case H-alpha in atom.h4)
ie = a.readvar("ie") #read the emergent intensity
snu = a.readvar("snu") #read the source function
zt1 = a.readvar("ie") #height of z(tau=1) 


#
#Some examples
#
z = a.geometry.z/1e8 #height scale in Mm
tmin = np.min(a.atmos.tg) 
cc = m3dc.cc #speed of light in cgs unit

plt.plot(a.d.l,ie[0,0])
plt.show()