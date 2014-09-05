import mass
import numpy as np
import pylab as plt
from os import path

fnames = ["20140724_ferrioxalate_pump_probe_100um_circ", "20140724_ferrioxalate_pump_probe_100um_circ_2"]
uc = []
pc = []
for fname in fnames:
    oname = mass.ljh_util.output_basename_from_ljh_fname(path.join("/Volumes/Drobo/exafs_data",fname))
    pname = oname+"_pumped_combined.spectrum"
    uname = oname+"_unpumped_combined.spectrum"
    pdata = np.loadtxt(pname)
    udata = np.loadtxt(uname)
    e=pdata[:,0]
    uc.append(pdata[:,1])
    pc.append(udata[:,1])


plt.figure()
for j in range(len(uc)):
    plt.plot(e,uc[j]/np.median(uc[j]))


ucs = uc[0]+uc[1]
pcs = pc[0]+pc[1]

plt.figure()
plt.plot(e, ucs)
plt.plot(e, pcs)
plt.plot(e, uc[1])
plt.plot(e, pc[1])
# np.savetxt(e, "energy")
# np.savetxt(ucs, "unpumped")
# np.savetxt(pcs, "pumped")