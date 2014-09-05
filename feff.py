import pylab as plt
import numpy as np
import scipy.signal

def load_data():
    data = np.loadtxt("feff_ferrioxalate_From_joel.dat")
    energy = data[:,0]
    mu = data[:,3]
    energy2 = np.arange(energy[0], energy[-1], 1)
    mu2 = np.interp(energy2, energy, mu)
    return energy2, mu2

def gauss(x, mean, std):
    return np.exp(-((x-mean)**2/(2*std**2)))/np.sqrt(2*np.pi*std**2)

def gauss_fwhm(x,mean, fhwm):
    return gauss(x,mean, fhwm/2.355)

def gaussian_fwhm(n, fwhm, binsize):
    std =  fwhm/(2.355*binsize)
    return scipy.signal.gaussian(n, std)/np.sqrt(2*np.pi*std**2)

energy,mu = load_data()
binsize = energy[1]-energy[0]
g = gauss_fwhm(energy, np.median(energy), 5)

plt.figure()
plt.plot(energy,mu,'-', label=0)
d = {}
for fwhm in np.arange(4,20.1,4):
    g2 = gaussian_fwhm(len(energy), fwhm, binsize)
    d[fwhm] = np.convolve(mu, g2,"same")
    plt.plot(energy,d[fwhm],'-', label=fwhm)

plt.xlabel("energy (eV)")
plt.ylabel("xi")
plt.legend(loc="lower right")
plt.grid("on")
plt.xlim(7120,7250)

for k in d.keys():
    np.savetxt("ferrioxalate_feff_%0.2feV_fwhm.spectrum"%k, np.vstack((energy, d[k])).T, header="energy, xi")