import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl

pumped_filenames = {
    "7-10": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150710_pumped_hist_0.5ev_bin_blue_cut.dat",
    "7-13": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150713_pumped_hist_0.5ev_bin_blue_matter_cut.dat",
    "7-14": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150714_pumped_hist_0.5ev_bin_blue_matter_cut.dat",
    #"7-15": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150714_2_pumped_hist_0.5ev_bin_matter_cut.dat",
    "7-16": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150716_pumped_hist_0.5ev_bin_matter_cut.dat",
    "7-17": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150717_pumped_hist_0.5ev_bin_blue_matter_cut.dat",
    "7-20": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150720_pumped_hist_0.5ev_bin_blue_matter_cut.dat",
    "7-21": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150721_pumped_hist_0.5ev_bin_blue_matter_cut.dat",
    #"7-02": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/20150702_pumped_hist_0.5ev_bin_blue_matter_cut.dat"
    #"7-02": "/Volumes/Drobo/exafs_analysis_2014/201507_run_coadding/201500702_500mMferrrio_160ps_delay_xas_pumped_combined_2.spectrum"
}


def get_unpumped_name(pumped_filename):
    return pumped_filename.replace("pumped","unpumped")


def load_energy_counts(filename):
    try:
        data = np.loadtxt(filename, delimiter=",")
    except:
        data = np.loadtxt(filename)
    energy = data[:, 0]
    counts = data[:, 1]
    return energy, counts


def rebin(energy, counts, shared_energy):
    binsize = energy[1]-energy[0]
    shared_binsize = shared_energy[1]-shared_energy[0]
    shared_bin_edges = np.arange(shared_energy[0]-shared_binsize/2., shared_energy[-1]+shared_binsize,shared_binsize)
    outcounts = np.zeros_like(shared_energy,dtype="int")

    ratio = int(shared_binsize/binsize)
    assert(ratio % 2 == 0 or ratio == 1)
    i_start = np.searchsorted(energy, shared_energy[0])
    if ratio > 1:
        for j in range(len(outcounts)):
            emid = np.mean(energy[j*ratio+i_start-ratio/2:j*ratio+i_start+ratio/2])
            assert(emid == shared_energy[j])
            outcounts[j] = np.sum(counts[j*ratio+i_start-ratio/2:j*ratio+i_start+ratio/2])
    elif ratio == 1:
        for j in range(len(outcounts)):
            emid = np.mean(energy[j+i_start])
            assert(emid == shared_energy[j])
            outcounts[j] = np.sum(counts[j+i_start])

    return outcounts

shared_energy= np.arange(2001, 15000, 2)
shared_pumped = np.zeros(len(shared_energy),dtype="int")
shared_unpumped = np.zeros(len(shared_energy),dtype="int")

xlo, xhi = 7000, 7200
ilo, ihi = np.searchsorted(shared_energy, xlo), np.searchsorted(shared_energy, xhi)

import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 80

fig = plt.figure(figsize=(10,14))

ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
cmap = brewer2mpl.get_map('Set1',"Qualitative",len(pumped_filenames))
i = 0
for (key, pf) in pumped_filenames.iteritems():
    energy_pumped0, counts_pumped0 = load_energy_counts(pf)
    energy_unpumped0, counts_unpumped0 = load_energy_counts(get_unpumped_name(pf))
    counts_unpumped = rebin(energy_unpumped0, counts_unpumped0, shared_energy)
    counts_pumped = rebin(energy_pumped0, counts_pumped0, shared_energy)
    shared_pumped += counts_pumped
    shared_unpumped += counts_unpumped

    ax1.plot(shared_energy[ilo:ihi], counts_pumped[ilo:ihi], lw=1.2, label="%s pumped" % key,
             drawstyle="steps", color=cmap.mpl_colors[i])
    ax1.plot(shared_energy[ilo:ihi], counts_unpumped[ilo:ihi], lw=1.8, label="%s unpumped" % key,
             drawstyle="steps", color=cmap.mpl_colors[i])

    ax2.plot(shared_energy[ilo:ihi],
             (counts_pumped[ilo:ihi]-counts_unpumped[ilo:ihi])/np.sqrt(counts_unpumped[ilo:ihi]+counts_pumped[ilo:ihi]),
             drawstyle="steps", color=cmap.mpl_colors[i])
    i += 1
ax1.legend(loc="lower left")
ax1.set_xlabel("energy (eV)")
ax2.set_xlabel("energy (eV)")
ax1.set_ylabel("counts per %0.2f eV bin"%(shared_energy[1]-shared_energy[0]))
ax2.set_ylabel("pumped-unpumped/sqrt(pumped+unpumped)")
ax1.grid(True)
ax2.grid(True)

fig.show()

fig = plt.figure(figsize=(10, 14))

ax = fig.add_subplot(211)
ax.plot(shared_energy[ilo:ihi], shared_unpumped[ilo:ihi],label="unpumped",drawstyle="steps")
ax.plot(shared_energy[ilo:ihi], shared_pumped[ilo:ihi],label="pumped",drawstyle="steps")
ax.legend(loc="upper right")
ax.set_xlabel("energy (eV)")
ax.set_ylabel("counts per %0.2f eV bin"%(shared_energy[1]-shared_energy[0]))
ax.set_xlim(xlo,xhi)
ax.grid(True)

ax = fig.add_subplot(212)
ax.plot(shared_energy[ilo:ihi], (shared_pumped[ilo:ihi]-shared_unpumped[ilo:ihi])/np.sqrt(shared_unpumped[ilo:ihi]+shared_pumped[ilo:ihi]),drawstyle="steps")
ax.set_xlabel("energy (eV)")
ax.set_ylabel("pumped-unpumped/sqrt(pumped+unpumped)")
ax.set_xlim(xlo, xhi)
ax.grid(True)

fig.show()
plt.draw()
