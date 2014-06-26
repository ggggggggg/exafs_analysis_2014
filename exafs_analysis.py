import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import traceback, sys



dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140617_laser_plus_calibronium_timing"
dir_n = "20140617_laser_plus_calibronium_timing_noise"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
chan_nums = available_chans[:9]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)

data = mass.TESGroup(pulse_files, noise_files)
data.summarize_data_tdm(peak_time_microsec=220.0)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts)
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
data.drift_correct()
pulse_timing.calc_laser_phase(data)
pulse_timing.choose_laser(data, "not_laser")
#ds.phase_correct2014_dataset(10, plot=True) # doesnt work right now
data.calibrate('p_filt_value_dc', ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'],
                        eps=10,mcs=20, excl=["MnKBeta", "FeKBeta"])

ds = data.channel[1]
exafs.timestructure_dataset(ds,"p_filt_value_dc")

mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_dc'], True)

pulse_timing.label_pumped_band_for_alternating_pump(data)
pulse_timing.choose_laser(data, "pumped")


data.pickle_datasets()


def write_histogram_dataset(ds, fname, type):

    fname,ext = path.splitext(fname)
    fname=fname+"_chan%d"%ds.channum+".spectrum"
    erange = 0,20000
    binsize = 5
    bin_edge = np.arange(binsize*0.5+erange[0], erange[1]+binsize*0.5, binsize)
    bin_centers = bin_edge[:-1]+binsize*0.5
    counts, bin_edge = np.histogram(ds.p_energy[ds.cuts.good()], bin_edge)
    np.savetxt(fname, np.vstack((bin_centers, counts)).T,fmt=("%0.1f", "%i"), header="energy bin centers (eV), counts per bin")



def write_histograms_dataset(ds, path):
    pulse_timing.choose_laser_dataset(ds, "pumped")

    pulse_timing.choose_laser_dataset(ds, "unpumped")

def write_histograms(data, path):
    pass

def combined_energies_hist(data, binSize=3, minE = 0, maxE=20000, outFileName = 'last_histogram'):
    bin_edges = np.arange(minE, maxE, binSize)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    counts= np.zeros_like(bin_centers, dtype=np.int32)
    for ds in data:
        c, b = np.histogram(ds.p_energy[ds.cuts.good()], bin_edges)
        counts += c
    if outFileName is not None:
        if not '.' in outFileName: outFileName += '.spectrum'
        np.savetxt(outFileName, np.vstack((bin_centers, counts)).T, fmt=('%f', '%i'), header = 'energy (eV), counts per bin')
    return counts, bin_centers

def plot_combined_spectrum(gencal, binSize = 3, minE = 0, maxE = 10000, ref_lines = []):
    counts, bin_centers = combined_energies_hist(gencal, binSize, minE, maxE)
    plt.figure()
    plt.plot(bin_centers, counts)
    plt.xlabel('energy (eV)')
    plt.ylabel('counts per %.2f eV bin'%(bin_centers[1]-bin_centers[0]))
    plt.title('coadded spectrum %d pixel'%(gencal.data.num_good_channels))
    for line in ref_lines:
        plt.plot(np.array([1, 1])*mass.calibration.energy_calibration.STANDARD_FEATURES[line], pylab.ylim())

