import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import traceback, sys


# load data
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140620_1M_ferrioxalate_straw/"
dir_n = "20140620_1M_ferrioxalate_straw_noise/"
# dir_p = "20140617_laser_plus_calibronium_timing/"
# dir_n = "20140617_laser_plus_calibronium_timing_noise/"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:80]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files, auto_pickle=True)

# analyze data
data.summarize_data_tdm(peak_time_microsec=220.0, forceNew=False)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=False)
pulse_timing.choose_laser(data, "not_laser")
data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=False)
# data.calibrate('p_filt_value_dc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=True, max_pulses_for_dbscan=1e5)
# data.calibrate('p_filt_value_phc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=True, max_pulses_for_dbscan=1e5)
data.time_drift_correct(forceNew=False)
data.calibrate('p_filt_value_tdc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=True, max_pulses_for_dbscan=1e5)
pulse_timing.label_pumped_band_for_alternating_pump(data, forceNew=False)
data.pickle_datasets()

# do some quality control on the data
pulse_timing.choose_laser(data, "laser")
exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location")
exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=8)

# # write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7200), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)


# # diagnostics
ds = data.first_good_dataset
pulse_timing.choose_laser(data,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")

mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_tdc'], True)
ds.compare_calibrations()
exafs.timestructure_dataset(ds,"p_filt_value_phc")
exafs.calibration_summary(data, "p_filt_value_tdc")
exafs.pulse_summary(data)
data.plot_count_rate()

# save plots
# exafs.save_all_plots(data)






def leftover_phc_single(ds, attr="p_filt_value_phc", feature="CuKAlpha", ax=None):
    cal = ds.calibration[attr]
    pulse_timing.choose_laser_dataset(ds, "not_laser")
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.plot(ds.p_promptness[ds.cuts.good()], getattr(ds, attr)[ds.cuts.good()],'.')
    # ax.set_xlabel("promptness")
    ax.set_ylabel(attr)
    ax.set_title("chan %d %s"%(ds.channum, feature))
    ax.set_ylim(np.array([.995, 1.005])*cal.name2ph(feature))
    index = np.logical_and(getattr(ds, attr)[ds.cuts.good()]>ax.get_ylim()[0], getattr(ds, attr)[ds.cuts.good()]<ax.get_ylim()[1])
    xmin = plt.amin(ds.p_promptness[ds.cuts.good()][index])
    xmax = plt.amax(ds.p_promptness[ds.cuts.good()][index])
    ax.set_xlim(xmin, xmax)

def leftover_phc(data):
    plt.figure()
    for j,ds in enumerate(data):
        if j ==5: break
        ax=plt.subplot(5,2,2*j+2)
        leftover_phc(ds,ax=ax)
        ax2=plt.subplot(5,2,2*j+1)
        leftover_phc(ds, "p_filt_value_dc",ax=ax2)

