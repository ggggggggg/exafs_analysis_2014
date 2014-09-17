import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import shutil
import traceback, sys


# load data
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140820_ferrioxalate_pp_4x100um_circ"
dir_n = "20140820_ferrioxalate_pp_4x100um_circ_noise"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files)
if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output

# analyze data
data.summarize_data(peak_time_microsec=500.0, forceNew=False)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=True) # forceNew is True by default for apply_cuts, unlike most else
ds = data.channel[1]
cutnum = ds.CUT_NAME.index("timestamp_sec")
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
# pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=False, sample_time_s=90)
pulse_timing.choose_laser(data, "not_laser")
data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=False, forceNew=False)
data.calibrate('p_filt_value_dc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.calibrate('p_filt_value_phc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.time_drift_correct(forceNew=False)
data.calibrate('p_filt_value_tdc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
pulse_timing.label_pumped_band_for_alternating_pump(data, forceNew=False)



# do some quality control on the data
pulse_timing.choose_laser(data, "laser")
# exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
exafs.quality_control_range(data, exafs.edge_center_func, "FeKEdge Location", range=(7121.18-2, 7121.18+2))
exafs.quality_control_range(data, exafs.fwhm_ev_7kev, "7keV res fwhm", range=(0, 12))


# # write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=0.1)
exafs.write_combined_energies_hists_randsplit(data, erange=(0,20000), binsize=5)
exafs.plot_sqrt_spectra(data)
exafs.plot_sqrt_spectra(data, erange = (7080, 7300))
exafs.plot_sqrt_spectra(data, erange = (6500, 7500))



# # diagnostics
ds = data.first_good_dataset
pulse_timing.choose_laser(data,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")
(edgeCenter, preHeight, postHeight, fwhm, bgSlope, chi2, bin_centers, xi) = exafs.fit_edge_in_energy_combined(data, "FeKEdge", doPlot=True)


mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_tdc'], True)
ds.compare_calibrations()
exafs.calibration_summary_compare(data)
exafs.timestructure_dataset(ds,"p_filt_value_phc")
exafs.calibration_summary(data, "p_filt_value_tdc")
exafs.pulse_summary(data)
exafs.leftover_phc(data)
data.plot_count_rate()

# save plots
exafs.save_all_plots(data)


def plot_spectra_error_bars(data, erange=(0,20000), binsize=5, ref_lines = [], chans=None, desc=""):
    pulse_timing.choose_laser(data,"pumped")
    pcounts, bin_centers = exafs.combined_energies_hist(data, erange, binsize, chans)
    pulse_timing.choose_laser(data,"unpumped")
    ucounts, bin_centers = exafs.combined_energies_hist(data, erange, binsize, chans)
    plt.figure()
    ax = plt.gca()
    ax.errorbar(bin_centers, ucounts,fmt='-b', yerr=np.sqrt(ucounts), label="UNPUMPED")
    ax.errorbar(bin_centers, pcounts,fmt='-r', yerr=np.sqrt(pcounts), label="PUMPED")
    ax.legend()
    ax.set_xlabel("energy (eV)")
    ax.set_ylabel("counts/%0.2f eV bin"%(bin_centers[1]-bin_centers[0]))
    ax.set_title("error bars are +/- sqrt(counts) "+desc)
    plt.figure()
    ax = plt.gca()
    ax.errorbar(bin_centers, pcounts-ucounts,fmt='-b', yerr=np.sqrt(0.5*ucounts+0.5*pcounts), label="U-P")
    ax.set_xlabel("energy (eV)")
    ax.set_ylabel("u-p/%0.2f eV bin"%(bin_centers[1]-bin_centers[0]))
    ax.set_title("error bars are +/- sqrt(0.5*u+0.5*p) "+desc)
    plt.figure()
    ax = plt.gca()
    ax.plot(bin_centers, (pcounts-ucounts)/np.sqrt(0.5*pcounts+0.5*ucounts))
    ax.grid("on")
    ax.set_xlabel('energy (eV)')
    ax.set_ylabel('(u-p)/sqrt(u/2+p/2) per %.2f eV bin'%(bin_centers[1]-bin_centers[0]))
    ax.set_title(" "+desc)




# start_time = np.int64(40332)
# durations = np.arange(30, 180,30)*60
# cutnum = ds.CUT_NAME.index("timestamp_sec")
# for duration in durations[::-1]:
#     for ds in data:
#         ds.cuts.clearCut(cutnum)
#         ds.cuts.cut(cutnum, np.logical_or(ds.p_timestamp<start_time, ds.p_timestamp>(start_time+duration)))
#     print(ds.cuts.good().sum())
#     plot_spectra_error_bars(data, erange = (7080, 7300))
#     plt.title("duration %d min"%(duration/60))


def findbreaks_dataset(ds):
    diff = np.diff(ds.p_timestamp)
    breaks = np.nonzero(diff>100*np.median(diff))[0]
    return breaks

def findstarts_dataset(ds):
    breaks = findbreaks_dataset(ds)
    starts = np.hstack((ds.p_timestamp[0],ds.p_timestamp[:][breaks+1]))
    return starts

def timecut_duration_after_starts_dataset(ds, duration_min, exclude, doPlot=False):
    duration = duration_min*60
    starts = findstarts_dataset(ds)
    cut = np.zeros(ds.p_timestamp.shape, dtype="bool")
    for j in xrange(len(starts)-1):
        cut = np.logical_or(cut, np.logical_and(ds.p_timestamp[:]>starts[j]+duration, ds.p_timestamp[:]<starts[j+1]))
    if exclude > 0:
        cut = np.logical_or(cut, ds.p_timestamp[:]>starts[-exclude])
    cutnum = ds.CUT_NAME.index("timestamp_sec")
    ds.cuts.clearCut(cutnum)
    ds.cuts.cut(cutnum, cut)
    if doPlot:
        cutn = np.nonzero(cut)[0]
        plt.figure()
        plt.plot(ds.p_timestamp,'.')
        plt.plot(cutn, ds.p_timestamp[:][cutn],'.r')
        plt.title("chan %d, duration_min %0.2f"%(ds.channum, duration_min))

def timecut_duration_after_starts(data, duration_min, exclude):
    for ds in data:
        timecut_duration_after_starts_dataset(ds, duration_min, exclude)

# for duration_min in np.arange(5,26,5):
#     timecut_duration_after_starts(data, duration_min, 3)
#     print(ds.cuts.good().sum())
#     plot_spectra_error_bars(data, erange = (7080, 7300), desc="%0.2f min"%duration_min)


def plot_count_rate(self, bin_s=60, title=""):
    bin_edge = np.arange(self.first_good_dataset.p_timestamp[0],
                         self.first_good_dataset.p_timestamp[-1], bin_s)
    bin_centers = bin_edge[:-1]+0.5*(bin_edge[1]-bin_edge[0])
    rates_all = np.array([ds.count_rate(False, bin_edge)[1] for ds in self])
    rates_good = np.array([ds.count_rate(True, bin_edge)[1] for ds in self])
    plt.figure()
    plt.subplot(311)
    plt.plot(bin_centers, rates_all.T)
    plt.ylabel("all by chan")
    plt.subplot(312)
    plt.plot(bin_centers, rates_good.T)
    plt.ylabel("good by chan")
    plt.subplot(313)
    print rates_all.sum(axis=-1).shape
    plt.plot(bin_centers, rates_all.sum(axis=0))
    plt.ylabel("all array")
    plt.grid("on")

    plt.figure()
    plt.plot([ds.channum for ds in self], rates_all.mean(axis=1),'o', label="all")
    plt.plot([ds.channum for ds in self], rates_good.mean(axis=1),'o', label="good")
    plt.xlabel("channel number")
    plt.ylabel("average trigger/s")
    plt.grid("on")
    plt.legend()


