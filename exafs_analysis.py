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
dir_p = "20140930_timing_for_real"
dir_n = "20140930_noise"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:30]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files)
if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output

# analyze data
data.summarize_data(peak_time_microsec=500.0, forceNew=False)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
ds = data.channel[1]
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
# pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=False)
pulse_timing.choose_laser(data, "not_laser")

data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=False, forceNew=False)
data.calibrate('p_filt_value_dc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


data.calibrate('p_filt_value_phc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.time_drift_correct(forceNew=False)
data.calibrate('p_filt_value_tdc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


pulse_timing.choose_laser(data, "laser")
# exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
# exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
# exafs.quality_control_range(data, exafs.edge_center_func, "FeKEdge Location", range=(7121.18-2, 7121.18+2))
# exafs.quality_control_range(data, exafs.fwhm_ev_7kev, "7keV res fwhm", range=(0, 12))


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



