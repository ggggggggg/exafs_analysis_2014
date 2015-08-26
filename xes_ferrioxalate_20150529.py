import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import shutil
import traceback, sys
from matplotlib.backends.backend_pdf import PdfPages
import datetime


# load data
dir_base = "/Volumes/Drobo/exafs_data"
# dir_p = "20150529_260mM_ferrioxalate_xes_50ps/20150529_114904_chan1.ljh"
dir_p = "20150528_260mM_ferrioxalate_xes_50ps/20150528_160413_chan1.ljh"

dir_n = "20150528_260mM_ferrioxalate_xes_50ps_noise/20150528_152056_chan3.ljh"
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
for ds in data:
    if "time_after_last_external_trigger" in ds.hdf5_group: del(ds.hdf5_group["time_after_last_external_trigger"])
# pulse_timing.apply_offsets_for_monotonicity(data)
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
pulse_timing.calc_laser_phase(data, forceNew=False)
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.compute_noise_spectra()
# ds = data.channel[1]
# data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
data.drift_correct(forceNew=False)
pulse_timing.choose_laser(data, "laser",keep_size=0.014)
# data.phase_correct2014(10, plot=False, forceNew=False, pre_sanitize_p_filt_phase=True)
data.calibrate('p_filt_value_dc', ['FeKAlpha', 'FeKBeta'],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)

# for ds in data:
#     try:
#
#         # if ds.channum == 145:
#         #     tr.print_diff()
#         pdb.set_trace()
#         ds.calibrate('p_filt_value_dc', ['FeKAlpha', 'FeKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
#     except:
#         pass
# data.calibrate('p_filt_value_phc', ['FeKAlpha', 'FeKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
# data.time_drift_correct(forceNew=False)
# data.calibrate('p_filt_value_tdc', ['FeKAlpha', 'FeKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)

# basic_cuts_local = mass.core.controller.AnalysisControl(
#     pulse_average=(0.0, None),
#     pretrigger_rms=(None, 30.0),
#     pretrigger_mean_departure_from_median=(-120.0, 120.0),
#     peak_value=(0.0, None),
#     postpeak_deriv=(None, 250.0),
#     rise_time_ms=(None, 0.6),
#     peak_time_ms=(None, 0.8),
#     timestamp_diff_sec=(0.008,None),
#     timestamp_sec=(None, 65000)
# )
# data.apply_cuts(basic_cuts_local) #this is just for the time cut

# do some quality control on the data
binsize=1
pulse_timing.choose_laser(data, "pumped",keep_size=0.014)
p,b = exafs.combined_energies_hist(data,binsize=binsize)
pulse_timing.choose_laser(data, "unpumped",keep_size=0.014)
u,b = exafs.combined_energies_hist(data,binsize=binsize)

plt.figure()
plt.step(b,p)
plt.step(b,u)
plt.xlabel("energy (eV)")
plt.ylabel("counts per %0.2f eV bin"%binsize)


# # # write histograms
# exafs.plot_combined_spectra(data, ref_lines=["FeKAlpha"],binsize=2)
# exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"])
# exafs.write_channel_histograms(data, erange=(0,20000), binsize=2)
# exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=2)
# exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=0.1)
# exafs.write_combined_energies_hists_randsplit(data, erange=(0,20000), binsize=2)
# exafs.plot_sqrt_spectra(data)
# exafs.plot_sqrt_spectra(data, erange = (7080, 7300))
# exafs.plot_sqrt_spectra(data, erange = (6500, 7500))
#
#
#
# # # diagnostics
# ds = data.first_good_dataset
# pulse_timing.choose_laser(data,"laser")
# exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
# exafs.fit_edges(data,"FeKEdge")
# (edgeCenter, preHeight, postHeight, fwhm, bgSlope, chi2, bin_centers, xi) = exafs.fit_edge_in_energy_combined(data, "FeKEdge", doPlot=True, bin_size_ev=2)
#
#
# mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_tdc'], True)
# ds.compare_calibrations()
# exafs.calibration_summary_compare(data)
# exafs.timestructure_dataset(ds,"p_filt_value_phc")
# exafs.calibration_summary(data, "p_filt_value_tdc")
# exafs.pulse_summary(data)
# exafs.leftover_phc(data)
# data.plot_count_rate()
# exafs.cut_vs_time_plot(ds)
#
# # save plots
# exafs.save_all_plots(data, dir_p, dir_n)



