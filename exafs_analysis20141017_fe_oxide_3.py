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
dir_p = "20141017_fe_oxide_3"
dir_n = "20141017_fe_oxide_3_noise"
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
pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=False)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
ds = data.channel[1]
cutnum = ds.CUT_NAME.index("timestamp_sec")
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
pulse_timing.choose_laser(data, "not_laser")
data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=False, forceNew=False, pre_sanitize_p_filt_phase=True)
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

basic_cuts_local = mass.core.controller.AnalysisControl(
    pulse_average=(0.0, None),
    pretrigger_rms=(None, 30.0),
    pretrigger_mean_departure_from_median=(-120.0, 120.0),
    peak_value=(0.0, None),
    postpeak_deriv=(None, 250.0),
    rise_time_ms=(None, 0.6),
    peak_time_ms=(None, 0.8),
    timestamp_diff_sec=(0.008,None),
    timestamp_sec=(None, 65000)
)
data.apply_cuts(basic_cuts_local) #this is just for the time cut

# do some quality control on the data
pulse_timing.choose_laser(data, "laser")
exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
# exafs.quality_control_range(data, exafs.edge_center_func, "FeKEdge Location", range=(7121.18-2, 7121.18+2))
exafs.quality_control_range(data, exafs.fwhm_ev_7kev, "7keV res fwhm", range=(0, 12))


# # write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=2)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=2)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=0.1)
exafs.write_combined_energies_hists_randsplit(data, erange=(0,20000), binsize=2)
exafs.plot_sqrt_spectra(data)
exafs.plot_sqrt_spectra(data, erange = (7080, 7300))
exafs.plot_sqrt_spectra(data, erange = (6500, 7500))



# # diagnostics
ds = data.first_good_dataset
pulse_timing.choose_laser(data,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")
(edgeCenter, preHeight, postHeight, fwhm, bgSlope, chi2, bin_centers, xi) = exafs.fit_edge_in_energy_combined(data, "FeKEdge", doPlot=True, bin_size_ev=2)


mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_tdc'], True)
ds.compare_calibrations()
exafs.calibration_summary_compare(data)
exafs.timestructure_dataset(ds,"p_filt_value_phc")
exafs.calibration_summary(data, "p_filt_value_tdc")
exafs.pulse_summary(data)
exafs.leftover_phc(data)
data.plot_count_rate()
exafs.cut_vs_time_plot(ds)

# save plots
exafs.save_all_plots(data, dir_p, dir_n)

pulse_timing.choose_laser(data, "laser")

plt.figure()
e_center = 7000
e_half_width = 50
for i, e_center in enumerate(np.arange(4000,10000,1000)):
    use = np.logical_and(ds.cuts.good(), np.abs(ds.p_energy-e_center)<e_half_width)

    use = np.logical_and(use, np.abs(ds.time_after_last_external_trigger-np.median(ds.time_after_last_external_trigger[use]))<10e-6)
    time_after_us = ds.time_after_last_external_trigger[use]*1e6
    p_filt_phase_us = -ds.p_filt_phase[use]*9.6
    plt.plot(time_after_us-np.median(time_after_us),p_filt_phase_us-np.median(p_filt_phase_us)+(i-3)*3,'.', label="%g eV"%e_center)
    plt.plot([-1000,1000], np.array([-1000,1000])+(i-3)*3, lw=2)

plt.grid("on")
plt.minorticks_on()
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot([-1000,1000], [-1000,1000], lw=2)
plt.ylabel("median subtracted negative p_filt_phase*9.6 (us)")
plt.xlabel("median subtracted time after last external trigger (us)")
plt.legend(bbox_to_anchor=(1,1,0,0), loc="upper left")
plt.title("%s, %g eV wide energy ranges"%(path.split(ds.filename)[-1], 2*e_half_width))
plt.gca().set_aspect("equal")

plt.figure()
e_center = 7000
e_half_width = 50
for i, e_center in enumerate(np.arange(4000,10000,1000)):
    use = np.logical_and(ds.cuts.good(), np.abs(ds.p_energy-e_center)<e_half_width)

    use = np.logical_and(use, np.abs(ds.time_after_last_external_trigger-np.median(ds.time_after_last_external_trigger[use]))<10e-6)
    time_after_us = ds.time_after_last_external_trigger[use]*1e6
    prompt = ds.p_promptness[use]*160
    plt.plot(time_after_us-np.median(time_after_us),prompt-np.median(prompt)+(i-3)*3,'.', label="%g eV"%e_center)
    plt.plot([-1000,1000], np.array([-1000,1000])+(i-3)*3, lw=2)

plt.grid("on")
plt.minorticks_on()
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.plot([-1000,1000], [-1000,1000], lw=2)
plt.ylabel("median subtracted p_promptness*160 (~us)")
plt.xlabel("median subtracted time after last external trigger (us)")
plt.legend(bbox_to_anchor=(1,1,0,0), loc="upper left")
plt.title("%s, %g eV wide energy ranges"%(path.split(ds.filename)[-1], 2*e_half_width))
plt.gca().set_aspect("equal")
