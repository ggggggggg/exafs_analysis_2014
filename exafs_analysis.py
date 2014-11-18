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
dir_p = "20141022_ch7_perp_normal_xtalk"
dir_n = "20140905_fe55_noise"
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
# for ds in data:
#     if "time_after_last_external_trigger" in ds.hdf5_group: del(ds.hdf5_group["time_after_last_external_trigger"])
# pulse_timing.apply_offsets_for_monotonicity(data)
# pulse_timing.calc_laser_phase(data, forceNew=False)
# data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else

ds1 = data.channel[7]
for ds in data:
    maxs = np.amax(ds.traces, axis=1)
    mins = np.amin(ds.traces, axis=1)

    r = maxs-mins

    use = np.logical_and(r<300, ds1.cuts.good())
    if np.sum(use)>0:
        ds.avg_xt_pulse = np.mean(ds.traces[r<200,:],axis=0)
    else:
        ds.avg_xt_pulse = np.mean(ds.traces[ds1.cuts.good(),:],axis=0)
        ds.avg_xt_pulse = ds.avg_xt_pulse*0.005

plt.figure()
for ds in data:
    plt.plot(ds.avg_xt_pulse-ds.avg_xt_pulse[:20].mean())
plt.grid("on")
plt.xlabel("sample number")
plt.ylabel("avg signal pulse when chan %g is hit"%ds1.channum)
plt.ylim(-40,40)

plt.figure()
s = [np.std(ds.avg_xt_pulse) for ds in data]
c = [ds.channum for ds in data]
plt.plot(c,s,'o')
plt.xlabel("channel number")
plt.ylabel("std of avg pulse when chan %g is hit"%ds1.channum)
plt.grid("on")

# load data
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20141022_ch7_perp_turboboost_xtalk"
dir_n = "20140905_fe55_noise"
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
# for ds in data:
#     if "time_after_last_external_trigger" in ds.hdf5_group: del(ds.hdf5_group["time_after_last_external_trigger"])
# pulse_timing.apply_offsets_for_monotonicity(data)
# pulse_timing.calc_laser_phase(data, forceNew=False)
# data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else

ds1 = data.channel[7]
for i in range(32): ds1.cuts.clearCut(i)
cutval = ds1.traces[:,133]-ds1.traces[:,132]
ds1.cuts.cut(0,np.abs(cutval-np.median(cutval))>30)
ds1.cuts.cut(1,np.abs(ds1.p_peak_index[:]-141)>5)
for ds in data:
    maxs = np.amax(ds.traces, axis=1)
    mins = np.amin(ds.traces, axis=1)

    r = maxs-mins

    use = np.logical_and(r<300, ds1.cuts.good())
    if np.sum(use)>0:
        ds.avg_xt_pulse = np.mean(ds.traces[r<200,:],axis=0)
    else:
        ds.avg_xt_pulse = np.mean(ds.traces[ds1.cuts.good(),:],axis=0)
        ds.avg_xt_pulse = ds.avg_xt_pulse*0.005

plt.figure()
for ds in data:
    plt.plot(ds.avg_xt_pulse-ds.avg_xt_pulse[-20:].mean())
plt.grid("on")
plt.xlabel("sample number")
plt.ylabel("avg signal pulse when chan %g relocks"%ds1.channum)
plt.ylim(-40,40)

plt.figure()
s = [np.std(ds.avg_xt_pulse) for ds in data]
c = [ds.channum for ds in data]
plt.plot(c,s,'o')
plt.xlabel("channel number")
plt.ylabel("std of avg pulse when chan %g is hit"%ds1.channum)
plt.grid("on")


# ds = data.channel[1]
# cutnum = ds.CUT_NAME.index("timestamp_sec")
# data.avg_pulses_auto_masks() # creates masks and compute average pulses
# data.plot_average_pulses(-1)
# data.compute_filters(f_3db=10000.0, forceNew=False)
# data.filter_data(forceNew=False)
# # pulse_timing.choose_laser(data, "not_laser")
# data.drift_correct(forceNew=False)
# data.phase_correct2014(10, plot=False, forceNew=False, pre_sanitize_p_filt_phase=True)
# data.calibrate('p_filt_value_dc', ['MnKAlpha', 'MnKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
# data.calibrate('p_filt_value_phc', ['MnKAlpha', 'MnKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
# data.time_drift_correct(forceNew=False)
# data.calibrate('p_filt_value_tdc', ['MnKAlpha', 'MnKBeta'],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
#
# c,e = exafs.combined_energies_hist(data, erange=(0,8000),binsize=1)
# plt.plot(e,c)
# plt.xlabel("energy (eV)")
# plt.ylabel("counts per %g eV bin"%(e[1]-e[0]))
# plt.grid("on")
# plt.yscale("log")
# plt.title("Fe 55 spectrum, in tupac 20140905")
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
# pulse_timing.choose_laser(data, "laser")
# exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
# exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
# # exafs.quality_control_range(data, exafs.edge_center_func, "FeKEdge Location", range=(7121.18-2, 7121.18+2))
# exafs.quality_control_range(data, exafs.fwhm_ev_7kev, "7keV res fwhm", range=(0, 12))


# # write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7050, 7500), ref_lines=["FeKEdge"])
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

plt.figure(figsize=(24,16))
for j,channum in enumerate([1,3,63,67]):
    ds = data.channel[channum]
    plt.subplot(220+j)
    tdiff = []
    # for i in np.arange(0, ds.nPulses, 40000)[:-1]:
    for i in np.arange(0, 1, 40000)[:-1]:
        traces = ds.traces[i:i+40000]
        s = np.mean(traces[ds.cuts.good()[i:i+40000],:], axis=0)
        plt.plot(s)
        tdiff.append(ds.p_timestamp[i+40000]-ds.p_timestamp[i])
    tdiff = np.array(tdiff)
    plt.xlabel("sample number (9.6 usec timebase)")
    plt.ylabel("mean of sample number among many measured good pulses")
    plt.title("channel %g, median time slice %0.0f s, res @ 7keV %0.2f"%(ds.channum, np.median(tdiff), ds.calibration["p_filt_value_phc"].energy_resolutions[6]))
    plt.grid("on")
    plt.xlim(0,520)
    plt.plot([0,520], np.array([1,1])*np.median(ds.p_pretrig_rms[ds.cuts.good()]),lw=2)

tdiff = []
for i in np.arange(0, ds.nPulses, 40000)[:-1]:
    traces = ds.traces[i:i+40000]
    s = np.std(traces[ds.cuts.good()[i:i+40000],:], axis=0)
    tdiff.append(ds.p_timestamp[i+40000]-ds.p_timestamp[i])
    ngood = ds.cuts.good()[i:i+40000].sum()
    rate = ngood/tdiff[-1]
    plt.plot(rate, s[70],'bo')
    plt.plot(rate, s[174],'ro')
    plt.plot(rate, s[278],'go')
    plt.plot(rate, s[290],'bx')
tdiff = np.array(tdiff)
plt.xlabel("good laser triggers/s")
plt.ylabel("sample rms")
plt.grid("on")
plt.title("channel %g, median time slice %0.0f s"%(ds.channum, np.median(tdiff)))


