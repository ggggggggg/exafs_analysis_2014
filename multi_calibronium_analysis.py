import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import shutil
import traceback, sys
import time

# long records
# load data
dirp  = {4:"20141016_calibronium_520_noise_4hz", 20:"20141016_calibronium_520_noise_20hz",
         40:"20141016_calibronium_520_noise_40hz", 60:"20141016_calibronium_520_noise_60hz"}
hdf5filenames = {4: u'/Volumes/Drobo/exafs_data/20141016_calibronium_520_noise_4hz/20141016_calibronium_520_noise_4hz_mass.hdf5',
 20: u'/Volumes/Drobo/exafs_data/20141016_calibronium_520_noise_20hz/20141016_calibronium_520_noise_20hz_mass.hdf5',
 40: u'/Volumes/Drobo/exafs_data/20141016_calibronium_520_noise_40hz/20141016_calibronium_520_noise_40hz_mass.hdf5',
 60: u'/Volumes/Drobo/exafs_data/20141016_calibronium_520_noise_60hz/20141016_calibronium_520_noise_60hz_mass.hdf5'}

for key, fname in hdf5filenames.items():
    try:
        os.remove(fname)
    except:
        pass
# load data
datas = {}


local_basic_cuts = mass.core.controller.AnalysisControl(
    pulse_average=(0.0, None),
    pretrigger_rms=(None, 30.0),
    pretrigger_mean_departure_from_median=(-40.0, 40.0),
    peak_value=(0.0, None),
    postpeak_deriv=(None, 250.0),
    rise_time_ms=(None, 0.6),
    peak_time_ms=(None, 0.8),
    timestamp_diff_sec=(0.015,None))

for key in dirp.keys():
    dir_p = dirp[key]
    dir_base = "/Volumes/Drobo/exafs_data"
    dir_n = "20141016_calibronium_520_noise"
    available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
    if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
    chan_nums = available_chans[:20]
    pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
    noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
    data = mass.TESGroup(pulse_files, noise_files)
    if "__file__" in locals():
        exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output
    data.summarize_data(peak_time_microsec=500.0, forceNew=False)
    data.apply_cuts(local_basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
    data.compute_noise_spectra()
    ds = data.channel[1]
    data.avg_pulses_auto_masks() # creates masks and compute average pulses
    data.plot_average_pulses(-1)
    data.compute_filters(f_3db=10000.0, forceNew=True)
    data.filter_data(forceNew=False)
    cutnum = ds.CUT_NAME.index("p_filt_phase")
    print("p_filt_phase_cut")
    for ds in data:
        ds.cut_parameter(ds.p_filt_phase, (-2,2), cutnum)

    data.drift_correct(forceNew=False)
    data.phase_correct2014(10, plot=False, forceNew=False)
    data.calibrate('p_filt_value_dc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                            size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                            excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


    data.calibrate('p_filt_value_phc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                            size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                            excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
    data.time_drift_correct(forceNew=True)
    data.calibrate('p_filt_value_tdc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                            size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                            excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
    datas[key] = data

vdv = [ds.filter.predicted_v_over_dv["noconst"] for ds in datas[4]]
# nonlin = [ds.calibration["p_filt_value_tdc"].cal.refined_peak_positions for ds in datas[4]]
outer_ds = datas[4].channel[1]
outer_cal = outer_ds.calibration["p_filt_value_tdc"]
for j, key in enumerate(sorted(datas.keys())):
    data = datas[key]
    trigger_rate = [ds.count_rate()[1].mean() for ds in data]
    tr_low = np.percentile(trigger_rate, 10)
    tr_med = np.percentile(trigger_rate, 50)
    tr_high = np.percentile(trigger_rate, 90)
    for i,energy in enumerate(outer_cal.peak_energies):
        eres = [ds.calibration["p_filt_value_tdc"].energy_resolutions[i] for ds in data]
        eres_low = np.percentile(eres, 10)
        eres_med = np.percentile(eres, 50)
        eres_high = np.percentile(eres, 90)
        if i == 0:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j], label="%0.0f trigger/s"%tr_med)
        else:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j])



        # for calname in ["p_filt_value_dc", "p_filt_value_phc", "p_filt_value_tdc"]:
        #     cal = ds.calibration[calname]
        #
        # plt.plot(cal.peak_energies, cal.energy_resolutions,'.', label=str(key)+calname)
plt.legend(loc="upper left")
plt.xlabel("energy (1/s)")
plt.ylabel("energy resolution (eV)")
plt.grid("on")
plt.ylim(3,18)
plt.minorticks_on()
plt.grid("on", "minor")

plt.figure()
vdv = [ds.filter.predicted_v_over_dv["noconst"] for ds in datas[4]]
# nonlin = [ds.calibration["p_filt_value_tdc"].cal.refined_peak_positions for ds in datas[4]]
outer_ds = datas[4].channel[1]
outer_cal = outer_ds.calibration["p_filt_value_tdc"]
for j, key in enumerate(sorted(datas.keys())):
    data = datas[key]
    trigger_rate = [ds.count_rate()[1].mean() for ds in data]
    tr_low = np.percentile(trigger_rate, 10)
    tr_med = np.percentile(trigger_rate, 50)
    tr_high = np.percentile(trigger_rate, 90)
    for i,energy in enumerate(outer_cal.peak_energies):
        eres = [ds.calibration["p_filt_value_tdc"].energy_resolutions[i] for ds in data]
        eres_low = np.percentile(eres, 10)
        eres_med = np.percentile(eres, 50)
        eres_high = np.percentile(eres, 90)
        if i == 0:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j], label="%0.0f trigger/s"%(tr_med))
        else:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j])



        # for calname in ["p_filt_value_dc", "p_filt_value_phc", "p_filt_value_tdc"]:
        #     cal = ds.calibration[calname]
        #
        # plt.plot(cal.peak_energies, cal.energy_resolutions,'.', label=str(key)+calname)
plt.legend(loc="upper left")
plt.xlabel("energy (1/s)")
plt.ylabel("energy resolution (eV)")
plt.title("only calibronium time since last pulse cut at %g ms"%(local_basic_cuts.cuts_prm["timestamp_diff_sec"][0]*1000))
plt.grid("on")
plt.ylim(3,18)
plt.minorticks_on()
plt.grid("on", "minor")

# data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
#
# plt.figure()
#
# ds.tdiff = np.r_[-0.01, np.diff(ds.p_timestamp)]
# plt.plot(ds.tdiff*1000, ds.p_pretrig_rms,'.')
# plt.plot(ds.tdiff[ds.cuts.good()]*1000, ds.p_pretrig_rms[ds.cuts.good()],'.')
# plt.xlabel("time since last trigger (ms)")
# plt.ylabel("pretrig rms, channl %g"%ds.channum)
# plt.grid("on")
#
# plt.figure()
# ds.tdiff = np.r_[-0.01, np.diff(ds.p_timestamp)]
# plt.plot(ds.tdiff*1000, ds.p_pretrig_mean,'.')
# plt.plot(ds.tdiff[ds.cuts.good()]*1000, ds.p_pretrig_mean[ds.cuts.good()],'.')
# plt.xlabel("time since last trigger (ms)")
# plt.ylabel("pretrig mean, channl %g"%ds.channum)
# plt.grid("on")
#
# for ds in data:
#     ds.tdiff = np.r_[-0.01, np.diff(ds.p_timestamp)]
#     plt.plot(ds.tdiff*1000, ds.p_energy,'.')
#     plt.plot(ds.tdiff[ds.cuts.good()]*1000, ds.p_energy[ds.cuts.good()],'.')
#     plt.xlabel("time since last trigger (ms)")
#     plt.ylabel("energy, channl %g"%ds.channum)
#     plt.grid("on")
# plt.xlim(5,25)
# plt.ylim(4000,9000)


# def hitcut_all(arrival_times, cut_before, cut_after):
#     """
#     :param arrival_times: a list of numpy arrays of timestamps for all channels in the same row (or any another grouping)
#     :param cut_before: scalar zero or greater, reject pulse if any other pulse arrives within this long before it in the whole grouping
#     :param cut_after: scalar zero or greater, reject pulse if any other pulse arrives within this long before it im the whole grouping
#     :return:
#     """
#     assert(cut_before>=0)
#     assert(cut_after>=0)
#     nrow = len(arrival_times)
#     arrival_times = [np.array(at,dtype="float") for at in arrival_times]
#     rejects = [np.zeros(len(row_times), dtype="bool") for row_times in arrival_times]
#     for reference_row in xrange(nrow):
#         for other_row in xrange(nrow):
#             if other_row == reference_row: continue
#             before_times, after_times = nearest_arrivals(arrival_times[reference_row], arrival_times[other_row])
#             rejects[reference_row] = np.logical_or(rejects[reference_row], before_times <= cut_before)
#             rejects[reference_row] = np.logical_or(rejects[reference_row], after_times <= cut_after)
#     return rejects
#
#
# def nearest_arrivals(reference_times, other_times):
#     """nearest_arrivals(reference_times, other_times)
#     reference_times - 1d array
#     other_times - 1d array
#     returns: before_times, after_times
#     before_times - d array same size as reference_times, before_times[i] contains the difference between
#     the closest lesser time contained in other_times and reference_times[i]  or inf
#     if there was no earlier time in other_times
#     note that before_times is always a positive number even though the time difference it represents is negative
#     after_times - 1d array same size as reference_times, after_times[i] contains the difference between
#     reference_times[i] and the closest greater time contained in other_times or a inf
#     number if there was no later time in other_times
#     """
#     nearest_after_index = np.searchsorted(other_times, reference_times)
#     # because both sets of arrival times should be sorted, there are faster algorithms than searchsorted
#     # for example: https://github.com/kwgoodman/bottleneck/issues/47
#     # we could use one if performance becomes an issue
#     last_index = np.searchsorted(nearest_after_index, other_times.size,side="left")
#     first_index = np.searchsorted(nearest_after_index, 1)
#
#     nearest_before_index = np.copy(nearest_after_index)
#     nearest_before_index[:first_index]=1
#     nearest_before_index-=1
#     before_times = reference_times-other_times[nearest_before_index]
#     before_times[:first_index] = np.Inf
#
#     nearest_after_index[last_index:]=other_times.size-1
#     after_times = other_times[nearest_after_index]-reference_times
#     after_times[last_index:] = np.Inf
#
#     return before_times, after_times
#
# numgood ={}
# numgood_after = {}
# for j, key in enumerate(sorted(datas.keys())):
#     print("key %g"%key)
#     data = datas[key]
#     numgood[key]=[]
#     numgood_after[key]=[]
#     for column_number in range(8):
#         print("calculing column %g"%column_number)
#         col_dss = [ds for ds in data if ds.column_number==column_number]
#         col_arrival_times = [ds.p_timestamp[:] for ds in col_dss]
#         rejects = hitcut_all(col_arrival_times, -0.0005, 0.0005)
#
#         for ds, rejects_ds in zip(col_dss, rejects):
#             cutnum = ds.CUT_NAME.index("timing")
#             ds.cuts.clearCut(cutnum)
#             numgood[key].append(ds.cuts.good().sum())
#             ds.cuts.cut(cutnum, rejects_ds)
#             numgood_after[key].append(ds.cuts.good().sum())
#
#
#     data.calibrate('p_filt_value_tdc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                             name_ext="_hitcut0.5ms",
#                             size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                             excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
#
#
# perc_past = {key:100*np.sum(numgood_after[key])/float(np.sum(numgood[key])) for key in datas.keys()}
#
vdv = [ds.filter.predicted_v_over_dv["noconst"] for ds in datas[4]]
# nonlin = [ds.calibration["p_filt_value_tdc"].cal.refined_peak_positions for ds in datas[4]]
outer_ds = datas[4].channel[1]
outer_cal = outer_ds.calibration["p_filt_value_tdc_hitcut0.5ms"]
for j, key in enumerate(sorted(datas.keys())):
    data = datas[key]
    trigger_rate = [ds.count_rate()[1].mean() for ds in data]
    tr_low = np.percentile(trigger_rate, 10)
    tr_med = np.percentile(trigger_rate, 50)
    tr_high = np.percentile(trigger_rate, 90)
    for i,energy in enumerate(outer_cal.peak_energies):
        eres = [ds.calibration["p_filt_value_tdc_hitcut0.5ms"].energy_resolutions[i] for ds in data]
        eres_low = np.percentile(eres, 10)
        eres_med = np.percentile(eres, 50)
        eres_high = np.percentile(eres, 90)
        if i == 0:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j], label="%0.0f trigger/s, %0.1f %% pass hitcut"%(tr_med, perc_past[key]))
        else:
            plt.errorbar(energy+j*15, eres_med,yerr=[[eres_med-eres_low],[eres_high-eres_low]], fmt='--o',lw=2, color=["c","m","y","k"][j])



        # for calname in ["p_filt_value_dc", "p_filt_value_phc", "p_filt_value_tdc"]:
        #     cal = ds.calibration[calname]
        #
        # plt.plot(cal.peak_energies, cal.energy_resolutions,'.', label=str(key)+calname)
plt.legend(loc="upper left")
plt.xlabel("energy (1/s)")
plt.ylabel("energy resolution (eV)")
plt.title("only calibronium with 0.5 ms before or after same column hitcut")
plt.grid("on")
plt.ylim(3,18)
plt.minorticks_on()
plt.grid("on", "minor")
#
# ds = datas[60].first_good_dataset