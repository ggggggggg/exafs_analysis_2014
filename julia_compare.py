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


# load data
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20141008_brown_ferrioxalate_straw_2mm_emission"
dir_n = "20141010_noise_good_120"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:1]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files)
if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output

# analyze data
data.summarize_data(peak_time_microsec=500.0, forceNew=True)

import h5py


jld = h5py.File("/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_mass_julia.hdf5")

jc = jld["chan1"]
ds = data.channel[1]
mc = ds.hdf5_group


for key in [ u'min_value',
 u'peak_index',
 u'peak_value',
 u'postpeak_deriv',
 u'pretrig_mean',
 u'pretrig_rms',
 u'pulse_average',
 u'pulse_rms',
 u'rise_time',
 u'timestamp']:

# for key in [ u'rise_time']:
    plt.figure()
    plt.title(mc.name)
    plt.plot(mc[key][:10000],jc[key][:10000],'.', label="python")
    plt.xlabel("mass %s"%key)
    plt.ylabel("julia %s"%key)
    plt.ylabel(key)
    plt.grid("on")
    plt.plot([plt.xlim()[0], plt.xlim()[1]],[plt.xlim()[0], plt.xlim()[1]],'b')
    # plt.legend()




data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
ds = data.channel[1]
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=True)
data.filter_data(forceNew=False)
# pulse_timing.apply_offsets_for_monotonicity(data)
# pulse_timing.calc_laser_phase(data, forceNew=False)
# pulse_timing.choose_laser(data, "laser")
#
# pulse_timing.plot_phase(data.first_good_dataset)

data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=False, forceNew=False)
data.calibrate('p_filt_value_dc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


data.calibrate('p_filt_value_phc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.time_drift_correct(forceNew=False)
data.calibrate('p_filt_value_tdc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


# pulse_timing.choose_laser(data, "laser")
# exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
# exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
# exafs.quality_control_range(data, exafs.edge_center_func, "FeKEdge Location", range=(7121.18-2, 7121.18+2))
# exafs.quality_control_range(data, exafs.fwhm_ev_7kev, "7keV res fwhm", range=(0, 12))


# # write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)
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


# load data
dir_base = "/Volumes/Drobo/exafs_data/mystery_data"
dir_p = "20141001_E"
dir_n = "20141001_D/20141001_D_chan23.noi"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
noise_files2 = [path.splitext(p)[0]+".noi" for p in noise_files]
data2 = mass.TESGroup(pulse_files, noise_files2)
if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data2.datasets[0].filename) #copy this script to mass_output

# analyze data
data2.summarize_data(peak_time_microsec=500.0, forceNew=False)
data2.compute_noise_spectra()
data2.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
ds = data2.channel[1]
data2.avg_pulses_auto_masks() # creates masks and compute average pulses
data2.plot_average_pulses(-1)
data2.compute_filters(f_3db=10000.0, forceNew=True)
data2.filter_data(forceNew=False)


data2.drift_correct(forceNew=False)
data2.phase_correct2014(10, plot=False, forceNew=False)
data2.calibrate('p_filt_value_dc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)


data2.calibrate('p_filt_value_phc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data2.time_drift_correct(forceNew=False)
data2.calibrate('p_filt_value_tdc',  ['FeKAlpha',"FeKBeta"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)




# data.calibrate('p_filt_value_dc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
#
#
# data.calibrate('p_filt_value_phc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
# data.time_drift_correct(forceNew=False)
# data.calibrate('p_filt_value_tdc',  ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
#                         size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
#                         excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)

import h5py


jld = h5py.File("/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_mass_julia.hdf5")

jc = jld["chan1"]
ds = data.channel[1]
mc = ds.hdf5_group


for key in [ u'min_value',
 u'peak_index',
 u'peak_value',
 u'postpeak_deriv',
 u'pretrig_mean',
 u'pretrig_rms',
 u'pulse_average',
 u'pulse_rms',
 u'rise_time',
 u'timestamp']:

# for key in [ u'rise_time']:
    plt.figure()
    plt.title(mc.name)
    plt.plot(mc[key][:10000],jc[key][:10000],'.', label="python")
    plt.xlabel("mass %s"%key)
    plt.ylabel("julia %s"%key)
    plt.ylabel(key)
    # plt.legend()


vdv = np.array([ds.filter.predicted_v_over_dv["noconst"] for ds in data])
peak_signal = np.array([ds.filter.peak_signal for ds in data])
chan = np.array([ds.channum for ds in data])
cals = np.array([ds.calibration["p_filt_value_dc"] for ds in data])
nonlin = np.array([np.log(c.peak_energies[1]/c.peak_energies[0])/np.log(c.refined_peak_positions[1]/c.refined_peak_positions[0]) for c in cals])
achieved = np.array([c.energy_resolutions[0] for c in cals])

eres = cals[0].peak_energies[0]/vdv/nonlin

plt.figure()
plt.plot(chan, eres,'o',label="predicted")
plt.plot(chan, achieved,'o', label="achieved")
plt.legend()
plt.xlabel("channel number")
plt.ylabel("predicted energy resolution at FeKAlpha")
plt.ylim(0,15)
plt.figure()
plt.plot(chan, peak_signal,'o')
plt.xlabel("channel number")
plt.ylabel("pulse height at FeKAlpha")


cutnum = ds.CUT_NAME.index("energy")
for ds in data:
    ds.cuts.clearCut(cutnum)
    ds.cuts.cut(cutnum, np.abs(ds.p_energy-mass.energy_calibration.STANDARD_FEATURES["FeKAlpha"])<100)

pr = ds.p_promptness[ds.cuts.good()]
pp = ds.p_filt_phase[ds.cuts.good()]
plt.figure()
plt.plot(ds.time_after_last_external_trigger[ds.cuts.good()]*1e6, (pr-pr.mean())/pr.std(),'.', label="p_promptness")
plt.plot(ds.time_after_last_external_trigger[ds.cuts.good()]*1e6, (pp-pp.mean())/pp.std(),'.', label="p_filt_phase")
plt.xlabel("time after last external trigger (us)")
plt.title("tupac 8x30  at FeKAlpha +/- 100eV")
plt.ylabel("arrival time indicator (arb)")
plt.legend()
plt.grid("on")

plt.figure()
titles=["tupac", "mystery"]
for i,d in enumerate([data]):
    # plt.subplot(120+i)
    for ds in d:
        if ds.filter.predicted_v_over_dv["noconst"]>0:
            fmin = 1/(ds.timebase*ds.nSamples)
            f = np.linspace(fmin*2, fmin*260, ds.nSamples/2)
            plt.plot(f,ds.noise_psd[1:])
    plt.xlabel("frequency (hz)")
    plt.ylabel("noise psd (arb)")
    plt.title(titles[i])
    plt.xscale("log")
    plt.yscale("log")
    plt.grid("on")
    plt.ylim(0.001, 0.1)
    plt.xlim(f[0],f[-1])

plt.figure()
titles=["tupac", "mystery"]
for i,d in enumerate([data]):
    # plt.subplot(120+i)
    for ds in d:
        if ds.filter.predicted_v_over_dv["noconst"]>0:
            plt.plot(ds.filter.avg_signal)
    plt.xlabel("sample number (9.6us timebase)")
    plt.ylabel("avg pulse")
    plt.title(titles[i])
    plt.grid("on")


plt.figure()
titles=["tupac", "mystery"]
for i,d in enumerate([data]):
    plt.plot([ds.channum for ds in d if ds.filter.predicted_v_over_dv["noconst"]>0],[ds.filter.peak_signal for ds in d if ds.filter.predicted_v_over_dv["noconst"]>0],'o', label=titles[i])

    plt.xlabel("pixel number (arb)")
    plt.ylabel("pulse peak signal (arb)")
    plt.legend()
    plt.grid("on")

plt.figure()
titles=["tupac", "mystery"]
for i,d in enumerate([data]):
    plt.plot([ds.channum for ds in d if ds.filter.predicted_v_over_dv["noconst"]>0],[ds.filter.predicted_v_over_dv["noconst"] for ds in d if ds.filter.predicted_v_over_dv["noconst"]>0],'o', label=titles[i])

    plt.xlabel("pixel number (arb)")
    plt.ylabel("predicted v/dv filt_noconst")
    plt.legend()
    plt.grid("on")