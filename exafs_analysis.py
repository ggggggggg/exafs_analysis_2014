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
data.summarize_data(peak_time_microsec=500.0, forceNew=True)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=True) # forceNew is True by default for apply_cuts, unlike most else
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
# pulse_timing.label_pumped_band_for_alternating_pump(data, forceNew=False)
for ds in data:
    if not "pumped_band_knowledge" in ds.hdf5_group:
        ds.hdf5_group["pumped_band_knowledge"] = 1
    if "p_filt_value" in ds.calibration:
        ds.calibration.pop("p_filt_value")


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

def calc_laser_phase_dataset(ds, forceNew=False, pump_period = 0.002):
    if "p_laser_phase" in ds.hdf5_group and not forceNew:
        ds.p_laser_phase = ds.hdf5_group["p_laser_phase"]
    else:
        med = periodic_median(ds.time_after_last_external_trigger[:],pump_period/2)
        phase = 2*((ds.time_after_last_external_trigger[:]+med)%pump_period)/pump_period
        ds.hdf5_group["p_laser_phase"] = phase
    return ds.p_laser_phase


def calc_laser_phase(data, forceNew=False, pump_period=0.002):
    #try to pick a reasonable dataset to get f0 and the spline from
    for ds in data:
        calc_laser_phase(ds, forceNew, pump_period)

def cut_on_last_external_trigger_diff_from_median_data(self, low, high, keep_inrange=True, mod_period=None):
    for ds in self:
        ds.cut_on_last_external_trigger_diff_from_median(low, high, keep_inrange, mod_period)

def cut_on_last_external_trigger_diff_from_median(self,low,high,keep_inrange=True, mod_period=None):
    if mod_period is not None:
        med = periodic_median(ds.time_after_last_external_trigger[:],mod_period)
        cuttime = (self.time_after_last_external_trigger[:]+med)%mod_period-med
    else:
        cuttime = self.time_after_last_external_trigger[:]
    if keep_inrange:
        cutmask = np.logical_and(cuttime>low, cuttime<high)
    else:
        cutmask = np.logical_and(cuttime<low, cuttime>high)
    CUT_INDEX = ds.CUT_NAME.index("timing")
    ds.cuts.clearCut(CUT_INDEX)
    ds.cuts.cut(CUT_INDEX, cutmask)

def periodic_median(timestamp, mod_period=0.001):
    # finds the offset required to make the median 0.5, the backs out what the true median must be to require that offset
    p0=0
    maxj = 3
    for j in xrange(maxj+1):
        phase = (timestamp+p0)%mod_period
        p0 -= (np.median(phase)-0.5*mod_period)+np.random.rand()*(0 if j==maxj else 0.001*mod_period)
        # the random bit is to try to avoid the case where the median is 0.5 due to half the population being
        # approx 0 and half being approx 1,
        # I tested without the random adding 10000 linearly increasing offsets to some actual data
        # and never observed the problem the random is trying to address
    return (0.5-p0)*mod_period

ds.cut_on_last_external_trigger_diff_from_median = cut_on_last_external_trigger_diff_from_median

cut_on_last_external_trigger_diff_from_median(ds,-10e-6, 10e-6)




extern_trig_row_counts, h5 = open_timing_file(data)
extern_trig_timestamp = extern_trig_row_counts[:]*ds.timebase/ds.pulse_records.datafile.number_of_rows

ds1 = data.channel[1]
ds2 = data.channel[31]
before1, after1 =nearest_arrivals(ds1.p_timestamp[ds1.cuts.good()], extern_trig_timestamp)
before1_phase, after1_phase =nearest_arrivals(ds1.p_timestamp[ds1.cuts.good()]+ds1.p_filt_phase[ds1.cuts.good()]*ds.timebase, extern_trig_timestamp)
before2, after2 =nearest_arrivals(ds2.p_timestamp[ds2.cuts.good()], extern_trig_timestamp)
before2_phase, after2_phase =nearest_arrivals(ds2.p_timestamp[ds2.cuts.good()]+ds2.p_filt_phase[ds2.cuts.good()]*ds.timebase, extern_trig_timestamp)

plt.figure()
hist1, bins = np.histogram(before1, np.linspace(0, 0.002, 1001))
hist1_phase, bins = np.histogram(before1_phase, np.linspace(0, 0.002, 1001))
hist2, bins = np.histogram(before2, np.linspace(0, 0.002, 1001))
hist2_phase, bins = np.histogram(before2_phase, np.linspace(0, 0.002, 1001))
bin_centers = bins[1:]-0.5*(bins[1]-bins[0])
plt.plot(bin_centers*1e6, hist1, label="channel %g"%ds1.channum)
plt.plot(bin_centers*1e6, hist1_phase, label="channel %g, with p_filt_phase"%ds1.channum)
plt.plot(bin_centers*1e6, hist2, label="channel %g"%ds2.channum)
plt.plot(bin_centers*1e6, hist2_phase, label="channel %g, with p_filt_phase"%ds2.channum)
plt.legend()
plt.xlabel("time difference from nearest previous extern trig (microseconds)")
plt.ylabel("number of xrays")
plt.grid("on")

ds = data.channel[1]
ds.pulse_records.datafile.read_segment(0)
row_count = ds.pulse_records.datafile.row_count
row_count = np.array(row_count,dtype="int64")
before,after = nearest_arrivals(row_count, extern_trig_row_counts[:])
fig = plt.figure()
plt.scatter(before[ds.cuts.good()], 30*ds.p_filt_phase[ds.cuts.good()], c=(ds.p_pulse_rms[ds.cuts.good()]/(2*np.median(ds.p_pulse_rms[ds.cuts.good()]))))
ind = plt.find(ds.cuts.good())
leftind = np.logical_and(np.abs(30*ds.p_filt_phase[:][ind]+12)<2, np.abs(before[ind]-3060)<5)
rightind = np.logical_and(np.abs(30*ds.p_filt_phase[:][ind]+12)<2, np.abs(before[ind]-3088)<5)
plt.plot(before[ind[rightind]],30*ds.p_filt_phase[:][ind[rightind]],'or')
plt.plot(before[ind[leftind]],30*ds.p_filt_phase[:][ind[leftind]],'ko')
plt.xlabel("row count (320 ns per unit)")
plt.ylabel("p_filt_phase*30 (~320 ns per unit)")
plt.grid("on")
plt.gca().set_aspect("equal")


plt.figure()
for i in ind[leftind]: plt.plot(ds.read_trace(i),'k.-')
for i in ind[rightind]: plt.plot(ds.read_trace(i),'r.-')
plt.xlabel("sample number")
plt.xlim(130,134)
plt.ylim(5000,9000)
plt.grid("on")


ds = data.channel[1]
ds.pulse_records.datafile.read_segment(0)
plt.plot(ds.pulse_records.datafile.datatimes_float-ds.pulse_records.datafile.datatimes_float_old,'.')

frame_count_temp = (np.array(ds.pulse_records.datafile.datatime_4usec_tics, dtype=np.int64)*40)//96
frame_count = frame_count_temp+np.sign((np.array(ds.pulse_records.datafile.datatime_4usec_tics, dtype=np.uint64)*40)%96)
frame_count2 = -((-np.array(ds.pulse_records.datafile.datatime_4usec_tics, dtype=np.uint64)*40)//96)
frame_count_float = np.ceil(ds.pulse_records.datafile.datatimes_float_old/ds.timebase)
plt.plot(frame_count-frame_count_float,'.')
plt.plot(frame_count*ds.timebase-ds.pulse_records.datafile.datatimes_float_old,'.')