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
dir_p = "201500702_500mMferrrio_160ps_delay_xas"
dir_n = "201500702_500mMferrrio_160ps_delay_xas_noise"
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



# do some quality control on the data
pulse_timing.choose_laser(data, "laser")
exafs.quality_control(data, exafs.edge_center_func, "FeKEdge Location", threshold=7)
exafs.quality_control(data, exafs.chi2_func, "edge fit chi^2", threshold=7)
exafs.quality_control_range(data, exafs.edge_drop_func, "edge drop", range=(0.02,4.0))
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

def chopper_period(rc):
    # rc is external trigger rowcount
    l = len(rc)
    period = []
    period_rc = []
    n = 1000 # number of external triggers to average over
    for i in xrange(0,l-n-1,n):
        period_rc.append( np.median(rc[i:i+n]) )
        period.append(np.median(np.diff(rc[i:i+n])))
    return np.array(period_rc), np.array(period)*500

def chopper_badness(rc):
    # rc is external trigger rowcount
    l = len(rc)
    drc = np.diff(rc)
    med_diff = np.median(drc)
    lo = int(1.1*med_diff)
    hi = int(10*med_diff)
    badness = []
    badness_rc = []
    n = 1000 # number of external triggers to average over
    for i in xrange(0,l-n-1,n):
        sub = rc[i:i+n]
        dsub = np.diff(sub)
        badness_rc.append( np.median(sub) )
        badness.append( any( np.logical_and(dsub>lo, dsub<hi ) ) )
    return np.array(badness_rc), np.array(badness)



def find_badranges(badness_rc, badness, recover_duration_s=10, pre_problem_buffer_s = 2):
    ranges = []
    lo,hi=0,0
    buildingrange=False
    recover_duration = int(recover_duration_s/320e-9) # how long it takes the chopper to get back to being good after it not longer appears bad
    pre_problem_buffer = int(pre_problem_buffer_s/320e-9)
    for i in range(len(badness_rc)):
        if not buildingrange and badness[i]:
            lo = badness_rc[i]-pre_problem_buffer
            buildingrange=True
        if buildingrange and badness[i]:
            hi = badness_rc[i]+recover_duration
        elif buildingrange and not badness[i] and badness_rc[i]>hi:
            ranges.append((lo,hi))
            buildingrange = False
    return ranges
badness_rc, badness = chopper_badness(ds.external_trigger_rowcount)
badranges = find_badranges(badness_rc, badness)
badrange_duration =  np.sum([r[1]-r[0] for r in badranges])
total_duration = badness_rc[-1]-badness_rc[0]
print("chopper badness cut fraction = %0.4f"%(badrange_duration/float(total_duration)))

plt.figure()
plt.plot(badness_rc*320e-9/3600, badness,".")
plt.xlabel("time (hour)")
plt.ylabel("chopper bad")
for r in badranges:
    lo,hi = r
    plt.plot(np.array([lo,hi])*320e-9/3600, [1.00,1.00],"r",lw=3)
plt.ylim(-0.01,1.01)

badranges_prepped_for_cut = [(lo*320e-9, hi*320e-9) for (lo,hi) in badranges]
badranges_prepped_for_cut.append("invert")

CUT_NUM = ds.CUT_NAME.index("timestamp_sec")
for ds in data:
    ds.cuts.clearCut(CUT_NUM)
    ds.cut_parameter(ds.p_timestamp, badranges_prepped_for_cut, CUT_NUM)

exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"],binsize=2)
exafs.plot_sqrt_spectra(data, erange = (7080, 7300), binsize=2)



#####################################################
# Whole spectrum
#####################################################
pulse_timing.choose_laser(data, "pumped")

N = 26000
pumped_hist_full = np.zeros(N, dtype=np.int)

for ds in data:
    p_energies = ds.p_energy[ds.good()]
    hist, bins = np.histogram(p_energies, bins=np.linspace(2000, 15000, N+1))
    pumped_hist_full += hist

pulse_timing.choose_laser(data, "unpumped")

N = 26000
unpumped_hist_full = np.zeros(N, dtype=np.int)

for ds in data:
    p_energies = ds.p_energy[ds.good()]
    hist, bins = np.histogram(p_energies, bins=np.linspace(2000, 15000, N+1))
    unpumped_hist_full += hist

import matplotlib.pyplot as pyplot
fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.step((bins[1:] + bins[:-1])/2, pumped_hist_full / 1000.0, c='orange', label="pumped", where='mid')
ax.step((bins[1:] + bins[:-1])/2, unpumped_hist_full / 1000.0, c='g', label="unpumped", where='mid')

ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"Count ($\times$ 1000 per 2 eV bin)")
ax.set_title("20150702 ferrioxalate 500 mM")

leg = ax.legend(numpoints=1, frameon=False, handlelength=0.5)
for l in leg.get_lines():
    l.set_linewidth(2)

ax.set_xlim(2000, 15000)

fig.show()

###########################################################
# Write to files.
###########################################################
with open("20150702_pumped_hist_0.5ev_bin_blue_matter_cut.dat", 'w') as f:
    for e, c in zip((bins[1:] + bins[:-1])/2, pumped_hist_full):
        f.write("{0:.2f},{1:d}\n".format(e, int(c)))

with open("20150702_unpumped_hist_0.5ev_bin_blue_matter_cut.dat", 'w') as f:
    for e, c in zip((bins[1:] + bins[:-1])/2, unpumped_hist_full):
        f.write("{0:.2f},{1:d}\n".format(e, int(c)))
