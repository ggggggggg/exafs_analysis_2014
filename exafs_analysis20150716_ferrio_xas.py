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
dir_p = "20150716_ferrio_500mM_xas"
dir_n = "20150716_ferrio_500mM_xas_noise"
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
pulse_timing.calc_laser_phase(data, forceNew=True)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
ds = data.channel[1]
cutnum = ds.CUT_NAME.index("timestamp_sec")
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
pulse_timing.choose_laser(data, "not_laser", keep_size=0.02, exclude_size=0.025, forceNew=True)
data.drift_correct(forceNew=True)
#data.phase_correct2014(10, plot=False, forceNew=True, pre_sanitize_p_filt_phase=True)
data.calibrate('p_filt_value_dc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.calibrate('p_filt_value_phc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)
data.time_drift_correct(forceNew=False)
data.calibrate('p_filt_value_tdc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=True, max_num_clusters = 18, plot_on_fail=False, max_pulses_for_dbscan=1e5)



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


exafs.plot_combined_spectra(data, erange = (7080, 7300), ref_lines=["FeKEdge"],binsize=2)
# exafs.plot_sqrt_spectra(data, erange = (7080, 7300), binsize=2)def chopper_period(rc):
#     # rc is external trigger rowcount
#     l = len(rc)
#     period = []
#     period_rc = []
#     n = 1000 # number of external triggers to average over
#     for i in xrange(0,l-n-1,n):
#         period_rc.append( np.median(rc[i:i+n]) )
#         period.append(np.median(np.diff(rc[i:i+n])))
#     return np.array(period_rc), np.array(period)*500
#
# def chopper_badness(rc):
#     # rc is external trigger rowcount
#     l = len(rc)
#     drc = np.diff(rc)
#     med_diff = np.median(drc)
#     lo = int(1.1*med_diff)
#     hi = int(10*med_diff)
#     badness = []
#     badness_rc = []
#     n = 1000 # number of external triggers to average over
#     for i in xrange(0,l-n-1,n):
#         sub = rc[i:i+n]
#         dsub = np.diff(sub)
#         badness_rc.append( np.median(sub) )
#         badness.append( any( np.logical_and(dsub>lo, dsub<hi ) ) )
#     return np.array(badness_rc), np.array(badness)
#
#
#
# def find_badranges(badness_rc, badness, recover_duration_s=10, pre_problem_buffer_s = 2):
#     ranges = []
#     lo,hi=0,0
#     buildingrange=False
#     recover_duration = int(recover_duration_s/320e-9) # how long it takes the chopper to get back to being good after it not longer appears bad
#     pre_problem_buffer = int(pre_problem_buffer_s/320e-9)
#     for i in range(len(badness_rc)):
#         if not buildingrange and badness[i]:
#             lo = badness_rc[i]-pre_problem_buffer
#             buildingrange=True
#         if buildingrange and badness[i]:
#             hi = badness_rc[i]+recover_duration
#         elif buildingrange and not badness[i] and badness_rc[i]>hi:
#             ranges.append((lo,hi))
#             buildingrange = False
#     return ranges
# badness_rc, badness = chopper_badness(ds.external_trigger_rowcount)
# badranges = find_badranges(badness_rc, badness)
# badrange_duration =  np.sum([r[1]-r[0] for r in badranges])
# total_duration = badness_rc[-1]-badness_rc[0]
# print("chopper badness cut fraction = %0.4f"%(badrange_duration/float(total_duration)))
#
# plt.figure()
# plt.plot(badness_rc*320e-9/3600, badness,".")
# plt.xlabel("time (hour)")
# plt.ylabel("chopper bad")
# for r in badranges:
#     lo,hi = r
#     plt.plot(np.array([lo,hi])*320e-9/3600, [1.00,1.00],"r",lw=3)
# plt.ylim(-0.01,1.01)
#
# badranges_prepped_for_cut = [(lo*320e-9, hi*320e-9) for (lo,hi) in badranges]
# badranges_prepped_for_cut.append("invert")
#
# CUT_NUM = ds.CUT_NAME.index("timestamp_sec")
# for ds in data:
#     ds.cuts.clearCut(CUT_NUM)
#     ds.cut_parameter(ds.p_timestamp, badranges_prepped_for_cut, CUT_NUM)

# talt_slope = 3.5368901445181e-8
#
# # 20150710 dataset
# for ds in data:
#     talt_1 = ds.time_after_last_external_trigger - ((ds.p_timestamp[...] -ds.p_timestamp[0]) * talt_slope)
#
#     hist, bins = np.histogram(talt_1, bins=np.linspace(-0.003, 0.003, 301))
#     lmf = (hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])
#     lm_hist = hist[lmf]
#     lmp = ((bins[2:-1] + bins[1:-2])/2)[lmf]
#     lmp = lmp[np.argsort(lm_hist)[-4:]]
#
#     lmp_2 = []
#
#     for p in lmp:
#         lmp_2.append(np.mean(talt_1[(talt_1 > (p - 30.0e-6)) & (talt_1 < (p + 30.0e-6))]))
#
#     lmp_2.sort()
#
#     talt_diff = np.zeros_like(talt_1)
#     too_low_flag = talt_1 < np.mean(lmp_2[:2])
#     talt_diff[too_low_flag] = lmp_2[3] - lmp_2[1]
#     talt_2 = talt_1 + talt_diff
#     too_low_flag = talt_2 < np.mean(lmp_2[1:3])
#     talt_diff = np.zeros_like(talt_2)
#     talt_diff[too_low_flag] = lmp_2[2] - lmp_2[0]
#     talt_3 = talt_2 + talt_diff
#
#     ds.time_after_last_external_trigger[...] = talt_3

for i in range(ds.pulse_records.n_segments):
    a, b = ds.read_segment(i)
    ds.hdf5_group['timestamp'][a:b] = ds.times

for ds in data:
    original_timestamp = ds.p_timestamp[...]
    new_timestamp = original_timestamp / np.float64(np.float32(9.6e-6)) * np.float64(9.6e-6)

    ds.p_timestamp[...] = new_timestamp

    del ds.hdf5_group['time_after_last_external_trigger']
    ds.time_after_last_external_trigger
    print("Chan {0:d} time_after_last_external_trigger is finished.".format(ds.channum))

import matplotlib.pyplot as pyplot

ds = data.channel[1]

fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.hist(ds.p_timestamp, bins=np.linspace(0, 56000, 56001), histtype='step')

fig.show()


from mass.core.ljh_util import load_aux_file

crate_unix_usecs, crate_frames = load_aux_file(os.path.join(dir_base, dir_p, "20150714_ferrio_500mM_xas_2"))
crate_unix_usecs = np.asarray(crate_unix_usecs, dtype=np.int64)
crate_frames = np.asarray(crate_frames, dtype=np.int64)

fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.scatter(crate_unix_usecs[::500], crate_frames[::500], s=2, edgecolors='None', c='orange', alpha=0.5)

fig.show()

vega_timestamps = []
vega_powers = []

with open("20150713_ferrioxalate_500mM_160ps_delay/vega_power_20150713.dat") as vega_file:
    for l in vega_file:
        t, p = [float(t) for t in l.strip().split('\t')]
        vega_timestamps.append(t)
        vega_powers.append(p)

import matplotlib.dates
import pytz

mtz = pytz.timezone('US/Mountain')

fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])

ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d %H:%M", tz=mtz))

#ax.plot(range(3))
ax.scatter([datetime.fromtimestamp(st, tz=mtz) for st in vega_timestamps], vega_powers, s=2, edgecolors='None', alpha=0.2)

for l in ax.get_xticklabels():
    l.set_rotation(45)

fig.show()

# Create p_unix_timestamp data set for each channel.
for ds in data:
    pulse_unix_timestamp = pulse_timing.extrap(ds.p_timestamp[...] / 9.6e-6, crate_frames[7089:],
                                               1.0e-6 * crate_unix_usecs[7089:])
    ds.p_unix_timestamp = ds.hdf5_group.require_dataset("unix_timestamp", ds.p_timestamp.shape, dtype=np.float64)
    ds.p_unix_timestamp[...] = pulse_unix_timestamp

for ds in data:
    pulse_vega_power = pulse_timing.extrap(ds.hdf5_group['unix_timestamp'][...], vega_timestamps, vega_powers)
    ds.p_vega_power = ds.hdf5_group.require_dataset("vega_power", ds.p_timestamp.shape, dtype=np.float64)
    ds.p_vega_power[...] = pulse_vega_power

unix_timestamp_bins = np.arange(
    18000,
    81000,
    1
)

matter_counts = np.zeros(unix_timestamp_bins.shape[0] - 1)

# This is a histogram of all pulses. (Good, bad, xsi, or laser)
for ds in data:
    hist, bins = np.histogram(ds.p_timestamp, bins=unix_timestamp_bins)
    matter_counts += hist

for ds in data:
    pulse_matter_count = pulse_timing.extrap(ds.p_timestamp[...],
                                             (unix_timestamp_bins[1:] + unix_timestamp_bins[:-1])/2,
                                             matter_counts)
    ds.p_array_countrate = ds.hdf5_group.require_dataset("array_countrate", ds.p_timestamp.shape, dtype=np.float64)
    ds.p_array_countrate = pulse_matter_count


def load_vega_power(filename):
    f = open(filename)

    vega_power = []
    vega_epochtime = []
    i = 0

    for line in f:
        try:
            a,b=line.split("\t")
            a = float(a)
            b = float(b)
            vega_epochtime.append(a)
            vega_power.append(b)
        except:
            pass

    return np.array(vega_epochtime),np.array(vega_power)


def find_badranges(vega_timestamp, vega_power, threshold_power_hi=0.3, threshold_power_lo=0.1):
    building_bad_region = False
    lo_extra = 10
    hi_extra = 0
    lo,hi=0,0
    badranges = []
    for i in xrange(len(vega_timestamp)):
        t,p = vega_timestamp[i], vega_power[i]
        power_in_range = threshold_power_lo<p<threshold_power_hi
        power_out_of_range = not power_in_range
        if power_out_of_range:
            if not building_bad_region:
                lo = t-lo_extra
                building_bad_region = True
            hi=max(lo,t+hi_extra)
        elif t>hi and building_bad_region:
                badranges.append((lo,hi))
                building_bad_region=False

    if building_bad_region:
        badranges.append((lo,hi))

    return join_overlapping_ranges(badranges)


def join_overlapping_ranges(ranges):
    outranges = []
    oldlo, oldhi = -1,-1
    building_lo, building_hi = -1,-1
    building_range = False
    for lo, hi in ranges:
        if building_range:
            if lo > building_hi:
                outranges.append((building_lo, building_hi))
                building_range = False
            else:
                building_hi = hi
        if not building_range:
            building_lo = lo
            building_hi = hi
            building_range = True
    if building_range:
        outranges.append((building_lo, building_hi))
    return outranges

vega_badranges = find_badranges(vega_timestamps, vega_powers, threshold_power_hi=0.2, threshold_power_lo=0.15)
matter_badranges = find_badranges((unix_timestamp_bins[1:] + unix_timestamp_bins[:-1])/2,
                                  matter_counts, threshold_power_hi=5500, threshold_power_lo=1000)

total_badranges = vega_badranges + matter_badranges

import itertools
import operator

while True:
    temp = list(total_badranges)
    for r, u in itertools.combinations(temp, 2):
        if (r[1] > u[0]) and (r[0] < u[1]):
            total_badranges.remove(r)
            total_badranges.remove(u)
            total_badranges.append((min(r[0], u[0]), max(r[1], u[1])))
            break
    else:
        break

total_badranges.sort(key=operator.itemgetter(0))

badranges_prepped_for_cut = matter_badranges
badranges_prepped_for_cut.append("invert")

CUT_NUM = ds.CUT_NAME.index("timestamp_sec")
for ds in data:
    ds.cuts.clearCut(CUT_NUM)
    ds.cut_parameter(ds.p_timestamp[...], badranges_prepped_for_cut, CUT_NUM)


### Edge drops

pulse_timing.choose_laser(data, "pumped")
N = 450

pumped_hist_f = np.zeros(N)
pumped_hist_s = np.zeros(N)

for ds in data:
    p_energies = ds.p_energy[ds.good()]
    hist, bins = np.histogram(p_energies[:p_energies.shape[0]/2], bins=np.linspace(6700, 7600, N+1))
    pumped_hist_f += hist
    hist, bins = np.histogram(p_energies[p_energies.shape[0]/2:], bins=np.linspace(6700, 7600, N+1))
    pumped_hist_s += hist

pulse_timing.choose_laser(data, "unpumped")
unpumped_hist_f = np.zeros(N)
unpumped_hist_s = np.zeros(N)

for ds in data:
    p_energies = ds.p_energy[ds.good()]
    hist, bins = np.histogram(p_energies[:p_energies.shape[0]/2], bins=np.linspace(6700, 7600, N+1))
    unpumped_hist_f += hist
    hist, bins = np.histogram(p_energies[p_energies.shape[0]/2:], bins=np.linspace(6700, 7600, N+1))
    unpumped_hist_s += hist

pumped_hist = pumped_hist_f + pumped_hist_s
unpumped_hist = unpumped_hist_f + unpumped_hist_s

import matplotlib as mpl
import brewer2mpl

mpl.rcParams['savefig.dpi'] = 80
bmap = brewer2mpl.get_map("Spectral", "Diverging", 8)

fig = pyplot.figure()
ax = fig.add_subplot(111)

N = 450
bins = np.linspace(6700, 7600, N + 1)
ax.step((bins[1:] + bins[:-1])/2, unpumped_hist / 1000, where='mid', color=bmap.mpl_colors[0], label="unpumped", alpha=0.8)
ax.step((bins[1:] + bins[:-1])/2, pumped_hist / 1000, where='mid', color=bmap.mpl_colors[-1], label="pumped", alpha=0.8)

leg = ax.legend(numpoints=1, frameon=False, handlelength=1.0)
for l in leg.get_lines():
    l.set_linewidth(2)

ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"Count ($\times$1000 per 2 eV bin)")
ax.set_title("20150716 500mM ferrioxalate with Matter count cuts")

ax.set_ylim(17.5, 29.5)

fig.show()

# Signals
fig = plt.figure()
ax = fig.add_subplot(111)

N = 450
bins = np.linspace(6700, 7600, N + 1)
ax.step((bins[1:] + bins[:-1])/2, (pumped_hist - unpumped_hist) / np.sqrt(pumped_hist + unpumped_hist), where='mid',
        color='g')

ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"(pumped - unpumped) / sqrt(pumped + unpumped")
ax.set_title("20150713 500mM ferrioxalate with Blue power cand Matter count cuts")

ax.set_ylim(-4, 4)

fig.show()

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

fig = pyplot.figure()
ax = fig.add_subplot(111)

ax.plot((bins[1:] + bins[:-1])/2, pumped_hist_full / 1000.0, c='orange', label="pumped")
ax.plot((bins[1:] + bins[:-1])/2, unpumped_hist_full / 1000.0, c='g', label="unpumped")

ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"Count ($\times$ 1000 per 0.5 eV bin)")
ax.set_title("20150714_2 ferrioxalate 500 mM with Matter count cuts")

leg = ax.legend(numpoints=1, frameon=False, handlelength=0.5)
for l in leg.get_lines():
    l.set_linewidth(2)

ax.set_xlim(2000, 15000)

fig.show()

###########################################################
# Write to files.
###########################################################
with open("20150716_pumped_hist_0.5ev_bin_matter_cut.dat", 'w') as f:
    for e, c in zip((bins[1:] + bins[:-1])/2, pumped_hist_full):
        f.write("{0:.2f},{1:d}\n".format(e, int(c)))

with open("20150716_unpumped_hist_0.5ev_bin_matter_cut.dat", 'w') as f:
    for e, c in zip((bins[1:] + bins[:-1])/2, unpumped_hist_full):
        f.write("{0:.2f},{1:d}\n".format(e, int(c)))

with open("20150717_ferrioxlate_pumped_coadded.dat", 'w') as f:
    for e, c in zip(np.linspace(2001, 14999, 6500), shared_pumped):
        f.write("{0:.2f},{1:d}\n".format(e, c))

with open("20150717_ferrioxlate_unpumped_coadded.dat", 'w') as f:
    for e, c in zip(np.linspace(2001, 14999, 6500), shared_unpumped):
        f.write("{0:.2f},{1:d}\n".format(e, c))
