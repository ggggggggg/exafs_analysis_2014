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


dir_base = "/Volumes/Drobo/exafs_data/"
dir_p = "20150826_iron_tris_xes_multi_delay/"
# dir_p = "20150808_50mM_irontris_xes_m20mm_55mm/"
# New dir_p here

dir_n = "20150826_iron_tris_xes_multi_delay_noise"
# New dir_n here

available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))

if len(available_chans) == 0:
    raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)

data = mass.TESGroup(pulse_files, noise_files)

if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output

# analyze data
data.summarize_data(peak_time_microsec=300.0, forceNew=False)

data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks()  # creates masks and compute average pulses
data.plot_average_pulses(-1)

data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)



ds = data.first_good_dataset

pulse_timing.calc_laser_phase(data, forceNew=True)

pulse_timing.choose_laser(data, "laser")
data.drift_correct(forceNew=True)
#data.phase_correct2014(10, plot=False, forceNew=False, pre_sanitize_p_filt_phase=True)
data.calibrate('p_filt_value_dc', ['FeKAlpha', 'FeKBeta'],
               size_related_to_energy_resolution=20.0, min_counts_per_cluster=20,
               excl=[],forceNew=True, max_num_clusters = 4, plot_on_fail=False, max_pulses_for_dbscan=1e5)



################################################################################
#
#  You need convert to unix timestamp, if you need to cut based on the vega power
#
################################################################################
crate_epoch_usec, crate_frame = mass.load_aux_file(ds.filename)
avg_crate_timebase = 1.0 * (crate_epoch_usec[-1] - crate_epoch_usec[0]) / (crate_frame[-1] - crate_frame[0])
crate_epoch_usec_dev = crate_epoch_usec - crate_epoch_usec[0] - avg_crate_timebase * (crate_frame - crate_frame[0])


##### CUT ON DELAY STAGE
delay_filename = path.join(dir_base, dir_p,"delay_stage_scan.log")

move_posix_ts, move_desc = [],[]
with open(delay_filename) as f:
    for line in f:
        text_date, posix_ts, movement = line.split("\t")
        move_posix_ts.append(float(posix_ts))
        move_desc.append(movement.strip())

start_end_posix_ts = []
start_end_position_string = []
i=0
j=1
while i<len(move_desc):
    # find a start
    if not move_desc[i].startswith("start"):
        i+=1
        j=1
        continue
    # find an end
    while i+j<len(move_desc):
        if move_desc[i+j].startswith("end"): break
        j+=1
    if i+j>=len(move_desc): break

    # now I know where a start end pair is
    start_end_posix_ts.append((move_posix_ts[i], move_posix_ts[i+j]))
    start_end_position_string.append((move_desc[i], move_desc[i+j]))
    i+=1

# turn move desc into float
start_end_position = [(float(a.split()[-1]), float(b.split()[-1])) for (a,b) in start_end_position_string]
start_end_ts = [(np.interp(a,crate_epoch_usec*1e-6, crate_frame*9.6e-6),np.interp(b,crate_epoch_usec*1e-6, crate_frame*9.6e-6)) for a,b in start_end_posix_ts]



acceptable_pos_error = 0.01
delay_ranges_nonunique = {}
for i in range(len(start_end_position)):
    # validate movement
    target_pos, final_pos = start_end_position[i]
    if np.abs(final_pos-target_pos)>acceptable_pos_error:
        # this is not a good range to use
        continue
    # create time range from end to next start, or None if there is no next start

    if start_end_ts[i][0]>=crate_frame[-1]*9.6e-6 or start_end_ts[i][1]<=crate_frame[0]*9.6e-6: continue # dont use ranges outside of the range we can convert betwen frame and posix timestamp in

    r = delay_ranges_nonunique.setdefault(target_pos,[]) # either get the existing list or make a new one if this is the first entry at that delay
    if i < len(start_end_position)-1:
        r.append((start_end_ts[i][1],start_end_ts[i+1][0]))
    else:
        r.append((start_end_ts[i][1], Inf))


delay_ranges = {k:sorted([a for a in set(delay_ranges_nonunique[k])],key=lambda x: x[0]) for k in delay_ranges_nonunique}


for k in delay_ranges:
    CUTNUM = len(ds.CUT_NAME)
    for ds in data:
        ds.cuts.clearCut(CUTNUM)
        ds.cut_parameter(ds.p_timestamp, delay_ranges[k], CUTNUM)
    delay_str = "delay_%g"%k
    delay_str.replace(".","p")

    ########################################################
    #
    #  Ka, Kb1,3, kb2,5 emission spectra (individual)
    #
    ########################################################
    run_title = "20150807 50mM irontris, delay stage %g mm"%(k)

    emin, emax = 2000, 15000

    N = int((emax - emin) / 0.5)

    pulse_timing.choose_laser(data, "pumped")
    pumped_hist = np.zeros(N)

    for ds in data:
        p_energies = ds.p_energy[ds.good()]
        hist, bins = np.histogram(p_energies, bins=np.linspace(emin, emax, N + 1))
        pumped_hist += hist

    pulse_timing.choose_laser(data, "unpumped")
    unpumped_hist = np.zeros(N)

    for ds in data:
        p_energies = ds.p_energy[ds.good()]
        hist, bins = np.histogram(p_energies, bins=np.linspace(emin, emax, N + 1))
        unpumped_hist += hist

    signals = (pumped_hist - unpumped_hist) / np.sqrt(pumped_hist + unpumped_hist)

    import brewer2mpl
    bmap = brewer2mpl.get_map("Spectral", "Diverging", 8)

    for emin, emax, label in zip([6370, 7020, 7070], [6430, 7090, 7130],
                                 [r"$\mathrm{K}\alpha$",
                                  r"$\mathrm{K}\beta_{1,3}$",
                                  r"$\mathrm{K}\beta_{2,5}$"]):
        emin_index = (emin - 2000) * 2
        emax_index = (emax - 2000) * 2
        ylims1 = (max(0, 0.9 * np.min(unpumped_hist[emin_index:emax_index]) -
                  0.1 * np.max(unpumped_hist[emin_index:emax_index])),
                 np.max(unpumped_hist[emin_index:emax_index]) * 1.1)
        ylims2 = np.array([-1, 1]) * 1.1 * (np.max(np.abs(signals[emin_index:emax_index])))

        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(211)

        bins = np.linspace(2000, 15000, N + 1)
        ax.step((bins[1:] + bins[:-1]) / 2, pumped_hist, where='mid', color=bmap.mpl_colors[-1], label="pumped",
                alpha=0.8)
        ax.step((bins[1:] + bins[:-1]) / 2, unpumped_hist, where='mid', color=bmap.mpl_colors[0], label="unpumped",
                alpha=0.8)

        leg = ax.legend(numpoints=1, frameon=False, handlelength=1.0)
        for l in leg.get_lines():
            l.set_linewidth(2)

        ax.text(0.05, 0.95, label, size=16, ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"Count (per 0.5 eV bin)")
        ax.set_title(run_title)

        ax.set_xlim(emin, emax)
        ax.set_ylim(*ylims1)

        ax = fig.add_subplot(212)

        ax.step((bins[1:] + bins[:-1]) / 2,
                signals,
                where='mid', color='g',
                alpha=1)

        ax.text(0.05, 0.95, label, size=16, ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"$\frac{\mathrm{p - u}}{\sqrt{\mathrm{p + u}}}$")
        ax.set_title(run_title)

        ax.set_xlim(emin, emax)
        ax.set_ylim(*ylims2)

        fig.show()

        fig.savefig("20150807_%s_%g.png"%(delay_str,emin),dpi=600)

    ###########################################################
    #
    # Write to files.
    # Make sure that you changed filenames!
    #
    ###########################################################

    with open("20150807_%s_pumped_hist_0.5ev_bin.dat"%delay_str, 'w') as f:
        for e, c in zip((bins[1:] + bins[:-1]) / 2, pumped_hist):
            f.write("{0:.2f},{1:d}\n".format(e, int(c)))

    with open("20150807_%s_unpumped_hist_0.5ev_bin.dat"%delay_str, 'w') as f:
        for e, c in zip((bins[1:] + bins[:-1]) / 2, unpumped_hist):
            f.write("{0:.2f},{1:d}\n".format(e, int(c)))


