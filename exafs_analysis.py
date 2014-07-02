import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import traceback, sys


# load data
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140620_1M_ferrioxalate_straw/"
dir_n = "20140620_1M_ferrioxalate_straw_noise/"
# dir_p = "20140617_laser_plus_calibronium_timing/"
# dir_n = "20140617_laser_plus_calibronium_timing_noise/"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
if len(available_chans)==0: raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:4]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files, auto_pickle=True)

# analyze data
data.summarize_data_tdm(peak_time_microsec=220.0, forceNew=True)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=True)
pulse_timing.choose_laser(data, "not_laser")
data.drift_correct(forceNew=False)
#ds.phase_correct2014_dataset(10, plot=True) # doesnt work right now
data.calibrate('p_filt_value_dc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        eps=5,mcs=20, excl=[],forceNew=True)
pulse_timing.label_pumped_band_for_alternating_pump(data, forceNew=False)
data.pickle_datasets()

# write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7200), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)


# diagnostics
ds = data.first_good_dataset
pulse_timing.choose_laser_dataset(ds,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")

mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_dc'], True)
exafs.timestructure_dataset(ds,"p_filt_value_dc")
exafs.calibration_summary(data, "p_filt_value_dc")

#save plots
#exafs.save_all_plots(data)


ljh_fname = ds.filename
crate_epoch_usec, crate_frame = mass.load_aux_file(ljh_fname)
starts, ends = pulse_timing.monotonic_frame_ranges(np.array(crate_frame, dtype=np.int))

# the ratio of diff(crate_epoch_usec) to diff(crate_frame) appears to follow a pattern  with period 4
# one sample with a much higher than average ratio, two with typical ratios, one with much lower ratio
# so I would like to resample both of them such that each sample is now the average of 4 (or a multiple of 4)
# other samples, maybe roughy 1 second is good
period_entries = 4 # psuedo-period in plot of diff(crate_epoch_usec), it was 4 when I looked, but it may not always be 4
resampling_period_s = 1
samples_per_newsample = int(period_entries*np.ceil(1e6*resampling_period_s/(period_entries*np.mean(np.diff(crate_epoch_usec)))))
resampled_crate_epoch = []
resampled_crate_frame = []
for j in range(len(starts)):
    resampled_crate_epoch.append(pulse_timing.downsampled(crate_epoch_usec[starts[j]:ends[j]], samples_per_newsample))
    resampled_crate_frame.append(pulse_timing.downsampled(crate_frame[starts[j]:ends[j]], samples_per_newsample))

start_frames = [resampled_crate_frame[0][0]]
offsets = []
for j in range(len(starts)):
    offsets.append(-resampled_crate_frame[j][0]+start_frames[j])
    if j != len(starts)-1:
        first_epoch_in_next = resampled_crate_epoch[j+1][0]
        start_frames.append(pulse_timing.extrap(np.array([first_epoch_in_next]), resampled_crate_epoch[j], resampled_crate_frame[j]+offsets[j]))

new_frame = [r+offsets[i] for i,r in enumerate(resampled_crate_frame)]

