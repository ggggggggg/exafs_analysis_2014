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
dir_p = "20140617_laser_plus_calibronium_timing"
dir_n = "20140617_laser_plus_calibronium_timing_noise"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
chan_nums = available_chans[:10]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)
data = mass.TESGroup(pulse_files, noise_files, auto_pickle=True)

# analyze data
data.summarize_data_tdm(peak_time_microsec=220.0)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
data.drift_correct()
pulse_timing.calc_laser_phase(data)
pulse_timing.choose_laser(data, "not_laser")
#ds.phase_correct2014_dataset(10, plot=True) # doesnt work right now
data.calibrate('p_filt_value_dc', ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'],
                        eps=10,mcs=20, excl=["MnKBeta", "FeKBeta"], forceNew=False)
pulse_timing.label_pumped_band_for_alternating_pump(data)
data.pickle_datasets()

# write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)


# diagnostics
ds = data.channel[1]
pulse_timing.choose_laser_dataset(ds,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")

mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_dc'], True)
exafs.timestructure_dataset(ds,"p_filt_value_dc")
exafs.calibration_summary(data, "p_filt_value_dc")

# save plots
exafs.save_all_plots_as_pdf(data)



