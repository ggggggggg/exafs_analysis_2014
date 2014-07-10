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
data.summarize_data_tdm(peak_time_microsec=220.0, forceNew=False)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks() # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
pulse_timing.apply_offsets_for_monotonicity(data)
pulse_timing.calc_laser_phase(data, forceNew=False)
pulse_timing.choose_laser(data, "not_laser")
data.drift_correct(forceNew=False)
data.phase_correct2014(10, plot=True) # doesnt work right now
data.calibrate('p_filt_value_dc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=True, max_pulses_for_dbscan=1e5)
data.calibrate('p_filt_value_phc', ['VKAlpha', 'MnKAlpha', 'MnKBeta', 'FeKAlpha', 'CoKAlpha', 'CoKBeta', 'CuKAlpha', "FeKBeta", "VKBeta","CuKBeta","ScKAlpha","NiKAlpha"],
                        size_related_to_energy_resolution=20.0,min_counts_per_cluster=20,
                        excl=[],forceNew=False, max_num_clusters = 18, plot_on_fail=True, max_pulses_for_dbscan=1e5)
pulse_timing.label_pumped_band_for_alternating_pump(data, forceNew=False)
data.pickle_datasets()

# write histograms
exafs.plot_combined_spectra(data, ref_lines=["FeKEdge"])
exafs.plot_combined_spectra(data, erange = (7080, 7200), ref_lines=["FeKEdge"])
exafs.write_channel_histograms(data, erange=(0,20000), binsize=5)
exafs.write_combined_energies_hists(data, erange=(0,20000), binsize=5)


# diagnostics
ds = data.first_good_dataset
pulse_timing.choose_laser(data,"laser")
exafs.fit_edge_in_energy_dataset(ds, "FeKEdge",doPlot=True)
exafs.fit_edges(data,"FeKEdge")

mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_phc'], True)
exafs.timestructure_dataset(ds,"p_filt_value_phc")
exafs.calibration_summary(data, "p_filt_value_phc")
exafs.pulse_summary(data)

#save plots
#exafs.save_all_plots(data)





