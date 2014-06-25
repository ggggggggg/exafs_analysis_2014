import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import traceback, sys



dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140617_laser_plus_calibronium_timing"
dir_n = "20140617_laser_plus_calibronium_timing_noise"
available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))
chan_nums = available_chans[:3]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)

data = mass.TESGroup(pulse_files, noise_files)
data.summarize_data_tdm(peak_time_microsec=220.0)
pulse_timing.calc_laser_phase(data)
data.compute_noise_spectra()
data.apply_cuts(exafs.basic_cuts)
data.avg_pulses_auto_masks(data) # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
data.drift_correct()
data.pickle_datasets()
pulse_timing.choose_laser(data, "not_laser")
#ds.phase_correct2014_dataset(10, plot=True) # doesnt work right now
data.calibrate('p_filt_value_dc', ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'],
                        eps=10,mcs=20, excl=["MnKBeta", "FeKBeta"])



ds = data.channel[1]
mass.calibration.young.diagnose_calibration(ds.calibration['p_filt_value_dc'], True)
data.pickle_datasets()

pulse_timing.label_pumped_band_for_alternating_pump(ds)
pulse_timing.choose_laser(data, "pumped")




