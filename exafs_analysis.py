import numpy as np
import pylab as plt
import mass
from os import path
import exafs
import traceback, sys



maxchan=99999
dir_base = "/Volumes/Drobo/exafs_data"
dir_p = "20140617_laser_plus_calibronium_timing"
dir_n = "20140617_laser_plus_calibronium_timing_noise"
chan_nums = (1,3,5)
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)

data = mass.TESGroup(pulse_files, noise_files)
data.summarize_data_tdm(peak_time_microsec=220.0)
data.compute_noise_spectra()
exafs.apply_cuts(data, exafs.basic_cuts)
exafs.avg_pulse(data) # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
