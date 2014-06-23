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
exafs.calc_laser_phase(data)
data.compute_noise_spectra()
exafs.apply_cuts(data, exafs.basic_cuts)
exafs.avg_pulse(data) # creates masks and compute average pulses
data.plot_average_pulses(-1)
data.compute_filters(f_3db=10000.0)
data.filter_data_tdm(forceNew=False)
exafs.drift_correct(data)
exafs.choose_laser(data, "not_laser")
# exafs.phase_correct(data, typical_resolution=10)



ds = data.first_good_dataset
d =good_pulses_data(ds)


# import sklearn.cluster
# res=10
# MIN_PCT = 2
# MIN_PULSES = 50
# min_samples = max(MIN_PULSES, int(0.5+ 0.01*MIN_PCT*N))
#
# for ds in data:
#     energy = ds.p_pulse_rms[ds.cuts.good()]
#     N = len(energy)
#
#     _core_samples, labels = sklearn.cluster.dbscan(energy.reshape((N,1)), eps=res,
#                                                    min_samples=min_samples)
#     labels = np.asarray(labels, dtype=int)
#     labelCounts,_ = np.histogram(labels, 1+labels.max(), [-.5, .5+labels.max()])
#     print 'Label counts: ', labelCounts
#
