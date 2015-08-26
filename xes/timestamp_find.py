import numpy as np
import pylab as plt
import time
import h5py

ljh_ts = np.array([165328.23048,241494.86127359999])

ljh_timestamp_offset = 1438800684.587908

aux_filename = "/Volumes/Drobo/exafs_data/20150807_50mM_irontris_xes_m20mm_55mm_correct_trigger/20150807_50mM_irontris_xes_m20mm_55mm_correct_trigger.timing_aux"

data = np.fromfile(aux_filename, dtype = "int64")
data.shape = (data.size/2,2)
crate_frame, crate_posix = data[:,0], data[:,1]
crate_ts = crate_frame*9.6e-6

ljh_posix = np.interp(ljh_ts, crate_ts, crate_posix*1e-6)
ljh_gmtime = [time.gmtime(p) for p in ljh_posix]

extern_trig_hdf5s = ["/Volumes/Drobo/exafs_data/20150808_50mM_irontris_xes_m20mm_55mm/20150808_50mM_irontris_xes_m20mm_55mm_extern_trig.hdf5",
                     "/Volumes/Drobo/exafs_data/20150807_50mM_irontris_xes_m20mm_55mm/20150807_50mM_irontris_xes_m20mm_55mm_extern_trig.hdf5"]

h5 =  h5py.File(extern_trig_hdf5s[0])
extern_ts = 9.6e-6*h5["trig_times"][:]/30


extern_gmtime = [time.gmtime(p) for p in np.interp([extern_ts[0], extern_ts[-1]], crate_ts, crate_posix*1e-6)]

delay_filename =