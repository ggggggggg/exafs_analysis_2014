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
data.pickle_datasets()
exafs.choose_laser(data, "not_laser")
#exafs.phase_correct2014_dataset(ds, 10, plot=True) # doesnt work right now

ds = data.channel[1]
from mass.calibration import young
ycal = young.EnergyCalibration(eps=10,mcs=20, excl=["MnKBeta", "FeKBeta"])
ycal.fit(ds.p_filt_value_dc[ds.cuts.good()], ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'])
young.diagnose_calibration(ycal, True)


# params: a 6-element sequence of [Resolution (fwhm), Pulseheight of the Kalpha1 peak,
#                 energy scale factor (pulseheight/eV), amplitude, background level (per bin),
#                 and background slope (in counts per bin per bin) ]
#                 If params is None or does not have 6 elements, then they will be guessed.

def is_calibrated(cal):
    if hasattr(cal,"npts"): # checks for Joe style calibration
        return False
    if cal.elements is None: # then checks for now many elements are fitted for
        return False
    return True

def calibrate(data, attr, line_names,name_ext="",eps=10, mcs=20, hw=200, excl=(), plot_on_fail=False, forceNew=False):
    for ds in data:
        calibrate_dataset(ds, attr, line_names,name_ext,eps, mcs, hw, excl, plot_on_fail, forceNew)

def calibrate_dataset(ds, attr, line_names,name_ext="",eps=10, mcs=20, hw=200, excl=(), plot_on_fail=False, forceNew=False):
    calname = attr+name_ext
    if ds.calibration.has_key(calname):
        cal = ds.calibration[calname]
        if is_calibrated(cal) and not forceNew:
            print("Not calibrating chan %d %s because it already exists"%(ds.channum, calname))
            return None
        # first does this already exist? if the calibration already exists and has more than 1 pt,
        # we probably dont need to redo it
    cal = young.EnergyCalibration(eps, mcs, hw, excl, plot_on_fail)
    cal.fit(getattr(ds, attr)[ds.cuts.good()], line_names)
    ds.calibration[calname]=cal
    return cal

def convert_to_energy_dataset(ds, attr, calname=None):
    if calname is None: calname = attr
    if not ds.calibration.has_key(calname):
        raise ValueError("For chan %d calibration %s does not exist"(ds.channum, calname))
    cal = ds.calibration[calname]
    ds.p_energy = cal.ph2energy(getattr(ds, attr))

def convert_to_energy(data, attr, calname=None):
    if calname is None: calname = attr
    print("for all channels converting %s to energy with calibration %s"%(attr, calname))
    for ds in data:
        convert_to_energy_dataset(ds, attr, calname)

ds = data.channel[1]
cal = calibrate_dataset(ds, 'p_filt_value_dc', ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'],
                        eps=10,mcs=20, excl=["MnKBeta", "FeKBeta"])
young.diagnose_calibration(cal, True)

#calibrate(data, 'p_filt_value_dc', ['MnKAlpha', 'CuKAlpha', 'VKAlpha', 'ScKAlpha', 'CoKAlpha', 'FeKAlpha', 'CuKBeta'])