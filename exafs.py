import numpy as np
import pylab as plt
import mass

basic_cuts = mass.core.controller.AnalysisControl(
    pulse_average=(0.0, None),
    pretrigger_rms=(None, 30.0),
    pretrigger_mean_departure_from_median=(-40.0, 40.0),
    peak_value=(0.0, None),
    max_posttrig_deriv=(None, 300.0),
    rise_time_ms=(None, 0.6),
    peak_time_ms=(None, 0.8))

def avg_pulse(data, max_pulses_to_use=7000):
    median_pulse_avg = np.array([np.median(ds.p_pulse_average[ds.good()]) for ds in data])
    masks = data.make_masks([.95, 1.05], use_gains=True, gains=median_pulse_avg)
    for m in masks:
        if len(m) > max_pulses_to_use:
            m[max_pulses_to_use:] = False
    data.compute_average_pulse(masks)

def apply_cuts(data, cuts):
    for ds in data:
        ds.apply_cuts(cuts)

def drift_correct(data):
    for ds in data:
        ds.drift_correct()

def phase_correct(data,typical_resolution):
    # note that typical resolution must be in units of p_pulse_rms
    for ds in data:
        ds.phase_correct2014(typical_resolution)

