import numpy as np
import pylab as plt
import mass
import pulse_timing

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
        ds.phase_correct2014(typical_resolution, plot=True)

def phase_correct2014_dataset(self, typical_resolution, plot=False):
    """Apply the phase correction that seems good for calibronium-like
    data as of June 2014. For more notes, do
    help(mass.core.analysis_algorithms.FilterTimeCorrection)

    <typical_resolution> should be an estimated energy resolution in UNITS OF
    self.p_pulse_rms. This helps the peak-finding (clustering) algorithm decide
    which pulses go together into a single peak.  Be careful to use a semi-reasonable
    quantity here.

    This version is for when segment 0 doesn't have enough data to train one,
    for example when only 3% of the data is calibration data.
    """
    data,g = good_pulses_data(self, max_records = 20000)
    prompt = self.p_promptness
    dataFilter = self.filter.filt_noconst
    tc = mass.core.analysis_algorithms.FilterTimeCorrection(
            data, prompt[g], self.p_pulse_rms[g], dataFilter,
            self.nPresamples, typicalResolution=typical_resolution)

    self.p_filt_value_phc = self.p_filt_value_dc - tc(prompt, self.p_pulse_rms)

    if plot:
        plt.clf()
        g = self.cuts.good()
        plt.plot(prompt[g], self.p_filt_value_dc[g], 'g.',label="dc")
        plt.plot(prompt[g], self.p_filt_value_phc[g], 'b.',label="phc")
        plt.legend()

def good_pulses_data(ds, max_records=20000):
    """
    :param ds: at dataset object
    :param max_records: roughly maximum records to includes, it can exceed this number by segment_size
    :return: data, g
    data is a (X,Y) array where X is number of records, and Y is number of samples per record
    g is a 1d array of booleans of size X
    if we could did load all of ds.data at once, this would be roughly equivalent to
    return ds.data[ds.cuts.good()], ds.cuts.good()
    """
    first, end = ds.pulse_records.read_segment(0)
    g = ds.cuts.good()
    data = ds.data[g[first:end]]
    for j in xrange(1, ds.pulse_records.n_segments):
        first, end = ds.pulse_records.read_segment(j)
        data = np.vstack((data, ds.data[g[first:end]]))
        if data.shape[0]>max_records:
            break
    return data, g[:end]

def calc_laser_phase(data):
    ds = data.first_good_dataset
    phase, spline = pulse_timing.calc_phase(ds.p_timestamp)
    for ds in data:
        if not hasattr(ds, "p_laser_phase"):
            ds.p_laser_phase = spline.phase(ds.p_timestamp)

def choose_laser(data, band, cut_lines=[0.47,0.515]):
    """
    uses the dataset.cuts object to mark bad all pulses not in a specific category related
    to laser timing
    :param data: a microcal TESChannelGroup object
    :param band: options: (1,2,"laser", "not_laser") for( band1, band2, band1 and band2, not_laser pulses)
    :param cut_lines: same as pulse_timing.phase_2band_find
    :return: None
    """
    band = str(band).lower()
    cutnum = data.first_good_dataset.CUT_NAME.index('timing')
    for ds in data:
        band1, band2, bandNone = pulse_timing.phase_2band_find(ds.p_laser_phase,cut_lines=cut_lines)
        if band == '1':
            ds.cuts.cut(cutnum, np.logical_not(band1))
        elif band == '2':
            ds.cuts.cut(cutnum, np.logical_not(band2))
        elif band == 'not_laser':
            ds.cuts.cut(cutnum, np.logical_not(bandNone))
        elif band == "laser":
            ds.cuts.cut(cutnum, bandNone)