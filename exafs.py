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

def is_drift_corrected(ds):
    return not all(ds.p_filt_value_dc == 0)

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

def timestructure_dataset(ds, calname="p_filt_value_dc"):
    pulse_timing.choose_laser_dataset(ds, "not_laser")
    cal = ds.calibration[calname]
    energy = ds.p_energy[ds.cuts.good()]
    cmap = plt.get_cmap()
    cmap = [cmap(i/float(len(cal.elements))) for i in xrange(len(cal.elements))]

    plt.figure()
    plt.plot(ds.p_timestamp[ds.cuts.good()], ds.p_energy[ds.cuts.good()],'.')
    plt.xlabel("frame timestamp (s)")
    plt.ylabel("p_energy")
    plt.title("chan %d, not_laser pulses selected"%ds.channum)

    for i,line_name in enumerate(cal.elements):
        low,high = mass.energy_calibration.STANDARD_FEATURES[line_name]*np.array([0.99, 1.01])
        use = np.logical_and(energy>low, energy<high)
        use_time = ds.p_timestamp[ds.cuts.good()][use]
        pfit = np.polyfit(use_time, energy[use],1)
        plt.plot(use_time, np.polyval(pfit, use_time),c=cmap[i], label=line_name+" %0.2f eV/hr"%(pfit[0]*3600))
        plt.plot(use_time, energy[use],'.',c=cmap[i])
        plt.legend()