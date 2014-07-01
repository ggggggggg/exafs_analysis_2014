import numpy as np
import pylab as plt
import mass
import pulse_timing
from scipy.optimize import curve_fit
import os
from os import path

basic_cuts = mass.core.controller.AnalysisControl(
    pulse_average=(0.0, None),
    pretrigger_rms=(None, 30.0),
    pretrigger_mean_departure_from_median=(-40.0, 40.0),
    peak_value=(0.0, None),
    max_posttrig_deriv=(None, 300.0),
    rise_time_ms=(None, 0.6),
    peak_time_ms=(None, 0.8))


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

from scipy.special import erf
def edge_model(x, edgeCenter, preHeight, postHeight, fwhm=7.0, bgSlope=0):
    # this model is a gaussian smoothed step edge according to wikipedia + a slope
    return 0.5*(postHeight-preHeight)*erf(0.707106781186*(x-edgeCenter)/float(1e-20+fwhm/2.3548201)) + 0.5*(preHeight+postHeight) + (x-edgeCenter)*bgSlope

def fit_edge_hist(bins, counts, fwhm_guess=10.0):
    if len(bins) == len(counts)+1: bins = bins[:-1]+0.5*(bins[1]-bins[0]) # convert bin edge to bin centers if neccesary
    pfit = np.polyfit(bins, counts, 3)
    edgeGuess = np.roots(np.polyder(pfit,2))
    try:
        preGuessX, postGuessX = np.sort(np.roots(np.polyder(pfit,1)))
    except:
        raise ValueError("failed to generate guesses")
    use = bins>(edgeGuess+2*fwhm_guess)
    if len(use)>4:
        pfit2 = np.polyfit(bins[use], counts[use],1)
        slope_guess = pfit2[0]
    else:
        slope_guess=1
    pGuess = np.array([edgeGuess, np.polyval(pfit,preGuessX), np.polyval(pfit,postGuessX),fwhm_guess,slope_guess],dtype='float64')

    try:
        pOut = curve_fit(edge_model, bins, counts, pGuess)
    except:
        raise ValueError("failed to fit data to edge_model")
    (edgeCenter, preHeight, postHeight, fwhm, bgSlope) = pOut[0]
    return (edgeCenter, preHeight, postHeight, fwhm, bgSlope)

def fit_edge_in_energy_dataset(ds, edge_name, width_ev=400, bin_size_ev=3, fwhm_guess=10.0, doPlot=False):
    if not edge_name[-4:].lower()=="edge": raise ValueError("%s is not an edge"%edge_name)
    low_energy, high_energy = mass.energy_calibration.STANDARD_FEATURES[edge_name] +  np.array([-0.5, 0.5])*width_ev
    bin_edges = np.arange(low_energy, high_energy, bin_size_ev)
    counts, bin_edges = np.histogram(ds.p_energy[ds.cuts.good()], bin_edges)
    if doPlot:
        bin_centers = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])
        plt.figure()
        plt.plot(bin_centers, counts)
    (edgeCenter, preHeight, postHeight, fwhm, bgSlope) = fit_edge_hist(bin_edges, counts, fwhm_guess)

    if doPlot:
        plt.plot(np.linspace(low_energy,high_energy,1000), edge_model(np.linspace(low_energy,high_energy,1000), edgeCenter, preHeight, postHeight, fwhm, bgSlope))
        plt.xlabel("energy (eV)")
        plt.ylabel("counts per %0.2f eV bin"%(bin_edges[1]-bin_edges[0]))
        plt.title("chan %d %s"%(ds.channum, edge_name))

    return (edgeCenter, preHeight, postHeight, fwhm, bgSlope)

def write_histogram_dataset(ds, fname, erange=(0,20000), binsize=5):
    fname+=".spectrum"
    bin_edge = np.arange(binsize*0.5+erange[0], erange[1]+binsize*0.5, binsize)
    bin_centers = bin_edge[:-1]+binsize*0.5
    counts, bin_edge = np.histogram(ds.p_energy[ds.cuts.good()], bin_edge)
    np.savetxt(fname, np.vstack((bin_centers, counts)).T,fmt=("%0.1f", "%i"), header="energy bin centers (eV), counts per bin")

def write_histograms_dataset(ds, erange=(0,20000), binsize=5):
    basename = mass.output_basename_from_ljh_fname(ds.filename)+"_chan%d"%ds.channum
    for laser_choice in ["laser", "not_laser","pumped", "unpumped"]:
        pulse_timing.choose_laser_dataset(ds, laser_choice)
        write_histogram_dataset(ds, basename+laser_choice, erange, binsize)


def write_channel_histograms(data, erange=(0,20000), binsize=5):
    for ds in data:
        write_histograms_dataset(ds, erange, binsize)

def write_combined_energies_hists(data, erange=(0,20000), binsize=5, chans=None):
    for laser_choice in ["laser", "not_laser","pumped", "unpumped"]:
        pulse_timing.choose_laser(data,laser_choice)
        counts, bin_centers = combined_energies_hist(data, erange, binsize, chans)
        basename = mass.output_basename_from_ljh_fname(data.first_good_dataset.filename)
        fname = basename+"_"+laser_choice+"_combined.spectrum"
        np.savetxt(fname, np.vstack((bin_centers, counts)).T,fmt=("%0.1f", "%i"), header="energy bin centers (eV), counts per bin")

def combined_energies_hist(data, erange=(0,20000), binsize=5, chans=None):
    bin_edges = np.arange(erange[0], erange[1], binsize)
    bin_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
    counts= np.zeros_like(bin_centers, dtype=np.int32)
    if chans==None: chans=[ds.channum for ds in data]
    for channum in chans:
        ds = data.channel[channum]
        c, b = np.histogram(ds.p_energy[ds.cuts.good()], bin_edges)
        counts += c
    return counts, bin_centers

def plot_combined_spectrum(data, erange=(0,20000), binsize=5, ref_lines = [], chans=None, label=""):
    counts, bin_centers = combined_energies_hist(data, erange, binsize, chans)
    plt.figure()
    plt.plot(bin_centers, counts)
    plt.xlabel('energy (eV)')
    plt.ylabel('counts per %.2f eV bin'%(bin_centers[1]-bin_centers[0]))
    nchans = (len(chans) if chans is not None else data.num_good_channels)
    plt.title('coadded %s spectrum %d pixel'%(label.upper(), nchans))
    for line in ref_lines:
        plt.plot(np.array([1, 1])*mass.calibration.energy_calibration.STANDARD_FEATURES[line], plt.ylim())

def plot_combined_spectra(data, erange=(0,20000), binsize=5, ref_lines = [], chans=None):
    for laser_choice in ["laser", "not_laser","pumped", "unpumped"]:
        pulse_timing.choose_laser(data,laser_choice)
        plot_combined_spectrum(data, erange, binsize, ref_lines, chans, laser_choice)

def save_all_plots(data):
    basename = mass.output_basename_from_ljh_fname(data.first_good_dataset.filename)
    dir, fname = path.split(basename)
    print("writing %d plots as png to %s"%(len(plt.get_fignums()), dir))
    for i in plt.get_fignums():
        print("writing plot %d of %d"%(i, len(plt.get_fignums())))
        plt.figure(i)
        plt.savefig(path.join(dir,'figure%d.png') % i, dpi=600)

def fit_edges(data,edge_name , width_ev=400, bin_size_ev=3, fwhm_guess=10.0, doPlot=True):
    fit_params = np.array([fit_edge_in_energy_dataset(ds, edge_name, width_ev, bin_size_ev, fwhm_guess) for ds in data])
    chans = [ds.channum for ds in data]
    if doPlot:
        plt.figure()
        plt.subplot(411)
        plt.plot(chans,fit_params[:,0],'o',label="edge_center")
        plt.ylabel("edge_center (eV)")
        plt.subplot(412)
        pre, post = fit_params[:,1], fit_params[:,2]
        delta_mu=np.log(pre/post)
        plt.plot(chans, delta_mu,'o',label="pre height")
        plt.ylabel("delta abs len")
        plt.subplot(413)
        plt.plot(chans, (pre+post)/2.0,'o')
        plt.ylabel("avg counts")
        plt.subplot(414)
        plt.plot(chans, fit_params[:,3],'o')
        plt.ylabel("energy resolution")
        plt.xlabel("channel number")
    return fit_params

# in development
def calibration_summary(data, calname):
    elements = [ds.calibration[calname].elements for ds in data]
    energy_resolution = [ds.calibration[calname].energy_resolutions for ds in data]
    refined_peak_positions = [ds.calibration[calname].refined_peak_positions for ds in data]

    energies = [mass.energy_calibration.STANDARD_FEATURES[name] for name in elements[0]]

    plt.figure()
    for j in xrange(len(elements)):
        plt.plot(energies, energy_resolution[j],'.')
    plt.xlabel("energy (eV)")
    plt.ylabel("fwhm res from calibration (eV)")
    plt.ylim(0,20)
    plt.grid("on")
