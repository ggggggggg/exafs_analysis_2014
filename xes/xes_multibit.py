import numpy as np
import pylab as plt
import mass
from os import path
import os
import exafs
import pulse_timing
import shutil
import traceback, sys
from matplotlib.backends.backend_pdf import PdfPages
import datetime


dir_base = "/Volumes/Drobo/exafs_data/"
dir_p = "20150902_iron_tris_xes_delay_m24p5/"
# dir_p = "20150808_50mM_irontris_xes_m20mm_55mm/"
# New dir_p here

dir_n = "20150902_iron_tris_xes_delay_m24p5_noise"
# New dir_n here
run_str = dir_p[:8]

available_chans = mass.ljh_get_channels_both(path.join(dir_base, dir_p), path.join(dir_base, dir_n))

if len(available_chans) == 0:
    raise ValueError("no channels have both noise and pulse data")
chan_nums = available_chans[:]
pulse_files = mass.ljh_chan_names(path.join(dir_base, dir_p), chan_nums)
noise_files = mass.ljh_chan_names(path.join(dir_base, dir_n), chan_nums)

data = mass.TESGroup(pulse_files, noise_files)

if "__file__" in locals():
    exafs.copy_file_to_mass_output(__file__, data.datasets[0].filename) #copy this script to mass_output

# analyze data
data.summarize_data(peak_time_microsec=300.0, forceNew=False)

data.apply_cuts(exafs.basic_cuts, forceNew=False) # forceNew is True by default for apply_cuts, unlike most else
data.avg_pulses_auto_masks()  # creates masks and compute average pulses
data.plot_average_pulses(-1)

data.compute_filters(f_3db=10000.0, forceNew=False)
data.filter_data(forceNew=False)
data.calc_external_trigger_timing(from_nearest=True,forceNew=True)
ds = data.first_good_dataset


data.register_categorical_cut_field("pump",["pumped","unpumped"])
data.register_categorical_cut_field("laser",["laser","not_laser"])

def calc_laser_cuts_dataset(ds, unpumped_location, include_size, exclude_size):
    pumped_bool = np.abs(ds.rows_from_nearest_external_trigger[:])<include_size
    unpumped_bool = np.abs(ds.rows_from_nearest_external_trigger[:]-unpumped_location)<include_size
    not_laser_bool = np.logical_and(np.abs(ds.rows_from_nearest_external_trigger[:])>exclude_size, np.abs(ds.rows_from_nearest_external_trigger[:]-unpumped_location)>exclude_size)
    ds.cuts.cut_categorical("pump",{"pumped":pumped_bool,
                                    "unpumped":unpumped_bool})
    ds.cuts.cut_categorical("laser",{"not_laser":not_laser_bool,
                                     "laser":pumped_bool|unpumped_bool})
def calc_laser_cuts(data, unpumped_location, include_size=50, exclude_size=60):
    for ds in data:
        calc_laser_cuts_dataset(ds, unpumped_location, include_size, exclude_size)
laser_period = np.median(np.diff(ds.external_trigger_rowcount[:1000]))/2.0
unpumped_location = np.median(ds.rows_from_nearest_external_trigger[np.abs(ds.rows_from_nearest_external_trigger[:]-laser_period)<100])
calc_laser_cuts(data, unpumped_location=unpumped_location)


# use laser status to set calibration choice
# calibration_labels = np.zeros(ds.nPulses)
# calibration_names = data.cut_field_categories("calibration")
# calibration_labels[ds.good(laser="not_laser")]=calibration_names["in"]
# calibration_labels[ds.good(laser="laser")]=calibration_names["out"]
# ds.cuts.cut("calibration",calibration_labels)
# {"calibration":"in"} is used by default for drift_correct, calibrate, phase_correct_2014
# you can pass a different choice using the keyword category, eg category = {"laser":"not_laser"}

# data.drift_correct(forceNew=False,category={"laser":"laser"})
# data.calibrate('p_filt_value_dc', ['FeKAlpha', 'FeKBeta'], size_related_to_energy_resolution=20.0,
#                excl=[],forceNew=False, plot_on_fail=False,category={"laser":"laser"})
data.drift_correct(forceNew=False,category={"laser":"laser"})
data.calibrate('p_filt_value_dc', ['FeKAlpha', 'FeKBeta'], size_related_to_energy_resolution=20.0,
               excl=[],forceNew=False, plot_on_fail=False,category={"laser":"laser"})


##### CUT ON DELAY STAGE
crate_epoch_usec, crate_frame = mass.load_aux_file(ds.filename)
avg_crate_timebase = 1.0 * (crate_epoch_usec[-1] - crate_epoch_usec[0]) / (crate_frame[-1] - crate_frame[0])
crate_epoch_usec_dev = crate_epoch_usec - crate_epoch_usec[0] - avg_crate_timebase * (crate_frame - crate_frame[0])
delay_filename = path.join(dir_base, dir_p,"delay_stage_scan.log")
if os.path.isfile(delay_filename):
    move_posix_ts, move_desc = [],[]
    with open(delay_filename) as f:
        for line in f:
            text_date, posix_ts, movement = line.split("\t")
            move_posix_ts.append(float(posix_ts))
            move_desc.append(movement.strip())

    move_rowcount = np.array(np.interp(move_posix_ts, crate_epoch_usec*1e-6, crate_frame)*ds.number_of_rows,dtype=np.int64)

    categories = {"uncategorized":0}
    move_category = []
    for md in move_desc:
        if md.startswith("end"):
            category = "%0.1f"%np.round(float(md.split()[-1][:-1]),2)
            if category not in categories:
                categories[category]=len(categories)+1
        elif md.startswith("start"):
            category = "uncategorized"
        move_category.append(category)
    data.register_categorical_cut_field("delay",[k for k,v in categories.items()])
    categories = data.cut_field_categories("delay")

    for ds in data:
        inds = np.searchsorted(ds.p_rowcount, move_rowcount)
        labels = np.zeros(ds.nPulses)
        for i in range(len(inds)-1):
            labels[inds[i]:inds[i+1]]=categories[move_category[i]]
        ds.cuts.cut("delay", labels)
else: # if the delay_stage_scan file doesn't exist, label every pulse "uncategorized"
    categories ={ "uncategorized":0}
    data.register_categorical_cut_field("delay",[k for k,v in categories.items()])

non_empty_delays = {k for k in data.cut_field_categories("delay") if ds.good(delay=k).sum()>0}

spectra = {}
for k in non_empty_delays:
    delay_str = "delay_"+k
    delay_str.replace(".","p")

    ########################################################
    #
    #  Ka, Kb1,3, kb2,5 emission spectra (individual)
    #
    ########################################################
    run_title = "%s 50mM irontris, delay stage %s mm"%(run_str,k)

    emin, emax = 2000, 15000

    N = int((emax - emin) / 0.5)
    bin_edges = np.linspace(2000, 15000, N + 1)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0

    pumped_hist = np.zeros(N,dtype=np.int64)
    unpumped_hist = np.zeros(N,dtype=np.int64)

    for ds in data:
        hist, bins = np.histogram(ds.p_energy[ds.good(delay=k,pump="pumped")], bins=bin_edges)
        pumped_hist += hist
        hist, bins = np.histogram(ds.p_energy[ds.good(delay=k,pump="unpumped")], bins=bin_edges)
        unpumped_hist += hist

    spectra[k] = {"bin_centers":bin_centers, "pumped":pumped_hist, "unpumped":unpumped_hist}

    signals = (pumped_hist - unpumped_hist) / np.sqrt(pumped_hist + unpumped_hist)

    import brewer2mpl
    bmap = brewer2mpl.get_map("Spectral", "Diverging", 8)

    for emin, emax, label in zip([6370, 7020, 7070], [6430, 7090, 7130],
                                 [r"$\mathrm{K}\alpha$",
                                  r"$\mathrm{K}\beta_{1,3}$",
                                  r"$\mathrm{K}\beta_{2,5}$"]):
        emin_index = (emin - 2000) * 2
        emax_index = (emax - 2000) * 2
        ylims1 = (max(0, 0.9 * np.min(unpumped_hist[emin_index:emax_index]) -
                  0.1 * np.max(unpumped_hist[emin_index:emax_index])),
                 np.max(unpumped_hist[emin_index:emax_index]) * 1.1)
        ylims2 = np.array([-1, 1]) * 1.1 * (np.max(np.abs(signals[emin_index:emax_index])))

        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(211)


        ax.step(bin_centers, pumped_hist, where='mid', color=bmap.mpl_colors[-1], label="pumped",
                alpha=0.8)
        ax.step(bin_centers, unpumped_hist, where='mid', color=bmap.mpl_colors[0], label="unpumped",
                alpha=0.8)

        leg = ax.legend(numpoints=1, frameon=False, handlelength=1.0)
        for l in leg.get_lines():
            l.set_linewidth(2)

        ax.text(0.05, 0.95, label, size=16, ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"Count (per 0.5 eV bin)")
        ax.set_title(run_title)

        ax.set_xlim(emin, emax)
        ax.set_ylim(*ylims1)

        ax = fig.add_subplot(212)

        ax.step((bins[1:] + bins[:-1]) / 2,
                signals,
                where='mid', color='g',
                alpha=1)

        ax.text(0.05, 0.95, label, size=16, ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel(r"$\frac{\mathrm{p - u}}{\sqrt{\mathrm{p + u}}}$")
        ax.set_title(run_title)

        ax.set_xlim(emin, emax)
        ax.set_ylim(*ylims2)

        fig.show()

        fig.savefig("%s_%s_%g.png"%(run_str,delay_str,emin),dpi=600)

    ###########################################################
    #
    # Write to files.
    # Make sure that you changed filenames!
    #
    ###########################################################

    with open("%s_%s_pumped_hist_0.5ev_bin.dat"%(run_str,delay_str), 'w') as f:
        for e, c in zip((bins[1:] + bins[:-1]) / 2, pumped_hist):
            f.write("{0:.2f},{1:d}\n".format(e, int(c)))

    with open("%s_%s_unpumped_hist_0.5ev_bin.dat"%(run_str,delay_str), 'w') as f:
        for e, c in zip((bins[1:] + bins[:-1]) / 2, unpumped_hist):
            f.write("{0:.2f},{1:d}\n".format(e, int(c)))


################
#  fit data
################

#load reference spectra
singlet = np.loadtxt("singlet.csv",delimiter=",",skiprows=6,usecols=(0,1))
quintet = np.loadtxt("quintet.csv",delimiter=",",skiprows=6,usecols=(0,1))

# indicies for fit range
lo, hi = 10072, 10148


def detector_response(binsize_ev, shift_ev, fwhm_ev, tail_size_ev, tail_fraction):
    sigma = fwhm_ev/2.355
    n = np.round(50/binsize_ev)
    x = np.arange(-n*binsize_ev-shift_ev,n*binsize_ev-shift_ev, binsize_ev)
    y = (1-tail_fraction)/sigma/np.sqrt(2*np.pi)*np.exp( -(x/sigma)**2/2 )
    y += tail_fraction*np.exp(x/tail_size_ev)*np.less_equal(x,0)/tail_size_ev
    return x,y

def fitfunc(x, ground, excited, shift_ev, f, amplitude, fwhm_ev, tail_size_ev, tail_fraction, bg, bg_slope):
    x_dr,dr = detector_response(x[1]-x[0], shift_ev, fwhm_ev, tail_size_ev, tail_fraction)
    y_ideal = f*excited+(1-f)*ground
    y = amplitude*np.convolve(dr, y_ideal, mode="full")
    indstart = len(dr)/2
    y=y[indstart:indstart+len(x)]
    y+=bg+bg_slope*(x-x[len(x)/2])
    return y
def make_theory_func(ground, excited):
    return lambda params,x: fitfunc(x, ground, excited, *params)
params_guess = [0.1, 0.5, 1800, 5, 24, 0.14, 40, 0]
params_names = ["shift_ev", "f", "amplitude", "fwhm", "tail_size_ev", "tail_fraction", "bg", "bg_slope"]

for k in spectra:
    bin_centers, pumped_hist, unpumped_hist = spectra[k]["bin_centers"][lo:hi], spectra[k]["pumped"][lo:hi], spectra[k]["unpumped"][lo:hi]

    singlet_intensity = np.interp(bin_centers, singlet[:,0], singlet[:,1])
    quintet_intensity = np.interp(bin_centers, quintet[:,0], quintet[:,1])
    theory_func = make_theory_func(singlet_intensity, quintet_intensity)

    # first fit the unpumped spectra holiding f=0
    fitter = mass.MaximumLikelihoodHistogramFitter(bin_centers, unpumped_hist, params_guess, theory_func)
    fitter.hold(params_names.index("tail_fraction"),0.0)
    fitter.hold(params_names.index("tail_size_ev"),24)
    fitter.hold(params_names.index("f"),0)
    fitter.setbounds(params_names.index("f"), 0, 1)
    fitter.setbounds(params_names.index("tail_fraction"), 0, .2)
    fitter.setbounds(params_names.index("tail_size_ev"), 10, 30)
    fitter.setbounds(params_names.index("fwhm"), 2, 8)
    fitter.setbounds(params_names.index("bg"), 0, 1e6)

    params,covar = fitter.fit()
    # done fitting unpumped spectra

    # now fit the pumped spectra, holding all params equal to fit from unpumped, except f
    params_guessp = params.copy()
    fitterp = mass.MaximumLikelihoodHistogramFitter(bin_centers, pumped_hist, params_guess, theory_func)
    for i, param in enumerate(params):
        if not params_names[i] == "f":
            fitterp.hold(i, param)
    paramsp, covarp = fitterp.fit()
    dfp = np.sqrt(covarp[params_names.index("f"), params_names.index("f")])
    # done fitting pumped spectra

    # make plot
    plt.figure()
    plt.plot(bin_centers, unpumped_hist,label="unpumped",drawstyle="steps-mid")
    plt.plot(bin_centers, theory_func(params,bin_centers),label="f=%0.2f"%params[1],lw=2)
    plt.plot(bin_centers, pumped_hist,label="pumped",drawstyle="steps-mid")
    plt.plot(bin_centers, theory_func(paramsp, bin_centers),label="f=%0.2f +/- %0.3f"%(paramsp[1], dfp),lw=2)
    plt.legend()
    plt.title("irontris %s, delay = %s"%(run_str, k))
    plt.savefig("irontris %s, delay = %s.png"%(run_str, k), dpi=100)

