import pylab as plt, numpy as np
import mass
import scipy.signal, scipy.optimize
import exafs


import scipy.signal, scipy.optimize

def periodic_median(timestamp, mod_period=0.001):
    # finds the offset required to make the median 0.5, the backs out what the true median must be to require that offset
    p0=0
    maxj = 3
    for j in xrange(maxj+1):
        phase = (timestamp+p0)%mod_period
        p0 -= (np.median(phase)-0.5*mod_period)+np.random.rand()*(0 if j==maxj else 0.001*mod_period)
        # the random bit is to try to avoid the case where the median is 0.5 due to half the population being
        # approx 0 and half being approx 1,
        # I tested without the random adding 10000 linearly increasing offsets to some actual data
        # and never observed the problem the random is trying to address
    return (0.5-p0)*mod_period

def calc_laser_phase_dataset(ds, forceNew=False, pump_period = 0.002):
    """calculate a number from 0-2 for each pulse
        designed so that numbers near 0.5 are pumped laser x-rays
        numbers near 1.5 are unpumped laser x-rays
        numbers far from either 0.5 or 1.5 are non-laser x-rays
    """
    if "p_laser_phase" in ds.hdf5_group and not forceNew:
        ds.p_laser_phase = ds.hdf5_group["p_laser_phase"]
    else:
        med = periodic_median(ds.time_after_last_external_trigger[:],pump_period/2)
        phase = 2*((ds.time_after_last_external_trigger[:]+med)%pump_period)/pump_period
        ds.hdf5_group["p_laser_phase"] = phase
        ds.p_laser_phase = ds.hdf5_group["p_laser_phase"]
    return ds.p_laser_phase


def calc_laser_phase(data, forceNew=False, pump_period=0.002):
    #try to pick a reasonable dataset to get f0 and the spline from
    for ds in data:
        calc_laser_phase_dataset(ds, forceNew, pump_period)

def calc_laser_cuts_dataset(ds, forceNew=False, keep_size=0.012, exclude_size=0.014):
    if "pumped_bool" in ds.hdf5_group and not forceNew:
        return
    for s in ["pumped_bool", "unpumped_bool", "not_laser_bool"]:
        if s in ds.hdf5_group: del(ds.hdf5_group[s])
    ds.hdf5_group["pumped_bool"] = np.abs(ds.p_laser_phase[:]-0.5)<keep_size
    ds.hdf5_group["unpumped_bool"] = np.abs(ds.p_laser_phase[:]-1.5)<keep_size
    ds.hdf5_group["not_laser_bool"] = np.logical_and(np.abs(ds.p_laser_phase[:]-0.5)>exclude_size, np.abs(ds.p_laser_phase[:]-1.5)>exclude_size)


def choose_laser_dataset(ds, band, keep_size=0.010, exclude_size=0.014,forceNew=False):
    """
    uses the dataset.cuts object to mark bad all pulses not in a specific category related
    to laser timing
    :param data: a microcal TESChannelGroup object
    :param band: options: (1,2,"laser", "not_laser") for( band1, band2, band1 and band2, not_laser pulses)
    :param cut_lines: same as phase_2band_find
    :return: None
    """
    calc_laser_cuts_dataset(ds, forceNew, keep_size, exclude_size) # knows to check hdf5 file first
    band = str(band).lower()
    cutnum = ds.CUT_NAME.index('timing')
    ds.cuts.clearCut(cutnum)

    if band =="pumped":
        tocut = ~(ds.hdf5_group["pumped_bool"][:])
    elif band == "unpumped":
        tocut = ~(ds.hdf5_group["unpumped_bool"][:])
    elif band == "laser":
        tocut = ~np.logical_or(ds.hdf5_group["pumped_bool"][:], ds.hdf5_group["unpumped_bool"][:])
    elif band == "not_laser":
        tocut = ~(ds.hdf5_group["not_laser_bool"][:])
    elif band == "all":
        return
    else:
        raise ValueError("%s is not a valid laser choie"%band)

    ds.cuts.cut(cutnum, tocut)

def choose_laser(data, band, keep_size=0.010, exclude_size=0.014, forceNew=False):
    print("Choosing otherwise good %s pulses via cuts."%band.upper())
    for ds in data:
        choose_laser_dataset(ds, band, keep_size, exclude_size, forceNew)

def plot_phase(ds):
    for i,b in enumerate(["laser", "pumped", "unpumped", "not_laser"]):
        choose_laser_dataset(ds, b)
        counts, bin_edges = np.histogram(ds.p_laser_phase[ds.cuts.good()], np.linspace(0,2,1000))
        bin_centers = bin_edges[1:]-0.5*(bin_edges[1]-bin_edges[0])
        if b == "laser":
            plt.plot(bin_centers, counts, label=b, lw=2.5)
        else:
            plt.plot(bin_centers, counts+i, label=b)
    plt.xlabel("laser phase (0.5 should be pumped, 1.5 unpumped)")
    plt.ylabel("number of good pulses per bin")
    plt.legend()