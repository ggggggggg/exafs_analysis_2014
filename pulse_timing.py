import pylab as plt, numpy as np
import mass
import scipy.signal, scipy.optimize
import exafs
import h5py



import scipy.signal, scipy.optimize

def downsampled(x, samples_per_newsample):
    resampled_len = int(np.floor(len(x)/samples_per_newsample))
    reshaped_x = np.reshape(x[:resampled_len*samples_per_newsample], (resampled_len, samples_per_newsample))
    return np.mean(reshaped_x, 1)

def contiguous_regions(condition, minlen=8):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""
    d = np.diff(condition)
    idx, = d.nonzero()
    idx += 1
    if condition[0]:
        idx = np.r_[0, idx]
    if condition[-1]:
        idx = np.r_[idx, condition.size] # Edit
    idx.shape = (-1,2)
    starts, ends = idx[:,0], idx[:,1]
    for j in range(len(starts))[::-1]:
        if ends[j]-starts[j]<minlen:
            ends.pop(j)
            starts.pop(j)
    return starts, ends

def monotonic_frame_ranges(frames, minlen=8):
    starts, ends = contiguous_regions(np.diff(frames)>0,minlen)
    return starts, ends


def monotonicity(ljh_fname):
    crate_epoch_usec, crate_frame = mass.load_aux_file(ljh_fname)
    starts, ends = monotonic_frame_ranges(np.array(crate_frame, dtype=np.int), minlen=8)

    # the ratio of diff(crate_epoch_usec) to diff(crate_frame) appears to follow a pattern  with period 4
    # one sample with a much higher than average ratio, two with typical ratios, one with much lower ratio
    # so I would like to resample both of them such that each sample is now the average of 4 (or a multiple of 4)
    # other samples, maybe roughy 1 second is good
    period_entries = 4 # psuedo-period in plot of diff(crate_epoch_usec), it was 4 when I looked, but it may not always be 4
    resampling_period_s = 1
    samples_per_newsample = int(period_entries*np.ceil(1e6*resampling_period_s/(period_entries*np.mean(np.diff(crate_epoch_usec)))))
    keep = ends-starts > samples_per_newsample
    starts = starts[keep]
    ends=ends[keep]
    resampled_crate_epoch = []
    resampled_crate_frame = []

    for j in range(len(starts)):
        resampled_crate_epoch.append(downsampled(crate_epoch_usec[starts[j]:ends[j]], samples_per_newsample))
        resampled_crate_frame.append(downsampled(crate_frame[starts[j]:ends[j]], samples_per_newsample))
    start_frames = [resampled_crate_frame[0][0]]
    offsets = []
    for j in range(len(starts)):
        offsets.append(-resampled_crate_frame[j][0]+start_frames[j])
        if j != len(starts)-1:
            first_epoch_in_next = resampled_crate_epoch[j+1][0]
            start_frames.append(extrap(np.array([first_epoch_in_next]), resampled_crate_epoch[j], resampled_crate_frame[j]+offsets[j]))

    new_frame = [r+offsets[i] for i,r in enumerate(resampled_crate_frame)]

    return offsets, np.hstack(resampled_crate_epoch), np.hstack(new_frame)


def apply_offsets_for_monotonicity_array(offsets, time_wo_offsets, test=False, forceNew=False):
    starts, ends = monotonic_frame_ranges(time_wo_offsets, minlen=0)
    if len(starts)>len(offsets):
        ems = ends-starts
        starts = sorted(starts[np.argsort(ems)[-len(offsets):]]) # drop the shortest regions
        ends = sorted(ends[np.argsort(ems)[-len(offsets):]]) # drop the shortest regions
    out = time_wo_offsets[:]
    for j in xrange(1,len(starts)):
        out[ends[j-1]+1:ends[j]+1]+=offsets[j]
    return out


def apply_offsets_for_monotonicity_dataset(offsets, ds, test=False, forceNew=False):
    if not "p_timestamp_raw" in ds.hdf5_group or forceNew: # only apply corrections once
        if not "p_timestamp_raw" in ds.hdf5_group: ds.p_timestamp_raw = ds.hdf5_group.create_dataset("p_timestamp_raw", data=ds.p_timestamp)
        time_w_offsets = apply_offsets_for_monotonicity_array(offsets, ds.p_timestamp_raw[:])
        ds.p_timestamp[:] = time_w_offsets
    else:
        ds.p_timestamp_raw = ds.hdf5_group["p_timestamp_raw"]

def apply_offset_for_monotonicity_external_trigger(offsets, ds):
    filename = mass.ljh_util.ljh_get_extern_trig_fname(ds.filename)
    h5 = h5py.File(filename)
    if "trig_times_w_offsets" in h5:
        del(h5["trig_times_w_offsets"])
    trig_times_wo_offsets = np.array(h5["trig_times"], np.int64) #convert to int64 so subtraction will give negative numbers
    h5["trig_times_w_offsets"] =apply_offsets_for_monotonicity_array(offsets, trig_times_wo_offsets)
    h5.close()


def apply_offsets_for_monotonicity(data, test=False, doPlot=True, forceNew=False):
    offsets, crate_epoch, crate_frame = monotonicity(data.first_good_dataset.filename) # offets has units of frame count
    number_of_rows = data.first_good_dataset.number_of_rows
    timebase = data.first_good_dataset.timebase
    offsets_rowcount = np.array([np.int64(o*number_of_rows) for o in offsets])
    offsets_seconds = (offsets_rowcount/number_of_rows-0.3)*timebase
    # I don't understand why the constant offset is neccesary, or why it has that value, and I don't know if it will work with other datasets
    print("applying time offsets to all datasets", offsets)
    # apply_offsets_for_monotonicity_dataset(offsets_seconds, ds, test, forceNew)
    for ds in data:
        try:
            apply_offsets_for_monotonicity_dataset(offsets_seconds, ds, test, forceNew)
        except:
            data.set_chan_bad(ds.channum, "apply offsets for monotonicity")
    apply_offset_for_monotonicity_external_trigger(offsets_rowcount, data.first_good_dataset)
    if doPlot:
        plt.figure()
        plt.plot([ds.channum for ds in data], [np.amax(np.diff(ds.p_timestamp[:])) for ds in data],'.')
        plt.xlabel("channel number")
        plt.ylabel("largest jump in p_timestamp")

def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y[x < xp[0]] = yp[0] + (x[x<xp[0]]-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
    y[x > xp[-1]]= yp[-1] + (x[x>xp[-1]]-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2])
    return y



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
        if "p_laser_phase" in ds.hdf5_group: del(ds.hdf5_group["p_laser_phase"])
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
    plt.title("channel %g"%ds.channum)