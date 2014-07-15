import pylab as plt, numpy as np
import mass
import scipy.signal, scipy.optimize
import exafs


import scipy.signal, scipy.optimize

def contiguous_regions(condition, minlen =8):
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


def apply_offsets_for_monotonicity_dataset(offsets, ds, test=False, forceNew=False):
    ds_frame = ds.p_timestamp/ds.timebase
    if not hasattr(ds, "p_timestamp_raw"): ds.p_timestamp_raw = ds.p_timestamp.copy()
    starts, ends = monotonic_frame_ranges(ds_frame, minlen=0)
    if all(ds.p_timestamp_raw==ds.p_timestamp) or forceNew: # only apply corrections once
        print("channel %d applying offsets for monotonicity"%ds.channum)
        if len(starts)>len(offsets):
            ems = ends-starts
            starts = sorted(starts[np.argsort(ems)[-len(offsets):]]) # drop the shortest regions
            ends = sorted(ends[np.argsort(ems)[-len(offsets):]]) # drop the shortest regions
        out = ds.p_timestamp_raw.copy()
        for j in xrange(1,len(starts)):
            out[ends[j-1]+1:ends[j]+1]+=offsets[j]*ds.timebase
        if not test:
            ds.p_timestamp = out
        else:
            plt.figure()
            plt.plot(ds.p_timestamp,'.')
            for j in xrange(1,len(starts)):
                plt.plot([ends[j-1], ends[j]], offsets[j]*ds.timebase*np.array([1,1]))
                plt.plot(out)
                plt.title(str(ends))
            print("offsets", offsets)
            print("starts", starts)
            print("ends", ends)
        return out


def apply_offsets_for_monotonicity(data, test=False, doPlot=True, forceNew=False):
    offsets, crate_epoch, crate_frame = monotonicity(data.first_good_dataset.filename)
    print("applying time offsets to all datasets", offsets)
    for ds in data:
        apply_offsets_for_monotonicity_dataset(offsets, ds, test, forceNew)
    if doPlot:
        plt.figure()
        plt.plot([ds.channum for ds in data], [np.amax(np.diff(ds.p_timestamp)) for ds in data],'.')
        plt.xlabel("channel number")
        plt.ylabel("largest jump in p_timestamp")


def find_f0(timestamp, f0_low, f0_high):
    f0_low, f0_high = np.sort([f0_low, f0_high])
    for j in xrange(5):
        freqs, psd = ls_psd(timestamp, f0_low, f0_high-f0_low)
        fstep = freqs[1]-freqs[0]
        f0_low = f0_low + freqs[np.argmax(psd)]-fstep
        f0_high = f0_low + 2*fstep
        #print(f0_low, f0_high)
    return f0_low+fstep

def ls_psd(timestamp, f0_guess, f0_guess_quality=1):
    timestamp = timestamp.copy()
    timestamp -= timestamp[0]
    timestamp_times_f0 = (timestamp-timestamp[0])*f0_guess
    freqs = np.linspace(f0_guess_quality/100.0, f0_guess_quality,50)
    psd = lombscarg(timestamp, timestamp_times_f0, freqs)
    return freqs, psd

def lombscarg(timestamp, timestamp_times_f0, f_hz):
    TWOPI = np.pi*2
    return scipy.signal.lombscargle(timestamp, np.cos(TWOPI*timestamp_times_f0), np.array(f_hz*TWOPI,ndmin=1))

def periodic_median(timestamp, f0):
    # finds the offset required to make the median 0.5, the backs out what the true median must be to require that offset
    p0=0
    maxj = 3
    for j in xrange(maxj+1):
        phase = (timestamp*f0+p0)%1
        p0 -= (np.median(phase)-0.5)+np.random.rand()*(0 if j==maxj else 0.001)
        # the random bit is to try to avoid the case where the median is 0.5 due to half the population being
        # approx 0 and half being approx 1,
        # I tested without the random adding 10000 linearly increasing offsets to some actual data
        # and never observed the problem the random is trying to address
    return 0.5-p0

def sampled_phase(timestamp, f0, sample_time_s = 60):
    t_sample = np.arange(timestamp[0], timestamp[-1], sample_time_s)
    i_sample = np.searchsorted(timestamp, t_sample)
    p = np.zeros(len(t_sample)-1, dtype=np.float64)
    for j in xrange(len(i_sample)-1):
        p[j] = periodic_median(timestamp[i_sample[j]:i_sample[j+1]],f0)
    return t_sample[:-1]+0.5*sample_time_s, p

def splined_phase(timestamp, f0, sample_time_s=60):
    t_sample, phase = sampled_phase(timestamp, f0, sample_time_s)
    phase = np.unwrap(phase*2*np.pi)/(2*np.pi)
    spline = mass.mathstat.CubicSpline(t_sample, phase)
    return spline

def calc_phase(timestamp,f0=None,flatten=True,num_bands=2,f_guess_range=(1000,1001),sample_time_s=60):
    """
    Taken an array of timestamps that contain a large but not 1 fraction of events
    happening at fixed period according to a clock with or without drift relative to the
    clock the timestamps are measured by.
    :param timestamp: numpy array of timestamps
    :param f0: optional frequency of events, its probably better to let phase calculate f0
    :param flatten: optional if True phase attempts to correct for the relative drift of the two clocks and
    centers one band on 0.5
    :param num_bands: int how many bands of events there are in the output, in case eg events are alternatley
    pumped and unpumped
    :param f_guess_range: used if f0 is None, must contain the actual frequency of events
    :param sample_time_s: when flattening, how frequently the drift between the two clocks is sampled.
    must be large enough to contain many timestamps in each division
    :return: phase, f0, spline
    :phase: numpy array of phases in units (0 to num_bands)
    :spline:
    """
    if f0 is None: f0 = find_f0(timestamp, f_guess_range[0],f_guess_range[1])
    if flatten:
        spline = splined_phase(timestamp, f0, sample_time_s)
        spline.phase = lambda timestamp: (0.5+(timestamp*f0-spline(timestamp)))%num_bands
        return spline.phase(timestamp),f0, spline
    else:
        return (timestamp*f0)%num_bands,f0, None

def phase_2band_find(phase, cut_lines=[.03,0.02]):
    bin_e = np.arange(0,1.01,0.01)
    counts, bin_e =np.histogram(phase%1, bin_e)
    bin_c = bin_e[1:]-(bin_e[1]-bin_e[0])*0.5
    maxbin = bin_c[np.argmax(counts)] # find the maximum bin
    a,b = maxbin-cut_lines[0], maxbin+cut_lines[1] # make cuts on either side of it
    band1 = np.logical_and(a<phase, phase<b)
    band2 = np.logical_and((a+1)<phase, phase<(b+1))
    bandNone = np.logical_not(np.logical_or(band1, band2))
    return band1, band2, bandNone

def periodogram2(timestamp, cut_lines = [0.47,0.515]):
    num_bands = 2
    phase,f0, spline = calc_phase(timestamp)
    band1, band2, bandNone = phase_2band_find(phase, cut_lines)
    plt.figure()
    plt.plot(timestamp, phase,'.')
    plt.plot(timestamp[band1], phase[band1],'.')
    plt.plot(timestamp[band2], phase[band2],'.')
    plt.plot(timestamp[bandNone], phase[bandNone],'.')
    for j in xrange(4):
        plt.plot([timestamp[0], timestamp[-1]], np.array([1,1])*(cut_lines[j%2]+(1 if j>1 else 0))%num_bands,'k')
    plt.xlabel("time (s)")
    plt.ylabel("flattened phase/2*pi")
    plt.title("f0=%f"%f0)

def periodogram(timestamp, cut_lines=[.03,0.02], flatten=True, split=True):
    num_bands = (2 if split else 1)
    phase,f0,spline = calc_phase(timestamp, flatten=flatten, num_bands = num_bands)
    plt.figure()
    plt.plot(timestamp, phase,'.')
    if flatten:
        for j in xrange(4):
            plt.plot([timestamp[0], timestamp[-1]], np.array([1,1])*(cut_lines[j%2]+(1 if j>1 else 0))%num_bands,'k')
    plt.xlabel("time (s)")
    plt.ylabel("%sphase/2*pi"%("flattened " if split else ""))
    plt.title("f0=%f"%f0)

def dataset_periodogram2(ds):
    periodogram2(ds.p_timestamp)

def extrap(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = np.interp(x, xp, yp)
    y[x < xp[0]] = yp[0] + (x[x<xp[0]]-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
    y[x > xp[-1]]= yp[-1] + (x[x>xp[-1]]-xp[-1])*(yp[-1]-yp[-2])/(xp[-1]-xp[-2])
    return y

def downsampled(x, samples_per_newsample):
    resampled_len = int(np.floor(len(x)/samples_per_newsample))
    reshaped_x = np.reshape(x[:resampled_len*samples_per_newsample], (resampled_len, samples_per_newsample))
    return np.mean(reshaped_x, 1)


def calc_laser_phase(data, forceNew=False):
    #try to pick a reasonable dataset to get f0 and the spline from
    for ds in data:
        print("looking at chan %d as potential source of phase spline"%ds.channum)
        try:
            phase,f0, spline = calc_phase(ds.p_timestamp) # for the purposes of finding f0
        except:
            print("chan %d rejected for failing to calculate phase"%ds.channum)
            continue
        band1, band2, bandNone = phase_2band_find(phase)
        if len(band1)/float(ds.nPulses) > 0.2 and len(band2)/float(ds.nPulses)>0.2 and len(bandNone)/float(ds.nPulses) >0.01:
            break
        else:
            print("chan %d rejected for not having a reasonable distribution of pulses in bands"%ds.channum)
    print("using spline from %d with %d pulses"%(ds.channum, ds.nPulses))
    for ds in data:
        if not hasattr(ds, "p_laser_phase") or forceNew:
            print("chan %d calculating laser phase"%ds.channum)
            ds.p_laser_phase = spline.phase(ds.p_timestamp)
        else:
            print("chan %d skipping calculate laser phase, already done"%ds.channum)

def choose_laser_dataset(ds, band, cut_lines=[.03,0.02]):
    """
    uses the dataset.cuts object to mark bad all pulses not in a specific category related
    to laser timing
    :param data: a microcal TESChannelGroup object
    :param band: options: (1,2,"laser", "not_laser") for( band1, band2, band1 and band2, not_laser pulses)
    :param cut_lines: same as phase_2band_find
    :return: None
    """
    band = str(band).lower()
    cutnum = ds.CUT_NAME.index('timing')
    band1, band2, bandNone = phase_2band_find(ds.p_laser_phase,cut_lines=cut_lines)
    ds.cuts.clearCut(cutnum)
    if band == "pumped":
        if not hasattr(ds, "pumped_band_knowledge"): raise ValueError("unknown which band is pumped, try calling label_pump_band_for_alternating_pump")
        band=str(ds.pumped_band_knowledge)
    if band == 'unpumped':
        if not hasattr(ds, "pumped_band_knowledge"): raise ValueError("unknown which band is pumped, try calling label_pump_band_for_alternating_pump")
        if ds.pumped_band_knowledge==1:
            band='2'
        else:
            band = '1'
    if band == '1':
        ds.cuts.cut(cutnum, np.logical_not(band1))
    elif band == '2':
        ds.cuts.cut(cutnum, np.logical_not(band2))
    elif band == 'not_laser':
        ds.cuts.cut(cutnum, np.logical_not(bandNone))
    elif band == "laser":
        ds.cuts.cut(cutnum, bandNone)
    else:
        raise ValueError("%s is not a valid choice for choose_laser_dataset"%band)

def choose_laser(data, band, cut_lines=[.03,0.02]):
    print("Choosing otherwise good %s pulses via cuts."%band.upper())
    for ds in data:
        choose_laser_dataset(ds, band, cut_lines)

def mic_triggers_as_timestamps(ds):
    offsets, crate_epoch_usec, crate_frame = monotonicity(ds.filename)


    mic_epoch_usec = mass.load_mic_file(ds.filename)
    measured_mic_latency = -0.18557235449876053
    mic_frame_adjustment = -measured_mic_latency/ds.timebase
    mic_frame = extrap(mic_epoch_usec, crate_epoch_usec, crate_frame)+mic_frame_adjustment
    return mic_frame*ds.timebase

def label_pumped_band_for_alternating_pump_datsaset(ds, pump_freq_hz=500, doPlot=True):
    mic_timestamps = mic_triggers_as_timestamps(ds)
    band1, band2, bandNone = phase_2band_find(ds.p_laser_phase)
    band1_timestamps = ds.p_timestamp[band1]
    band2_timestamps = ds.p_timestamp[band2]
    #cut out mic_timestamps that come before or after ds timestamps
    mic_timestamps = mic_timestamps[np.logical_and(mic_timestamps>ds.p_timestamp[0], mic_timestamps<ds.p_timestamp[-2])]
    mic_index_band1 = np.searchsorted(band1_timestamps, mic_timestamps)
    mic_index_band2 = np.searchsorted(band2_timestamps, mic_timestamps)
    band1_med_diff = np.abs(0.5-periodic_median((band1_timestamps[mic_index_band1]-mic_timestamps), pump_freq_hz))
    band2_med_diff = np.abs(0.5-periodic_median((band2_timestamps[mic_index_band2]-mic_timestamps), pump_freq_hz))
    diff_diff = np.abs(band1_med_diff-band2_med_diff)

    if band1_med_diff<band2_med_diff:
    # xrays occuring simultaneous to a microphone trigger should have phase 1 or 0
    # xrays occuring on a laser pulse not simultaneous to a microphone trigger should have phase 0.5
    # for alternating pumped - unpumped
        pumped_band = 2
    else:
        pumped_band = 1
    ds.pumped_band_knowledge = pumped_band
    if doPlot:
        a,b="pumped", "unpumped"
        if pumped_band==2: a,b=b,a
        plt.figure()
        #plt.plot(mic_timestamps,'.')
        plt.plot(band1_timestamps[mic_index_band1],(pump_freq_hz*(band1_timestamps[mic_index_band1]-mic_timestamps))%1,'.',label="band1 %s"%a)
        plt.plot(band2_timestamps[mic_index_band2],(pump_freq_hz*(band2_timestamps[mic_index_band2]-mic_timestamps))%1,'.',label="band2 %s"%b)
        plt.xlabel("frame time (s)")
        plt.ylabel("x-ray phase difference from nearest microphone timestamps")
        plt.legend()
    if diff_diff < 0.4:
        raise ValueError("ambiguous which band is pumped")
    return pumped_band

def label_pumped_band_for_alternating_pump(data, pump_freq_hz=500, doPlot=True, forceNew=False):
    pre_knowledge = [ds.pumped_band_knowledge for ds in data if ds.pumped_band_knowledge is not None]
    pumped_band = None
    if len(pre_knowledge)>0:
        if all([pre_knowledge[i] == pre_knowledge[0] for i in xrange(len(pre_knowledge))]):
            pumped_band = pre_knowledge[0]
    if pumped_band is None or forceNew:
        for ds in data:
            try:
                pumped_band = label_pumped_band_for_alternating_pump_datsaset(ds, pump_freq_hz, doPlot)
                break
            except:
                pass
        if pumped_band is None: raise ValueError("failed to assign pumped_band with any dataset")
    else:
        print("skipping labeling of pumped band, because the band is already labeled")
    for ds in data:
        ds.pumped_band_knowledge=pumped_band