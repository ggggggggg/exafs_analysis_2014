import pylab as plt, numpy as np
import mass
import scipy.signal, scipy.optimize


import scipy.signal, scipy.optimize

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
        spline.f0 = f0
        spline.phase = lambda timestamp: (0.5+(timestamp*f0-spline(timestamp)))%num_bands
        return spline.phase(timestamp), spline
    else:
        return (timestamp*f0)%num_bands, f0, None

def phase_2band_find(phase, cut_lines=[0.47,0.515]):
    a,b = np.amin(cut_lines), np.amax(cut_lines)
    band1 = np.logical_and(a<phase, phase<b)
    band2 = np.logical_and((a+1)<phase, phase<(b+1))
    bandNone = np.logical_not(np.logical_or(band1, band2))
    return band1, band2, bandNone

def periodogram2(timestamp, cut_lines = [0.47,0.515]):
    num_bands = 2
    phase, spline = calc_phase(timestamp)
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
    plt.title("f0=%f"%spline.f0)

def periodogram(timestamp, cut_lines = [0.47,0.515], flatten=True, split=True):
    num_bands = (2 if split else 1)
    phase, spline = calc_phase(timestamp, flatten=flatten, num_bands = num_bands)
    plt.figure()
    plt.plot(timestamp, phase,'.')
    if flatten:
        for j in xrange(4):
            plt.plot([timestamp[0], timestamp[-1]], np.array([1,1])*(cut_lines[j%2]+(1 if j>1 else 0))%num_bands,'k')
    plt.xlabel("time (s)")
    plt.ylabel("%sphase/2*pi"%("flattened " if split else ""))
    plt.title("f0=%f"%spline.f0)

def dataset_periodogram2(ds):
    periodogram2(ds.p_timestamp)