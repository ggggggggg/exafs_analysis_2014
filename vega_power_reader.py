import numpy as np
import pylab as plt

filename = "/Users/oneilg/vega_power_20150710.dat"

def load_vega_power(filename):
	f = open(filename)

	vega_power = []
	vega_epochtime = []
	i = 0

	for line in f:
		try:
			a,b=line.split("\t")
			a = float(a)
			b = float(b)
			vega_epochtime.append(a)
			vega_power.append(b)
		except:
			pass


	return np.array(vega_epochtime),np.array(vega_power)

def find_badranges(vega_timestamp, vega_power, threshold_power_hi=0.3, threshold_power_lo=0.1):
	building_bad_region = False
	lo_extra = 10
	hi_extra = 0
	lo,hi=0,0
	badranges = []
	for i in xrange(len(vega_timestamp)):
		t,p = vega_timestamp[i], vega_power[i]
		power_in_range = threshold_power_lo<p<threshold_power_hi
		power_out_of_range = not power_in_range
		if power_out_of_range:
			if not building_bad_region:
				lo = t-lo_extra
				building_bad_region = True
			hi=max(lo,t+hi_extra)
		elif t>hi and building_bad_region:
				badranges.append((lo,hi))
				building_bad_region=False

	if building_bad_region:
		badranges.append((lo,hi))

	return join_overlapping_ranges(badranges)

def join_overlapping_ranges(ranges):
	outranges = []
	oldlo, oldhi = -1,-1
	building_lo, building_hi = -1,-1
	building_range = False
	for lo,hi in ranges:
		if building_range:
			if lo > building_hi:
				outranges.append((building_lo, building_hi))
				building_range = False
			else: 
				building_hi = hi
		if not building_range:
			building_lo = lo
			building_hi = hi
			building_range = True
	if building_range:
		outranges.append((building_lo, building_hi))
	return outranges



def plot_badranges(badranges, ypos,linespec = "r-"):
	for (lo,hi) in badranges:
		plt.plot([lo,hi],np.array([ypos,ypos])+np.random.rand()*0.0001,linespec,lw=4)

vega_epochtime, vega_power = load_vega_power(filename)
badranges = find_badranges(vega_epochtime, vega_power, 0.3)
badranges_joined = join_overlapping_ranges(badranges)

plt.figure()
plt.plot(vega_epochtime, vega_power)
plt.ylabel("vega_power (mW)")
plt.xlabel("epoch time")
plot_badranges(badranges, 0.3)


july10noon = 1436554800.0
july11noon = july10noon + 24*3600

badtime = np.sum([hi-lo for (lo,hi) in badranges if lo>july10noon and hi<july11noon])

print(1-(badtime/(july11noon-july10noon)))

tlo = july10noon
tduration = 3600
hour = []
percent_bad = []
for i in xrange(int((vega_epochtime[-1]-july10noon)/tduration)):
	tlo = july10noon+tduration*i
	thi = tlo+tduration
	badtime = 0
	for lo,hi in badranges:
		if lo<tlo and hi>thi:
			badtime = tduration
			break
		if thi>lo>tlo or thi>hi<tlo:
			badtime+=min(hi,thi)-max(lo,tlo)
	percent_bad.append(100*badtime/float(tduration))

x = np.arange(round(vega_epochtime[0]), vega_epochtime[-1],1)
y = np.zeros(len(x),dtype="int")
for lo,hi in badranges:
	indlo = np.searchsorted(x,lo)
	indhi = np.searchsorted(x,hi)
	y[indlo:indhi] = 1

import scipy.signal
gauss_fwhm_s = 60*60
gauss_std_s = int(gauss_fwhm_s/2.35)
gauss_len = int(gauss_std_s*5)
gauss = scipy.signal.gaussian(gauss_len,gauss_std_s,sym=True)
yconv = np.convolve(y,gauss,mode="same")
yconv2 = yconv/np.amax(yconv)

plt.figure()
plt.plot(vega_epochtime, vega_power)
plt.ylabel("vega_power (mW)")
plt.xlabel("epoch time")
plot_badranges(badranges, 0.3)
plt.plot(x,yconv2)

# badranges_prepped_for_cut.append("invert")

# CUT_NUM = ds.CUT_NAME.index("timestamp_sec")
# for ds in data:
#     ds.cuts.clearCut(CUT_NUM)
#     ds.cut_parameter(ds.p_timestamp, badranges_prepped_for_cut, CUT_NUM)