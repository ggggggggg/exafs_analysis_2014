import numpy as np
import pylab as plt
import mass

filepairs = []
filepairs.append(["20150807_delay_-20_unpumped_hist_0.5ev_bin.dat","20150807_delay_-20_pumped_hist_0.5ev_bin.dat"])
filepairs.append(["20150807_delay_55_unpumped_hist_0.5ev_bin.dat","20150807_delay_55_pumped_hist_0.5ev_bin.dat"])
filepairs.append(["20150803_unpumped_hist_0.5ev_bin.dat","20150803_pumped_hist_0.5ev_bin.dat"])
filepairs.append(["20150731_unpumped_hist_0.5ev_bin.dat","20150731_pumped_hist_0.5ev_bin.dat"])
filepairs.append(["20150731_2_unpumped_hist_0.5ev_bin.dat","20150731_2_pumped_hist_0.5ev_bin.dat"])
filepairs.append(["20150730_unpumped_hist_0.5ev_bin.dat","20150730_pumped_hist_0.5ev_bin.dat"])
# filepairs.append(["20150807_delay_-20_unpumped_hist_0.5ev_bin.dat","20150807_delay_55_unpumped_hist_0.5ev_bin.dat"])
fname = filepairs[2][0]
data = np.loadtxt(fname, delimiter=",")
binc_all, counts_all = data[:,0],data[:,1]
fnamep = filepairs[2][1]
datap = np.loadtxt(fnamep, delimiter=",")
binc_allp, counts_allp = datap[:,0],datap[:,1]


singlet = np.loadtxt("singlet.csv",delimiter=",",skiprows=6,usecols=(0,1))
singlet_energy = singlet[:,0]
singlet_intensity_ = singlet[:,1]
quintet = np.loadtxt("quintet.csv",delimiter=",",skiprows=6,usecols=(0,1))
quintet_energy = quintet[:,0]
quintet_intensity_ = quintet[:,1]


indlo, indhi = 10072, 10148
energy=binc_all[indlo:indhi]
counts = counts_all[indlo:indhi]
countsp = counts_allp[indlo:indhi]

singlet_intensity = np.interp(energy, singlet_energy, singlet_intensity_)
quintet_intensity = np.interp(energy, quintet_energy, quintet_intensity_)

plt.figure()
f=0.2
plt.plot(energy, 1800*((1-f)*singlet_intensity+f*quintet_intensity)+40)
plt.plot(energy, counts)



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
theory_func = make_theory_func(singlet_intensity, quintet_intensity)


y_ff = fitfunc(energy, singlet_intensity, quintet_intensity, 0.0, 0.0, 1800, 5, 15, 0.1, 40, 0)
y_ff_2 = fitfunc(energy, singlet_intensity, quintet_intensity, 1.0,  0.0, 1800, 5, 15, 0.1, 40, 0)
y_ff_3 = fitfunc(energy, singlet_intensity, quintet_intensity, -1.0,  0.0, 1800, 5, 15, 0.1, 40, 0)

x_dr,y_dr = detector_response(0.5, 0.0, 5, 20, 0.15)


plt.figure()
# plt.plot(x_dr+np.median(energy),y_dr)
plt.plot(energy, y_ff)
plt.plot(energy, y_ff_2)
plt.plot(energy, y_ff_3)
plt.plot(energy, counts)

params_guess = [0.1, 0.5, 1800, 5, 24, 0.14, 40, 0]
params_names = ["shift_ev", "f", "amplitude", "fwhm", "tail_size_ev", "tail_fraction", "bg", "bg_slope"]
theory_func = make_theory_func(singlet_intensity, quintet_intensity)
fitter = mass.MaximumLikelihoodHistogramFitter(energy, counts, params_guess, theory_func)
fitter.hold(params_names.index("tail_fraction"),0.0)
fitter.hold(params_names.index("tail_size_ev"),24)
fitter.hold(params_names.index("f"),0)
fitter.setbounds(params_names.index("f"), 0, 1)
fitter.setbounds(params_names.index("tail_fraction"), 0, .2)
fitter.setbounds(params_names.index("tail_size_ev"), 10, 30)
fitter.setbounds(params_names.index("fwhm"), 2, 8)
fitter.setbounds(params_names.index("bg"), 0, 1e6)

params,covar = fitter.fit()
params0 = params.copy()
params0[params_names.index("f")]=0
params1 = params.copy()
params1[params_names.index("f")]=.3

params_guessp = params
fitterp = mass.MaximumLikelihoodHistogramFitter(energy, countsp, params_guess, theory_func)
fitterp.setbounds(params_names.index("f"), 0, 1)
fitterp.setbounds(params_names.index("tail_fraction"), 0, .2)
fitterp.setbounds(params_names.index("tail_size_ev"), 10, 30)
fitterp.setbounds(params_names.index("fwhm"), 2, 8)
fitterp.setbounds(params_names.index("bg"), 0, 1e6)
for i, param in enumerate(params):
    if not params_names[i] == "f":
        print("hold", i, param, params_names[i])
        fitterp.hold(i, param)
paramsp, covarp = fitterp.fit()
dfp = np.sqrt(covarp[params_names.index("f"), params_names.index("f")])

plt.figure()
plt.plot(energy, counts,label="unpumped",drawstyle="steps-mid")
plt.plot(energy, theory_func(params,energy),label="f=%0.2f"%params[1],lw=2)
# plt.plot(energy, theory_func(params0,energy))
# plt.plot(energy, theory_func(params1,energy))
plt.plot(energy, countsp,label="pumped",drawstyle="steps-mid")
plt.plot(energy, theory_func(paramsp, energy),label="f=%0.2f +/- %0.3f"%(paramsp[1], dfp),lw=2)
plt.legend()
for v,n in zip(params, params_names):
    print(v,n)
for v,n in zip(paramsp, params_names):
    print(v,n)


# slope_dpulseheight_denergy = 1.0
# params_guess = [None] * 8
# # resolution guess parameter should be something you can pass
# params_guess[0] = 10 * slope_dpulseheight_denergy  # resolution in pulse height units
# params_guess[1] = 6404  # Approximate peak position
# params_guess[2] = slope_dpulseheight_denergy  # energy scale factor (pulseheight/eV)
# hold = [2]  #hold the slope_dpulseheight_denergy constant while fitting
# hold = None
#
#
# indlo, indhi = 8720, 8880
# fitter = mass.FeKAlphaFitter()
# # params_guess = [3,None, 1, 0, None, None, .1, 10]
# # hold = [2]
# fitter.fit(counts_all[indlo:indhi], binc_all[indlo:indhi], params=params_guess, plot=True, vary_bg=True, vary_tail=True,hold=hold)
# fitter.fit(counts_all[indlo:indhi], np.arange(indhi-indlo), params=params_guess, plot=True, vary_bg=True, vary_tail=True,hold=hold)

plt.close("all")
for fp in filepairs:
    data = np.loadtxt(fp[0], delimiter=",")
    binc_all, counts_all = data[:,0],data[:,1]
    datap = np.loadtxt(fp[1], delimiter=",")
    binc_allp, counts_allp = datap[:,0],datap[:,1]
    indlo, indhi = 10072, 10148
    energy=binc_all[indlo:indhi]
    counts = counts_all[indlo:indhi]
    countsp = counts_allp[indlo:indhi]

    params_guess = [0.1, 0.5, 1800, 5, 24, 0.14, 40, 0]
    params_names = ["shift_ev", "f", "amplitude", "fwhm", "tail_size_ev", "tail_fraction", "bg", "bg_slope"]

    fitter = mass.MaximumLikelihoodHistogramFitter(energy, counts, params_guess, theory_func)
    fitter.hold(params_names.index("tail_fraction"),0.0)
    fitter.hold(params_names.index("tail_size_ev"),24)
    fitter.hold(params_names.index("f"),0)
    fitter.setbounds(params_names.index("f"), -0.2, 1)
    fitter.setbounds(params_names.index("tail_fraction"), 0, .2)
    fitter.setbounds(params_names.index("tail_size_ev"), 10, 30)
    fitter.setbounds(params_names.index("fwhm"), 2, 8)
    fitter.setbounds(params_names.index("bg"), 0, 1e6)

    params,covar = fitter.fit()

    params_guessp = params.copy()
    params_guessp[params_names.index("f")] =0.5
    fitterp = mass.MaximumLikelihoodHistogramFitter(energy, countsp, params_guess, theory_func)
    fitterp.setbounds(params_names.index("f"), -0.2, 1)
    fitterp.setbounds(params_names.index("tail_fraction"), 0, .2)
    fitterp.setbounds(params_names.index("tail_size_ev"), 10, 30)
    fitterp.setbounds(params_names.index("fwhm"), 2, 8)
    fitterp.setbounds(params_names.index("bg"), 0, 1e6)
    for i, param in enumerate(params):
        if not params_names[i] == "f":
            print("hold", i, param, params_names[i])
            fitterp.hold(i, param)
    paramsp, covarp = fitterp.fit()
    dfp = np.sqrt(covarp[params_names.index("f"), params_names.index("f")])

    # plt.figure()
    # plt.title("%s\n%s"%(fp[0], fp[1]))
    # plt.plot(energy, counts,label="unpumped",drawstyle="steps-mid")
    # plt.plot(energy, theory_func(params,energy),label="f=%0.2f"%params[1],lw=2)
    # plt.plot(energy, countsp,label="pumped",drawstyle="steps-mid")
    # plt.plot(energy, theory_func(paramsp, energy),label="f=%0.2f +/- %0.3f"%(paramsp[1], dfp),lw=2)
    # plt.xlabel("energy (eV)")
    # plt.ylabel("counts per 0.5 eV bin")
    # plt.text(0.05,0.95,"unpumped fit params\n"+"\n".join(["%s %0.2f"%(n,v) for n,v in zip(params_names, params)]), transform=plt.gca().transAxes, va="top")
    # plt.legend()
    # plt.savefig(fp[0][:fp[0].find("u")-1], dpi=100)

    plt.figure()
    plt.plot(energy, countsp)
    plt.plot(energy, theory_func(paramsp, energy),label="f=%0.2f +/- %0.3f"%(paramsp[1], dfp),lw=2)
    plt.plot(energy, theory_func([params[0], 1, paramsp[1]*params[2], params[3],params[4],params[5],0, 0], energy) )
    plt.plot(energy, theory_func([params[0], 0, (1-paramsp[1])*params[2], params[3],params[4],params[5],0, 0], energy) )
    plt.plot(energy, theory_func([params[0], 0, 0, params[3],params[4],params[5],params[6], params[7]], energy) )
    plt.xlabel("energy (eV)")
    plt.ylabel("counts per 0.5 eV bin")
    break


def delay_as_offset(t0_ps=50, tau_ps=660, n_points=3, t0_2xmm=-20):
    tau_ps = float(tau_ps)
    delay_ps = -np.log(np.linspace(np.exp(-t0_ps/tau_ps),0,n_points+1))[:-1]*tau_ps
    n_index_air = 1.0003
    c_2xmm_per_ps = 299792458*1e-12*1e3/2/n_index_air
    delay_2xmm = t0_2xmm+delay_ps*c_2xmm_per_ps
    delay_f = np.exp(-delay_ps/tau_ps)
    print("delay_2xmm", delay_2xmm)
    print("delay_ps", delay_ps)
    print("delay_f", delay_f)
    print("t0_2xmm", t0_2xmm)
    return delay_2xmm, delay_ps, delay_f


fs = 24
plt.figure()
t=np.arange(-200,600)
# plt.plot(t,np.zeros_like(t),'k')
plt.plot(t,0.42/np.exp(-50/765.)*np.exp(-t/765.)*np.less_equal(0,t),"b",lw=2)
# plt.plot(t,0.42/np.exp(-50/660.)*np.exp(-t/660.)*np.less_equal(0,t),"--",lw=2)
plt.errorbar([-100,50,50+495],[-0.07,0.42,0.22],yerr=[0.095,0.092,0.096],fmt="r.",markersize=16,lw=2,clip_on=False)
plt.xlabel("xray arrival-pump arrival (ps)",fontsize=fs)
plt.ylabel("excited state fraction",fontsize=fs)
plt.xlim(-110,600)
plt.ylim(-0.1,0.55)
a = plt.gca()
plt.tick_params(axis='both', which='major', labelsize=fs)
a.set_yticklabels(["","0.0","0.1","0.2","0.3","0.4","0.5"],fontsize=fs)
a.set_xticklabels(["-1 ms","-1 ms","0","100","200","300","400","500","600"],fontsize=fs)
plt.gcf().subplots_adjust(bottom=0.15)
