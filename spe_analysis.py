import numpy as np
import scipy.special as spec
from scipy.stats import norm
import matplotlib.pyplot as plt
# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit
from iminuit.cost import LeastSquares

import iminuit
print("iminuit version:", iminuit.__version__)


"""
fit_spe_v1 : fit di stefania senza il contributo di 3 pe
fit_spe_v2 : fit di stefania 
fit_spe_v3 : fit di lorenzo 
"""

def _gaussian(x, mu, sigma, N):
    """
    Gaussian function for fit
    :param x:
    :param mu: Mean
    :param sig: Sigma
    :param N: Normalization
    :return: Gaussian(x,Mu,Sigma,N)
    """
    gaus = N / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
    return gaus


def _pedestal_function(x,mu_b,sigma_b,Delta,f_b1,f_0PE):
    """
    :param x:
    :param mu_b:
    :param sigma_b:
    :param Delta:
    :param f_b1:
    :return:
    """
    shifted=mu_b-Delta
    return f_0PE*(f_b1 * norm.pdf(x,loc=mu_b,scale=sigma_b) + (1. - f_b1) * norm.pdf(x,loc=shifted,scale=sigma_b))


def poisson(x, f, G, mu, sigma_b):
    """
    :param x:
    :param f:
    :param G:
    :param mu:
    :param sigma_b:
    :return: function continuous poisson
    """
    mu_1PE = f * G
    s2D = f * 0.435
    sigma_1PE = np.sqrt(f**2 * G + sigma_b**2 + G * s2D**2)
    gamma_correction = spec.gamma(1. + (x - mu) * mu_1PE / (sigma_1PE**2))
    exponential_contribution = np.exp(-(mu_1PE / sigma_1PE)** 2)
    return mu_1PE / (sigma_1PE**2) * exponential_contribution * (((mu_1PE / sigma_1PE))**(2*(x - mu) * mu_1PE / sigma_1PE**2)) / gamma_correction

def cassetta(x, f, G, mu, sigma_b):
    """
    :param x:
    :param f:
    :param G:
    :param mu:
    :param sigma_b:
    :return:
    """
    fact = 1.2
    muR = (G + 0.375) * f
    muL = f / 2.25
    s2D = f * 0.435
    sigmaR = np.sqrt(G * (2 *f**2 + s2D**2) + sigma_b**2) * fact
    sigmaL = np.sqrt(sigma_b**2 + s2D**2) * fact

    return 0.25/ (muR - muL) * (1. + spec.erf((x - mu - muL) / sigmaL)) * (1. - spec.erf((x - mu - muR) / sigmaR))

def _SPE(x, f, G, mu, sigma_b,f_1PE,f_1PE_SA):
    return f_1PE*(f_1PE_SA * cassetta(x, f, G, mu, sigma_b)+(1-f_1PE_SA)*poisson(x, f, G, mu, sigma_b))


def _val_2PE(x, f, G, mu, sigma_b, f_1PE_SA,f_2PE,f_3PE):
    s2D = f * 0.435
    mu_1PE = f * G
    muL_1PE_SA = f / 2.25
    muR_1PE_SA = (G + 0.375) * f
    sigma_1PE = np.sqrt(f * f * G + sigma_b * sigma_b + G * s2D * s2D)
    mu_2PE = 2 * ((muR_1PE_SA + muL_1PE_SA) / 2 * f_1PE_SA + mu_1PE * (1 - f_1PE_SA))
    mu_3PE = 3. * mu_2PE/2
    first_term = (f_1PE_SA * (muR_1PE_SA - muL_1PE_SA)**2) / 12
    second_term = (1 - f_1PE_SA) * (sigma_1PE** 2 - sigma_b** 2)
    third_term = f_1PE_SA * (1 - f_1PE_SA) * ((muR_1PE_SA + muL_1PE_SA) / 2 - mu_1PE)**2
    sigma_2PE = first_term + second_term + third_term
    sigma_2PE = np.sqrt(2 * sigma_2PE + sigma_b** 2)
    sigma_3PE = sigma_2PE
    sigma_3PE = np.sqrt(3. * sigma_3PE + sigma_b * sigma_b)
    val2pe = 1. / np.sqrt(2. * np.pi) / sigma_2PE * np.exp(-0.5 * ((x - mu - mu_2PE) / sigma_2PE) * ((x - mu - mu_2PE) / sigma_2PE))
    #val3pe = 1. / np.sqrt(2. * np.pi) / sigma_3PE * np.exp(-0.5 * ((x - mu - mu_3PE) / sigma_3PE) * ((x - mu - mu_3PE) / sigma_3PE))
    return f_2PE * val2pe + f_3PE * 0#val3pe




def global_method_1(x,N,f_1PE,f_1PE_SA,f_2PE,f_3PE,f_b1,mu_b,sigma_b,Delta,G,f ):
    f_0PE = 1 - f_1PE - f_2PE - f_3PE
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)
    val_base= _pedestal_function(x,mu_b,sigma_b,Delta,f_b1,f_0PE)
    val_single_pe= _SPE(x, f, G, mu, sigma_b,f_1PE,f_1PE_SA)
    val_more_pe= _val_2PE(x, f, G, mu, sigma_b, f_1PE_SA,f_2PE,f_3PE)
    val = f_0PE * val_base + val_single_pe + val_more_pe
    return N* val


def fit_spe_v1(bc,bh,bw=7):
    mask_zero_values = bh != 0
    x = bc[mask_zero_values]
    y = bh[mask_zero_values]
    y_err = np.sqrt(y*bw)/bw
    y = y
    least_squares = LeastSquares(x, y, y_err, global_method_1)
    m = Minuit(least_squares, N=251380,f_1PE = 0.4,f_1PE_SA = 0.4,f_2PE = 0.01,f_3PE = 0.,
               f_b1 = 0.8,mu_b = 9.71809,sigma_b = 15.9888,
               Delta = 5,G = 15.6,f = 23.033)
    m.limits['N'] = (10000., np.inf)
    m.limits['f_1PE'] = (0.1, 0.6)
    m.limits['f_1PE_SA'] = (0.3, 0.8)
    m.limits['f_2PE'] = (0.005, 0.2)

    # m.fixed['f_3PE'] = 0.
    m.limits['f_3PE'] = (0, 0.1)
    m.limits['f_b1'] = (0., 1)
    m.limits['mu_b'] = (-10, 20)
    m.limits['sigma_b'] = (0, 20)
    m.limits['Delta'] = (1, 20)
    m.limits['G'] = (10, 20)
    m.limits['f'] = (10, 30)
    
    #m.limits['f_3PE'] = (0,0.05)


    m.migrad()
    m.minos()
    """
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    
    for i in range(len(m.covariance[0])):
        for j in range(len(m.covariance[0])):
            correlation = m.covariance[i, j] / np.sqrt(m.covariance[i, i] * m.covariance[j, j])
            print("correlation between ", m.parameters[i],m.parameters[j], "= ",correlation)
    """
    return m.values,m.errors, m.fval, len(x)-m.nfit

def fit_spe_v2(bc,bh,bw=7):
    mask_zero_values = bh > 0
    x = bc[mask_zero_values]
    y = bh[mask_zero_values]
    y_err = np.sqrt(y*bw)/bw
    y = y
    least_squares = LeastSquares(x, y, y_err, global_method_1)
    m = Minuit(least_squares, N=251380,f_1PE = 0.3,f_1PE_SA = 0.4,f_2PE = 0.01,f_3PE = 0.0,
               f_b1 = 0.858905,mu_b = 9.71809,sigma_b = 15.9888,
               Delta = 5,G = 15.6,f = 23.033)

    m.limits['N'] = (10000, np.inf)
    m.limits['f_1PE'] = (0., 0.8)
    m.limits['f_1PE_SA'] = (0., 0.8)
    m.limits['f_2PE'] = (0.0, 0.1)
    m.fixed['f_3PE']= True
    #m.limits['f_3PE'] = (0, 0.005)
    m.limits['f_b1'] = (0., 1.)
    m.limits['mu_b'] = (-10, 50)
    m.limits['sigma_b'] = (5, 20)
    m.limits['Delta'] = (0, 50)
    m.limits['G'] = (10, 30)
    m.limits['f'] = (10, 35)

    m.migrad(ncall=1000000)
    m.minos()
    """
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    """
    return m.values,m.errors, m.fval, len(x)-m.nfit




def calc_gain(bc, err, err2,*pars ):
    N = pars[0]
    f_1PE = pars[1]
    f_1PE_SA = pars[2]
    f_2PE = pars[3]
    f_3PE = pars[4]
    f_0PE = 1 - f_1PE - f_2PE - f_3PE
    f_b1 = pars[5]
    mu_b = pars[6]
    sigma_b = pars[7]
    Delta = pars[8]
    G = pars[9]
    f = pars[10]
    
    sigma_G = err#err[9]
    sigma_f = err2#err[10]
    fact = 1.2
    muR = (G + 0.375) * f
    muL = f / 2.25
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)
    mu_1PE = f * G
    #gain = (f_1PE_SA*(muR + muL)/2) + mu_1PE*(1-f_1PE_SA)
    s2D = f * 0.435
    sigmaR = np.sqrt(G * (2 * f ** 2 + s2D ** 2) + sigma_b ** 2) * fact

    sigmaL = np.sqrt(sigma_b ** 2 + s2D ** 2) * fact

    mu_Cassetta = (muR + muL) / 2. + (sigmaR ** 2 - sigmaL ** 2) / (4. * (muR - muL))

    sigma2_Cassetta = (sigmaR ** 2 * muR + 2. / 3. * muR ** 3 - sigmaL ** 2 * muL - 2. / 3. * muL ** 3) / (
                2. * (muR - muL)) - mu_Cassetta ** 2

    # mean and sigma of the 1PE (Cassetta + Poisson)
    sigma2_mu1PE = (G * sigma_f)**2 + (f * sigma_G)**2

    gain = mu_Cassetta * f_1PE_SA + mu_1PE * (1 - f_1PE_SA)  # === GAIN !!!!!!!!
    sigma_gain = np.sqrt(sigma2_Cassetta)*f_1PE_SA+(1-f_1PE_SA)*np.sqrt(sigma2_mu1PE)
    return gain, mu_1PE , sigma_gain

def conversion(value):
    R = 50
    samp = 2 ** 14
    e = 1.6021766208
    V = 2
    dt = 2
    adc_to_pc = value * V / samp / R * dt * 10 ** 3
    return adc_to_pc / e


def area_spe(bc,*pars):
    N = pars[0]
    f_1PE = pars[1]
    f_1PE_SA = pars[2]
    f_b1 = pars[5]
    mu_b = pars[6]
    sigma_b = pars[7]
    Delta = pars[8]
    G = pars[9]
    f = pars[10]
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)
    y=N*_SPE(bc, f, G, mu, sigma_b, f_1PE, f_1PE_SA)
    area=np.sum(y)
    return area

def area_dpe(bc,*pars):
    N = pars[0]
    f_1PE_SA = pars[2]
    f_2PE = pars[3]
    f_3PE = pars[4]
    f_b1 = pars[5]
    mu_b = pars[6]
    sigma_b = pars[7]
    Delta = pars[8]
    G = pars[9]
    f = pars[10]
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)
    y=N*_val_2PE(bc, f, G, mu, sigma_b, f_1PE_SA,f_2PE,f_3PE)
    area=np.sum(y)
    return area

def efficiency(bhc, area_spe, area_dpe):
    return (np.sum(bhc)-area_dpe) / (area_spe)

def which_column(channel):
    pmt_in_even = [val for i, val in enumerate(np.split(np.arange(2000, 2120), 20)) if
                       (i + 1) % 2 == 0]
    pmt_in_odd = [val for i, val in enumerate(np.split(np.arange(2000, 2120), 20)) if
                        (i + 1) % 2 != 0]
    if channel+2000 in [item for sublist in pmt_in_even for item in sublist]:
        return "even"
    elif channel+2000 in [item for sublist in pmt_in_odd for item in sublist]:
        return "odd"

def plot_functions_fit_v1(bin_centers,bin_heights,bin_heights_cut,chi_square,ndof,pars,errors,channel,run_id,fit_type="v1"):
    N = pars[0]
    f_1PE = pars[1]
    f_1PE_SA = pars[2]
    f_2PE = pars[3]
    f_3PE = pars[4]
    f_0PE = 1 - f_1PE - f_2PE - f_3PE
    f_b1 = pars[5]
    mu_b = pars[6]
    sigma_b = pars[7]
    Delta = pars[8]
    G = pars[9]
    f = pars[10]
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)

    fig = plt.figure(figsize=(10, 8), dpi=200)
    grid = plt.GridSpec(5, 5, figure=fig, hspace=0, wspace=0)
    main_axis = plt.subplot(grid[1:5, 0:4])
    residuals_plot = plt.subplot(grid[:1, 0:4])
    results_table = plt.subplot(grid[1:, 4:])

    # ----------------
    # Main Plot stuff:
    # ----------------

    main_axis.hist(bin_centers, bins=len(bin_centers), range=(bin_centers[0], bin_centers[-1]), weights=bin_heights, histtype="step")
    main_axis.hist(bin_centers, bins=len(bin_centers), range=(bin_centers[0], bin_centers[-1]), weights=bin_heights_cut,
                   histtype="stepfilled" ,label= "Threshold",alpha=0.2)

    main_axis.plot(bin_centers, global_method_1(bin_centers, *pars), label='SPE fit', color='k')
    main_axis.plot(bin_centers, N * _pedestal_function(bin_centers, mu_b, sigma_b, Delta, f_b1, f_0PE), label="Pedestal",
                   color='orange',
                   ls='--')
    main_axis.plot(bin_centers, N * f_1PE * (1-f_1PE_SA) * poisson(bin_centers, f, G, mu, sigma_b),
                   label='Single Photo Electron',
                   color='red',
                   ls='-.')
    main_axis.plot(bin_centers, N * f_1PE * f_1PE_SA * cassetta(bin_centers, f, G, mu, sigma_b),
                   label='Sub-amplified',
                   color='green',
                   ls=':')
    main_axis.plot(bin_centers, N * _val_2PE(bin_centers, f, G, mu, sigma_b, f_1PE_SA, f_2PE, f_3PE),
                   label='Double Photo Electron',
                   color='brown',
                   ls='-')
    main_axis.grid('darkergray', linestyle='--')
    main_axis.grid('darkergray', linestyle=':', which='minor')

    main_axis.set_ylabel('Number of Entries')
    main_axis.set_xlabel('Integrated Charge [ADC x Samples]')

    main_axis.set_ylim(1, 10 ** 4)
    main_axis.set_yticks([10, 10 ** 2, 10 ** 3, 10 ** 4], [10, 10 ** 2, 10 ** 3, 10 ** 4])
    main_axis.set_yscale('log')


    # --------------
    # Residuals
    # --------------
    mask=bin_heights>1
    mask=bin_heights>1
    residuals = (bin_heights[mask] - global_method_1(bin_centers[mask] , *pars)) / np.sqrt(bin_heights[mask] )
    residuals_plot.plot(bin_centers[mask] , residuals,
                        ls='',
                        marker='.'
                        )

    residuals_plot.set_xticklabels([])
    residuals_plot.set_ylim(-5,5)
    residuals_plot.set_yticks([-2.5,0,2.5], [-2.5,0,2.5])
    residuals_plot.set_ylabel('Residuals')
    residuals_plot.grid('darkergray', ls='--')
    residuals_plot.hlines(0, *main_axis.get_xlim())
    residuals_plot.set_xlim(main_axis.get_xlim())
    residuals_plot.set_title(f"Channel {channel} - Fit Type - {fit_type} - Column Type - {which_column(channel)}")
    # ------------------
    # Fit Results table:
    # ------------------
    results_table.set_xticklabels([])
    results_table.set_yticklabels([])
    results_table.set_xticks([])
    results_table.set_yticks([])
    gain,_,err_gain = calc_gain(bin_centers,errors[9],errors[10], *pars)
    eff = efficiency(bin_heights_cut, area_spe(bin_centers, *pars), area_dpe(bin_centers, *pars))*100
    results_table.text(-0.18, 0.3, r'''
            Fit Parameters:
            $Norm: {0:.0f}\pm{11:.0f} $
            $f{{1pe}}: {1:.3f}\pm{12:.3f} $
            $f_{{subamp}}: {2:.3f}\pm{13:.3f} $
            $f_{{2pe}}: {3:.3f}\pm{14:.3f} $
            $f_{{Ped}}: {5:.3f}\pm{16:.3f} $
            $\mu_{{ped}}: {6:.3f}\pm{17:.3f} $
            $\sigma_{{ped}}: {7:.3f}\pm{18:.3f} $
            $\Delta: {8:.3f}\pm{19:.3f} $
            $G: {9:.3f}\pm{20:.3f} $
            $f: {10:.3f}\pm{21:.3f} $
            
            Fit statistics:
            $\chi^2: {22:.2f} $
            $ndof: {23:.2f} $
            '''.format(*pars,*errors,chi_square,ndof),transform=results_table.transAxes)

    results_table.axis('off')

    fig.savefig(f"./figures/{run_id}/{channel}_{fit_type}.png")
    plt.close()

    return gain,err_gain



"""
def new_gain_calculation(bin_centers,pars):
    N = pars[0]
    f_1PE = pars[1]
    f_1PE_SA = pars[2]
    f_b1 = pars[5]
    mu_b = pars[6]
    sigma_b = pars[7]
    Delta = pars[8]
    G = pars[9]
    f = pars[10]
    mu = f_b1 * mu_b + (1. - f_b1) * (mu_b - Delta)

    arrays = N * f_1PE * (1 - f_1PE_SA) * poisson(bin_centers, f, G, mu,
                                                      sigma_b) + N * f_1PE * f_1PE_SA * cassetta(bin_centers, f, G,
                                                                                                     mu, sigma_b)
    value = np.sum([bin_centers[i] * arrays[i] for i in range(0, len(arrays))])
    total = np.sum(
        N * f_1PE * (1 - f_1PE_SA) * poisson(bin_centers, f, G, mu, sigma_b) + N * f_1PE * f_1PE_SA * cassetta(
            bin_centers, f, G, mu, sigma_b))
    gain=1/(value/total)
    return gain
"""