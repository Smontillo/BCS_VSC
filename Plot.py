import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numba as nb
import time
from scipy.special import erfi
import BCS as par
# =====================================================
def DeltaAn(ω):
    PreF   = 0.5 * par.N * par.g0**2 * np.sqrt(np.exp(-par.β * par.ω0))
    Afac   = 1/(ω - par.ωc)
    Bfac   = np.sqrt(np.pi/(2 * par.σ**2)) * np.exp ((par.ω0 - par.ωc)**2/(2 * par.σ**2)) * erfi((par.ω0 - par.ωc)/np.sqrt(2*par.σ**2))
    return PreF * (Afac + Bfac)/par.cm2au

def V_U_An(ω):
    u2 = 0.5 * (1 + (ω - μ)/((ω - μ)**2 + DeltaAn(ω*par.cm2au)**2)**0.5)
    v2 = 0.5 * (1 - (ω - μ)/((ω - μ)**2 + DeltaAn(ω*par.cm2au)**2)**0.5)
    return u2, v2

def Gauss_Func(ω):
    PreF = 1/np.sqrt(2 * np.pi * (par.σ/par.cm2au)**2)
    Exp  = np.exp(-0.5 * (ω - ω0)**2/(par.σ/par.cm2au)**2)
    return PreF * Exp

def CritFreq():
    Num = par.N * (par.g0[0]/par.cm2au)**2 
    Ξ   = np.sqrt(np.pi/(2 * σ**2)) * np.exp ((ω0 - ωc)**2/(2 * σ**2)) * erfi((ω0 - ωc)/np.sqrt(2*σ**2))
    Den = 2 * (ω0 + (1/par.β*par.cm2au)) - par.N * (par.g0[0]/par.cm2au)**2 * Ξ
    return Num/Den

def OrderMu(ω_lst):
    n_exc = np.sum(np.exp(-par.β * par.cm2au * ω_lst)/(1 + np.exp(-par.β * par.cm2au * ω_lst)))
    α     = (par.N - 2 * n_exc) / par.N 
    return ω0 + np.sqrt(1/(1-α**2)) * α * DeltaAn(ω0)


col = ['#3498db', '#9b59b6', '#e74c3c', '#e67e22', '#34495e', '#1abc9c']
# =====================================================
μ     = np.load('./Par.npy')[0][0]
Data  = np.load('./Numerical_Data.npy')
ω_lst  = Data[:,0]
un     = Data[:,1]
vn     = Data[:,2]
Δa     = Data[:,3]
Ea     = Data[:,4]
ΔAn    = DeltaAn(ω_lst*par.cm2au)
ω0     = par.ω0/par.cm2au
ωc     = par.ωc/par.cm2au
σ      = par.σ/par.cm2au
u2,v2  = V_U_An(ω_lst)
μO     = OrderMu(ω_lst)
Crit_ω = CritFreq()
print(Crit_ω)
# =====================================================
# u_n², v_n² vs. (ω0 - μ)
# =====================================================
fig, ax = plt.subplots(1,3,figsize = (9,3), dpi = 300)
fig.subplots_adjust(wspace=0.1)
ax[1].plot((ω_lst - ωc),np.conjugate(vn)*vn, ls = '-', lw = 3, label = r'$|v_n|^2$', c = f'{col[0]}')
ax[1].plot((ω_lst - ωc),v2, ls = '--', lw = 1.2, c = 'black')
ax[1].plot((ω_lst - ωc),np.conjugate(un)*un, ls = '-', lw = 3, label = r'$|u_n|^2$', c = f'{col[2]}')
ax[1].plot((ω_lst - ωc),u2, ls = '--', lw = 1.2, c = 'black')
# ax.axvline((ωc-ωc), ls = '--', lw = 0.5, c = 'black', label = r'$\omega_c$')
# ax[1].grid()
ax2 = ax[1].twinx()
ax2.plot(np.NaN, np.NaN, ls= '-',label='Simulation', c='black')
ax2.plot(np.NaN, np.NaN, ls= '--',label='Theory', c='black')
ax2.get_yaxis().set_visible(False)
ax[1].set_xlabel(r'$ω - ω_c$ (cm$^{-1}$)')
ax[1].set_xlim(-4,4)
ax[1].legend(frameon=False, fontsize=8, handlelength=1.5, handletextpad=0.5, loc='center left')
ax2.legend(frameon=False, fontsize=8, handlelength=1.5, handletextpad=0.5, loc='center right')
ax[1].tick_params(axis='x', labelsize=11)
ax[1].tick_params(axis='y', labelsize=11)
# =====================================================
# Δn vs. (ω0 - μ)
# =====================================================
Δ  = np.ma.masked_where(np.abs(Δa) > 2010, Δa)
ΔA = np.ma.masked_where(np.abs(ΔAn) > 2010, ΔAn)
ax[0].plot((ω_lst-ωc), Δ, ls = '-', lw = 3, c = f'{col[0]}', label = 'Simulation')
ax[0].plot((ω_lst-ωc), ΔA, ls = '-', lw = 2, c = f'{col[2]}', label = 'Theory')
ax[0].set_xlabel(r'$\omega - ω_c$ (cm$^{-1}$)')
ax[0].set_ylabel(r'$\Delta_n$ (cm$^{-1}$)')
ax[0].set_xlim(-5,5)
ax[0].set_ylim(-150,150)
ax[0].tick_params(axis='x', labelsize=11)
ax[0].tick_params(axis='y', labelsize=11)
# ax[0].grid()
ax[0].legend(frameon=False,  fontsize=8, handlelength=1.5, handletextpad=0.5)
# =====================================================
# DISTRIBUTION VS. GAP ENERGY
# =====================================================
ω_con = np.linspace(ω_lst[0],ω_lst[-1],200)
ax[2].axhline(np.sqrt(np.exp(-(par.β*par.cm2au) * ω0)) * (ω0 + (1/(par.β*par.cm2au))), lw=1.5, c=col[4])#, label=r'$e^{β\hbar\omega_0/2}\cdot (k_BT+\hbar\omega_0)$')
ax[2].plot((ω_lst-ωc), np.abs(Δ), ls = '-', lw = 3, c = f'{col[0]}', label = 'Simulation')
ax[2].plot((ω_lst-ωc), np.abs(ΔA), ls = '-', lw = 2, c = f'{col[2]}', label = 'Theory')
ax[2].set_ylabel(r'$|\Delta_n|$ (cm$^{-1}$)')
ax[2].axvline(Crit_ω, ls='--', lw=1, c='black')
ax[2].axvline(-Crit_ω, ls='--', lw=1, c='black')
ax[2].set_ylim(10,2E3)
ax[2].set_xlim(-0.7,0.7)
ax[2].set_xlabel('ω - ω$_c$ (cm⁻¹)')
ax[2].set_yscale('log')
ax[2].tick_params(axis='x', labelsize=11)
ax[2].tick_params(axis='y', labelsize=11)
ax[2].legend(frameon=False,  fontsize=8, handlelength=1.5, handletextpad=0.5)
# ax[2].grid()
plt.tight_layout()
plt.savefig('./Images/Fig1.png', dpi = 300, bbox_inches = 'tight')
# plt.savefig('./Images/Comb_nogrid.svg', dpi = 300, bbox_inches = 'tight')
plt.close()
