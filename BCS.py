import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numba as nb
import time
from scipy.special import erfi
# =====================================================
# FUNCTIONS
# =====================================================
@nb.jit(nopython=True, fastmath=True)
def plotLoss(loss):
    fig, ax = plt.subplots(figsize = (3,3), dpi=100)
    ax.plot(range(len(loss)), loss, ls = '-', lw = 1, c = '#e74c3c')#marker = 'o', markersize = 1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    plt.savefig('Images/Loss.png', dpi=300, bbox_inches='tight')
    plt.close()

@nb.jit(nopython=True, fastmath=True)
def fields(var, g0):
    μ, Sg, Sgd = var
    ξ          = (ω_lst - μ)   
    A          = 0.5 * g0 * (Sg / δ0 + Sgd)
    B          = g0**2/δ0  * 0.5
    Δa         = A.copy()
    Ea         = np.sqrt((ξ)**2 + Δa**2)
    Δa         = 2 * A * Ea  / (2 * Ea + B)
    Ea         = np.sqrt((ξ)**2 + Δa**2)
    uv         = 0.5 * Δa / Ea
    v2         = 0.5 * (1 - (ξ) / Ea)
    return Δa, ξ, uv, v2

@nb.jit(nopython=True, fastmath=True)
def n_of_mu(m, Sg, Sgd, g0):
        return fields([m, Sg, Sgd], g0)[3].sum()

@nb.jit(nopython=True, fastmath=True)
def mu_bisection(μ, Sg, Sgd, target, g0, max_bisect=100):
    Δa0, ξ0, uv0, v20 = fields([μ, Sg, Sgd], g0)
    scale             = np.max(np.abs(Δa0)) + 1.0 * cm2au
    low               = np.min(ω_lst) - 10.0 * scale * cm2au
    high              = np.max(ω_lst) + 10.0 * scale * cm2au
    n_low, n_high     = n_of_mu(low, Sg, Sgd, g0), n_of_mu(high, Sg, Sgd, g0)
    # ENLARGE BRACKET
    if not (n_low <= target <= n_high):
        width         = 100.0 * scale
        low, high     = low - width, high + width
        n_low, n_high = n_of_mu(low, Sg, Sgd, g0), n_of_mu(high, Sg, Sgd, g0)
        if not (n_low <= target <= n_high):
            raise RuntimeError("Could not bracket mu for the requested n_target.")
    # BISECTION
    for _ in range(max_bisect):
        mid      = 0.5 * (low + high)
        n_mid    = n_of_mu(mid, Sg, Sgd, g0)
        if np.abs(n_mid - target) <= 1e-12: #* max(1.0, N):
            return mid
        if n_mid < target:
            low  = mid
        else:
            high = mid
    return 0.5 * (low + high)

@nb.jit(nopython=True, fastmath=True)
def Opt_BCS(μ, Sg, Sgd, n_target, tol, max_iter, mix, verbose, g0):
    loss = []
    if verbose:
        print("# MINIMIZATION STARTS ===============================================")
    for it in range(max_iter):
        # Given current Sg,Sg_over_delta, solve for mu
        μ_new = mu_bisection(μ, Sg, Sgd, n_target, g0)
        # Recompute fields at μ_new
        Δa, ξ, uv, v2 = fields([μ_new, Sg, Sgd], g0)
        # Update the two scalar order parameters
        Sg_new  = np.dot(g0, uv)
        Sgd_new = np.dot(g0 / δ0, uv)
        # Mix for stability
        μ_upd   = (1.0 - mix) * μ + mix * μ_new
        Sg_upd  = (1.0 - mix) * Sg + mix * Sg_new
        Sgd_upd = (1.0 - mix) * Sgd + mix * Sgd_new
        # Check convergence
        err     = max(abs(μ_upd - μ), abs(Sg_upd - Sg), abs(Sgd_upd - Sgd))
        loss.append(err)
        if verbose:
            if it % 10 == 0:
                occ = v2.sum()
                # print(f"it {it} | err={err}")
        μ, Sg, Sgd = μ_upd, Sg_upd, Sgd_upd
        if err < tol:
            break
    else:
        occ = v2.sum()
        # plotLoss(loss)
        # print(f"it {it} | err={err}")
        raise RuntimeError("Did not converge; try smaller mix or better initial guess.")
    occ = v2.sum()
    # print(f"it {it} | err={err}")
    # Final fields and (u,v) with consistent signs
    Δa, ξ, uv, v2    = fields([μ, Sg, Sgd],g0)
    if verbose:
        print("# MINIMIZATION ENDS ===============================================")
    v                = np.sqrt(v2)
    # Assign sign to v so that u*v has the sign of uv (u >= 0 by choice)
    u                = np.sqrt(1.0 - v2)
    sign_uv          = np.sign(uv + 0.0)
    v                = sign_uv * v
    # plotLoss(loss)
    return μ, u, v, Δa, uv, Sg, Sgd, err

@nb.jit(nopython=True, fastmath=True)
def GaussDist(σ,ε):
    ω_lst  = np.zeros(N)
    ran    = 200 * cm2au
    ωMin   = ωc - ran
    ωMax   = ωc + ran
    i_n    = 0
    while i_n < N:
        sampledEnergy = np.random.normal(ω0, σ)
        if  (np.abs(sampledEnergy-ωc)) > ε and (ωMin < sampledEnergy < ωMax): 
        # if sampledEnergy != ωc: # (np.abs(sampledEnergy-ωc)) > ε and (ωMin < sampledEnergy < ωMax):
            ω_lst[i_n] = sampledEnergy
            i_n += 1
    ω_lst  = np.sort(ω_lst)
    δ0     = ω_lst - ωc
    n_exc  = np.sum(np.exp(-beta * ω_lst)/(1+np.exp(-beta * ω_lst)))
    # plotDist(ω_lst,σ)
    return ω_lst, δ0, n_exc

@nb.jit(nopython=True, fastmath=True)
def plotDist(ω_lst,σ):
    fig, ax = plt.subplots(figsize=(3,3), dpi=150)
    count, bins, ignored = plt.hist(ω_lst/cm2au, 70, density=True, color='red')#, label = f'{np.round(ΩR_lst[j]/cm2au,1)}')
    plt.plot(bins, 1/(σ/cm2au * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - (ω0/cm2au))**2 / (2 * (σ/cm2au)**2) ),
            linewidth=1, color='black')
    plt.axvline(ωc/cm2au, ls = '--', c = 'black', lw =1, label = '$\omega_c$')
    plt.xlabel('$\omega$ (cm⁻¹)')
    plt.legend(frameon=False, fontsize=5, title = r'σ (cm⁻¹)', title_fontsize=6, handlelength=1.5, handletextpad=0.5)
    plt.tight_layout()
    plt.savefig('./Images/FreqDist.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

@nb.jit(nopython=True, fastmath=True)
def MinFunc(μ0, Sg0, Sgd0, n_target, tol, max_iter, mix, verbose, g0):
    μ, Sg, Sgd          = μ0, Sg0, Sgd0
    μ, u_n, v_n, Δa, uv, Sg, Sgd, err = Opt_BCS(μ, Sg, Sgd, n_target, tol, max_iter, mix, verbose, g0)
    Ea                  = np.sqrt((ω_lst - μ)**2 + Δa**2)
    return μ, u_n, v_n, Δa, Ea, Sg, Sgd, err

# @nb.jit(nopython=True, fastmath=True)
def Gap_An(ω,σ):
    PreF = 0.5 * N * g0[0]**2 * np.sqrt(np.exp(-beta * ω0))
    Afac = 1 / (ω - ωc)
    Bfac = np.sqrt(np.pi / (2 * σ**2)) * np.exp((ω0 - ωc)**2 / (2 * σ**2)) * erfi((ω0 - ωc) / np.sqrt(2 * σ**2))
    return PreF * (Afac + Bfac)

# @nb.jit(nopython=True, fastmath=True)
def InitCond(ω_lst,δ0,g0,n_exc,σ):
    # GAP EQUATION
    Δω  = Gap_An(ω_lst,σ)
    Δω0 = Gap_An(ω0,σ)
    # μ
    α   = (N - 2 * n_exc) / N
    μ   = ω0 + np.sqrt(1 / (1 - α**2)) * α * Δω0
    # v_n and u_n
    vn  = np.sqrt(0.5 * (1 - (ω_lst - μ) / (np.sqrt((ω_lst - μ)**2 + Δω**2))))
    un  = np.sqrt(0.5 * (1 + (ω_lst - μ) / (np.sqrt((ω_lst - μ)**2 + Δω**2))))
    # Sg and Sgd
    Sg  = np.sum(g0 * vn * un) 
    Sgd = np.sum(g0 * vn * un / δ0) 
    return μ, Sg, np.abs(Sgd)*100
# =====================================================
# PHYSICAL CONSTANTS
# =====================================================
eV2au  = 0.036749405469679
meV2au = 0.036749405469679 / 1000
fs2au  = 41.341                           # 1 fs = 41.341 a.u.
ps2au  = 41.341 * 1000                          # 1 fs = 41.341 a.u.
cm2au  = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
autoK  = 3.1577464e+05 
temp   = 300 / autoK
beta   = 1 / temp 
β      = beta
# =====================================================
# PARAMETERS
# =====================================================
N      = int(1E7)
ω0     = 1000 * cm2au
ωc     = ω0 
σ      = 20 * cm2au
ε      = 0.01 * cm2au 

# RABI SPLITTING VALUES
ΩR     = 50 * cm2au 
g0     = ΩR / (2 * N**0.5) * np.ones(N)
print(f'Number of molecules    → {N:.1e}')
print(f'σ                      → {σ/cm2au:.3f} cm⁻¹')
print(f'Perturbation Par.      → {g0[0]/ε:.3f} cm⁻¹')
print(f'g0                     → {g0[0]/cm2au:.6f} cm⁻¹')
print('======================================================')
Min     = False
verbose = False                                              # PRINT ERROR AT EACH STEP OF MINIMIZATION
Plot    = False
# =====================================================
# MINIMIZATION
# =====================================================
if __name__ == "__main__":
    if Min:
        st_time = time.time()
        # CREATE FREQUENCY DISTRIBUTION → GAUSSIAN
        ω_lst, δ0, n_exc = GaussDist(σ,ε)
        n_target            = n_exc                                             # TARGET NUMBER OF EXCITATIONS
        tol                 = 1E-11                                             # TOLERANCE
        max_iter            = 500                                               # MAXIMUM NUMBER OF ITERATIONS
        mix                 = 0.6                                               # MIXING BETWEEN NEW AND PREVIOUS VALUES OF OPTIMIZED VARIABLES
        # MINIMIZATION
        μ0, Sg0, Sgd0    = InitCond(ω_lst,δ0,g0,n_exc,σ)
        μ, un, vn, Δa, Ea, Sg, Sgd, err = MinFunc(μ0, Sg0, Sgd0, n_target, tol, max_iter, mix, verbose, g0)
        print(f'Σv_n²       → {np.sum(vn**2):.4f}')
        print(f'Num. Exc.   → {n_exc:.4f}')
        print(f'μ           → {μ/cm2au:.3} cm⁻¹')
        print(f'Error       → {err:.3e}')
        print('======================================================')
        np.save('Par.npy', np.c_[μ/cm2au,Sg,Sgd])
        np.save('Numerical_Data.npy', np.c_[ω_lst/cm2au, un, vn, Δa/cm2au, Ea/cm2au])
        fn_time = time.time()
        print(f'Time → {(fn_time - st_time)/60:.3} min')
    # =====================================================
    # PLOTTING
    # =====================================================
    col = ['#3498db', '#9b59b6', '#e74c3c', '#e67e22', '#34495e', '#1abc9c']
    if Plot:
        # =====================================================
        # HISTOGRAM
        # =====================================================
        fig, ax = plt.subplots(figsize=(3,3), dpi=150)
        count, bins, ignored = plt.hist(ω_lst/cm2au, 70, density=True, color=f'{col[0]}')#, label = f'{np.round(ΩR_lst[j]/cm2au,1)}')
        plt.plot(bins, 1/(σ/cm2au * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - (ω0/cm2au))**2 / (2 * (σ/cm2au)**2) ),
                linewidth=1, color='black')
        plt.axvline(ωc/cm2au, ls = '--', c = 'black', lw =1, label = '$\omega_c$')
        plt.xlabel('$\omega$ (cm⁻¹)')
        plt.legend(frameon=False, fontsize=5, handlelength=1.5, handletextpad=0.5)
        plt.tight_layout()
        plt.savefig('./Images/FreqDist.png', dpi = 300, bbox_inches = 'tight')