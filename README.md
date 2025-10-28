# BCS_VSC
Minimization of the Bardeen–Cooper–Schrieffer (BCS) like equations for the Tavis-Cummings Hamiltonian for Vibrational Strong Coupling (VSC). 

This code produces Figure 1 of the paper **"Vibrational Strong Coupling in Cavity QED forms a Macroscopic Quantum State"** 
$⟹$ https://chemrxiv.org/engage/chemrxiv/article-details/68e9945e5dd091524fdf4fbb 
---

We solve the BCS equations by minimizing a fixed-point map over the global parameters $ g_{\mu m}^{\ast} v_m $.
To achieve this, we define the auxiliary variables

$$
\begin{aligned}
S_g &= \sum_m g_{\mu m}^{\ast} v_m, \\
S_{g/\delta} &= \sum_m \frac{g_{\mu m}^{\ast} v_m}{\delta_m}.
\end{aligned}
$$
Then the gap equation can be rewritten as
$$ 
\begin{aligned}
\Delta_n &= \sum_{m\neq n}G_{nm}u_m^*v_m  = \frac{g_n}{2} \left(\sum_m \frac{g_mu_m^*v_m}{\delta_n} + \sum_m \frac{g_mu_m^*v_m}{\delta_m}\right) = \frac{g_n}{2} \left(\frac{S_g}{\delta_n} + S_{g/\delta}\right).
\end{aligned}
$$

One can thus minimize \( S_g \) and \( S_{g/\delta} \) while enforcing that the
expectation value of the phonon excitation number \( \bar{n} \) is constrained to the Boltzmann distribution value:

$$
\sum_n |v_n|^2
= \sum_n \frac{e^{-\beta\hbar\widetilde{\omega}_n}}{1 + e^{-\beta\hbar\widetilde{\omega}_n}}
= \frac{1}{2}\!\left(1 - \frac{\widetilde{\omega}_n - \mu}{E_n}\right).
$$

We achieve this conditional minimization by monotone bisection of the chemical potential $\mu$. Given the terms $\mu$, $S_g$ and $S_{g,\delta}$, the gaps are updated according to Eq. 2 and the variable $u_n^*v_n$ is recomputed for each cycle as
$$
\begin{aligned}
u_n^*v_n = \frac{\Delta_n}{2E_n}, \quad E_n = \sqrt{(\widetilde{\omega}_n-\mu)^2+\Delta_n^2},
\end{aligned}
$$

which is then used to update $S_g$ and $S_{g,\delta}$. The update of each of the minimized variables is done through linear mixing, where the old and new values of the given variable are combined according to the mixing value $\alpha$ as $x = (1 - \alpha)\cdot x^\mathrm{old} + \alpha\cdot  x^\mathrm{new}$, to improve stabilization.
This process is repeated until $\max\left(|\Delta\mu|,|\Delta S_g|,|S_{g/\delta}|\right) < \epsilon$ and $|\sum_n |v_n|^2 - \bar{n}|< \epsilon$, where $\epsilon$ is the tolerance parameter.

To perform the simulations, we sample $N$ vibrational frequencies $\omega_n$ from a Gaussian distribution with standard deviation $\sigma$. The Schrieffer–Wolff transformation used in this paper requires the perturbation parameter $|\lambda_{n}|=\left|g_{n}/\delta_{n}\right| \ll 1$, therefore, during the sampling we only keep the frequencies that fulfill the condition $|\omega_n - \omega_c| > g_n$.

