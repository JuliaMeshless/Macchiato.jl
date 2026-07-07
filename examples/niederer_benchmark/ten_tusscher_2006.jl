# ten Tusscher & Panfilov (2006) epicardial cell model
# "Alternans and spiral breakup in a human ventricular tissue model"
# Am J Physiol Heart Circ Physiol 291:H1088-H1100
#
# 19 state variables per point, stored as a flat vector:
#   [V(1:N), Xr1, Xr2, Xs, m, h, j, d, f, f2, fCass, s, r, Ca_i, Ca_SR, Ca_ss, R_prime, Na_i, K_i]

const NSTATES = 19

# Physical constants
const R_gas = 8314.472   # mJ/(mol·K)
const Temp = 310.0       # K
const FF = 96485.3415    # C/mol
const RTOF = R_gas * Temp / FF
const FORT = 1.0 / RTOF

# Cell geometry
const Cm_cell = 0.185    # µF/cm²
const V_c = 16404.0      # µm³
const V_sr = 1094.0      # µm³
const V_ss = 54.68       # µm³

# External concentrations (mM)
const K_o = 5.4
const Na_o = 140.0
const Ca_o = 2.0

# Maximum conductances — epicardial variant
const G_Na = 14.838
const G_bNa = 0.00029
const G_CaL = 3.98e-5
const G_bCa = 0.000592
const G_to = 0.294       # epicardial
const G_Ks = 0.392       # epicardial
const G_Kr = 0.153
const G_K1 = 5.405
const G_pCa = 0.1238
const G_pK = 0.0146
const P_kna = 0.03

# Pump and exchanger
const P_NaK = 2.724
const K_mK = 1.0
const K_mNa = 40.0
const k_NaCa = 1000.0
const K_sat = 0.1
const α_NaCa = 2.5
const γ_NaCa = 0.35
const Km_Ca = 1.38
const Km_Nai = 87.5
const K_pCa = 0.0005

# Calcium dynamics
const k1_prime = 0.15
const k2_prime = 0.045
const k3 = 0.06
const k4 = 0.005
const EC = 1.5
const max_sr = 2.5
const min_sr = 1.0
const V_rel = 0.102
const V_xfer = 0.0038
const K_up = 0.00025
const V_leak = 0.00036
const Vmax_up = 0.006375

# Buffering
const Buf_c = 0.2
const K_buf_c = 0.001
const Buf_sr = 10.0
const K_buf_sr = 0.3
const Buf_ss = 0.4
const K_buf_ss = 0.00025

"""
    tt06_initial_conditions(N::Int) -> Vector{Float64}

Return the initial state vector for N spatial points (19N entries).
Values from Niederer et al. (2011), Phil. Trans. R. Soc. A 369:4331-4351, Table 2.
"""
function tt06_initial_conditions(N::Int)
    u = zeros(NSTATES * N)
    for i in 1:N
        u[i] = -85.23            # V (mV)
        u[1N + i] = 0.00621      # Xr1
        u[2N + i] = 0.4712       # Xr2
        u[3N + i] = 0.0095       # Xs
        u[4N + i] = 0.00172      # m
        u[5N + i] = 0.7444       # h
        u[6N + i] = 0.7045       # j
        u[7N + i] = 3.373e-5     # d
        u[8N + i] = 0.7888       # f
        u[9N + i] = 0.9755       # f2
        u[10N + i] = 0.9953      # fCass
        u[11N + i] = 0.999998    # s
        u[12N + i] = 2.42e-8     # r
        u[13N + i] = 0.000126    # Ca_i (mM)
        u[14N + i] = 3.64        # Ca_SR (mM)
        u[15N + i] = 0.00036     # Ca_ss (mM)
        u[16N + i] = 0.9073      # R_prime
        u[17N + i] = 8.604       # Na_i (mM)
        u[18N + i] = 136.89      # K_i (mM)
    end
    return u
end

"""
    tt06_reaction_point!(du, u, i, N, stim_dvdt)

Compute the TT06 reaction for a single spatial point `i`. Writes derivatives into
`du` at the appropriate flat-vector indices. `stim_dvdt` is the pre-computed stimulus
contribution to dV/dt (zero when not stimulated or outside the stimulus region).
"""
function tt06_reaction_point!(du, u, i, N, stim_dvdt)
    @inbounds begin
        V = u[i]
        # Clamp gates and concentrations into physical ranges. Forward Euler on the
        # TT06 fast gates/ion concentrations can undershoot by O(1e-10)–O(1e-300)
        # per step, which then raises DomainError in downstream `log`. Clamping is
        # the standard cardiac-modelling workaround when using explicit integration.
        Xr1 = clamp(u[1N + i], 0.0, 1.0)
        Xr2 = clamp(u[2N + i], 0.0, 1.0)
        Xs = clamp(u[3N + i], 0.0, 1.0)
        m_gate = clamp(u[4N + i], 0.0, 1.0)
        h_gate = clamp(u[5N + i], 0.0, 1.0)
        j_gate = clamp(u[6N + i], 0.0, 1.0)
        d_gate = clamp(u[7N + i], 0.0, 1.0)
        f_gate = clamp(u[8N + i], 0.0, 1.0)
        f2gate = clamp(u[9N + i], 0.0, 1.0)
        fCass = clamp(u[10N + i], 0.0, 1.0)
        s_gate = clamp(u[11N + i], 0.0, 1.0)
        r_gate = clamp(u[12N + i], 0.0, 1.0)
        Ca_i = max(u[13N + i], 1.0e-10)
        Ca_SR = max(u[14N + i], 1.0e-10)
        Ca_ss = max(u[15N + i], 1.0e-10)
        R_prim = clamp(u[16N + i], 0.0, 1.0)
        Na_i = max(u[17N + i], 1.0e-10)
        K_i = max(u[18N + i], 1.0e-10)

        # Nernst potentials
        E_Na = RTOF * log(Na_o / Na_i)
        E_K = RTOF * log(K_o / K_i)
        E_Ks = RTOF * log((K_o + P_kna * Na_o) / (K_i + P_kna * Na_i))
        E_Ca = 0.5 * RTOF * log(Ca_o / Ca_i)

        # --- Ionic currents ---

        # Fast Na+ current
        I_Na = G_Na * m_gate^3 * h_gate * j_gate * (V - E_Na)

        # Background Na+ current
        I_bNa = G_bNa * (V - E_Na)

        # L-type Ca2+ current (GHK formulation)
        Veff = V - 15.0
        if abs(Veff) < 1.0e-6
            # L'Hôpital limit at Veff → 0: Veff/(exp(2*Veff*FORT)-1) → 1/(2*FORT)
            I_CaL = G_CaL * d_gate * f_gate * f2gate * fCass *
                2.0 * FF * (0.25 * Ca_ss * exp(2.0 * Veff * FORT) - Ca_o)
        else
            exp_2Veff = exp(2.0 * Veff * FORT)
            I_CaL = G_CaL * d_gate * f_gate * f2gate * fCass *
                4.0 * Veff * FF^2 / (R_gas * Temp) *
                (0.25 * Ca_ss * exp_2Veff - Ca_o) /
                (exp_2Veff - 1.0)
        end

        # Background Ca2+ current
        I_bCa = G_bCa * (V - E_Ca)

        # Transient outward current (epicardial)
        I_to = G_to * r_gate * s_gate * (V - E_K)

        # Slow delayed rectifier K+ current
        I_Ks = G_Ks * Xs^2 * (V - E_Ks)

        # Rapid delayed rectifier K+ current
        I_Kr = G_Kr * sqrt(K_o / 5.4) * Xr1 * Xr2 * (V - E_K)

        # Inward rectifier K+ current
        α_K1 = 0.1 / (1.0 + exp(0.06 * (V - E_K - 200.0)))
        β_K1 = (
            3.0 * exp(0.0002 * (V - E_K + 100.0)) +
                exp(0.1 * (V - E_K - 10.0))
        ) /
            (1.0 + exp(-0.5 * (V - E_K)))
        xK1_inf = α_K1 / (α_K1 + β_K1)
        I_K1 = G_K1 * xK1_inf * (V - E_K)

        # Na+/K+ pump current
        Vfrt = V * FORT
        I_NaK = P_NaK * K_o / (K_o + K_mK) *
            Na_i / (Na_i + K_mNa) /
            (1.0 + 0.1245 * exp(-0.1 * Vfrt) + 0.0353 * exp(-Vfrt))

        # Na+/Ca2+ exchanger current
        exp_gVfrt = exp(γ_NaCa * Vfrt)
        exp_gm1Vfrt = exp((γ_NaCa - 1.0) * Vfrt)
        I_NaCa = k_NaCa *
            (exp_gVfrt * Na_i^3 * Ca_o - exp_gm1Vfrt * Na_o^3 * Ca_i * α_NaCa) /
            (
            (Km_Nai^3 + Na_o^3) * (Km_Ca + Ca_o) *
                (1.0 + K_sat * exp_gm1Vfrt)
        )

        # Sarcolemmal Ca2+ pump
        I_pCa = G_pCa * Ca_i / (K_pCa + Ca_i)

        # Plateau K+ current
        I_pK = G_pK * (V - E_K) / (1.0 + exp((25.0 - V) / 5.98))

        # Total ionic current (µA/cm²)
        I_ion = I_Na + I_bNa + I_CaL + I_bCa + I_to + I_Ks + I_Kr + I_K1 +
            I_NaK + I_NaCa + I_pCa + I_pK

        # --- Gating variable kinetics: dx/dt = (x_inf - x) / tau_x ---

        # Xr1
        xr1_inf = 1.0 / (1.0 + exp((-26.0 - V) / 7.0))
        α_xr1 = 450.0 / (1.0 + exp((-45.0 - V) / 10.0))
        β_xr1 = 6.0 / (1.0 + exp((V + 30.0) / 11.5))
        tau_xr1 = α_xr1 * β_xr1
        dXr1 = (xr1_inf - Xr1) / tau_xr1

        # Xr2
        xr2_inf = 1.0 / (1.0 + exp((V + 88.0) / 24.0))
        α_xr2 = 3.0 / (1.0 + exp((-60.0 - V) / 20.0))
        β_xr2 = 1.12 / (1.0 + exp((V - 60.0) / 20.0))
        tau_xr2 = α_xr2 * β_xr2
        dXr2 = (xr2_inf - Xr2) / tau_xr2

        # Xs
        xs_inf = 1.0 / (1.0 + exp((-5.0 - V) / 14.0))
        α_xs = 1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0))
        β_xs = 1.0 / (1.0 + exp((V - 35.0) / 15.0))
        tau_xs = α_xs * β_xs + 80.0
        dXs = (xs_inf - Xs) / tau_xs

        # m (Na activation)
        m_inf = (1.0 / (1.0 + exp((-56.86 - V) / 9.03)))^2
        α_m = 1.0 / (1.0 + exp((-60.0 - V) / 5.0))
        β_m = 0.1 / (1.0 + exp((V + 35.0) / 5.0)) +
            0.1 / (1.0 + exp((V - 50.0) / 200.0))
        tau_m = α_m * β_m
        dm = (m_inf - m_gate) / tau_m

        # h (Na fast inactivation)
        h_inf = (1.0 / (1.0 + exp((V + 71.55) / 7.43)))^2
        if V >= -40.0
            α_h = 0.0
            β_h = 0.77 / (0.13 * (1.0 + exp(-(V + 10.66) / 11.1)))
        else
            α_h = 0.057 * exp(-(V + 80.0) / 6.8)
            β_h = 2.7 * exp(0.079 * V) + 3.1e5 * exp(0.3485 * V)
        end
        tau_h = 1.0 / (α_h + β_h)
        dh = (h_inf - h_gate) / tau_h

        # j (Na slow inactivation)
        j_inf = h_inf
        if V >= -40.0
            α_j = 0.0
            β_j = 0.6 * exp(0.057 * V) / (1.0 + exp(-0.1 * (V + 32.0)))
        else
            α_j = (-2.5428e4 * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V)) *
                (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23)))
            β_j = 0.02424 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14)))
        end
        tau_j = 1.0 / (α_j + β_j)
        dj = (j_inf - j_gate) / tau_j

        # d (ICaL activation)
        d_inf = 1.0 / (1.0 + exp((-8.0 - V) / 7.5))
        α_d = 1.4 / (1.0 + exp((-35.0 - V) / 13.0)) + 0.25
        β_d = 1.4 / (1.0 + exp((V + 5.0) / 5.0))
        γ_d = 1.0 / (1.0 + exp((50.0 - V) / 20.0))
        tau_d = α_d * β_d + γ_d
        dd = (d_inf - d_gate) / tau_d

        # f (ICaL voltage inactivation)
        f_inf = 1.0 / (1.0 + exp((V + 20.0) / 7.0))
        α_f = 1102.5 * exp(-(V + 27.0)^2 / 225.0)
        β_f = 200.0 / (1.0 + exp((13.0 - V) / 10.0))
        γ_f = 180.0 / (1.0 + exp((V + 30.0) / 10.0)) + 20.0
        tau_f = α_f + β_f + γ_f
        df = (f_inf - f_gate) / tau_f

        # f2 (ICaL Ca-dependent inactivation)
        f2_inf = 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33
        α_f2 = 562.0 * exp(-(V + 27.0)^2 / 240.0)
        β_f2 = 31.0 / (1.0 + exp((25.0 - V) / 10.0))
        γ_f2 = 80.0 / (1.0 + exp((V + 30.0) / 10.0))
        tau_f2 = α_f2 + β_f2 + γ_f2
        df2 = (f2_inf - f2gate) / tau_f2

        # fCass (ICaL fast Ca-dependent inactivation in subspace)
        fCass_inf = 0.6 / (1.0 + (Ca_ss / 0.05)^2) + 0.4
        tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05)^2) + 2.0
        dfCass = (fCass_inf - fCass) / tau_fCass

        # s (Ito inactivation — epicardial)
        s_inf = 1.0 / (1.0 + exp((V + 20.0) / 5.0))
        tau_s = 85.0 * exp(-(V + 45.0)^2 / 320.0) +
            5.0 / (1.0 + exp((V - 20.0) / 5.0)) + 3.0
        ds = (s_inf - s_gate) / tau_s

        # r (Ito activation)
        r_inf = 1.0 / (1.0 + exp((20.0 - V) / 6.0))
        tau_r = 9.5 * exp(-(V + 40.0)^2 / 1800.0) + 0.8
        dr = (r_inf - r_gate) / tau_r

        # --- Calcium dynamics ---

        # Ryanodine receptor
        kcasr = max_sr - (max_sr - min_sr) / (1.0 + (EC / Ca_SR)^2)
        k1 = k1_prime / kcasr
        k2 = k2_prime * kcasr
        O = k1 * Ca_ss^2 * R_prim
        dR_prime = -k2 * Ca_ss * R_prim + k4 * (1.0 - R_prim)

        # SR Ca release, uptake, leak
        I_rel = V_rel * O * (Ca_SR - Ca_ss)
        I_up = Vmax_up / (1.0 + K_up^2 / Ca_i^2)
        I_leak = V_leak * (Ca_SR - Ca_i)
        I_xfer = V_xfer * (Ca_ss - Ca_i)

        # Buffering factors
        Ca_i_bufc = 1.0 / (1.0 + Buf_c * K_buf_c / (Ca_i + K_buf_c)^2)
        Ca_sr_bufsr = 1.0 / (1.0 + Buf_sr * K_buf_sr / (Ca_SR + K_buf_sr)^2)
        Ca_ss_bufss = 1.0 / (1.0 + Buf_ss * K_buf_ss / (Ca_ss + K_buf_ss)^2)

        # Calcium concentration ODEs
        dCa_i = Ca_i_bufc * (
            I_leak - I_up + I_xfer -
                (I_bCa + I_pCa - 2.0 * I_NaCa) * Cm_cell / (2.0 * V_c * FF) * 1.0e6
        )
        dCa_SR = Ca_sr_bufsr * (V_c / V_sr) * (I_up - I_rel - I_leak)
        dCa_ss = Ca_ss_bufss * (
            -I_CaL * Cm_cell / (2.0 * V_ss * FF) * 1.0e6 +
                I_rel * V_sr / V_ss - I_xfer * V_c / V_ss
        )

        # Sodium and potassium concentration ODEs
        dNa_i = -(I_Na + I_bNa + 3.0 * I_NaK + 3.0 * I_NaCa) *
            Cm_cell / (V_c * FF) * 1.0e6
        dK_i = -(I_K1 + I_to + I_Kr + I_Ks + I_pK - 2.0 * I_NaK) *
            Cm_cell / (V_c * FF) * 1.0e6

        # --- Write derivatives ---
        # dV/dt = -(sum of ionic currents) + stimulus. Currents are already in pA/pF
        # (≡ mV/ms); Cm_cell appears only in the concentration-update conversions
        # below. This matches the TT06 CellML membrane equation.
        du[i] = -I_ion + stim_dvdt
        du[1N + i] = dXr1
        du[2N + i] = dXr2
        du[3N + i] = dXs
        du[4N + i] = dm
        du[5N + i] = dh
        du[6N + i] = dj
        du[7N + i] = dd
        du[8N + i] = df
        du[9N + i] = df2
        du[10N + i] = dfCass
        du[11N + i] = ds
        du[12N + i] = dr
        du[13N + i] = dCa_i
        du[14N + i] = dCa_SR
        du[15N + i] = dCa_ss
        du[16N + i] = dR_prime
        du[17N + i] = dNa_i
        du[18N + i] = dK_i
    end

    return nothing
end
