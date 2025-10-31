import numpy as np
import matplotlib.pyplot as plt
import pulse_generators as pg
import gaspype as gp
from scipy.optimize import brentq


try:
    # SciPy ≥1.14
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    # SciPy <1.14
    from scipy.integrate import cumtrapz




class Ignitor():
    """
    Represents an ignitor model defining time-dependent combustion properties,
    including mass flow, specific enthalpy, enthalpy rate, and derived quantities
    such as cumulative mass and energy release. Supports gas-phase product tracking
    and thermodynamic integration over time.

    The ignitor may represent a real ignition charge or a modeled energy source,
    defined by its time-dependent thermodynamic behavior.

    Attributes
    ----------
    # --- User-defined on initialization ---
    name : str
        Descriptive name of the ignitor.
    h_curve : np.ndarray
        Specific enthalpy [J/kg] as a function of time.
    m_dot_curve : np.ndarray
        Total mass flow rate [kg/s] as a function of time.
    t_vector : np.ndarray
        Time vector [s], aligned with h_curve and m_dot_curve.
    H_dot_curve : np.ndarray
        Instantaneous enthalpy flow rate [W] = m_dot * h.

    # --- Defined after combustion setup (via setup_combustion_products) ---
    total_combustion_moles : float
        Total moles of combustion products (gas + condensed).
    total_combustion_mass : float
        Total mass [kg] of all combustion products (gas + condensed).
    total_gas_mass : float
        Total gaseous combustion mass [kg].
    gas_combustion_products_per_kg : object
        Mixture object (from `gaspype`) representing the normalized gaseous
        combustion products per kilogram of total products.
    gas_mas_ratio : float
        Ratio of gaseous mass to total combustion mass.
    gm_dot_curve : np.ndarray
        Gas-phase mass flow rate [kg/s] as a function of time.

    # --- Derived cumulative quantities (computed via compute_mass_integrals_arrays) ---
    mass_integral_curve : np.ndarray
        Cumulative total combusted mass [kg] vs. time.
    gas_mass_integral_curve : np.ndarray
        Cumulative released gas mass [kg] vs. time.
    H_integral_curve : np.ndarray
        Cumulative enthalpy release [J] vs. time.

    # --- Derived analysis quantities ---
    get_m_dot(t) : float or np.ndarray
        Interpolated total mass flow [kg/s] at time t.
    get_gm_dot(t) : float or np.ndarray
        Interpolated gas mass flow [kg/s] at time t.
    get_h(t) : float or np.ndarray
        Interpolated specific enthalpy [J/kg] at time t.
    get_H_dot(t) : float or np.ndarray
        Interpolated enthalpy rate [W] at time t.
    get_m_integral(t) : float or np.ndarray
        Interpolated total combusted mass [kg] up to time t.
    get_gm_integral(t) : float or np.ndarray
        Interpolated released gas mass [kg] up to time t.
    get_H_integral(t) : float or np.ndarray
        Interpolated total enthalpy release [J] up to time t.
    get_m_interval(t1, t2) : float
        Total mass combusted [kg] between t1 and t2.
    get_gm_interval(t1, t2) : float
        Gas mass released [kg] between t1 and t2.
    get_H_interval(t1, t2) : float
        Enthalpy released [J] between t1 and t2.
    get_fluid_interval(t1, t2) : object
        Mixture object representing combustion gas released between t1 and t2.

    # --- Visualization ---
    plot_ignitor_curves()
        Plots mass flow, specific enthalpy, and power vs. time.
        If available, also plots cumulative mass release curves.

    Notes
    -----
    - The ignitor supports both solid and gaseous combustion components.
    - Integration is performed using SciPy’s trapezoidal integration (`cumtrapz` or `cumulative_trapezoid`).
    - Time interpolation uses `numpy.interp` with zero-extrapolation outside the domain.
    - The class assumes the combustion product object supports scalar multiplication
      and provides a `.get_mass()` method for normalization.
    """
    def __init__(self, h_curve, m_dot_curve, t_vector,name="Unconspicuous ignitor", initial_T=293.15):
        """
                Initialize the ignitor with given time-dependent curves.

                Parameters
                ----------
                h_curve : np.ndarray
                    Specific enthalpy [J/kg] vs. time.
                m_dot_curve : np.ndarray
                    Mass flow rate [kg/s] vs. time.
                t_vector : np.ndarray
                    Time vector [s], same length as h_curve and m_dot_curve.
                name : str, optional
                    Descriptive name for this ignitor instance.
                """
        self.name = name
        self.h_curve = h_curve
        self.m_dot_curve = m_dot_curve
        self.t_vector = t_vector
        self.H_dot_curve = np.multiply(m_dot_curve, self.h_curve)
        self.cp = None

    def plot_ignitor_curves(self):
        """
        Plot the ignitor curves vs. time:
          - Top:    Mass flow rate (kg/s)
          - Middle: Specific enthalpy (J/kg)
          - Bottom: Power (W = kg/s * J/kg)

        If available, also plots the cumulative integrals of:
          - total combusted mass
          - released gas mass

        Inputs
        ------
        None

        Outputs
        -------
        None
        (Displays matplotlib plots.)
        """
        t = self.t_vector
        m = self.m_dot_curve
        h = self.h_curve
        Hdot = self.H_dot_curve

        # --- Main 3 plots: m_dot, h, and power ---
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
        fig.suptitle(self.name, fontsize=14)

        axs[0].plot(t, m)
        axs[0].set_ylabel('mass flow (kg/s)')
        axs[0].grid(True)

        axs[1].plot(t, h)
        axs[1].set_ylabel('specific enthalpy (J/kg)')
        axs[1].grid(True)

        axs[2].plot(t, Hdot)
        axs[2].set_ylabel('power (W)')
        axs[2].set_xlabel('time (s)')
        axs[2].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # --- Cumulative integrals plot (if available) ---
        if hasattr(self, "mass_integral_curve"):
            plt.figure(figsize=(8, 4))
            plt.plot(t, self.mass_integral_curve, label="Total mass combusted")
            if hasattr(self, "gas_mass_integral_curve"):
                plt.plot(t, self.gas_mass_integral_curve, label="Gas mass released")
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative mass (kg)")
            plt.title(f"{self.name} – Cumulative Combustion Mass")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ No integral curves computed yet. Run setup_combustion_products() first.")

    def setup_combustion_products(self, combustion_products, total_combustion_moles, total_combustion_mass,
                                  total_gas_mass):
        """
        Define combustion product properties and compute gas-phase ratios.

        Inputs
        ------
        combustion_products : object
            Gas-phase mixture object representing combustion products.
            Must implement `get_mass()` and scalar multiplication.
        total_combustion_moles : float
            Total moles of products generated per ignition event.
        total_combustion_mass : float
            Total mass [kg] of all combustion products (gas + condensed).
        total_gas_mass : float
            Total gas-phase mass [kg] of combustion products.

        Outputs
        -------
        None
        (Sets internal attributes and computes gas mass flow curves.)
        """
        self.total_combustion_moles = total_combustion_moles
        self.total_combustion_mass = total_combustion_mass
        self.total_gas_mass = total_gas_mass

        # Start with gaseous products mixture
        self.gas_combustion_products_per_kg = combustion_products

        # Mass of this defined gas mixture (before scaling)
        mass_computed = self.gas_combustion_products_per_kg.get_mass()

        # Mass of gas mixture per kg of total combustion
        gas_mass_per_kg_total = total_gas_mass / total_combustion_mass
        self.gas_mas_ratio = gas_mass_per_kg_total
        self.gm_dot_curve = self.m_dot_curve*self.gas_mas_ratio

        # Scale so the fluid corresponds to the correct gas mass per kg of total mixture
        scaling_factor = gas_mass_per_kg_total / mass_computed
        self.gas_combustion_products_per_kg = self.gas_combustion_products_per_kg * scaling_factor
        self.compute_mass_integrals_arrays()
        print("new mass", self.gas_combustion_products_per_kg.get_mass())

    def get_m_dot(self, t):
        """
        Return the instantaneous total mass flow rate [kg/s] at a given time.

        Inputs
        ------
        t : float or np.ndarray
            Time(s) [s] at which to evaluate the mass flow.

        Outputs
        -------
        m_dot : float or np.ndarray
            Interpolated mass flow rate [kg/s].
        """
        return np.interp(t, self.t_vector, self.m_dot_curve, left=0.0, right=0.0)

    def get_gm_dot(self, t):
        """
                Return the instantaneous gas-phase mass flow rate [kg/s] at a given time.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate gas mass flow.

                Outputs
                -------
                gm_dot : float or np.ndarray
                    Interpolated gas-phase mass flow rate [kg/s].
                """
        return np.interp(t, self.t_vector, self.gm_dot_curve, left=0.0, right=0.0)

    def get_gm_integral(self, t):
        """
                Return the cumulative gas mass released [kg] at a given time.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate the cumulative gas mass.

                Outputs
                -------
                gm_integral : float or np.ndarray
                    Interpolated cumulative gas mass [kg].
                """
        return np.interp(t, self.t_vector, self.gas_mass_integral_curve, left=0.0, right=0.0)

    def get_m_integral(self, t):
        """
                Return the cumulative total combusted mass [kg] at a given time.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate cumulative mass.

                Outputs
                -------
                m_integral : float or np.ndarray
                    Interpolated cumulative total mass [kg].
                """
        return np.interp(t, self.t_vector, self.mass_integral_curve, left=0.0, right=0.0)

    def get_H_integral(self, t):
        """
                Return the cumulative enthalpy release [J] at a given time.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate cumulative enthalpy.

                Outputs
                -------
                H_integral : float or np.ndarray
                    Interpolated cumulative enthalpy [J].
                """
        return np.interp(t, self.t_vector, self.H_integral_curve, left=0.0, right=0.0)

    def get_h(self, t):
        """
                Return the specific enthalpy [J/kg] at a given time.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate the enthalpy.

                Outputs
                -------
                h : float or np.ndarray
                    Interpolated specific enthalpy [J/kg].
                """
        return np.interp(t, self.t_vector, self.h_curve, left=self.h_curve[0], right=self.h_curve[-1])

    def get_H_dot(self, t):
        """
                Return the instantaneous enthalpy rate (power) [W] = m_dot * h.

                Inputs
                ------
                t : float or np.ndarray
                    Time(s) [s] at which to evaluate power.

                Outputs
                -------
                H_dot : float or np.ndarray
                    Interpolated enthalpy rate [W].
                """
        return np.interp(t, self.t_vector, self.H_dot_curve, left=0.0, right=0.0)

    def compute_mass_integrals_arrays(self):
        """
                Compute cumulative integrals of:
                  - total combusted mass
                  - released gas mass
                  - total enthalpy release

                over time using trapezoidal integration.

                Inputs
                ------
                None
                (Uses existing self.t_vector, self.m_dot_curve, self.H_dot_curve.)

                Outputs
                -------
                None
                (Stores results as arrays aligned with self.t_vector:)
                    self.mass_integral_curve [kg]
                    self.gas_mass_integral_curve [kg]
                    self.H_integral_curve [J]
                """
        t = self.t_vector
        m_dot = self.m_dot_curve
        gm_dot = getattr(self, "gm_dot_curve", None)
        H_dot = self.H_dot_curve

        # Integrate mass flow curves with respect to time
        self.mass_integral_curve = np.concatenate(
            ([0.0], cumtrapz(m_dot, t))
        )

        self.H_integral_curve = np.concatenate(
            ([0.0], cumtrapz(H_dot, t))
        )

        if gm_dot is not None:
            self.gas_mass_integral_curve = np.concatenate(
                ([0.0], cumtrapz(gm_dot, t))
            )
        else:
            self.gas_mass_integral_curve = np.zeros_like(t)

        print("Total mass combusted:", self.mass_integral_curve[-1], "kg")
        print("Total gas mass released:", self.gas_mass_integral_curve[-1], "kg")

    def get_H_interval(self, t1, t2):
        """
                Return total enthalpy released [J] between two times.

                Inputs
                ------
                t1 : float
                    Start time [s].
                t2 : float
                    End time [s].

                Outputs
                -------
                delta_H : float
                    Enthalpy released [J] between t1 and t2.
                """
        return self.get_H_integral(t2) - self.get_H_integral(t1)
    def get_m_interval(self, t1, t2):
        """
                Return total mass combusted [kg] between two times.

                Inputs
                ------
                t1 : float
                    Start time [s].
                t2 : float
                    End time [s].

                Outputs
                -------
                delta_H : float
                    Mass combusted [kg] between t1 and t2.
                """
        return self.get_m_integral(t2) - self.get_m_integral(t1)
    def get_gm_interval(self, t1, t2):
        """
                Return total gas mass released [kg] between two times.

                Inputs
                ------
                t1 : float
                    Start time [s].
                t2 : float
                    End time [s].

                Outputs
                -------
                delta_gm : float
                    Gas mass released [kg] between t1 and t2.
                """
        return self.get_gm_integral(t2) - self.get_gm_integral(t1)


    def get_fluid_interval(self, t1, t2):
        """
                Return the equivalent gas mixture (combustion products) for the
                enthalpy released between two time points.

                Inputs
                ------
                t1 : float
                    Start time [s].
                t2 : float
                    End time [s].

                Outputs
                -------
                fluid : object
                    Scaled combustion product mixture corresponding to mass
                    released between t1 and t2.
                """
        return self.gas_combustion_products_per_kg*self.get_m_interval(t1, t2)
    def set_cp(self, cp):
        self.cp = cp
    def set_conductivity_factor(self, conductivity_factor):
        self.conductivity_factor = conductivity_factor

class Nozzle:
    def __init__(self, A_throat, A_chamber, A_exit, name="Unconspicuous Nozzle"):
        self.name = name
        self.A_throat = A_throat
        self.A_chamber = A_chamber
        self.A_exit = A_exit

    def get_mach(self, P_t, P, gamma, R_bar):
        """Return Mach number from stagnation and static pressures."""
        self.gamma = gamma
        self.R_bar = R_bar
        M = np.sqrt((2 / (gamma - 1)) * ((P_t / P) ** ((gamma - 1) / gamma) - 1))
        return M

    def _A_over_Astar(self, M, gamma):
        """Area/Mach relation A/A* as a function of M (works for M>0)."""
        return (1.0 / M) * ((2.0 / (gamma + 1.0) * (1.0 + (gamma - 1.0) / 2.0 * M**2))
                             ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))

    def get_M_throat(self, P_t, P_e, gamma):
        """
        Robust computation of throat Mach (subsonic root) for an unchoked nozzle.
        Returns M_throat (<1) or 1.0 if the nozzle is choked.
        """

        # --- 0. Basic checks ---
        if P_t <= 0 or P_e <= 0:
            raise ValueError("Pressures must be positive.")

        # --- 1. Check choking condition ---
        crit_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))  # p_e/p_t_crit
        # choked if p_e/p_t <= crit_ratio  <=>  P_t/P_e >= 1/crit_ratio
        if (P_e / P_t) <= crit_ratio:
            # Choked: throat Mach = 1
            self.M_e = None
            self.M_throat = 1.0
            return 1.0

        # --- 2. Compute exit Mach number M_e from isentropic relation ---
        # Guard against tiny numerical negatives inside sqrt
        ratio = (P_t / P_e)
        if ratio <= 1.0:
            # physically the flow is extremely weak; set M_e ~ 0 (practically)
            M_e = 1e-8
        else:
            val = (2.0 / (gamma - 1.0)) * (ratio ** ((gamma - 1.0) / gamma) - 1.0)
            if val <= 0.0:
                M_e = 1e-8
            else:
                M_e = np.sqrt(val)

        # --- 3. Compute A* from the exit condition:
        A_e_over_Astar = self._A_over_Astar(M_e, gamma)
        if A_e_over_Astar <= 0:
            raise RuntimeError("Computed non-positive A_e/A*; check inputs.")
        A_star = self.A_exit / A_e_over_Astar

        # target area ratio we need the throat to have:
        target = self.A_throat / A_star
        print(target)
        # if target <= 1:
        #     target = 1

        # --- 4. Solve for the subsonic Mach M_t such that A/A* = target ---
        # The A/A* function is >0 and has two branches. We want the subsonic root in (0,1).
        def resid(M):
            return self._A_over_Astar(M, gamma) - target

        # bracket in (M_low, M_high) with M_low very small, M_high slightly <1
        M_low = 1e-8
        M_high = 1.0 - 1e-8

        # Ensure there is a sign change on the chosen bracket; if not, raise informative error
        f_low = resid(M_low)
        f_high = resid(M_high)
        if np.isnan(f_low) or np.isnan(f_high):
            raise RuntimeError("Residue evaluated to NaN; check gamma and geometry.")
        if f_low * f_high > 0:
            # No sign change: likely unphysical geometry or pressures, or extremely near-choke.
            # Provide helpful diagnostics instead of silent failure.
            raise RuntimeError(
                "Unable to find subsonic throat Mach: resid has same sign at bracket ends.\n"
                f"Diagnostics: target A/A* = {target:.6g}, A_e/A* (from exit) = {A_e_over_Astar:.6g}, "
                f"M_e = {M_e:.6g}."
            )

        M_throat = brentq(resid, M_low, M_high, xtol=1e-12, rtol=1e-10, maxiter=100)

        # store and return
        self.M_e = M_e
        self.M_throat = M_throat
        return M_throat

    def get_m_dot(self, P_t, T_t, R_bar, P_e, gamma):
        """
        Compute mass flow rate through the throat.
        """
        self.gamma = gamma
        self.R_bar = R_bar

        # Compute Mach numbers
        self.Me = self.get_mach(P_t, P_e, gamma=self.gamma, R_bar=self.R_bar)
        self.M_throat = self.get_M_throat(P_t, P_e, gamma=self.gamma)

        # --- Mass flow formula
        M = self.M_throat
        m_dot = ((self.A_throat * P_t) / np.sqrt(T_t)) * np.sqrt(self.gamma / self.R_bar) * \
                M * (1 + (self.gamma - 1) / 2 * M ** 2) ** (-(self.gamma + 1) / (2 * (self.gamma - 1)))

        self.m_dot = m_dot
        return m_dot

    import numpy as np
    import matplotlib.pyplot as plt

    def plot_nozzle_performance_vs_chamber(self, P_e, T_t, R_bar, gamma, P_t_range):
        """
        Plot Mach number (throat & exit) and mass flow vs chamber (stagnation) pressure.

        Parameters
        ----------
        P_e : float
            Exit/ambient static pressure [Pa]
        T_t : float
            Chamber (stagnation) temperature [K]
        R_bar : float
            Gas constant [J/(kg*K)]
        gamma : float
            Ratio of specific heats
        P_t_range : array-like
            Range of chamber pressures [Pa] to evaluate
        """
        self.gamma = gamma
        self.R_bar = R_bar
        M_e_list = []
        M_t_list = []
        m_dot_list = []
        ratios = np.array(P_t_range) / P_e  # for optional plotting on same axis

        for P_t in P_t_range:
            # Exit Mach (isentropic)
            M_e = self.get_mach(P_t, P_e, gamma=self.gamma, R_bar=self.R_bar)

            # Throat Mach (subsonic branch before choking)
            try:
                M_t = self.get_M_throat(P_t, P_e, gamma)
            except Exception:
                M_t = np.nan

            # If nozzle is choked (M_t -> 1), clamp value
            if M_t >= 1.0:
                M_t = 1.0

            # Mass flow rate at throat
            m_dot = self.get_m_dot(P_t, P_t, R_bar, P_e, gamma)

            M_e_list.append(M_e)
            M_t_list.append(M_t)
            m_dot_list.append(m_dot)

        # --- Compute critical choking pressure ratio ---
        crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        P_t_choke = P_e / crit_ratio

        # --- Plot setup ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(P_t_range, M_t_list, label="M_throat", color='tab:blue', linewidth=2)
        ax1.plot(P_t_range, M_e_list, label="M_exit", color='tab:cyan', linestyle='--', linewidth=2)
        ax1.axvline(P_t_choke, color='gray', linestyle=':', label='Choking onset')

        ax1.set_xlabel(r"Chamber Pressure $P_t$ [Pa]")
        ax1.set_ylabel("Mach number", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # --- Mass flow curve on right axis ---
        ax2 = ax1.twinx()
        ax2.plot(P_t_range, m_dot_list, label=r"Mass Flow $\dot{m}$", color='tab:red', linewidth=2)
        ax2.set_ylabel(r"Mass Flow Rate $\dot{m}$ [kg/s]", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"Nozzle Performance vs Chamber Pressure — {self.name}")
        plt.tight_layout()
        plt.show()

# Testing code for the Ignitor class:

# t_vector_example = np.linspace(0, 2, 2000)
# m_dot_curve_example = pg.smooth_pulse(t_vector_example, 2, 0.2, 0.2, 3/1000) # [kg/s]
# h_curve_example = np.full(t_vector_example.shape, 2.12*1000*1000) # Black powder constant enthalpy generation
#
# ign = Ignitor(h_curve_example, m_dot_curve_example, t_vector_example,name="Test ignitor")
#
# # Set up the ignition gas mixture (CO2 + N2)
# combustion_moles_example = 16
# combustion_products_mass = 1.203   # total combustion products (kg)
# combustion_products_gas_mass = 0.404   # gaseous part only (kg)
#
# # If the massflow was measured only using the gas part then all the mass is gas
#
# products_example = gp.fluid({
#     'CO2': 6/combustion_moles_example,
#     'N2': 5/combustion_moles_example
# })
#
# # Run the setup method
# ign.setup_combustion_products(
#     combustion_products=products_example,
#     total_combustion_moles=combustion_moles_example,
#     total_combustion_mass=combustion_products_mass,
#     total_gas_mass=combustion_products_gas_mass
# )
#
# ign.plot_ignitor_curves()
#
# print(ign.get_m_dot(np.pi/2))   # mass flow
# print(ign.get_h(np.pi/2))       # specific enthalpy
# print(ign.get_H_dot(np.pi/2))   # power

# Nozzle testing code

gamma = 1.4
R = 287
P_e = 101325       # ambient pressure [Pa]
T_t = 300          # K
A_t = 0.0005       # m²
A_e = 0.0008       # m²
A_c = 0.0010       # m²

nzl = Nozzle(A_t, A_c, A_e)

# Sweep chamber pressure from 0.5 atm up to ~10 atm
P_t_range = np.linspace(1.1e5, 2e5, 1000)

nzl.plot_nozzle_performance_vs_chamber(P_e, T_t, R, gamma, P_t_range)
