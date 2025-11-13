import numpy as np
import matplotlib.pyplot as plt
import pulse_generators as pg
import gaspype as gp
from scipy.optimize import brentq
from scipy.optimize import root_scalar


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
        self.T = initial_T
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
    def set_timing(self, ignition_start):
        self.ignition_start = ignition_start
        self.compute_mass_integrals_arrays()
    def update(self, t, dt):
        if t>self.ignition_start:
            t = t-self.ignition_start
            m_released = self.get_gm_interval(t, t+dt)
            fluid_released = self.get_fluid_interval(t, t+dt)
            remaining_ignitor_mass = self.get_m_integral(t+dt)
            heat_released = self.get_H_interval(t, t+dt)
        else:
            m_released = 0
            fluid_released = None
            remaining_ignitor_mass = None
            heat_released = 0
        return m_released, fluid_released, remaining_ignitor_mass, heat_released



class Nozzle():
    def __init__(self, A_throat, A_chamber, A_exit , name="Unconspicuous Nozzle"):
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
    def mach_from_area_ratio(self, A_Astar, gamma, bracket=(1e-6, 1.0) ):
        # if A_Astar <1:
        #     raise RuntimeError("A_Astar must be greater than 1.")
        def f(M):
            term = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M ** 2)
            return (1 / M ** 2) * (term ** ((gamma + 1) / (gamma - 1))) - (A_Astar) ** 2


        sol = root_scalar(f, bracket=bracket, method='brentq')
        # if not sol.converged:
        #     raise RuntimeError("Root solver did not converge")
        # return sol.root

        # M_range = np.linspace(0.1, 2, 500)
        # f_values = [f(M) for M in M_range]
        #
        # plt.figure(figsize=(7, 4))
        # plt.plot(M_range, f_values, label=r"$f(M)$")
        # if sol.converged:
        #     plt.plot(sol.root, 0, "or")
        # plt.axhline(0, color='k', linestyle='--', linewidth=1)
        # plt.title(f"f(M) vs M  |  A/A*={A_Astar}, γ={gamma}")
        # plt.xlabel("Mach number M")
        # plt.ylabel(
        #     r"$f(M) = \frac{1}{M^2}\left[\frac{2}{\gamma+1}(1+\frac{\gamma-1}{2}M^2)\right]^{\frac{\gamma+1}{\gamma-1}} - (A/A^*)^2$")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.show()
        return sol.root
    def _A_over_Astar(self, M, gamma):
        """Area/Mach relation A/A* as a function of M (works for M>0)."""
        term = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M ** 2)
        A_over_A_star = np.sqrt((1 / M ** 2) * (term ** ((gamma + 1) / (gamma - 1))))
        return A_over_A_star
    def get_A_star(self, A, gamma, M):
        self.gamma = gamma
        A_over_Astar = self._A_over_Astar(M, gamma)
        A_star = A/A_over_Astar
        return A_star
    def get_M_throat(self, P_t, P_e, gamma, R_bar):
        """
        Return throat Mach number.
        """
        self.R_bar = R_bar
        M_e = self.get_M_e_opt(gamma, P_t, P_e)
        A_star = self.get_A_star(self.A_exit, gamma, M_e)
        # print(self.A_throat/ A_star)
        if A_star > self.A_throat:
            M_throat = 1
        else:
            M_throat = self.mach_from_area_ratio(self.A_throat/ A_star, gamma)
        # print(M_throat)
        return M_throat
    def get_m_dot(self, P_t, T_t, R_bar, P_e, gamma):
        """
        Compute mass flow rate through the throat.
        """
        crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        P_crit = crit_ratio * P_t
        self.gamma = gamma
        self.R_bar = R_bar

        # Compute Mach numbers
        self.Me = self.get_M_e_opt(gamma, P_t, P_e)
        try:
            self.M_throat = self.get_M_throat(P_t, P_e, self.gamma, self.R_bar)
        except Exception:
            self.M_throat = 1

        # --- Mass flow formula
        if self.M_throat < 1:
            M = self.M_throat
            m_dot = ((self.A_throat * P_t) / np.sqrt(T_t)) * np.sqrt(self.gamma / self.R_bar) * M * (
                        1 + (self.gamma - 1) / 2 * M ** 2) ** (-(self.gamma + 1) / (2 * (self.gamma - 1)))

        else:
            M = 1
            m_dot= ((self.A_throat * P_t) / np.sqrt(T_t)) * np.sqrt(self.gamma / self.R_bar) * (((gamma+1)/2)**((gamma+1)/(2-2*gamma)))

        # print(m_dot)
        self.m_dot = m_dot
        return m_dot
    def get_M_e_opt(self, gamma, P_t, P_e):
        self.gamma = gamma
        M_e_opt = np.sqrt((2 / (gamma - 1)) * ((P_t / P_e) ** ((gamma - 1) / gamma) - 1))
        return M_e_opt
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
            M_e = self.get_M_e_opt(gamma, P_t, P_e)

            # Throat Mach (subsonic branch before choking)
            try:
                M_t = self.get_M_throat(P_t, P_e, gamma, R_bar)
            except Exception:
                M_t = np.nan

            # If nozzle is choked (M_t -> 1), clamp value
            if M_t >= 1.0:
                M_t = 1.0
                M_e = np.nan

            # Mass flow rate at throat

            m_dot = self.get_m_dot(P_t, T_t, R_bar, P_e, gamma)


            M_e_list.append(M_e)
            M_t_list.append(M_t)
            m_dot_list.append(m_dot)

        # --- Compute critical choking pressure ratio ---

        # --- Plot setup ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(P_t_range, M_t_list, label="M_throat", color='tab:blue', linewidth=2)
        ax1.plot(P_t_range, M_e_list, label="M_exit_optimum", color='tab:cyan', linestyle='--', linewidth=2)

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
    def plot_nozzle_performance_vs_exit(self, P_t, T_t, R_bar, gamma, P_e_range):
        """
        Plot Mach number (throat & exit) and mass flow vs exit pressure.

        Parameters
        ----------
        P_t : float
            Chamber (stagnation) pressure [Pa]
        T_t : float
            Chamber (stagnation) temperature [K]
        R_bar : float
            Gas constant [J/(kg*K)]
        gamma : float
            Ratio of specific heats
        P_e_range : array-like
            Range of exit/ambient pressures [Pa] to evaluate
        """
        self.gamma = gamma
        self.R_bar = R_bar
        M_e_list = []
        M_t_list = []
        m_dot_list = []
        ratios = np.array(P_t) / np.array(P_e_range)  # for optional plotting

        for P_e in P_e_range:
            # Exit Mach (isentropic)
            M_e = self.get_M_e_opt(gamma, P_t, P_e)

            # Throat Mach (subsonic branch before choking)
            try:
                M_t = self.get_M_throat(P_t, P_e, gamma, R_bar)
            except Exception:
                M_t = np.nan

            # If nozzle is choked (M_t -> 1), clamp value
            if M_t >= 1.0:
                M_t = 1.0
                M_e = np.nan

            # Mass flow rate at throat
            m_dot = self.get_m_dot(P_t, T_t, R_bar, P_e, gamma)

            M_e_list.append(M_e)
            M_t_list.append(M_t)
            m_dot_list.append(m_dot)

        # --- Compute critical choking pressure ratio ---
        crit_ratio = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        # P_e_choke = P_t * crit_ratio

        # --- Plot setup ---
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(P_e_range, M_t_list, label="M_throat", color='tab:blue', linewidth=2)
        ax1.plot(P_e_range, M_e_list, label="M_exit_optimum", color='tab:cyan', linestyle='--', linewidth=2)
        # ax1.axvline(P_e_choke, color='gray', linestyle=':', label='Choking onset')

        ax1.set_xlabel(r"Exit Pressure $P_e$ [Pa]")
        ax1.set_ylabel("Mach number", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # --- Mass flow curve on right axis ---
        ax2 = ax1.twinx()
        ax2.plot(P_e_range, m_dot_list, label=r"Mass Flow $\dot{m}$", color='tab:red', linewidth=2)
        ax2.set_ylabel(r"Mass Flow Rate $\dot{m}$ [kg/s]", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"Nozzle Performance vs Exit Pressure — {self.name}")
        plt.tight_layout()
        plt.show()

class Injector():
    def __init__(self, Cd, ManifoldPressure, Fluid_1kg,total_Area, ManifoldT, name="Unconspicuous Injector"):
        self.Cd = Cd #discharge coefficient
        self.ManifoldPressure = ManifoldPressure
        self.ManifoldT = ManifoldT
        self.total_Area = total_Area
        self.fluid = Fluid_1kg
        self.name = name
        self.m_dot = 0
    def get_choked_massflow(self, T):
        """
        Compute the choked mass flow at a given fluid temperature T [K].
        Uses Gaspype for Cp, Cv, and density. Handles multi-species fluids safely.
        """
        try:
            # Number of moles in the fluid
            n = self.fluid.get_n()

            # Cp and Cv per mole
            Cp_mol = self.fluid.get_cp(t=T)  # J/mol·K
            Cv_mol = Cp_mol - 8.314  # J/mol·K

            # Specific heat ratio
            gamma = Cp_mol / Cv_mol

            # Density at T and manifold pressure
            rho = self.fluid.get_density(t=T, p=self.ManifoldPressure)  # kg/m³

            # Choked flow factor
            gamma_function = (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))

            # Choked mass flow (kg/s)
            self.choked_massflow = self.Cd * self.total_Area * np.sqrt(
                gamma * rho * self.ManifoldPressure * gamma_function)

        except Exception as e:
            print(f"[WARNING] Failed to compute choked mass flow: {e}")
            self.choked_massflow = 0.0

        return self.choked_massflow

    def get_massflow(self, ChamberPressure, T):
        rho = self.fluid.get_density(t=T, p=self.ManifoldPressure)
        pressure_delta = self.ManifoldPressure - ChamberPressure
        self.m_dot=self.Cd*self.total_Area*np.sqrt(2*rho*pressure_delta)
        choked_m_dot = self.get_choked_massflow(T)
        if self.m_dot > choked_m_dot:
            m_dot = choked_m_dot
            # print("choked flow reached at injector")
        return self.m_dot
    def set_timing(self, valve_open):
        self.valve_open_t = valve_open
    def update(self, t, dt, chamber_P):
        T = self.ManifoldT
        if t>self.valve_open_t:
            self.m_dot = self.get_massflow(chamber_P, T)
            m_added = self.m_dot*dt
            fluid_added = self.fluid*m_added
        else:
            m_added, fluid_added, T = 0, None, T
        return m_added, fluid_added, T


class Grain():
    def __init__(self, mass, area, cp, cp_gas,h_fg, h_sf, heat_transfer_coefficient, T_melt, T_gas ,initial_T=293.15,name="Unconspicuous Grain"):
        self.area = area
        self.cp = cp
        self.h_fg = h_fg
        self.name = name
        self.k_h = heat_transfer_coefficient
        self.T_melt = T_melt
        self.T = initial_T
        self.thermal_mass = mass
        self.liquid_mass = 0
        self.h_sf = h_sf
        self.T_gas = T_gas
        self.cp_gas = cp_gas
        self.grain_state = 0 # 0 is solid, 1 is melting, 2 is gassing, 3 is burning, 4 is burned
    def update(self, dt, T_ambient):
        # Assumes pressures don't vary much
        Q = dt*self.k_h*self.area*(T_ambient-self.T)
        gas_released = 0
        if  (self.T+ Q/self.cp) < self.T_melt and self.thermal_mass>0:
            self.T = self.T+ Q/(self.cp*self.thermal_mass)
        elif self.T+ Q/(self.cp*self.thermal_mass) < self.T_gas and self.thermal_mass>0:
            self.liquid_mass += Q*dt/self.h_sf
            self.thermal_mass -= Q*dt/self.h_sf
            self.grain_state = 1
        elif self.T+ Q/(self.cp*self.thermal_mass) < self.T_gas and self.thermal_mass<0:
            self.T = self.T + Q / (self.cp_gas * self.liquid_mass)
        elif self.T+ Q/(self.cp*self.thermal_mass) > self.T_gas and self.liquid_mass > 0:
            gas_released = Q * dt / self.h_sf
            self.liquid_mass -= Q * dt / self.h_sf
            self.grain_state = 2
        else:
            self.grain_state = 4
            gas_released = 0
            # print("Grain thermal mass has runned out")
        return gas_released, Q
    def setup_burn(self, output_fluid_per_mole, stoich_OF_mass, reaction_enthalpy, gas_flash_T, A_constant, E_a):
        self.flashT = gas_flash_T
        self.out_fluid_per_mole = output_fluid_per_mole
        self.stoich_OF_mass = stoich_OF_mass
        self.reaction_enthalpy = reaction_enthalpy

class Chamber():
    def __init__(self, volume, initialP, initial_fluid, initial_T=293.15, name="Unconspicuous Chamber"):
        self.volume = volume
        self.P = initialP
        self.T = initial_T
        self.name = name
        self.fluid = initial_fluid
        self.mass = self.fluid.get_density(t=initial_T, p=initialP)*self.volume

class Engine():
    def __init__(self, Ignitor, Nozzle, Injector, Grain, Chamber, ambient_P=101325):
        self.Ignitor = Ignitor
        self.Nozzle = Nozzle
        self.Injector = Injector
        self.Grain = Grain
        self.Chamber = Chamber
        self.Time = 0
        self.AmbientPressure = ambient_P
        self.ox_mass = 0
        self.gas_f_mass = 0

    def _scalar_cp_mass(self, fluid, T):
        """
        Return a scalar mass-based specific heat capacity [J/kg/K] for `fluid` at temperature T.

        This function:
          - coerces T to a float robustly,
          - safely calls fluid.get_cp(T) (which may return ndarray or scalar, in J/mol/K),
          - converts molar Cp -> mass Cp via molar mass (kg/mol),
          - falls back gracefully if an array is returned.
        """
        import numpy as _np

        # 1) Make T a plain float (handle arrays, lists, etc.)
        try:
            T_scalar = float(_np.asarray(T).flat[0])
        except Exception:
            # As a last resort, use current chamber T if available
            try:
                T_scalar = float(self.Chamber.T)
            except Exception:
                T_scalar = 300.0  # reasonable default

        # 2) Ask gaspype for molar Cp (J/mol/K) at scalar T
        try:
            cp_molar = fluid.get_cp(T_scalar)
        except Exception as e:
            # If gaspype errors (out-of-range), clamp T to a safer value
            # Try using median of fluid temp grid if available, else fallback
            try:
                # Many gas libs expose T limits; try crude fallback:
                cp_molar = fluid.get_cp(max(200.0, min(5000.0, T_scalar)))
            except Exception:
                # give up and return a default cp_mass
                return 1000.0  # J/kg/K fallback (survivable default)

        cp_molar = _np.asarray(cp_molar)

        # 3) If cp_molar is an array, try to collapse it to a scalar (weighted average)
        if cp_molar.size > 1:
            try:
                # If fluid.get_n() returns array of molar amounts/fractions, use it to weight cp
                n = _np.asarray(fluid.get_n())
                # If n is scalar but cp array, fall back to mean
                if _np.ndim(n) == 0 or n.size == 1:
                    cp_molar_scalar = float(cp_molar.mean())
                else:
                    # If n and cp_molar align, weight them
                    # If n are molar amounts, sum-weight; if fractions, normalize first
                    n_sum = n.sum()
                    if n_sum == 0:
                        cp_molar_scalar = float(cp_molar.mean())
                    else:
                        cp_molar_scalar = float((cp_molar * n).sum() / n_sum)
            except Exception:
                cp_molar_scalar = float(cp_molar.mean())
        else:
            cp_molar_scalar = float(cp_molar)

        # 4) Convert molar Cp (J/mol/K) to mass Cp (J/kg/K)
        try:
            molar_mass = float(fluid.get_molar_mass())  # kg/mol
            if molar_mass == 0:
                return 1000.0
            cp_mass = cp_molar_scalar / molar_mass
        except Exception:
            # If get_molar_mass fails, fallback to dividing by 0.029 (approx air)
            cp_mass = cp_molar_scalar / 0.029

        return cp_mass
    def update(self, dt):

        # Ignitor stuff
        ig_m_released, ig_fluid_released, ig_remaining_ignitor_mass, ig_heat_released = self.Ignitor.update(self.Time, dt)

        # Injector stuff
        in_m_added, in_fluid_added, in_T = self.Injector.update(self.Time, dt, self.Chamber.P)
        self.ox_mass += in_m_added

        # Grain stuff

        gas_prop_released, Q_consumed = self.Grain.update(dt, self.Chamber.T)
        self.gas_f_mass += gas_prop_released

        # TO DO: SET UP ENTHALPY AND GASSES OF BURN
        # Grain burn stuff... In the long run I'll use arrhenius to find how much reacts
        burned_fuel_mass = 0

        added_burn_gas = None
        added_burn_enthalpy = 0

        if gas_prop_released>0 and self.Chamber.T>self.Grain.flashT:
            required_ox_mass = self.gas_f_mass * self.Grain.stoich_OF_mass

            if self.ox_mass > required_ox_mass:
                burned_fuel_mass = self.gas_f_mass
                self.gas_f_mass = 0
                self.ox_mass -= required_ox_mass
            else:
                burned_fuel_mass = self.ox_mass/self.Grain.stoich_OF_mass
                self.ox_mass = 0
                self.gas_f_mass -= burned_fuel_mass
            self.Grain.grain_state = 3
            added_burn_enthalpy = self.Grain.reaction_enthalpy*burned_fuel_mass
            added_burn_gas = self.Grain.out_fluid_per_mole*(1/self.Grain.out_fluid_per_mole.get_mass())



        # Update values

        if in_fluid_added is not None:
            n_chamber = self.Chamber.fluid.get_n()
            # cp1 = self.Chamber.fluid.get_cp(self.Chamber.T) * n_chamber
            cp1 = self._scalar_cp_mass(self.Chamber.fluid, self.Chamber.T)
            m1 = self.Chamber.mass
            m2 = in_m_added
            n_injector = in_fluid_added.get_n()
            # cp2 = in_fluid_added.get_cp(in_T) * n_injector
            cp2 = self._scalar_cp_mass(in_fluid_added, in_T)
            self.Chamber.fluid  = self.Chamber.fluid + in_fluid_added
            T1 = self.Chamber.T
            T2 = in_T
            self.Chamber.T = (m1*cp1*T1 + m2*cp2*T2)/(m1*cp1+m2*cp2)

        if ig_fluid_released is not None:
            self.Chamber.fluid  = self.Chamber.fluid + ig_fluid_released

        if added_burn_gas is not None:
            self.Chamber.fluid = self.Chamber.fluid + added_burn_gas

        n_chamber=self.Chamber.fluid.get_n()
        # cp = self.Chamber.fluid.get_cp(self.Chamber.T)*n_chamber
        cp = self._scalar_cp_mass(self.Chamber.fluid, self.Chamber.T)
        self.Chamber.T = self.Chamber.T + (ig_heat_released+added_burn_enthalpy)/cp

        # --- Update chamber pressure ---
        R_bar = 8.314
        R = 8.314 / self.Chamber.fluid.get_molar_mass()  # specific gas constant (J/kg·K)
        rho = self.Chamber.mass / self.Chamber.volume
        self.Chamber.P = np.sum(self.Chamber.fluid.get_n()) * R_bar * self.Chamber.T/self.Chamber.volume

        # --- Compute nozzle mass flow and thrust ---
        gamma = self.Chamber.fluid.get_cp(t=self.Chamber.T) / ( self.Chamber.fluid.get_cp(t=self.Chamber.T) - self.Chamber.fluid.get_molar_mass()/self.Chamber.fluid.get_mass() * 0 )
        P_t = self.Chamber.P
        T_t = self.Chamber.T
        P_e = self.AmbientPressure
        m_dot_noz = self.Nozzle.get_m_dot(P_t, T_t, R_bar, P_e, gamma)
        # --- Deplete chamber mass due to nozzle flow ---
        m_expelled = (m_dot_noz * dt)
        # print("C mass: ", self.Chamber.mass)
        # print("Mass expelled: ", m_expelled)
        if m_expelled > self.Chamber.mass:
            m_expelled = self.Chamber.mass
        self.Chamber.mass -= m_expelled
        chamber_unit_fluid = self.Chamber.fluid/self.Chamber.fluid.get_mass()
        fluid_expelled = m_expelled*chamber_unit_fluid
        self.Chamber.fluid = self.Chamber.fluid - fluid_expelled
        # --- Update chamber pressure again after outflow ---
        rho = max(self.Chamber.mass / self.Chamber.volume, 1e-9)
        self.Chamber.P = np.sum(self.Chamber.fluid.get_n()) * R_bar * self.Chamber.T/self.Chamber.volume



        self.Time += dt
    def run_ignition_sequence(self,t_start, t_end,
                              ignition_start, injector_start, shutdown, dt=1e-3,
                              verbose=True):
        """
        Simulate a full ignition sequence for the self object.
        Parameters
        ----------
        t_start : float
            End time of simulation [s].
        t_end : float
            End time of simulation [s].
        dt : float
            Time step [s].
        ignition_start, injector_start, grain_ignite, shutdown : float
            Event times (s) for plotting reference lines.
        Returns
        -------
        results : dict
            Dictionary with time history of key variables.
        """
        # --- Storage arrays ---
        self.Time = t_start
        times, pressures, temps, mdots, machs, grain_states, grain_ts, grain_mass, grain_liquid_mass, mdots_ox = [], [], [], [], [], [], [], [], [], []
        for step in range(int((t_end-t_start) / dt)):
            t = self.Time
            # Example: turn on subsystems at event times
            if hasattr(self.Ignitor, "active"):
                self.Ignitor.active = t >= ignition_start
            if hasattr(self.Injector, "active"):
                self.Injector.active = t >= injector_start
            # if hasattr(self.Grain, "ignited"):
            #     self.Grain.ignited = t >= grain_ignite
            # --- Run self update ---
            self.update(dt)
            # --- Record ---
            times.append(self.Time)
            pressures.append(self.Chamber.P)
            temps.append(self.Chamber.T)
            grain_states.append(self.Grain.grain_state)
            grain_ts.append(self.Grain.T)
            grain_mass.append(self.Grain.thermal_mass)
            grain_liquid_mass.append(self.Grain.liquid_mass)
            # Recompute nozzle quantities for logging
            gamma = self.Chamber.fluid.get_cp(t=self.Chamber.T) / (self.Chamber.fluid.get_cp(
                t=self.Chamber.T) - self.Chamber.fluid.get_molar_mass() / self.Chamber.fluid.get_mass() * 0)
            R_bar = 8.314 / self.Chamber.fluid.get_molar_mass()  # specific gas constant (J/kg·K)
            M_e = self.Nozzle.get_M_throat(self.Chamber.P, self.AmbientPressure, gamma, R_bar)
            mdots.append(self.Nozzle.get_m_dot(self.Chamber.P, self.Chamber.T, R_bar, self.AmbientPressure, gamma))
            mdots_ox.append(self.Injector.m_dot)
            machs.append(M_e)
            if verbose and step % 10 == 0:
                print(f"t={t:.3f}s, P={self.Chamber.P / 1e6:.3f} MPa, T={self.Chamber.T:.1f} K \n")
            if t > t_end:
                break
            # --- Plot results (2 columns × 3 rows) ---
        fig, axs = plt.subplots(3, 3, figsize=(30, 12), sharex=True)
        axs = axs.flatten()

        axs[0].plot(times, np.array(pressures) / 1e5)
        axs[0].set_ylabel("Chamber Pressure [bar]")
        axs[0].grid(True)

        axs[1].plot(times, temps)
        axs[1].set_ylabel("Chamber Temperature [K]")
        axs[1].grid(True)

        axs[2].plot(times, mdots)
        axs[2].set_ylabel("Nozzle Mass Flow [kg/s]")
        axs[2].grid(True)

        axs[3].plot(times, machs)
        axs[3].set_ylabel("Nozzle Throat Mach")
        axs[3].grid(True)

        axs[4].plot(times, grain_ts)
        axs[4].set_ylabel("Grain Temperature [K]")
        axs[4].grid(True)

        axs[5].step(times, grain_states, where="post")
        axs[5].set_ylabel("Grain State")
        axs[5].set_xlabel("Time [s]")
        axs[5].set_yticks([0, 1, 2, 3, 4])
        axs[5].set_yticklabels(["Solid", "Melting", "Gassing", "Burning", "Burned"])
        axs[5].grid(True)

        axs[6].plot(times, grain_mass, label="Total Grain Mass")
        axs[6].plot(times, grain_liquid_mass, label="Liquid Grain Mass")
        axs[6].set_ylabel("Grain Mass [kg]")
        axs[6].grid(True)
        axs[6].legend()

        axs[7].plot(times, mdots_ox)
        axs[7].set_ylabel("Injector Mass Flow [kg/s]")
        axs[7].grid(True)
        # --- Vertical event lines and grain state transitions ---
        for ax in axs:
            # Core ignition/injection/shutdown events
            for ev_time, label in [
                (ignition_start, "Ignitor ON"),
                (injector_start, "Main Valve ON"),
                (shutdown, "Shutdown")
            ]:
                ax.axvline(ev_time, color='r', linestyle='--', alpha=0.7)
                ax.text(ev_time, ax.get_ylim()[1] * 0.95, label, rotation=90,
                        va='top', ha='right', fontsize=8, color='r')

            # --- Grain state transitions ---
            grain_colors = {
                (0, 1): "black",
                (1, 2): "orange",
                (2, 3): "red",
                (3, 4): "blue"
            }
            grain_labels = {
                (0, 1): "Melting start",
                (1, 2): "Gassing start",
                (2, 3): "Burning start",
                (3, 4): "Burnout"
            }

            # detect transitions in grain_states
            if len(grain_states) > 1:
                for i in range(1, len(grain_states)):
                    prev_state = grain_states[i - 1]
                    curr_state = grain_states[i]
                    transition = (prev_state, curr_state)
                    if transition in grain_colors:
                        color = grain_colors[transition]
                        label = grain_labels[transition]
                        t_transition = times[i]
                        ax.axvline(t_transition, color=color, linestyle='--', alpha=0.8, linewidth=2)
                        ax.text(
                            t_transition, ax.get_ylim()[1] * 0.9, label,
                            rotation=90, va='top', ha='right', fontsize=8, color=color
                        )
        plt.tight_layout()
        plt.show()
        # --- Return results for post-processing ---
        return {
            "time": np.array(times),
            "P": np.array(pressures),
            "T": np.array(temps),
            "m_dot": np.array(mdots),
            "Mach": np.array(machs)
        }








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

# gamma = 1.4
# R = 287
# P_e = 101325       # ambient pressure [Pa]
# T_t = 1000          # K
# A_t = 0.0005       # m²
# A_e = 1.5*A_t
# A_c = 0.0010       # m²
#
# nzl = Nozzle(A_t, A_c, A_e)
#
# # Sweep chamber pressure
#
# P_t_range = np.linspace(P_e+0.1, 2*P_e, 1000)
#
# nzl.plot_nozzle_performance_vs_chamber(P_e, T_t, R, gamma, P_t_range)
#
# P_t = 2e5           # Chamber pressure [Pa] (~5 atm)
#
# # Sweep exit pressure from very low (high expansion) up to near chamber pressure
# P_e_range = np.linspace(1e5, P_t - 1e3, 500)  # Pa
#
# # Plot nozzle performance vs exit pressure
# nzl.plot_nozzle_performance_vs_exit(P_t, T_t, R, gamma, P_e_range)
