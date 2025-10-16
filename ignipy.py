import numpy as np
import matplotlib.pyplot as plt
import pulse_generators as pg
import gaspype as gp
try:
    # SciPy ≥1.14
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    # SciPy <1.14
    from scipy.integrate import cumtrapz




class Ignitor():

    def __init__(self, h_curve, m_dot_curve, t_vector,name="Unconspicuous ignitor"):
        self.name = name
        self.h_curve = h_curve
        self.m_dot_curve = m_dot_curve
        self.t_vector = t_vector
        self.H_dot_curve = np.multiply(m_dot_curve, self.h_curve)

    def plot_ignitor_curves(self):
        """
        Plot the ignitor curves vs time:
          - top:    mass flow (kg/s)
          - middle: specific enthalpy (J/kg)
          - bottom: power / enthalpy generation (W = kg/s * J/kg)
        Also plots the integrated total and gas mass released over time.
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
        Sets up the combustion product gas mixture normalized per kg of total combustion products.
        Only the gaseous species are included in combustion_products.
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
        """Return mass flow rate [kg/s] at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.m_dot_curve, left=0.0, right=0.0)

    def get_gm_dot(self, t):
        """Return gas mass flow rate [kg/s] at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.gm_dot_curve, left=0.0, right=0.0)

    def get_gm_integral(self, t):
        """Return gas mass flow rate [kg/s] at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.gas_mass_integral_curve, left=0.0, right=0.0)
    def get_m_integral(self, t):
        """Return gas mass flow rate [kg/s] at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.mass_integral_curve, left=0.0, right=0.0)

    def get_h(self, t):
        """Return specific enthalpy [J/kg] at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.h_curve, left=self.h_curve[0], right=self.h_curve[-1])

    def get_H_dot(self, t):
        """Return enthalpy rate (power) [W] = m_dot * h at time t via linear interpolation."""
        return np.interp(t, self.t_vector, self.H_dot_curve, left=0.0, right=0.0)

    def compute_mass_integrals_arrays(self):
        """
        Compute cumulative integrals of:
          - total combusted mass
          - released gas mass
        over time using trapezoidal integration.

        Stores results as arrays aligned with t_vector:
            self.mass_integral_curve [kg]
            self.gas_mass_integral_curve [kg]
        """
        t = self.t_vector
        m_dot = self.m_dot_curve
        gm_dot = getattr(self, "gm_dot_curve", None)

        # Integrate mass flow curves with respect to time
        self.mass_integral_curve = np.concatenate(
            ([0.0], cumtrapz(m_dot, t))
        )

        if gm_dot is not None:
            self.gas_mass_integral_curve = np.concatenate(
                ([0.0], cumtrapz(gm_dot, t))
            )
        else:
            self.gas_mass_integral_curve = np.zeros_like(t)

        print("Total mass combusted:", self.mass_integral_curve[-1], "kg")
        print("Total gas mass released:", self.gas_mass_integral_curve[-1], "kg")


# Testing code for the Ignitor class:

t_vector_example = np.linspace(0, 2, 2000)
m_dot_curve_example = pg.smooth_pulse(t_vector_example, 2, 0.2, 0.2, 3/1000) # [kg/s]
h_curve_example = np.full(t_vector_example.shape, 2.12*1000*1000) # Black powder constant enthalpy generation

ign = Ignitor(h_curve_example, m_dot_curve_example, t_vector_example,name="Test ignitor")

# Set up the ignition gas mixture (CO2 + N2)
combustion_moles_example = 16
combustion_products_mass = 1.203   # total combustion products (kg)
combustion_products_gas_mass = 0.404   # gaseous part only (kg)

# If the massflow was measured only using the gas part then all the mass is gas

products_example = gp.fluid({
    'CO2': 6/combustion_moles_example,
    'N2': 5/combustion_moles_example
})

# Run the setup method
ign.setup_combustion_products(
    combustion_products=products_example,
    total_combustion_moles=combustion_moles_example,
    total_combustion_mass=combustion_products_mass,
    total_gas_mass=combustion_products_gas_mass
)

ign.plot_ignitor_curves()

print(ign.get_m_dot(np.pi/2))   # mass flow
print(ign.get_h(np.pi/2))       # specific enthalpy
print(ign.get_H_dot(np.pi/2))   # power