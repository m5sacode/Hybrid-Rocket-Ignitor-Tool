import numpy as np
import ignipy as ip
import gaspype as gp
import pulse_generators as pg

# -------- IGNITOR STUFF ----------

# time base
t_vector = np.linspace(0, 5, 2000)
dt = t_vector[1] - t_vector[0]

# --- steel wool parameters ---
steel_mass = 10/1000         # kg (20 g total steel consumed)
h_per_kg_steel = 9.58e6    # J/kg  (approx heat release for Fe -> Fe2O3)
burn_duration = 5 # s


m_dot_curve = pg.smooth_rise_linear_fall(t_vector, burn_duration, 0.2, steel_mass)

# constant enthalpy per unit mass of fuel (J/kg)
h_curve = np.full_like(t_vector, h_per_kg_steel)

# make ignitor object
ignitor = ip.Ignitor(h_curve, m_dot_curve, t_vector, name=f"Steel wool ignitor ({steel_mass*1000} g)")

# combustion product setup:
combustion_products_mass = steel_mass * (159.6882/111.69)   # ~ mass of Fe2O3 produced (kg)
combustion_products_gas_mass = 0.5/1000   # kg

# gas composition
products_gas = gp.fluid({'N2': 0.79, 'O2': 0.21})

# call setup: note total_combustion_moles isn't very meaningful for a metal oxide system
moles_fe2o3 = combustion_products_mass / 0.1596882   # mol
ignitor.setup_combustion_products(
    combustion_products=products_gas,
    total_combustion_moles=moles_fe2o3,
    total_combustion_mass=combustion_products_mass,
    total_gas_mass=combustion_products_gas_mass
)

ignitor.plot_ignitor_curves()


# Black powder
# t_vector_example = np.linspace(0, 1, 2000)
# m_dot_curve_example = pg.smooth_pulse(t_vector_example, 1, 0.2, 0.2, 2/1000) # [kg/s]
# h_curve_example = np.full(t_vector_example.shape, 2.12*1000*1000) # Black powder constant enthalpy generation
#
# ignitor = ip.Ignitor(h_curve_example, m_dot_curve_example, t_vector_example,name="Test ignitor")
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
# ignitor.setup_combustion_products(
#     combustion_products=products_example,
#     total_combustion_moles=combustion_moles_example,
#     total_combustion_mass=combustion_products_mass,
#     total_gas_mass=combustion_products_gas_mass
# )
#
# ignitor.plot_ignitor_curves()

# -------- NOZZLE STUFF ----------

throat_area = 0.00007088218 # m^2
exit_area = 0.00027464588 # m^2
chamber_area = 0.00192442184 # m^2

nozzle = ip.Nozzle(throat_area, chamber_area, exit_area)

# -------- INJECTOR STUFF ----------

Cd = 0.735
Manifold_pressure = 15 *100000 # Pa, ignition valve
# Manifold_pressure = 30 *100000 # Pa, Main valve
total_area = 0.000004155 # m^2
manifoldT = 233.15 # K
# manifoldT = 33.15 # K
fluid1kg = gp.fluid({'N2O':1})
fluid1kg = fluid1kg / fluid1kg.get_mass()

injector = ip.Injector(Cd, Manifold_pressure, fluid1kg,total_area, manifoldT, name="Mk2 single port injector")

# -------- GRAIN STUFF ----------

grain_mass = (50/1000)/1 #kg
grain_thermal_mass = grain_mass
grain_area = np.pi * (30/2000)**2 * 110/1000

# ---------- FDM (3D-printed) ABS ----------

# T_pyro= 616.15
T_pyro=378.0 # override to assume that local t is higher
abs_grain = ip.Grain(
    mass= grain_mass,
    thermal_mass=grain_thermal_mass,
    area= grain_area,
    cp=1500.0,
    cp_gas=1000.0,
    h_fg=3.070e6,                        # J/kg (measured enthalpy of gasification, FDM ABS). :contentReference[oaicite:8]{index=8}
    h_sf=1,
    heat_transfer_coefficient=300.0,     #W/m2K
    T_melt=378.0,
    T_gas=T_pyro,
    initial_T=293.15,
    name="ABS_FDM_paper"
)

abs_grain.setup_burn(
    output_fluid_per_mole=gp.fluid({'CO2':0.158,'H2O':0.085,'N2':0.757}),
    stoich_OF_mass=3.99,
    reaction_enthalpy=3.68e7, #J/kg
    gas_flash_T=750.0,
    A_constant=1e-11,
    E_a=1.2e5
)


# -------- CHAMBER STUFF ----------

volume = 0.0002116864 # m^3
initialP = 101325 # Pa, ambient
initialT = 293.15 # K

air_composition = {'N2': 0.78084, 'O2': 0.20946, 'Ar': 0.00934, 'CO2': 0.00036}

initial_fluid = gp.fluid(air_composition)
initial_fluid = initial_fluid * (volume/initial_fluid.get_v(initialT, initialP))

# print("Volume error: ", 100*((volume-initial_fluid.get_v(initialT, initialP))/volume), "%")

chamber = ip.Chamber(volume, initialP, initial_fluid, initial_T=initialT)


# -------- ENGINE SETUP ----------

ChimeraMk2 = ip.Engine(ignitor, nozzle, injector, abs_grain, chamber)

# -------- TIMINGS ----------
ignition = -3.6 # s
mv_open = 0.0 # s
shutdown = 4


# -------- RUN ----------

t_start = ignition-0.1
t_end = shutdown+0.5
# t_end = 0.0

dt=1/1000000

ChimeraMk2.run_ignition_sequence(t_start, t_end, ignition, mv_open, shutdown, dt=dt)
