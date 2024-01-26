import cubic
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.integrate import solve_ivp

## Solve equations for albedo and CO2 partial pressure

# Dimensional parameters
Q = 1361                                        # (W/m^2)
sigma = 5.67e-8                                 # (W/m^2)
c = 10e6                                        # (J/K/m^2)
g = 9.81                                        # (m/s^2)
T0 = 288                                        # (K)
Tm = 273.16                                     # (K)
DelT_c = 13.7                                   # (K)
DelT = 10                                       # (K)
p0 = 42                                         # (Pa)
gamma0 = 0.64
gamma1 = 0.8e-3                                 # (1/Pa)
W0 = 3.8e11 / (365 * 24 * 3600)                 # (kg/s)
v_vulcanism = 0.141e12 / (365 * 24 * 3600)      # (kg/s)
t_i = 10e3 * 365 * 24 * 3600                    # (s)
M_a = 29e-3                                     # (kg/mol)
M_co2 = 44e-3                                   # (kg/mol)
A_E = 5.1e14                                    # (m^2)
A = 0.0028                                      # (mol/kg)
k1 = 1.3e-6                                     # (mol/kg)
k2 = 9.1e-10                                    # (mol/kg)
kh = 3.3e-7                                     # (mol/kg/Pa)
h = 7.3e11 / (365 * 24 * 3600)                  # (kg/s/Pa)
rho0 = 1e3                                      # (kg/m^3)
V0 = 1.35e18                                    # (m^3)
b = 10.2e12 / (365 * 24 * 3600)                 # (kg/s)

# Dimensionless parameters
ap = 0.85
am = 0.15
mu = 0.3
eps = (c * DelT_c * g * M_a * h) / (A_E * M_co2 * sigma * gamma0 * T0**4)
q = Q / (4 * sigma * gamma0 * T0**4)
lambda_ = gamma1 * p0 / gamma0
nu = DelT_c / T0
alpha = (g * M_a * h * t_i) / (A_E * M_co2)
w = W0 / (h * p0)
xi = (rho0 * V0 * g * M_a * kh * k1) / (A_E * k2)
beta = (M_co2 * b * kh * k1) / (h * k2)
A_new = (A * k2) / (kh * k1 * p0)       # Non-dimensional alkalinity
t_scale = (A_E * M_co2) / (g * M_a * h)

# Data to include time-dependent fossil fuel burning
t_years = np.array([2023-2023, 2025-2023, 2030-2023, 2035-2023, 2050-2023,
                    2056-2023, 2063-2023, 2068-2023, 2073-2023, 2078-2023,
                    2083-2023, 2088-2023, 2093-2023, 2098-2023, 2103-2023,
                    2108-2023, 2113-2023, 2118-2023, 2123-2023])
t_s = t_years * 365 * 24 * 3600 / t_scale   # Scaled time
co2_2023 = 36.8e12 / (365 * 24 * 3600)
co2_peak2025 = 39e12 / (365 * 24 * 3600)
co2_2030 = 16.65e12 / (365 * 24 * 3600)
co2_2035 = 15e12 / (365 * 24 * 3600)
constant_value = 0

# Vals is a list containing the CO2 emissions for each year in t_years
vals = np.array([co2_2023, co2_peak2025, co2_2030, co2_2035])
zeros = np.zeros(len(t_s) - len(vals)) + constant_value
vals = np.concatenate([vals, zeros]) + v_vulcanism

## Functions
def B(theta):
    '''
    Temperature-dependence of equilibrium albedo
    '''
    return ap - (ap - am) * (1 + np.tanh((T0 - Tm + DelT_c*theta)/DelT)) / 2

def emissions(t):
    '''
    Returns the non-dimensional value of v adapted to UN goals using cubic splines
    '''
    return cubic.cubic_splines(t, t_s, vals) / (h * p0)

def dadt(a, p, theta, C, t):
    '''
    Dimensionless Albedo differential equation
    '''
    return (B(theta) - a) / alpha

def Theta(a, p, theta, C, t):
    '''
    Dimensionless Temp. expanding to first-order in nu
    '''
    return (q * (1 - a) - (1 - lambda_*p) * (1 + nu*theta)**4) / eps

def dpdt(a, p, theta, C, t):
    '''
    Dimensionless Pressure differential equation
    '''
    return emissions(t)[0] - w*(p**mu)*np.exp(theta) - p + ((2*C - A_new) ** 2 / (A_new - C))

def dcdt(a, p, theta, C, t):
    '''
    Dimensionless DIC differential equation
    '''
    return (w*(p**mu)*np.exp(theta) + p - ((2*C - A_new)**2 / (A_new - C)) - beta * C) / xi

def dydt(t, y):
    '''
    System of all four equations
    '''
    a, p, theta, C = y
    return [dadt(a, p, theta, C, t), dpdt(a, p, theta, C, t), Theta(a, p, theta, C, t), dcdt(a, p, theta, C, t)]


## Simulation

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.set_figwidth(15)
fig.set_figheight(10)

length_trajectory_years = 100                                               # Simulation duration
length_trajectory = length_trajectory_years * 365 * 24 * 3600 / t_scale     # Scaled duration

# Initial (current) dimensionless data
p_0 = 1
a0 = 0.29
theta0 = 0
c0 = (2.3e-3 * k2) / (kh * k1 * p0)
y0 = [a0, p_0, theta0, c0]
# Solve equations
sol = solve_ivp(dydt, [0, length_trajectory], y0=y0, t_eval=np.linspace(0, length_trajectory, 1000))
a = sol.y[0, :]
p = sol.y[1, :]
theta = sol.y[2, :]
c = sol.y[3, :]
t = sol.t

# Plot solutions
ax1.plot(t*t_scale/(365*24*3600), a, 'k-', linewidth=2)
ax2.plot(t*t_scale/(365*24*3600), p*p0, 'k-', linewidth=2)
ax3.plot(t*t_scale/(365*24*3600), T0 + DelT_c*theta - 273.16, 'k-', linewidth=2)
ax4.plot(t*t_scale/(365*24*3600), c*kh*k1*p0/k2, 'k-', linewidth=2)

ax1.set_xlabel('Time (years)', fontsize='15')
ax1.set_ylabel('Albedo', fontsize='15')
ax1.ticklabel_format(useOffset=False, style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax1.locator_params(tight=True, nbins=4)
ax1.tick_params(axis='x', labelsize='14')
ax1.tick_params(axis='y', labelsize='14')

ax2.set_xlabel('Time (years)', fontsize='15')
ax2.set_ylabel(r'CO$_2$ partial pressure (Pa)', fontsize='15')
ax2.tick_params(axis='x', labelsize='14')
ax2.tick_params(axis='y', labelsize='14')

ax3.set_xlabel('Time (years)', fontsize='15')
ax3.set_ylabel('Temperature (°C)', fontsize='15')
ax3.tick_params(axis='x', labelsize='14')
ax3.tick_params(axis='y', labelsize='14')

ax4.set_xlabel('Time (years)', fontsize='15')
ax4.set_ylabel(r'Dissolved inorganic carbon in the ocean (mol kg$^{-1}$)', fontsize='13')
ax4.ticklabel_format(useOffset=False, style='sci', axis='y', scilimits=(0,0))
ax4.tick_params(axis='x', labelsize='14')
ax4.tick_params(axis='y', labelsize='14')

plt.show()
