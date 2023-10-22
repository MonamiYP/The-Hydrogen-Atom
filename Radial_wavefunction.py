import matplotlib.pyplot as plt
from scipy import constants
from scipy.special import eval_genlaguerre
import math
import numpy as np

EPSILON_0 = constants.epsilon_0
H_BAR = constants.hbar
ELECTRON_CHARGE = constants.elementary_charge
ELECTRON_MASS = constants.electron_mass

BOHR_RADIUS = (4 * math.pi * EPSILON_0 * H_BAR**2) / (ELECTRON_CHARGE**2 * ELECTRON_MASS)

def lageurre(n, k, x):
    return eval_genlaguerre(n, k, x)

def radial_equation(n, l, r):
    root_term = math.sqrt( (2/(n*BOHR_RADIUS))**3 * math.factorial(n-l-1) / (2*n*((math.factorial(n+l))**3)))
    exp_term = math.exp(-r/(n*BOHR_RADIUS))
    l_power_term = (2*r/(n*BOHR_RADIUS))**l
    lageurre_term = lageurre(n-l-1, 2*l+1, 2*r/(n*BOHR_RADIUS))

    R = root_term * exp_term * l_power_term * lageurre_term

    return R

nl_vals = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
plt.figure()
for index, nl in enumerate(nl_vals):
    x = np.linspace(0, 20, 100)
    R = np.empty(len(x))
    for i in range(len(x)):
        R[i] = radial_equation(nl[0], nl[1], x[i]*BOHR_RADIUS)
    ax = plt.subplot(3, 2, index+1)
    ax.plot(x, R)

    plt.title(f"Radial wavefunction L({nl[0]},{nl[1]})")
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel("r/a₀")
    plt.ylabel("R(r)")
plt.show()

plt.figure()
for index, nl in enumerate(nl_vals):
    x = np.linspace(0, 20, 100)
    R = np.empty(len(x))
    for i in range(len(x)):
        R[i] = radial_equation(nl[0], nl[1], x[i]*BOHR_RADIUS)
    ax = plt.subplot(3, 2, index+1)
    ax.plot(x, 4*math.pi*x**2*R**2)

    plt.title(f"Probability density L({nl[0]},{nl[1]})")
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel("r/a₀")
    plt.ylabel("4πr$^2$R(r)$^2$")
plt.show()

