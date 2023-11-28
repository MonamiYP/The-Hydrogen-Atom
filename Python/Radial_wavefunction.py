import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import constants
from scipy.special import eval_genlaguerre

BOHR_RADIUS = (4 * np.pi * constants.epsilon_0 * constants.hbar**2) / (constants.elementary_charge**2 * constants.electron_mass)

def radial_equation(n, l, r):
    R = 2*r/(n*BOHR_RADIUS)
    root_term = np.sqrt( (2/(n*BOHR_RADIUS))**3 * math.factorial(n-l-1) / (2*n*((math.factorial(n+l))))) # Cubed or not cubed? Different sources give different answers, check again later.
    exp_term = np.exp(-R/2)
    l_power_term = R**l
    lageurre_term = eval_genlaguerre(n-l-1, 2*l+1, R)

    return root_term * exp_term * l_power_term * lageurre_term

nl_vals = [(1,0), (2,0), (2,1), (3,0), (3,1), (3,2)]
plt.figure()
for index, nl in enumerate(nl_vals):
    x = np.linspace(0, 20, 100)
    R = np.empty(len(x))
    for i in range(len(x)):
        R[i] = radial_equation(nl[0], nl[1], x[i]*BOHR_RADIUS)
    ax = plt.subplot(3, 2, index+1)
    ax.plot(x, R/1e16*4)

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
    ax.plot(x, 4*np.pi*x**2*R**2/1e32)

    plt.title(f"Probability density L({nl[0]},{nl[1]})")
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel("r/a₀")
    plt.ylabel("P")
plt.show()

