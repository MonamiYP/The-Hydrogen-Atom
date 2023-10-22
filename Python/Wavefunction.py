import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

BOHR_RADIUS = (4 * np.pi * sp.constants.epsilon_0 * sp.constants.hbar**2) / (sp.constants.elementary_charge**2 * sp.constants.electron_mass)

def radial_equation(n, l, r):
    R = 2*r/(n*BOHR_RADIUS)
    constant_term = np.sqrt( (2/(n*BOHR_RADIUS))**3 * sp.special.factorial(n-l-1) / (2*n*((sp.special.factorial(n+l))))) # Cubed or not cubed? Different sources give different answers, check again later.
    lageurre_term = sp.special.eval_genlaguerre(n-l-1, 2*l+1, R)

    return constant_term * np.exp(-R/2) * R**l * lageurre_term

def angular_equation(l, m, theta, phi):
    m = np.abs(m)
    constant_term = (-1)**m * np.sqrt( (2*l+1)*sp.special.factorial(l-m) / (4*np.pi*sp.special.factorial(l+m)) )
    legrende = sp.special.lpmv(m, l, np.cos(theta))
    return constant_term * np.real(np.exp(1.j * m * phi)) * legrende

def wavefunction(n, l, m, r, theta, phi):
    psi = radial_equation(n, l, r) * angular_equation(l, m, theta, phi)
    return psi

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + x**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    return r, theta, phi

def plot_probability_density(n, l, m):

    max_radius = 50 # r/a_0
    resolution = 100
    x = y = np.linspace(-max_radius, max_radius, resolution)
    x, y = np.meshgrid(x, y)
    r = np.sqrt((x**2 + y**2))

    theta = np.arctan(x / (y + np.finfo(np.float32).eps))

    psi = wavefunction(n, l, m, r * BOHR_RADIUS, theta, 0)
    probability_density = psi**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(np.sqrt(probability_density))
    plt.show()

plot_probability_density(4, 3, 0)