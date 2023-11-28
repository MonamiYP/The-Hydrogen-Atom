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
    return constant_term * np.abs(np.exp(1.j * m * phi)) * legrende # Returns magnitude, in the future expand to include phase

def wavefunction(n, l, m, r, theta, phi):
    psi = radial_equation(n, l, r) * angular_equation(l, m, theta, phi)
    return psi

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    
    return r, theta, phi

def plot_probability_density_3d(n, l, m, radius=50):

    max_radius = radius # r/a_0
    resolution = 100
    x = y = z = np.linspace(-max_radius, max_radius, resolution)

    coords = []
    probability = []

    for ix in x:
        for iy in y:
            for iz in z:
                coords.append(str((ix, iy, iz)))
                r, theta, phi = cartesian_to_spherical(ix, iy, iz)
                psi = wavefunction(n, l, m, r * BOHR_RADIUS, theta, phi)
                probability.append(psi**2)
    
    probability = probability/sum(probability)
    coord = np.random.choice(coords, size=10000, replace=True, p=probability)
    coord_matrix = [i.split(',') for i in coord]
    coord_matrix = np.matrix(coord_matrix)
    x_coords = [float(i.item()[1:]) for i in coord_matrix[:,0]] 
    y_coords = [float(i.item()) for i in coord_matrix[:,1]] 
    z_coords = [float(i.item()[0:-1]) for i in coord_matrix[:,2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, alpha=0.05, s=2)
    ax.set_title(f"Hydrogen probability density (n={n}, l={l}, m={m})")

    ax.set_xticks([-radius,-radius/2, 0,radius/2,radius])
    ax.set_yticks([-radius,-radius/2, 0,radius/2,radius])
    ax.set_zticks([-radius,-radius/2, 0,radius/2,radius])

    ax.axes.set_xlim3d(left=-radius+1, right=radius-1) 
    ax.axes.set_ylim3d(bottom=-radius+1, top=radius-1) 
    ax.axes.set_zlim3d(bottom=-radius+1, top=radius-1) 
    plt.show()

def plot_wavefunction_2d(n, l, m):
    max_radius = 40 # r/a_0
    resolution = 1000
    x = y = np.linspace(-max_radius, max_radius, resolution)
    x, y = np.meshgrid(x, y)
    r = np.sqrt((x**2 + y**2))

    theta = np.arctan(x / (y + np.finfo(np.float32).eps))

    psi = wavefunction(n, l, m, r * BOHR_RADIUS, theta, 0)
    probability_density = psi**2
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"(n={n}, l={l}, m={m})")
    plt.imshow(np.sqrt(probability_density))
    plt.show()

plot_wavefunction_2d(4, 1, 1)
#plot_probability_density_3d(5, 3, 0, 50)