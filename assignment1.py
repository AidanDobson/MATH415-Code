import numpy as np
import scipy.sparse as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#The approximation can be rearranged to """" = U_{j}^{n} + kf(U_{j}^{n})
#This function computes that RHS
def rhs(previous_solution_vector,k):
    return previous_solution_vector + k*4*previous_solution_vector*(1-np.square(previous_solution_vector))

def assemble_tridiagonal(upper,middle,lower,factor,dimension):
    upper_diagonal = upper*np.ones(dimension - 1)
    middle_diagonal = middle*np.ones(dimension)
    lower_diagonal = lower*np.ones(dimension - 1)

    A = factor*(np.diag(upper_diagonal, +1) + np.diag(middle_diagonal) + np.diag(lower_diagonal, -1))

    return A

#Question 3.
h = 1/10
k = 1/10
n = int(20/h)

ratio = (h**2)/k
grid = np.linspace(-10,10,n+1)
tolerance = 10**(-6)

#We initialize with an initial guess for the values
previous_iteration = 2*np.ones(n+1)
current_iteration = previous_iteration

factor_matrix = assemble_tridiagonal(1, ratio + 2, 1,1/ratio,n + 1)
factor_matrix[0,1] = -2
factor_matrix[n,n-1] = -2

print(factor_matrix)

#Compress the matrix for scipy's sparse solver
csr_factor_matrix = scp.csr_matrix(factor_matrix)

rhs_vector = rhs(current_iteration,k)

current_iteration,previous_iteration = scp.linalg.spsolve(csr_factor_matrix,rhs_vector), current_iteration
step_counter = 1

solutions = [previous_iteration,current_iteration]
times = [0,k]
while np.linalg.norm(current_iteration - previous_iteration) > tolerance:
    rhs_vector = rhs(current_iteration,k)
    current_iteration,previous_iteration = scp.linalg.spsolve(csr_factor_matrix,rhs_vector), current_iteration
    solutions.append(current_iteration)
    times.append(times[-1] + k)
    step_counter += 1

print(f'Converged in {step_counter} steps after {step_counter*k} units of time.')

#Create the mesh for the surface plot
X, T = np.meshgrid(grid, times)

plt.plot(grid,current_iteration)
plt.xlabel("x", size= 40)
plt.ylabel("u(x,t)",size = 40)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface_plt = ax.plot_surface(T,np.array(solutions),X, cmap = cm.coolwarm)
fig.colorbar(surface_plt, shrink=0.5, aspect=5)
ax.view_init(elev=-70,azim=170,roll = -100)
plt.xlabel("Time")
plt.ylabel("Solution")
ax.set_zlabel("x")
plt.show()

