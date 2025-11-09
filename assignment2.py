import numpy as np
import scipy.integrate as integrate
import scipy.sparse as scp
import matplotlib.pyplot as plt

#Using the second order centre difference to discretise u_xx

def construct_tridiagonal(coefficients,N,h):
    upper = coefficients[0]*np.ones(N-2)
    diagonal = coefficients[1]*np.ones(N-1)
    lower = coefficients[2]*np.ones(N-2)

    diagonals = [upper,diagonal,lower]
    offsets = [+1,0,-1]

    A = (1/(h**2))*scp.diags(diagonals,offsets,format='csr')

    return A

def initial1(x):
    return np.sin(2*np.pi*x) + 0.5*np.sin(np.pi*x)

def grid_points(n):
    h = 0.1/(2**n)
    return np.linspace(h, 1 - h, 10*(2**n) - 1)

def plot_time(solutions, t):

    line = solutions.t
    #Find the position in the solution where t appears
    index = 0

    while line[index] != t:
        index += 1

    solution_values = []

    for solution in solutions.y:
        solution_values.append(solution[index])

    return solution_values

def plot_times(solutions, t_values, n):

    points = grid_points(n)

    for t in t_values:
        solution_values = plot_time(solutions, t)
        plt.plot(points, solution_values,label = f"t = {t}")

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()

#Question 1.
#The setup for this system is quite straightforward since the BCs are 0
n = 5
h = 0.1/(2**n)

A = construct_tridiagonal([1,-2,1],10*(2**n),h)
boundary1 = 0

boundary2 = 0
def system1(t,u):
    v = np.zeros(len(u))
    v[0] = boundary1/(h**2)
    v[-1] = boundary2/h**2
    return A*u + v

#The initial conditions
y0 = initial1(grid_points(n))
tspan = [0,1]
t_eval = [0,0.2,0.4,0.8,1]

solution = integrate.solve_ivp(system1, [0,1],y0,'Radau',t_eval)
t_values = [0, 0.2, 0.4, 0.8, 1]

plot_times(solution, t_values, n)

#Question 2.
#The main matrix is the same as in (1):

def g1(solution,h):
    #This is the other term in the standard MOL discretisation
    g = [solution[1]**2]

    for u in range(len(solution) - 2):
        g.append(solution[u+2]**2 - solution[u]**2)

    g.append(-(solution[-1]**2))
    g = -(1/(4*h))*np.array(g)
    
    return g

n = 3
h = 0.1/(2**n)
epsilon = 0.01

A = epsilon*construct_tridiagonal([1,-2,1],10*(2**n),h)

def system2(t,u,h):
    v = np.zeros(len(u))
    v[0] = boundary1/(h**2)
    v[-1] = boundary2/h**2

    return A*u + g1(u,h) + v

y0 = initial1(grid_points(n))
tspan = [0,1]
t_eval = [0,0.2,0.4,0.8,1]

solution = integrate.solve_ivp(system2, [0,1],y0,'Radau',t_eval,args=[h])
t_values = [0, 0.2, 0.4, 0.8, 1]

plot_times(solution, t_values, n)

#Question 3.
"""n = 5
zeta = (0.1)/(2**n)
N = 10*2**n + 1
epsilon = 0.1

def ux(index,u,x):
    if index == 1:
        return ((3/2)*u[0] - 2*u[1] + (1/2)*u[2])/((3/2)*x[0] - 2*x[1] + (1/2)*x[2])
    
    elif index == N:
        return ((3/2)*u[N-3] - 2*u[N-4] + (1/2)*u[N-5])/((3/2)*x[N-3] - 2*x[N-4] + (1/2)*x[N-5])
    
    elif index == 2:
        return u[index - 1]/x[index - 1]
    elif index == N - 1:
        return (-u[index - 3])/(1 - x[index-3])
    else:
        return ((u[index-1]) - u[index - 3])/(x[index-1] - x[index - 3])
    
def uxx(index,u,x):
    if index == 1:
        return 2*((x[1]-x[0])*(u[2]-u[0])-(x[2]-x[0])*(u[1]-u[0]))/((x[2]-x[0])*(x[1]-x[0])*(x[2]-x[1]))

    elif index == N:
        return 2*((x[N-4]-x[N-3])*(u[N-5]-u[N-3])-(x[N-5]-x[N-3])*(u[N-4]-u[N-3]))/((x[N-5]-x[N-3])*(x[N-4]-x[N-3])*(x[N-5]-x[N-4]))
    elif index == 2:
        return (2/(x[index-1]))*((u[index-1] - u[index - 2])/(x[index-1] - x[index - 2]) - (u[index - 2])/(x[index - 2]))
    
    elif index == N-1:
        return (2/(1 - x[index - 3]))*((-u[index - 2])/(1 - x[index - 2]) - (u[index - 2] - u[index - 3])/(x[index - 2] - x[index - 3]))

    else:
        return (2/(x[index-1] - x[index - 3]))*((u[index-1] - u[index - 2])/(x[index-1] - x[index - 2]) - (u[index - 2] - u[index - 3])/(x[index - 2] - x[index - 3]))
    

def p1(index,u,x):
    return np.sqrt(1 + (np.abs(ux(index,u,x)))**2)

def p2(index, u, x):
    return (1 + (np.abs(uxx(index,u,x)))**2)**0.25

def dxt(index,u,x,i):

    p = p1
    if i == 2:
        p = p2

    coefficient = 1/(p(index,u,x)*0.01*(epsilon**2))
    
    if index == N-1:
        term1 = (p(index + 1,u,x) - p(index,u,x))*(1 - x[index - 2])/2
    else:
        term1 = (p(index + 1,u,x) - p(index,u,x))*(x[index-1] - x[index - 2])/2
    term2 = (p(index,u,x) + p(index-1,u,x))*(x[index-2]-x[index-3])/2
    

    return coefficient*(term1 - term2)


def system3(t,u, N,i):

    actual_x = u[:N-2]
    actual_u = u[N-2:]

    vector = np.zeros(2*N - 4)

    new_x = [dxt(2,actual_u,actual_x,i)]
    new_u = []

    coefficient1 = (actual_u[1]/actual_x[1])*new_x[0]
    coefficient2 = 2*epsilon/(actual_x[1])
    term1 = (actual_u[1] - actual_u[0])/(actual_x[1]-actual_x[0])
    term2 = actual_u[0]/actual_x[0]
    term3 = ((actual_u[1])**2)/(2*actual_x[1])

    new_u.append(coefficient1 + coefficient2*(term1 - term2) - term3)

    for j in range(3,N-1):
        new_x.append(dxt(j,actual_u,actual_x,i))

        index = j - 2
        coefficient1 = new_x[index]*(actual_u[index + 1] - actual_u[index - 1])/(actual_x[index + 1] - actual_x[index -1])
        coefficient2 = 2*epsilon/(actual_x[index + 1] - actual_x[index - 1])

        term1 = (actual_u[index + 1] - actual_u[index])/(actual_x[index + 1] - actual_x[index])
        term2 = (actual_u[index] - actual_u[index - 1])/(actual_x[index] - actual_x[index -1])

        term3 = ((actual_u[index+1])**2 - (actual_u[index - 1])**2)/(2*(actual_x[index + 1] - actual_x[index - 1]))
        new_u.append(coefficient1 + coefficient2*(term1 - term2) - term3)
    
    new_x.append(dxt(N-1,actual_u,actual_x,i))
    
        
    coefficient1 = (-actual_u[N-4]/(1 - actual_x[N-4]))*new_x[N-3]
    coefficient2 = 2*epsilon/(1 - actual_x[N-4])
    term1 = (-actual_u[N-3])/(1-actual_x[N-3])
    term2 = (actual_u[N-3] - actual_u[N-4])/(actual_x[N-3] - actual_x[N-4])
    term3 = (-((actual_u[N-4])**2))/(2*(1 - actual_x[N-4]))

    new_u.append(coefficient1 + coefficient2*(term1 - term2) - term3)

    vector[:N-2] = new_x
    vector[N-2:] = new_u

    return vector

y0 = np.zeros(2*N - 4)
x0 = grid_points(n)
y0[:N-2] = x0

y0[N-2:] = initial1(x0)
t_eval = [0,0.2,0.4,0.8,1]


solution = integrate.solve_ivp(system3, [0,1],y0,t_eval=t_eval,args=[N,1])
t_values = [0, 0.2, 0.4, 0.8, 1]
print(solution.sol)
def plot_time_adaptive(solutions, t):

    line = solutions.t
    #Find the position in the solution where t appears
    index = 0
    while line[index] != t:
        index += 1

    solution_values = []

    for solution in solutions.y[N-2:]:
        solution_values.append(solution[index])

    x_values = solutions.y[:N-2]
    return solution_values,x_values

def plot_times_adaptive(solutions, t_values):

    for t in t_values:
        solution_values,points = plot_time_adaptive(solutions, t)
        plt.plot(points, solution_values,label = f"t = {t}")

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()

plot_times_adaptive(solution,t_values)"""