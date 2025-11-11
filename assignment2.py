import numpy as np
import scipy.integrate as integrate
import scipy.sparse as scp
import matplotlib.pyplot as plt
import time

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

n = 6
h = 0.1/(2**n)
epsilon = 0.001

A = epsilon*construct_tridiagonal([1,-2,1],10*(2**n),h)

def system2(t,u,h):
    print(t)
    v = np.zeros(len(u))
    v[0] = boundary1/(h**2)
    v[-1] = boundary2/h**2

    return A*u + g1(u,h) + v

y0 = initial1(grid_points(n))
tspan = [0,1]
t_eval = [0,0.2,0.4,0.8,1]
time1 = time.time()
solution = integrate.solve_ivp(system2, [0,1],y0,'Radau',t_eval,args=[h])
t_values = [0, 0.2, 0.4, 0.8, 1]
print(time.time() - time1)
plot_times(solution, t_values, n)

#Question 3.

#Including the boundaries, there are exactly N points going from i = 0 to i = N-1
#At the boundaries, x0 = 0, xN-1 = 1, and u0 = uN-1 = 0
n = 4
xi = (0.1)/(2**n)
N = 10*(2**n) + 1
epsilon = 0.0001

def ux(index,u,x):
    if index == 0:
        numerator = (3/2)*u[0] - 2*u[1] + (1/2)*u[2]
        denominator = (3/2)*x[0] - 2*x[1] + (1/2)*x[2]
    elif index == N-1:
        numerator = (3/2)*u[-1] - 2*u[-2] + (1/2)*u[-3]
        denominator = (3/2)*x[-1] - 2*x[-2] + (1/2)*x[-3]
    else:
        numerator = u[index + 1] - u[index - 1]
        denominator = x[index + 1] - x[index - 1]
    
    return numerator/denominator
    
def uxx(index,u,x):

    factor = 2
    if index == 0:
        numerator = (x[1] - x[0])*(u[2] - u[0]) - (x[2]-x[0])*(u[1] - u[0])
        denominator = (x[2] - x[0])*(x[1] - x[0])*(x[2] - x[1])
    
    elif index == N-1:
        numerator = (x[-2] - x[-1])*(u[-3] - u[-1]) - (x[-3] - x[-1])*(u[-2] - u[-1])
        denominator = (x[-3] - x[-1])*(x[-2] - x[-1])*(x[-3] - x[-2])

    else:
        term1 = (u[index + 1] - u[index])/(x[index + 1] - x[index])
        term2 = (u[index] - u[index - 1])/(x[index] - x[index - 1])

        numerator = term1 + term2
        denominator = (x[index + 1] - x[index - 1])

    return factor*numerator/denominator

    

def p1(index,u,x):
    return np.sqrt(1 + (np.abs(ux(index,u,x)))**2)

def p2(index, u, x):
    return (1 + (np.abs(uxx(index,u,x)))**2)**0.25

def dxt(index,u,x,i):

    p = p1
    if i == 2:
        p = p2

    coefficient = 1/(2*p(index,u,x)*0.01*(xi**2))
    
    term1 = (p(index + 1,u,x) + p(index,u,x))*(x[index + 1] - x[index])
    term2 = (p(index, u, x) + p(index - 1,u,x))*(x[index] - x[index - 1])
    

    return coefficient*(term1 - term2)


def system3(t,current_solution, i):
    print(t)
    actual_x = current_solution[:N-2]
    actual_u = current_solution[N-2:]

    x = np.zeros(N)
    x[-1] = 1
    x[1:-1] = actual_x

    u = np.zeros(N)
    u[1:-1] = actual_u

    new_x = []
    new_u = []

    for j in range(1,N-1):
        new_x.append(dxt(j,u,x,i))

        numerator1 = u[j+1] - u[j-1]
        denominator1 = x[j+1] - x[j-1]
        term1 = new_x[-1]*numerator1/denominator1

        term2a = (u[j+1]-u[j])/(x[j+1]-x[j])
        term2b = (u[j] - u[j-1])/(x[j] - x[j-1])
        denominator2 = x[j+1] - x[j-1]
        term2 = 2*epsilon*(term2a - term2b)/denominator2

        numerator2 = (u[j+1])**2 - (u[j-1])**2
        denominator3 = 2*(x[j+1] - x[j-1])
        term3 = numerator2/denominator3

        new_u.append(term1 + term2 - term3)

    vector = np.zeros(2*N-4)
    vector[:N-2] = new_x
    vector[N-2:] = new_u

    return vector

y0 = np.zeros(2*N - 4)
x0 = grid_points(n)
y0[:N-2] = x0

y0[N-2:] = initial1(x0)
t_eval = [0,0.2,0.4,0.8,1]

time1 = time.time()
if __name__ == '__main__':
    solution = integrate.solve_ivp(system3, [0,1],y0,'Radau',t_eval,args=[2])
print(time.time() - time1)
t_values = [0, 0.2, 0.4, 0.8, 1]
def plot_time_adaptive(solutions, t):

    line = solutions.t
    #Find the position in the solution where t appears
    index = 0
    while line[index] != t:
        index += 1

    solution_values = []
    x_values = []

    for i, solution in enumerate(solutions.y[N-2:]):
        solution_values.append(solution[index])
        x_values.append(solutions.y[:N-2][i][index])

    
    return solution_values,x_values

def plot_times_adaptive(solutions, t_values):

    for t in t_values:
        solution_values,points = plot_time_adaptive(solutions, t)
        
        plt.plot(points, solution_values,label = f"t = {t}", marker = 'D')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.show()

plot_times_adaptive(solution,t_values)