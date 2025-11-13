import numpy as np
import scipy.sparse as scp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Some general function for the questions
def solve(factor_matrix,rhs):
    #Compress the matrix, since it's sparse, to make solving faster
    csr_factor_matrix = scp.csr_matrix(factor_matrix)

    solution = scp.linalg.spsolve(csr_factor_matrix,rhs)

    return np.array(solution)

def correct(un, current, k,  f):
    return un + k*f(current)

def f1(u):
    return 100*np.sin(u)

def df1(u):
    return 100*np.cos(u)

def newton(initial,f,df,n):
    current_value = initial
    for i in range(n):
        current_value = initial - f(current_value)/df(current_value)
    
    return current_value

def construct_tridiagonal(coefficients,N):
    upper = coefficients[0]*np.ones(N-2)
    diagonal = coefficients[1]*np.ones(N-1)
    lower = coefficients[2]*np.ones(N-2)

    A = (np.diag(upper,+1) + np.diag(diagonal) + np.diag(lower,-1))

    return A

#Question 1.d
n = 1000
k = 0.001

solutions = [1]
times = [0]
uhat = solutions[-1] + k*f1(solutions[-1])

for i in range(n-1):
    uhat = solutions[-1] + k*f1(uhat)

solutions.append(uhat)
times.append(k)

while abs(solutions[-1] - solutions[-2]) > 10e-6:
    uhat = solutions[-1] + k*f1(solutions[-1])

    for i in range(n-1):
        uhat = solutions[-1] + k*f1(uhat)

    times.append(times[-1] + k)
    solutions.append(uhat)

plt.plot(times,solutions)
plt.xlabel('t')
plt.ylabel('u')
plt.show()
print(len(solutions))
print(solutions)

#Question 1.e
#k = 0.016
solutions = [1]
uhat = solutions[-1] + k*f1(solutions[-1])

solutions.append(newton(uhat,f1,df1,n-1))

while abs(solutions[-1] - solutions[-2]) > 10e-6:
    uhat = solutions[-1] + k*f1(solutions[-1])
    solutions.append(newton(uhat,f1,df1,n-1))

print(solutions)
print(len(solutions))
#Question 2.

def grid_points(xbounds,h):
    points = [xbounds[0]]
    while round(points[-1],8) <= xbounds[1] - h:
        points.append(points[-1] + h)
    return np.array(points)



def rhs(m,previous_solution,xbounds,boundary_function,time):
    new_rhs = []
    
    for i in range(len(previous_solution) - 2):
        new_rhs.append(m*previous_solution[i] + (1-2*m)*previous_solution[i+1] + m*previous_solution[i+2])

    new_rhs[0] += m*boundary_function(xbounds[0],time)
    new_rhs[-1] += m*boundary_function(xbounds[1],time)

    return new_rhs


def cn(k,h,kappa,boundary_function, initial_function, xbounds):
    m = (k*kappa)/(2*(h**2))
    time = 0
    times = [0]
    solutions = [initial_function(grid_points(xbounds,h))]
    
    time += k

    while time < 1:
        times.append(time)
        b = rhs(m, solutions[-1],xbounds, boundary_function, time)
        A = construct_tridiagonal([-m, 1, -m],int((xbounds[1]-xbounds[0])/h))

        partial_solution = solve(A,b)
        full_solution = np.zeros(int((xbounds[1]-xbounds[0])/h) + 1)
        full_solution[0] = boundary_function(xbounds[0],time)
        full_solution[-1] = boundary_function(xbounds[1],time)
        
        full_solution[1:-1] = partial_solution
        solutions.append(full_solution)
        
        time += k
        
    return solutions,times

def exact_solution(x,t):
    return (1/(np.sqrt(4*150*0.02*t + 1)))*np.exp((-150*((x-0.4)**2))/(4*150*0.02*t + 1))

def b1(x,t):
    return exact_solution(x,t)

def i1(x):
    return exact_solution(x,0)

def error(solution,time,n,xbounds):
    x_values = np.linspace(xbounds[0],xbounds[1],n+1)

    return np.linalg.norm(exact_solution(x_values,time) - solution)
    

n = 100
h = 1/(n)
k = 4*h
xbounds = [0,1]

grid = np.linspace(0,1,n+1)

solutions, times = cn(k,h,0.02,b1,i1,xbounds)

X, T = np.meshgrid(grid, times)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface_plt = ax.plot_surface(T,np.array(solutions),X, cmap = cm.coolwarm)
fig.colorbar(surface_plt, shrink=0.5, aspect=5)
ax.view_init(elev=-70,azim=170,roll = -100)
plt.xlabel("Time")
plt.ylabel("Solution")
ax.set_zlabel("x")
plt.show()

h_values = np.logspace(0,-3,60)
errors = []

for h in h_values:
    k = 4*h
    print(h)
    solution = cn(k,h,0.02,b1,i1,xbounds)[0][-1]

    errors.append(error(solution,0.5,int(1/h),xbounds))

plt.loglog(h_values,errors)
plt.show()

#def forward_euler(initial_value)
   

