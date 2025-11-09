import numpy as np
import scipy.sparse as scp
import matplotlib.pyplot as plt

#Some general helper functions for the whole section

def solve(factor_matrix,rhs):
    #Compress the matrix, since it's sparse, to make solving faster
    csr_factor_matrix = scp.csr_matrix(factor_matrix)

    solution = scp.linalg.spsolve(csr_factor_matrix,rhs)

    return solution

#Computes the exact solution based on the function in the question. Notice that since the boundary conditions are Dirichlet, we don't need the values at the boundaries, since their value is 0.
def grid_points(dx,dy,xbounds,ybounds):
    xinputs = []
    yinputs = []

    x_steps = int((xbounds[1] - xbounds[0])/dx - 1)
    y_steps = int((ybounds[1] - ybounds[0])/dy - 1)

    for xstep in range(x_steps):
        for ystep in range(y_steps):
            xinputs.append(xbounds[0] + dx*(xstep + 1))
            yinputs.append(ybounds[0] + dy*(ystep + 1))
    
    return np.array(xinputs),np.array(yinputs)

def compute_dimension(dx,dy,xbounds,ybounds):
    x_dimension = int((xbounds[1]-xbounds[0])/dx) - 1
    y_dimension = int((ybounds[1] - ybounds[0])/dy) - 1

    return x_dimension*y_dimension, [x_dimension,y_dimension]
def boundary_function(x,y):
    return np.exp(x+y/2)
    #return 1
def five_point_stencil(dx,dy,xbounds,ybounds):

    matrix_dimension,dimensions = compute_dimension(dx,dy,xbounds,ybounds)

    #Create the factors for the diagonals of the matrix
    f1 = (dx)**2
    f2 = (dy)**2
    f3 = -2*(dx**2 + dy**2)
    f4 = (dy)**2
    f5 = (dx)**2 

    #Create the diagonals of the matrix
    d1 = f1*np.ones(matrix_dimension - dimensions[0])
    d2 = f2*np.ones(matrix_dimension - 1)
    main_d = f3*np.ones(matrix_dimension)
    d4 = f4*np.ones(matrix_dimension - 1)
    d5 = f5*np.ones(matrix_dimension - dimensions[0])

    #Create the matrix:
    A = np.diag(d1, dimensions[0]) + np.diag(d2, +1) + np.diag(main_d) + np.diag(d4, -1) + np.diag(d5, - dimensions[0])

    #Correct the areas where we stitch together copies of T:
    for i in range(int(matrix_dimension/dimensions[0]) - 1):
        A[(i+1)*dimensions[0],(i+1)*dimensions[0] - 1] = 0
        A[(i+1)*dimensions[0] - 1,(i+1)*dimensions[0]] = 0

    return A

def nine_point_stencil(h,xbounds,ybounds):
    matrix_dimension,dimensions = compute_dimension(h,h,xbounds,ybounds)  

    #Create the diagonals of the matrix
    d1 = 1*np.ones(matrix_dimension - dimensions[0] - 1)
    d2 = 4*np.ones(matrix_dimension - dimensions[0])
    d3 = 1*np.ones(matrix_dimension - dimensions[0] + 1)

    d4 = 4*np.ones(matrix_dimension - 1)
    main_d = -20*np.ones(matrix_dimension)
    d6 = 4*np.ones(matrix_dimension - 1)

    d7 = 1*np.ones(matrix_dimension - dimensions[0] + 1)
    d8 = 4*np.ones(matrix_dimension - dimensions[0])
    d9 = 1*np.ones(matrix_dimension - dimensions[0] - 1)

    #Create the matrix:
    A = np.diag(d1, dimensions[0] + 1) + np.diag(d2, dimensions[0]) + np.diag(d3,dimensions[0] - 1) + np.diag(d4,1) + np.diag(main_d) + np.diag(d6,-1) + np.diag(d7,1 - dimensions[0]) + np.diag(d8, -dimensions[0]) + np.diag(d9, -dimensions[0] - 1)

    for n in range(dimensions[0]):
        A[n*dimensions[0], dimensions[0]*(n+1) - 1] = 0
        A[dimensions[0]*(n+1) - 1, n*dimensions[0]] = 0

    for n in range(dimensions[0] - 1):
        A[n*dimensions[0], dimensions[0]*(n+2) - 1] = 0
        A[dimensions[0]*(n+2) - 1, n*dimensions[0]] = 0

        A[dimensions[0]*(n+1) - 1, dimensions[0]*(n+1)] = 0
        A[dimensions[0]*(n+1), dimensions[0]*(n+1) - 1] = 0

    for n in range(dimensions[0] - 2):
        A[dimensions[0]*(n+1) - 1, (n+2)*dimensions[0]] = 0
        A[(n+2)*dimensions[0], dimensions[0]*(n+1) - 1] = 0

    return A


def exact_solution(dx,dy,xbounds,ybounds):
    x, y = grid_points(dx,dy,xbounds,ybounds)
    return np.exp(x + y/2)

def error(dx,dy,xbounds,ybounds,solution):
    real_solution = exact_solution(dx,dy,xbounds,ybounds)
    return np.linalg.norm(real_solution - solution,np.inf)

#Compute what we need to modify f by based on the boundary conditions using the five point stencil method
def fp_boundaries(dx,dy,xbounds,ybounds):
    matrix_dimension,dimensions = compute_dimension(dx,dy,xbounds,ybounds)

    boundaries = np.zeros(matrix_dimension)

    for row in range(dimensions[1]):
        x = xbounds[0] + dx
        y = ybounds[0] + (1+row)*dy
        
        left_term = boundary_function(x-dx,y)
        boundaries[dimensions[0]*row] += (-(dy**2))*left_term

        x += (dimensions[0] - 1)*dx

        right_term = boundary_function(x + dx, y)
        boundaries[dimensions[0]*(row + 1) - 1] += (-(dy**2))*right_term

    
    for column in range(dimensions[0]):
        x = xbounds[0] + (1+column)*dx
        y = ybounds[0] + dy
        
        bottom_term = boundary_function(x,y-dy)
        boundaries[column] += (-(dx**2))*bottom_term

        y += (dimensions[1] - 1)*dy

        top_term = boundary_function(x, y+dy)
        boundaries[column + dimensions[0]*(dimensions[1]-1)] += (-(dx**2))*top_term
    
    return np.array(boundaries)

def np_boundaries(h, xbounds, ybounds):
    matrix_dimension,dimensions = compute_dimension(h,h,xbounds,ybounds)

    boundaries = np.zeros(matrix_dimension)

    for row in range(dimensions[1]):
        x = xbounds[0] + h
        y = ybounds[0] + (1+row)*h
        
        b_left_term = boundary_function(x-h,y-h)
        m_left_term = boundary_function(x-h,y)
        t_left_term = boundary_function(x-h,y+h)

        boundaries[dimensions[0]*row] -= b_left_term + 4*m_left_term + t_left_term

        x += (dimensions[0] - 1)*h

        b_right_term = boundary_function(x + h, y-h)
        m_right_term = boundary_function(x+h, y)
        t_right_term = boundary_function(x+h,y+h)

        boundaries[dimensions[0]*(row + 1) - 1] -= b_right_term + 4*m_right_term + t_right_term

    for column in range(dimensions[0]):
        x = xbounds[0] + (1+column)*h
        y = ybounds[0] + h
        
        l_bottom_term = boundary_function(x-h,y-h)
        m_bottom_term = boundary_function(x, y-h)
        r_bottom_term = boundary_function(x + h, y-h)

        boundaries[column] -= l_bottom_term + 4*m_bottom_term + r_bottom_term

        y += (dimensions[1] - 1)*h

        l_top_term = boundary_function(x-h, y+h) 
        m_top_term = boundary_function(x, y+h)
        r_top_term = boundary_function(x+h,y+h)

        boundaries[column + dimensions[0]*(dimensions[1]-1)] -= l_top_term + 4*m_top_term + r_top_term

    boundaries[0] += boundary_function(xbounds[0],ybounds[0])
    boundaries[dimensions[0] - 1] += boundary_function(xbounds[1], ybounds[0])
    boundaries[dimensions[0]*(dimensions[0]-1)] += boundary_function(xbounds[0],ybounds[1])
    boundaries[-1] += boundary_function(xbounds[1],ybounds[1])

    return boundaries
def ftest(x,y,h):
    return (h**2)*np.ones(len(x))

def test_boundary(dx,xbounds,ybounds):
    return np.zeros((int((xbounds[1]-xbounds[0])/h) - 1)**2)

def grid_points(dx,dy,xbounds,ybounds):
    xinputs = []
    yinputs = []

    x_steps, y_steps = compute_dimension(dx,dy,xbounds,ybounds)[1]

    for xstep in range(x_steps):
        for ystep in range(y_steps):
            xinputs.append(xbounds[0] + dx*(xstep + 1))
            yinputs.append(ybounds[0] + dy*(ystep + 1))
    
    return np.array(xinputs),np.array(yinputs)

def rhs(dx,dy,xbounds,ybounds,boundary_fn, f):
    if boundary_fn == fp_boundaries:
        boundary_terms = boundary_fn(dx,dy,xbounds,ybounds)
    else: 
        boundary_terms = boundary_fn(dx,xbounds,ybounds)

    x, y = grid_points(dx,dy,xbounds,ybounds)

    if f == f1:
        initial_rhs = f(x,y,dx,dy)
    else:
        initial_rhs = f(x,y,dx)
    
    return np.array(initial_rhs + boundary_terms)

def f1(x,y,dx,dy):
    return ((dx)**2)*((dy)**2)*1.25*np.exp(x + y/2)

def f2(x,y,h):
    return (6*(h**2))*1.25*np.exp(x + y/2)

def f3(x,y,h):
    return ((h**2)/12 + (h**2)/48 + 1)*f2(x,y,h)

#Question 1.a

#With dx = dy = 0.1
xbounds = [0,1]
ybounds = [0,1]

dx = 0.1
dy = 0.1

A = five_point_stencil(dx,dy,xbounds,ybounds)
b = rhs(dx,dy,xbounds,ybounds,fp_boundaries,f1)

solution = solve(A,b)



#With dx = dy = 0.2:
dx = 0.01
dy = 0.01

A = five_point_stencil(dx,dy,xbounds,ybounds)
b = rhs(dx,dy,xbounds,ybounds,fp_boundaries,f1)

solution = solve(A,b)



#Question 1.b

#With the domain [0,1]*[0,2]
xbounds = [0,1]
ybounds = [0,2]

dx = 0.1
dy = 0.1

A = five_point_stencil(dx,dy,xbounds,ybounds)
b = rhs(dx,dy,xbounds,ybounds,fp_boundaries,f1)

solution = solve(A,b)

#print(error(dx,dy,xbounds,ybounds,solution))

#Question 1.c
#With dx = 0.1 and dy = 0.25

dx = 0.1
dy = 0.25

A = five_point_stencil(dx,dy,xbounds,ybounds)
b = rhs(dx,dy,xbounds,ybounds,fp_boundaries,f1)

solution = solve(A,b)

#print(error(dx,dy,xbounds,ybounds,solution))

#Question 2.i

#With h = 0.1

xbounds = [0,1]
ybounds = [0,1]

h = 0.1

A = nine_point_stencil(h,xbounds,ybounds)
b = rhs(h,h,xbounds,ybounds,np_boundaries,f2)

solution = solve(A,b)



#With h = 0.01:

h = 0.01

A = nine_point_stencil(h,xbounds,ybounds)
b = rhs(h,h,xbounds,ybounds,np_boundaries,f2)

solution = solve(A,b)




#Question 2.ii

h = 0.1

A = nine_point_stencil(h,xbounds,ybounds)
b = rhs(h,h,xbounds,ybounds,np_boundaries,f3)

solution = solve(A,b)



h = 0.01

A = nine_point_stencil(h,xbounds,ybounds)
b = rhs(h,h,xbounds,ybounds,np_boundaries,f3)

solution = solve(A,b)



"""h_values = np.logspace(-1,-2,20)
errors = []
for h in h_values:
    A = nine_point_stencil(h,xbounds,ybounds)
    b = rhs(h,h,xbounds,ybounds,np_boundaries,f3)
    errors.append(error(h,h,xbounds,ybounds,solve(A,b)))

plt.loglog(h_values,errors)
plt.show()"""


n = 32
h = 1/(n+1)
xbounds = [0,1]
ybounds = [0,1]

A = ((n+1)**4)*five_point_stencil(h,h,xbounds,ybounds)

b = np.array(((n+1)**4)*rhs(h,h,xbounds,ybounds,fp_boundaries,f1))

solution = solve(A,b)
print(solution)
print(exact_solution(h,h,xbounds,ybounds))
print(error(h,h,xbounds,ybounds,solution))
n = 100
h = 1/(n+1)
A = ((n+1)**4)*five_point_stencil(h,h,xbounds,ybounds)

b = np.array((n+1)**4)*rhs(h,h,xbounds,ybounds,fp_boundaries,f1)

solution = solve(A,b)
print(solution)
print(exact_solution(h,h,xbounds,ybounds))
print(error(h,h,xbounds,ybounds,solution))




new_solution = solution.reshape((n,n))
N = new_solution.shape[0]
U_full = np.zeros((N+2, N+2))
U_full[1:-1, 1:-1] = new_solution
# Create full grid including boundaries
x_full = np.linspace(0, 1, N+2)
y_full = np.linspace(0, 1, N+2)
Xf, Yf = np.meshgrid(x_full, y_full, indexing='ij')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(Xf, Yf, U_full, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.title('Numerical solution of Poisson\'s equation with Dirichlet BCs')
plt.tight_layout()
plt.show()
