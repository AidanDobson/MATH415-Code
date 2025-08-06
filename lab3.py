import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg as sci
from scipy.optimize import fsolve


def construct_five_point_laplacian(N,h):
    main_diag = -4*np.ones(N**2)
    main_off_diag = np.ones(N**2 - 1)
    secondary_off_diag = np.ones(N**2 - N)

    offsets = [-N,-1,0,1,N]
    diagonals = [secondary_off_diag,main_off_diag,main_diag,main_off_diag,secondary_off_diag]

    A = scipy.sparse.diags(diagonals,offsets,format='csr')

    return A

#Compute the terms on the diagonals of the matrix. 
def compute_coefficients(h, c1, c2):
    return [0.5*(2*c1 - h*c2), 0.5*(2*(h**2)-4*c1), 0.5*(2*c1 + h*c2)]

def function_vector(N, h, starting_point, ending_point, function,c1,c2,in1,in2,coef_function=compute_coefficients):
    grid_points = np.linspace(starting_point,ending_point,N+1)[1:-1]
    coefficients = coef_function(h,c1,c2)
    
    function_vector = [function(grid_points[0])-(1/(h**2))*coefficients[0]*in1]

    for point in grid_points[1:-1]:
        function_vector.append(function(point))

    function_vector.append(function(grid_points[0])-(1/(h**2))*coefficients[2]*in2)
    return np.array(function_vector)

def solve_dirichlet_bvp(initial_1,initial_2,h,function,coefs,coef_function=compute_coefficients):
    starting_point, in1 = initial_1
    ending_point, in2 = initial_2
    c1,c2 = coefs
    N = int((ending_point-starting_point)/h)

    coefficients = coef_function(h,c1,c2)

    matrix = scipy.sparse.csr_matrix(construct_tridiagonal(coefficients,N,h))
    rhs = function_vector(N,h,starting_point,ending_point,function,c1,c2,in1,in2,coef_function)
    
    interior_sols = sci.spsolve(matrix,rhs)

    solution = np.zeros(N+1)
    solution[1:-1] = interior_sols
    solution[0] = in1
    solution[-1] = in2

    return np.linspace(starting_point,ending_point,N+1),solution