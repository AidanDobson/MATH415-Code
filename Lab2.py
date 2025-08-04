import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg as sci

def construct_tridiagonal(coefficients,N,h):
    upper = coefficients[0]*np.ones(N-2)
    diagonal = coefficients[1]*np.ones(N-1)
    lower = coefficients[2]*np.ones(N-2)

    A = (1/(h**2))*(np.diag(upper,+1) + np.diag(diagonal) + np.diag(lower,-1))

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

#Question 2.b
def function(x):
    return 0

grid_points1, solution1 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.1,1))
grid_points2, solution2 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.01,1))
grid_points3, solution3 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.001,1))
grid_points4, solution4 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.0001,1))
grid_points5, solution5 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.00001,1))

plt.plot(grid_points1,solution1,label = "Epsilon = 0.1")
plt.plot(grid_points2,solution2,label = "Epsilon = 0.01")
plt.plot(grid_points3,solution3,label = "Epsilon = 0.001")
plt.plot(grid_points4,solution4,label = "Epsilon = 0.0001")
plt.plot(grid_points5,solution5,label = "Epsilon = 0.00001")

plt.legend()
plt.show()

#Question 2.c

"""def compute_coefficients2(h, c1, c2):
    return [c1,h**2 - h*c2 - 2*c1, c1 + h*c2]

grid_points1, solution1 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.1,1),coef_function=compute_coefficients2)
grid_points2, solution2 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.01,1),coef_function=compute_coefficients2)
grid_points3, solution3 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.001,1),coef_function=compute_coefficients2)
grid_points4, solution4 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.0001,1),coef_function=compute_coefficients2)
grid_points5, solution5 = solve_dirichlet_bvp((0,0),(1,1),1/100,function,(0.00001,1),coef_function=compute_coefficients2)

plt.plot(grid_points1,solution1,label = "Epsilon = 0.1")
plt.plot(grid_points2,solution2,label = "Epsilon = 0.01")
plt.plot(grid_points3,solution3,label = "Epsilon = 0.001")
plt.plot(grid_points4,solution4,label = "Epsilon = 0.0001")
plt.plot(grid_points5,solution5,label = "Epsilon = 0.00001")
plt.legend()
plt.show()"""

    

    


