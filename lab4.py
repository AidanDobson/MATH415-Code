import numpy as np
import matplotlib.pyplot as plt



#Question 2

def forward_euler(step_size,steps,f,initial_conditions):

    t0,u0 = initial_conditions
    t_values = [t0]
    u_values = [u0]

    for step in range(steps):
        u_values.append(u_values[-1] + step_size*f(t_values[-1],u_values[-1]))
        t_values.append(t_values[-1] + step_size)

    return t_values,u_values


#Question 2.a

def derivative1(t,u):
    return 10*u

def actual_solution(t,u):
    return np.exp(10*t)

step_size = 0.01
initial_conditions = (0,1)
num_steps = int(10/step_size)
output = forward_euler(step_size,num_steps,derivative1,initial_conditions)
plt.plot(output[0],output[1])

actual_solns = []
for value in output[0]:
    actual_solns.append(actual_solution(value,0))

plt.plot(output[0],actual_solns)
plt.show()

