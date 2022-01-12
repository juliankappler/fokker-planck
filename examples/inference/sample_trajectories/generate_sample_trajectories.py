#!/usr/bin/env python

import numpy as np
import pickle
import datetime

######################################################################
# This code generates sample trajectories of various lengths, which  #
# are used by the example notebook                                   #
#    "Parameter inference via Kramers-Moyal coefficients.ipynb"      #
# to demonstrate parameter inference via Kramers-Moyal coefficients. #
######################################################################

Fn = 'sample_trajectories.pkl' # filename for the output data

# initialize random number generator
rng = np.random.default_rng()

# Set simulation parameters
dt = 1e-5 # timestep for simulation
dt_out = 1e-4 # timestep of saved trajectories
stride_out = int(np.round(dt_out/dt))
N_sim = 4000 # number of trajectories to be generated

# For each trajectory, we use a random number of timesteps. This is to
# show that the parameter inference works with trajectory samples of
# various lengths.
# The number of timesteps for each trajectory is sampled from a uniform
# distribution on the interes from N_steps_min to N_steps_max, which are
# defined here:
N_steps_mean = int(np.round(0.05/dt))
N_steps_min = N_steps_mean*0.5
N_steps_max = N_steps_mean*1.5
# generate an array that contains the number of steps of all trajectories:
N_steps = rng.integers(low=N_steps_min,
                        high=N_steps_max,
                        size=N_sim)

# For each simulation, the initial condition is drawn from a uniform
# distribution on [x_L, x_R], using the following values:
x_L = -1.5
x_R = 1.5

# Definition of the parameters of the Langevin equation.
#
# We simulate the Ito-Langevin equation
#          dX_t = a(X_t) * dt  +  sqrt(2*D) * dW_t,
# where dX_t is the increment of the reaction coordinate at time t,
# a(x) is the drift, D is the diffusivity (which we assume to be a constant
# number, meaning we consider additive noise), and dW_t is the increment
# of the Wiener process.
#
# We consider a constant diffusivity
D = 1.
# and a gradient drift a(x) = -dU/dx that originates from a double-well potential
# U(x) = U0 * ( x**2 - 1 )**2, so that a(x) = -4 * U0 * ( x**2 - 1 ) * x.
U0 = 2.
a = lambda x: -4*U0*(x**2 - 1)*x


trajectories = []
print("{time}\tStarting simulation of {total} trajectories...".format(
                total =N_sim,
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
for i in range(N_sim):
    print("{time}\tRunning simulation {cur_int} of {total}...".format(cur_int=i+1,
                    total =N_sim,
                time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                end='\r')
    #
    # generate array containing current trajectory
    current_trajectory = np.zeros(N_steps[i]+1,dtype=float)
    # generate initial condition for current trajectory
    current_trajectory[0] = rng.random()*(x_R-x_L) + x_L
    #
    # generate all random numbers for the current simulation
    random_numbers_for_current_simulation = rng.normal(size=N_steps[i])
    random_numbers_for_current_simulation *= np.sqrt(2*D*dt)
    #
    # run simulation using Euler-Maruyama algorithm
    for j,current_x in enumerate(current_trajectory[:-1]):
        current_trajectory[j+1] = dt*a(current_x) \
                            + random_numbers_for_current_simulation[j] \
                            + current_x
    #
    # append current trajectory to list of trajectories
    trajectories.append(current_trajectory[::stride_out])

print("{time}\tFinished simulation {total} trajectories.              ".format(
                total =N_sim,
            time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            end='\n')

# save resulting trajectories
pickle.dump(trajectories,open(Fn,'wb'))
