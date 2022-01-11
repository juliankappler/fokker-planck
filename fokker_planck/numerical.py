#!/usr/bin/env python

import numpy as np
import time
import types

class fokker_planck:
	'''
	This is the base class that contains the shared codebase
	'''
	def __init__(self,parameters):
		'''
		Set simulation parameters, which are provided
		as dictionary in the argument parameters
		'''
		#
		self.set_Nx = False
		self.set_dt = False
		self.set_Nt = False
		#
		self.Nx = 'not set'
		self.dt = 'not set'
		self.Nt = 'not set'
		self.saving_stride = 1
		#
		self.set_parameters(parameters=parameters)


	def set_parameters(self,parameters):
		'''
		Change parameters of an existing instance of this class
		'''
		#
		try:
			self.verbose = parameters['verbose']
		except KeyError:
			self.verbose = True
		#
		try:
			self.Nx = int(parameters['Nx'])
			self.set_Nx = True
		except KeyError:
			pass
		#
		try:
			self.dt = parameters['dt']
			self.set_dt = True
		except KeyError:
			pass
		#
		try:
			self.Nt = int(parameters['Nt'])
			self.set_Nt = True
		except KeyError:
			pass
		#
		try:
			self.saving_stride = int(parameters['saving_stride'])
		except KeyError:
			pass

		if self.set_Nx:
			self.dx = 2./(self.Nx + 1)
			self.one_by_dx_squared = 1/self.dx**2
			self.x_array = np.arange(self.Nx+2,dtype=float)*self.dx - 1.
		if self.set_dt:
			self.t_array = np.arange(self.Nt+1,dtype=float)*self.dt
		if self.set_Nx and self.set_dt:
			self.P_array = np.zeros([self.Nt//self.saving_stride + 1,self.Nx+2],
							dtype=float)
			#
			if self.dt/self.dx**2 > 0.5:
				print("Warning: dx and dt are set such that the forward euler "\
					+ "algorithm for free diffusion is unstable:\n\tdt/dx**2 ="\
					+ "{0:3.5f} > 0.5. Simulation ".format(self.dt/self.dx**2) \
					+ "will likely be unstable.")

		# set boundary conditions
		try:
			self.boundary_condition_left = parameters['boundary_condition_left']
		except KeyError:
			self.boundary_condition_left = 'absorbing'

		try:
			self.boundary_condition_right = parameters['boundary_condition_right']
		except KeyError:
			self.boundary_condition_right = 'absorbing'

		#
		if self.boundary_condition_left == 'robin':
			try:
				self.Al = parameters['Al']
			except KeyError:
				raise RuntimeError("Robin boundary condition chosen for left" \
				+ " boundary, but no coefficient vector Al provided")

		if self.boundary_condition_right == 'robin':
			try:
				self.Ar = parameters['Ar']
			except KeyError:
				raise RuntimeError("Robin boundary condition chosen for right" \
				+ " boundary, but no coefficient vector Ar provided")
		#
		if self.boundary_condition_left == 'periodic':
			if self.boundary_condition_right != 'periodic':
				raise RuntimeError("Left boundary condition is set to periodic"\
				+ ", but right boundary condition is not.")
		if self.boundary_condition_right == 'periodic':
			if self.boundary_condition_left != 'periodic':
				raise RuntimeError("Right boundary condition is set to periodic"\
				+ ", but left boundary condition is not.")


	def get_parameters(self,print_parameters=False):
		'''
		return parameters of an existing instance of this class
		'''
		output_dictionary = {'verbose':self.verbose,
				'Nx':self.Nx,
				'dt':self.dt,
				'Nt':self.Nt,
				'saving_stride':self.saving_stride,
				'boundary_condition_left':self.boundary_condition_left,
				'boundary_condition_right':self.boundary_condition_right,
			 }
		if print_parameters:
			print("Parameters set for this instance:")
			print("verbose       = {0}".format(self.verbose))
			print("Nx            = {0}".format(self.Nx))
			print("Nt            = {0}".format(self.Nt))
			print("dt            = {0}".format(self.dt))
			print("saving_stride = {0}".format(self.saving_stride))
			print("boundary_condition_left = {0}".format(\
						self.boundary_condition_left))
			print("boundary_condition_right = {0}".format(\
						self.boundary_condition_right))
		return output_dictionary

	def print_time_remaining(self,step,end='\r'):
		elapsed_time = time.time() - self.system_time_at_start_of_simulation
		m_elapsed, s_elapsed = divmod(elapsed_time, 60)
		h_elapsed, m_elapsed = divmod(m_elapsed, 60)
		remaining_time = (self.Nt/step -1)*elapsed_time
		m_remaining, s_remaining = divmod(remaining_time, 60)
		h_remaining, m_remaining = divmod(m_remaining, 60)
		print("Running simulation. Progress: {0}%, elapsed time: {1:d}:{2:02d}:{3:02d}, remaining time: {4:d}:{5:02d}:{6:02d}\t\t\t".format(int(step/self.Nt*100.),
																														int(np.round(h_elapsed)),int(np.round(m_elapsed)),int(np.round(s_elapsed)),
																														int(np.round(h_remaining)),int(np.round(m_remaining)),int(np.round(s_remaining))),
			  end=end)


	def get_boundary_value_left(self,current_P):
		#
		if self.boundary_condition_left == 'absorbing':
			return 0.
		elif self.boundary_condition_left == 'periodic':
			return self.dt * self.one_by_dx_squared \
				* ( -self.dx * ( self.a_array[1]*current_P[1] - \
								 self.a_array[-2]*current_P[-2] \
								 ) /2. \
					+ ( self.D_array[1] * current_P[1] \
						- 2 * self.D_array[0] * current_P[0] \
						+ self.D_array[-2]*current_P[-2] \
						)
				)
		elif self.boundary_condition_left == 'no-flux':
			A11_times_two_dx = 2 * self.a_array[0] * self.dx \
							 + self.D_array[2] - 4 * self.D_array[1] \
							 + 3 * self.D_array[0]
			A12 = -self.D_array[0]
			B1_times_two_dx = 0
		elif self.boundary_condition_left == 'robin':
			A11_times_two_dx = 2 * self.Al[0] * self.dx
			A12 = self.Al[1]
			B1_times_two_dx = 2 * self.Al[2] * self.dx
		else:
			raise RuntimeError("No valid boundary condition for left boundary provided")
		#
		if A11_times_two_dx == 3*A12:
			raise RuntimeError("Given boundary condition does not specify P(x = -1,t)")
		#
		return (B1_times_two_dx + A12*( current_P[2] - 4*current_P[1] ) ) \
				/ ( A11_times_two_dx - 3 * A12 )


	def get_boundary_value_right(self,current_P):
		#
		if self.boundary_condition_right == 'absorbing':
			return 0.
		elif self.boundary_condition_right == 'periodic':
			return self.dt * self.one_by_dx_squared \
				* ( -self.dx * ( self.a_array[1]*current_P[1] - \
								 self.a_array[-2]*current_P[-2] \
								 ) /2. \
					+ ( self.D_array[1] * current_P[1] \
						- 2 * self.D_array[0] * current_P[0] \
						+ self.D_array[-2]*current_P[-2] \
						)
				)
		elif self.boundary_condition_right == 'no-flux':
			A21_times_two_dx = 2 * self.a_array[self.Nx+1] * self.dx \
					- 3 * self.D_array[self.Nx+1] + 4 * self.D_array[self.Nx] \
					- self.D_array[self.Nx-1]
			A22 = -self.D_array[self.Nx+1]
			B2_times_two_dx = 0
		elif self.boundary_condition_right == 'robin':
			A21_times_two_dx = 2 * self.Ar[0] * self.dx
			A22 = self.Ar[1]
			B2_times_two_dx = self.Ar[2] * self.dx
		else:
			raise RuntimeError("No valid boundary condition for right boundary provided")
		#
		if A21_times_two_dx == -3*A22:
			raise RuntimeError("Given boundary condition does not specify P(x = 1,t)")
		#
		return (B2_times_two_dx + A22*( 4*current_P[self.Nx] - current_P[self.Nx-1] ) ) \
				/ ( A21_times_two_dx + 3 * A22 )


	def return_results(self):
		output_dictionary = {'t':self.t_array[::self.saving_stride],
							'x':self.x_array,
							'y':self.P_array,
							'dx':self.dx,'Nx':self.Nx,
							'dt':self.dt,'Nt':self.Nt,
							'saving_stride':self.saving_stride,
							'boundary_condition_left':self.boundary_condition_left,
							'boundary_condition_right':self.boundary_condition_right,
							}
		return output_dictionary



class forward_euler(fokker_planck):
	#

	def simulate(self,D,a,P0):
		#
		# check if all parameters are set
		if self.set_Nx == False:
			raise RuntimeError("Parameter Nx not set")
		if self.set_dt == False:
			raise RuntimeError("Parameter dt not set")
		if self.set_Nt == False:
			raise RuntimeError("Parameter Nt not set")
		#
		#
		# check if the diffusivity is a constant number, a constant function, or
		# a time-dependent function:
		if isinstance(D,float):
			if self.verbose:
				print('Found temporally and spatially constant diffusivity D')
			self.D_array = D*np.ones(self.Nx+2,dtype=float)
			D_time_dependent = False
		elif isinstance(D,types.FunctionType):
			try:
				self.D_array = D(self.x_array,0)
				if self.verbose:
					print('Found time-dependent diffusivity function D(x,t)')
				D_time_dependent = True
			except TypeError:
				self.D_array = D(self.x_array)
				if self.verbose:
					print('Found time-independent diffusivity function D(x)')
				D_time_dependent = False

		# check if the drift is a constant number, a constant function, or
		# a time-dependent function:
		if isinstance(a,float):
			if self.verbose:
				print('Found temporally and spatially constant drift a')
			self.a_array = a*np.ones(self.Nx+2,dtype=float)
			a_time_dependent = False
		elif isinstance(a,types.FunctionType):
			try:
				self.a_array = a(self.x_array,0)
				if self.verbose:
					print('Found time-dependent drift function a(x,t)')
				a_time_dependent = True
			except TypeError:
				self.a_array = a(self.x_array)
				if self.verbose:
					print('Found time-independent drift function a(x)')
				a_time_dependent = False
		# if boundary conditions are periodic, check whether provided drift
		# and diffusivity are periodic (a(-1) = a(1))
		#
		if len(P0) == self.Nx:
			self.P_array[0,1:-1] = P0.copy()
		elif len(P0) == (self.Nx+2):
			self.P_array[0] = P0.copy()
			if verbose:
				print("Provided boundary values from initial condition will "\
					+ "be discarded.")
		else:
			raise RuntimeError("Invalid initial condition. Initial condition " \
					+ "must be array of length Nx or Nx+2.")
		# set initial values for P( x = -1 ) and P( x = 1 )
		if self.boundary_condition_left == 'periodic':
			if self.P_array[0,0] != self.P_array[0][-1]:
				raise RuntimeError("Periodic boundary conditions chosen, but"\
				+ " boundary values of initial condition are not identical")
		else:
			self.P_array[0,0] = self.get_boundary_value_left(self.P_array[0])
			self.P_array[0,-1] = self.get_boundary_value_right(self.P_array[0])



		self.system_time_at_start_of_simulation = time.time()

		current_P = self.P_array[0].copy()
		for step,next_time in enumerate(self.t_array[1:]):
			if self.verbose:
				if ( step % int(self.Nt/100) == 0 ) and step > 0:
					self.print_time_remaining(step)
			#

			# calculate values of interior points in next timestep
			next_P = current_P.copy()
			next_P[1:-1] +=  self.dt * self.one_by_dx_squared \
				* ( -self.dx * ( self.a_array[2:]*current_P[2:] - \
								 self.a_array[:-2]*current_P[:-2] \
								 ) /2. \
					+ ( self.D_array[2:] * current_P[2:] \
						- 2 * self.D_array[1:-1] * current_P[1:-1] \
						+ self.D_array[:-2]*current_P[:-2] \
						)
				)
			#
			# Set boundary conditions at next time step
			# For periodic boundary conditions, we use the current diffusivity
			# and drift in the boundary conditions. For all other boundary
			# conditions, we use the diffusivity and drift of the next timestep.
			# This means for periodic boundary conditions we update the
			# boundary points *before* updating D_array and a_array, whereas
			# for all other boundary conditions we set the boundary points
			# *after* updating D_array and a_array:
			#
			# update boundary points if periodic
			if self.boundary_condition_left == 'periodic':
				next_P[0] = current_P[0] + self.get_boundary_value_left(current_P)
				next_P[-1] = next_P[0]
				if step % 10000 == 0:
					print(next_P[0],next_P[-1])
			#
			# update D and a (only if they are time-dependent)
			if D_time_dependent:
				self.D_array = D(self.x_array,next_time)
			if a_time_dependent:
				self.a_array = a(self.x_array,next_time)
			#
			# update boundary points if not periodic
			if self.boundary_condition_left != 'periodic':
				next_P[0] = self.get_boundary_value_left(next_P)
				next_P[-1] = self.get_boundary_value_right(next_P)
			#
			# save to output array
			if (step+1)%self.saving_stride == 0:
				self.P_array[(step+1)//self.saving_stride] = next_P.copy()
			#
			current_P = next_P
		#
		if self.verbose:
			self.print_time_remaining(step+1,end='\n')
		return self.return_results()
