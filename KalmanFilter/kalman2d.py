""" 
2D Kalman Filter
Author: Ankush Gola

Tracks the 2D (x, y) position of an object.

Based on the Kalman Filer equations on http://en.wikipedia.org/wik/Kalman_filter
Some code taken from http://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python0
"""

import numpy as np
import matplotlib.pyplot as plt

class Kalman2D:
	"""
	2D Kalman Filter implementation
	"""
	def __init__(self, _X, _P, _Mot = np.matrix('0. 0. 0. 0.').T, _Q = np.matrix(np.eye(4))):
		"""
		constructor

		X: initial state 4-vector [x, y, x', y']
		P: initial uncertainty covariance matrix
		Mot: external motion added to state vector X
		Q: Motion noise
		"""
		self.X = _X
		self.P = _P
		self.Mot = _Mot
		self.Q = _Q

		# state transition matrix
		self.F = np.matrix('1. 0. 1. 0.; 0. 1. 0. 1.; 0. 0. 1. 0.; 0. 0. 0. 1.')

		# observation matrix
		self.H = np.matrix('''
			1. 0. 0. 0.;
			0. 1. 0. 0.''')

	def update(self, Meas, R):
		"""
		transition states, update matrices

		Meas: measured position
		R: Measurement noise
		"""
		Y = np.matrix(Meas).T - self.H * self.X
		S = self.H * self.P * (self.H.T) + R  # residual covarianced
		K = self.P * self.H.T * S.I    # Kalman gain
		self.X = self.X + K*Y
		I = np.matrix(np.eye(self.F.shape[0])) # identity matrix
		self.P = (I - K*self.H)*self.P

		# PREDICT x, P based on motion
		self.X = self.F*self.X + self.Mot
		self.P = self.F*self.P*self.F.T + self.Q


