import sys
sys.path.append('../KalmanFilter')
import numpy as np
import matplotlib.pyplot as plt
from kalman2d import Kalman2D

def demo_kalman_2d():
	kalman = Kalman2D(np.matrix('0. 0. 0. 0.').T, np.matrix(np.eye(4))*1000)
	N = 20
	true_x = np.linspace(0.0, 10.0, N)
	true_y = true_x**2
	observed_x = true_x + 0.05*np.random.random(N)*true_x
	observed_y = true_y + 0.05*np.random.random(N)*true_y
	plt.plot(observed_x, observed_y, 'ro')
	result = []
	R = 0.01**2
	for meas in zip(observed_x, observed_y):
		kalman.update(meas, R)
		result.append((kalman.X[:2]).tolist())
	kalman_x, kalman_y = zip(*result)
	plt.plot(kalman_x, kalman_y, 'g-')
	plt.show()

demo_kalman_2d()