import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from scipy.integrate import solve_ivp
from scipy import linalg as la

def calculate_a(N, alpha):

  a = np.zeros(N)
  a[0] = 1

  for i in range(1,int(N/2)-1):
    a[i] = a[i-1]*alpha

  a[int(N/2)-1] = -(np.sum(a) - 1 - N/4)
  a = np.sqrt(a)

  j = 1
  for i in range(int(N/2), N):
    a[i] = -a[int(N/2)-j]
    j = j + 1

  return a

def L1_sigma(p, N, grad):
  # p is the position of all robots
  # n is the dimension (2,3)
  # sigma is the scalar field

  L1 = np.zeros(2)
  for i in range(0,N):
    L1 = L1 + grad.dot(p[:,i])*p[:,i]

  return L1 / N

a_x1d = calculate_a(16,0.85)
#a_y1d = calculate_a(18, 0.87)
a_y1d = calculate_a(16, 0.86)

a_x1d = calculate_a(6,0.8)
a_y1d = calculate_a(6,0.6)

print(a_x1d)
print(a_y1d)

a_x = a_x1d.reshape(1, np.size(a_x1d))
a_x = np.vstack((a_x1d, np.zeros((1, np.size(a_x1d)))))

a_y = a_y1d.reshape(1, np.size(a_y1d))
a_y = np.vstack((np.zeros((1, np.size(a_y1d))), a_y1d))

a = np.hstack((a_x,a_y))

grad = np.array([1,1])
L1 = L1_sigma(a_x, np.size(a_x[0]), grad) + L1_sigma(a_y, np.size(a_x[0]), grad)

print(L1)
