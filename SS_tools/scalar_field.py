import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize

from mpl_toolkits.axes_grid1 import make_axes_locatable

from SS_tools.toolbox import *

# Scalar field color map
MY_CMAP = alpha_cmap(plt.cm.jet, 0.3)

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

# Matrix Q dot each row of X
Q_prod_xi = lambda Q,X: (Q @ X.T).T

# ----------------------------------------------------------------------
# General class for scalar field
# ----------------------------------------------------------------------
"""
Class for Gaussian scalar field.
  * max_intensity: maximum intensity of the Gaussian.
  * mu: central point of the Gaussian.
  * dev: standard deviation.
  * R: field rotation matrix.
"""
class sigma:
  def __init__(self, sigma_func, mu=None):
    self.sigma_func = sigma_func
    self.rot = None # Variable to rotate the field from outside
    if mu is None:
      x0 = self.sigma_func.x0 # Ask for help to find minimum
      self.mu = minimize(lambda x: -self.value(np.array([x])), x0).x
    else:
      self.mu = minimize(lambda x: la.norm(self.grad(np.array([x]))), mu).x

  """
  Evaluation of the scalar field for a vector of values.
  """
  def value(self, X):
    if self.rot is not None:
      X = Q_prod_xi(self.rot, X-self.mu) + self.mu
    return self.sigma_func.eval(X)

  """
  Gradient vector of the scalar field for a vector of values.
  """
  def grad(self, X):
    if self.rot is not None:
      X = Q_prod_xi(self.rot, X-self.mu) + self.mu
      grad = self.sigma_func.grad(X)
      return Q_prod_xi(self.rot.T, grad)
    else:
      return self.sigma_func.grad(X)

  """
  Function to draw the scalar field.
  """
  def draw(self, fig=None, ax=None, xlim=30, ylim=30, cmap=MY_CMAP, n=256, contour_levels=0, contour_lw=0.3):
    if fig == None:
      fig = plt.figure(figsize=(16, 9), dpi=100)
      ax = fig.subplots()
    elif ax == None:
      ax = fig.subplots()

    # Calculate
    x = np.linspace(self.mu[0] - xlim, self.mu[0] + xlim, n)
    y = np.linspace(self.mu[1] - ylim, self.mu[1] + ylim, n)
    X, Y = np.meshgrid(x, y)

    P = np.array([list(X.flatten()), list(Y.flatten())]).T
    Z = self.value(P).reshape(n,n)

    # Draw
    ax.plot(self.mu[0], self.mu[1], "+k")
    color_map = ax.pcolormesh(X, Y, Z, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)

    cbar = fig.colorbar(color_map, cax=cax)
    cbar.set_label(label='$\sigma$ [u]', labelpad=10)

    if contour_levels != 0:
      contr_map = ax.contour(X, Y, Z, contour_levels, colors="k", linewidths=contour_lw, linestyles="-", alpha=0.2)
      return color_map, contr_map
    else:
      return color_map,

  def draw_imshow(self, fig=None, ax=None, xlim=30, ylim=30, cmap=MY_CMAP, n=256, make_im=True):
    # Calculate
    x = np.linspace(self.mu[0] - xlim, self.mu[0] + xlim, n)
    y = np.linspace(self.mu[1] - ylim, self.mu[1] + ylim, n)
    X, Y = np.meshgrid(x, y)

    P = np.array([list(X.flatten()), list(Y.flatten())]).T
    Z = self.value(P).reshape(n,n)

    if make_im:
      extent = np.min(x), np.max(x), np.min(y), np.max(y)

      if fig == None:
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.subplots()
      elif ax == None:
        ax = fig.subplots()

      # Draw
      ax.plot(self.mu[0], self.mu[1], "+k")
      im = ax.imshow(Z, cmap=cmap, interpolation="nearest", extent=extent)

      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='2%', pad=0.05)

      cbar = fig.colorbar(im, cax=cax)
      cbar.set_label(label='$\sigma$ [u]', labelpad=10)
      return im

    else:
      return Z

  """
  Function to draw the gradient at a point.
    * x: point where the gradient is to be drawn 
    * ax: axis to plot on.
    * width: arrow size.
    * scale: scales the length of the arrow (smaller for larger scale values).
    * zorder: overlay order in the plot.
  """
  def draw_grad(self, x, ax, width=0.002, scale=30, zorder=2, alpha=1):
    if type(x) == list:
      grad_x = self.grad(np.array(x))[0]
    else:
      return None
    grad_x_unit = grad_x/la.norm(grad_x)
    quiver = ax.quiver(x[0], x[1], grad_x_unit[0], grad_x_unit[1],
                        width=width, scale=scale, zorder=zorder, alpha=alpha)
    return quiver

# ----------------------------------------------------------------------
# Scalar fields used in simulations
# (all these classes needs eval and grad functions)
# ----------------------------------------------------------------------

# Exponential function with quadratic form: exp(r) = e^((r - mu)^t @ Q @ (r - mu)).
exp = lambda X,Q,mu: np.exp(np.sum((X - mu) * Q_prod_xi(Q,X - mu), axis=1))

"""
Gaussian function.
  * a: center of the Gaussian.
  * dev: models the width of the Gaussian.
"""
class sigma_gauss:
  def __init__(self, mu=[0,0], n=2, max_intensity=100, dev=10, S=None, R=None):
    self.n = n
    self.max_intensity = max_intensity
    self.dev = dev

    # Variables needed for the sigma class
    self.x0  = mu
    # ---

    if S is None:
      S = -np.eye(n)
    if R is None:
      R = np.eye(n)
    self.Q = R.T@S@R/(2*self.dev**2)

  def eval(self, X):
    X = two_dim(X)
    sigma = self.max_intensity * exp(X,self.Q,self.x0) / np.sqrt(2*np.pi*self.dev**2)
    return sigma

  def grad(self, X):
    X = two_dim(X)
    return Q_prod_xi(self.Q,X-self.x0) * self.eval(X)

"""
Non-convex function: two Gaussians plus factor * norm.
  * k: norm factor.
  * dev: models the scale of the distribution while maintaining its properties.
"""
# Default parameters
S1 = 0.9*np.array([[1/np.sqrt(30),0], [0,1]])
S2 = 0.9*np.array([[1,0], [0,1/np.sqrt(15)]])
A = (1/np.sqrt(2))*np.array([[1,-1], [1,1]])
a = np.array([1, 0])
b = np.array([0,-2])

class sigma_nonconvex:
  def __init__(self, k, mu=[0,0], dev=1, a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
    self.k = k
    self.dev = dev

    # Variables needed for the sigma class
    self.x0 = mu
    self.rot = np.eye(2)
    # ---

    if type(a) == list:
      a = np.array(a)
    if type(b) == list:
      b = np.array(b)
    self.a = a
    self.b = b
    self.Qa = Qa
    self.Qb = Qb

  def eval(self, X):
    X = two_dim(X)
    X = (X - self.x0)/self.dev
    sigma = - 2 - exp(X,self.Qa,self.a) - exp(X,self.Qb,self.b) + self.k*la.norm(X, axis=1)
    return -sigma

  def grad(self, X):
    X = two_dim(X)
    X = (X - self.x0)/self.dev
    alfa = 0.0001 # Trick to avoid x/0
    sigma_grad = - Q_prod_xi(self.Qa,X-self.a) * exp(X,self.Qa,self.a)[:,None] \
                 - Q_prod_xi(self.Qb,X-self.b) * exp(X,self.Qb,self.b)[:,None] \
                 + self.k * X / (la.norm(X, axis=1)[:,None] + alfa)
    return -sigma_grad

"""
Analyzing the previous case, we propose a function that allows us to play much more with the 
generation of scalar fields.
  * k: norm factor.
  * dev: models the scale of the distribution while maintaining its properties.
"""

class sigma_fract:
  def __init__(self, k, mu=[0,0], dev=[1,1], a=a, b=b, Qa=-S1, Qb=-A.T@S2@A):
    self.k = k
    self.dev = dev

    # Variables necesarias para la clase sigma
    self.x0 = mu
    self.rot = np.eye(2)
    # ---

    if type(a) == list:
      a = np.array(a)
    if type(b) == list:
      b = np.array(b)
    self.a = a
    self.b = b
    self.Qa = Qa
    self.Qb = Qb

  def eval(self, X):
    X = two_dim(X)
    X = (X - self.x0)
    c1 = - exp(X/self.dev[0],self.Qa,self.a) - exp(X/self.dev[0],self.Qb,self.b)
    c2 = - exp(X/self.dev[1],self.Qa,self.a) - exp(X/self.dev[1],self.Qb,self.b)
    x_dist = la.norm(X, axis=1)
    sigma = - 2 + 2*c1 + c2 + self.k*x_dist
    return -sigma

  def grad(self, X):
    X = two_dim(X)
    X = (X - self.x0)/self.dev
    alfa = 0.0001 # Trick to avoid x/0
    sigma_grad = - Q_prod_xi(self.Qa,X-self.a) * exp(X,self.Qa,self.a)[:,None] \
                 - Q_prod_xi(self.Qb,X-self.b) * exp(X,self.Qb,self.b)[:,None] \
                 + self.k * X / (la.norm(X, axis=1)[:,None] + alfa)
    return -sigma_grad