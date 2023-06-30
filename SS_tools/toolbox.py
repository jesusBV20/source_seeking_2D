import os
import numpy as np
from numpy import linalg as la

# Graphic tools
import matplotlib.pylab as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# ----------------------------------------------------------------------
# Mathematical tools
# ----------------------------------------------------------------------

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Affine transformation
M_scale = lambda s, n=2: np.eye(n)*s

# Rotation matrix
M_rot = lambda psi: np.array([[np.cos(psi), -np.sin(psi)], \
                              [np.sin(psi),  np.cos(psi)]])

# Angle between two vectors (matrix computation)
def angle_of_vectors(A,B):
    cosTh = np.sum(A*B, axis=1)
    sinTh = np.cross(A,B, axis=1)
    theta = np.arctan2(sinTh,cosTh)
    return theta

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

"""
Unicycle patch.
  * XY: position [X, Y] of the patch 
  * yaw: heading of the unicycle
"""
def unicycle_patch(XY, yaw, color, size=1, lw=0.5):
    Rot = np.array([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw), np.cos(yaw)]])

    apex = 45*np.pi/180 # 30 degrees apex angle
    b = np.sqrt(1) / np.sin(apex)
    a = b*np.sin(apex/2)
    h = b*np.cos(apex/2)

    z1 = size*np.array([a/2, -h*0.3])
    z2 = size*np.array([-a/2, -h*0.3])
    z3 = size*np.array([0, h*0.6])

    z1 = Rot.dot(z1)
    z2 = Rot.dot(z2)
    z3 = Rot.dot(z3)

    verts = [(XY[0]+z1[1], XY[1]+z1[0]), \
             (XY[0]+z2[1], XY[1]+z2[0]), \
             (XY[0]+z3[1], XY[1]+z3[0]), \
             (0, 0)]

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, fc=color, lw=lw)

"""
Compute a 1D range zoomed around center.
(moded from https://gist.github.com/dukelec/e8d4171ef4d12f9998295cfcbe3027ce)
  * begin: The begin bound of the range.
  * end: The end bound of the range.
  * center: The center of the zoom (i.e., invariant point)
  * scale_factor: The scale factor to apply.
:return: The zoomed range (min, max)
"""
def _zoom_range(begin, end, center, scale_factor):
  if begin < end:
      min_, max_ = begin, end
  else:
      min_, max_ = end, begin

  old_min, old_max = min_, max_

  offset = (center - old_min) / (old_max - old_min)
  range_ = (old_max - old_min) / scale_factor
  new_min = center - offset * range_
  new_max = center + (1. - offset) * range_

  if begin < end:
      return new_min, new_max
  else:
      return new_max, new_min

"""
Apply alpha to the desired color map
https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
----------------------------------------------------------------------

When using pcolormesh, directly applying alpha can cause many problems.
The ideal approach is to generate and use a pre-diluted color map on a white background.
"""
def alpha_cmap(cmap, alpha):
  # Get the colormap colors
  my_cmap = cmap(np.arange(cmap.N))

  # Define the alphas in the range from 0 to 1
  alphas = np.linspace(alpha, alpha, cmap.N)

  # Define the background as white
  BG = np.asarray([1., 1., 1.,])

  # Mix the colors with the background
  for i in range(cmap.N):
      my_cmap[i,:-1] = my_cmap[i,:-1] * alphas[i] + BG * (1.-alphas[i])

  # Create new colormap which mimics the alpha values
  my_cmap = ListedColormap(my_cmap)
  return my_cmap

"""
Check if the dimensions are correct and adapt the input to 2D.
"""
def two_dim(X):
  if type(X) == list:
    return np.array([[X]])
  elif len(X.shape) < 2:
    return np.array([X])
  else:
    return X

"""
Función para dar formato a los plots de datos
"""
def fmt_data_axis(ax, ylabel = "", xlabel = "", title = "",
                  xlim = None, ylim = None, invy = True, d=2):
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)

  if xlim is not None:
    ax.set_xlim(xlim)
  if ylim is not None:
    ax.set_ylim(ylim)

  if invy:
    ax.yaxis.tick_right()
    
  ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.{}f'.format(d)))
  ax.grid(True)

"""
Create a new directory if it doesn't exist
"""
def createDir(dir):
  try:
    os.mkdir(dir)
    print("¡Directorio '{}' creado!".format(dir))
  except:
    print("¡El directorio '{}' ya existe!".format(dir))


# ----------------------------------------------------------------------
# Utility functions for simulations
# ----------------------------------------------------------------------

"""
Funtion to compute L_sigma.
  * X: (N x 2) matrix of agents position from the centroid
  * sigma: (N) vector of simgma_values on each row of X
"""
def L_sigma(X, sigma, denom=None):
    N = X.shape[0]
    l_sigma_hat = sigma[:,None].T @ X
    if denom == None:
        x_norms = np.zeros((N))
        for i in range(N):
            x_norms[i] = X[i,:] @ X[i,:].T
            D_sqr = np.max(x_norms)
        l_sigma_hat = l_sigma_hat / (N * D_sqr)
    else:
        l_sigma_hat = l_sigma_hat/denom
    return l_sigma_hat.flatten()

"""
Function to generate a uniform circular distribution.
  * N: number of points
  * n: dimension of the real space
  * rc0: position in the real space of the central point
  * r: radius of the circle
  * h: minimum radius of appearance with respect to the center
  * border_noise: noise generated at the boundary to avoid a perfect circle
"""
#TODO: Adjust the distribution to be uniform in curved space
def circular_distrib(N, n, rc0, r, h=0, border_noise=0.1):
  rand_ang = 2*np.pi*np.random.rand(N)
  rand_dirs = np.array([np.cos(rand_ang), np.sin(rand_ang)]).T
  rand_rads = (r - h)*np.random.rand(N) + h + border_noise*np.random.rand(N)
  X0 = rand_rads[:,None] * rand_dirs/la.norm(rand_dirs, axis=1)[:,None]
  return rc0 + X0

"""
Function to generate uniform rectangular distributions.
  * N: number of points
  * n: dimension of the real space
  * rc0: position in the real space of the central point
"""
def XY_distrib(N, n, rc0, lims, border_noise=0.1, scale=1):
  X0 = (np.random.rand(N,n) - 0.5)*2
  for i in range(n):
    X0[:,i] = X0[:,i] * lims[i]
  X0 = X0 @ M_scale(scale, n)
  return rc0 + X0

"""
Batman distribution in 2D :)
  * N: number of points
  * rc0: position in the real space of the central point
  * lims = [xlim, ylim]: width and length of the distribution
"""
def batman_distrib(N, rc0, lims, scale=1):
  x_filt = []
  y_filt = []

  min_eq_value1 = -0.05
  min_eq_value2 = -0.5

  while(len(x_filt)<N):
    X = (np.random.rand(N,2) - 0.5)*2*8

    x_ = X[:,0]
    y_ = X[:,1]

    eq1 = lambda x,y: ((x/7)**2 * np.sqrt(abs(abs(x)-3)/(abs(x)-3)) + (y/3)**2 * np.sqrt(abs(y+3/7*np.sqrt(33))/(y+3/7*np.sqrt(33))) - 1)
    eq2 = lambda x,y: (abs(x/2) - ((3*np.sqrt(33)-7)/112) * x**2 - 3 + np.sqrt(1-(abs(abs(x)-2)-1)**2) - y )
    eq3 = lambda x,y: (9*np.sqrt(abs((abs(x)-1)*(abs(x)-.75)) / ((1-abs(x))*(abs(x)-.75))) - 8*abs(x) - y)
    eq4 = lambda x,y: (3*abs(x) + .75 * np.sqrt(abs((abs(x)-.75)*(abs(x)-.5)) / ((.75-abs(x))*(abs(x)-.5))) - y )
    eq5 = lambda x,y: (2.25*np.sqrt(abs((x-.5)*(x+.5)) / ((.5-x)*(.5+x))) - y)
    eq6 = lambda x,y: (6*np.sqrt(10)/7 + (1.5-.5*abs(x)) * np.sqrt(abs(abs(x)-1)/(abs(x)-1)) - (6*np.sqrt(10)/14) * np.sqrt(4-(abs(x)-1)**2) - y)

    #eq1
    x = x_[np.logical_or(x_<-4, x_>4)]
    y = y_[np.logical_or(x_<-4, x_>4)]
    eq = eq1(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value1)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    x = x_[np.logical_and(y_>=0, np.logical_and(x_>-4, x_<4))]
    y = y_[np.logical_and(y_>=0, np.logical_and(x_>-4, x_<4))]
    eq = eq1(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value1)
    x_filt.extend(x[z])
    y_filt.extend(y[z])


    #eq2
    x = x_[np.logical_and(y_<0, np.logical_and(x_>=-4, x_<=4))]
    y = y_[np.logical_and(y_<0, np.logical_and(x_>=-4, x_<=4))]
    eq = eq2(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    #eq3
    x = x_[np.logical_and(y_>0, np.logical_and(x_>=-1, x_<=-0.75))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>=-1, x_<=-0.75))]
    eq = eq3(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    x = x_[np.logical_and(y_>0, np.logical_and(x_>=0.75, x_<=1))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>=0.75, x_<=1))]
    eq = eq3(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    #eq4
    x = x_[np.logical_and(y_>0, np.logical_and(x_>-0.75, x_<0.75))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>-0.75, x_<0.75))]
    eq = eq4(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    #eq5
    x = x_[np.logical_and(y_>0, np.logical_and(x_>-0.5, x_<0.5))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>-0.5, x_<0.5))]
    eq = eq5(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    #eq6
    x = x_[np.logical_and(y_>0, np.logical_and(x_>-3, x_<-1))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>-3, x_<-1))]
    eq = eq6(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

    x = x_[np.logical_and(y_>0, np.logical_and(x_>1, x_<3))]
    y = y_[np.logical_and(y_>0, np.logical_and(x_>1, x_<3))]
    eq = eq6(x,y)

    z = np.logical_and(eq<0, eq>min_eq_value2)
    x_filt.extend(x[z])
    y_filt.extend(y[z])

  X0 = np.array([x_filt[0:N],y_filt[0:N]]).T

  # Apply scaling and translation
  for i in range(2):
    X0[:,i] = X0[:,i]/8 * lims[i]
  X0 = X0 @ M_scale(scale, 2)
  return rc0 + X0