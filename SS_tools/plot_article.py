import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend # two legends

from SS_tools.toolbox import *
from SS_tools.scalar_field import *

KW_ARROW = {"lw":3, "hw":0.7, "hl":1}
KW_PATCH = {"size":1.5, "lw":0.5}

KW_ARROW_TINY = {"lw":2, "hw":0.05, "hl":0.1}
KW_PATCH_TINY = {"size":0.15, "lw":0.2}

def kw_arrow_dyn(scale):
    return {"lw":2*scale**(1/5), "hw":0.05*scale, "hl":0.1*scale}

def kw_patch_dyn(scale):
    return {"size":0.15*scale, "lw":0.2*scale**(1/2)}

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

"""
Function to easy plot a 2D vector
"""
def vector2d(axis, P0, Pf, c="k", ls="-", lw = 0.7, hw=0.1, hl=0.2):
    axis.arrow(P0[0], P0[1], Pf[0], Pf[1],
                lw=lw, color=c, ls=ls,
                head_width=hw, head_length=hl, 
                length_includes_head=True)

"""
Function to plot the S region
"""
def plot_sregion(ax,pc,l1,l2):
    l1_, l2_ = 1.3*l1, 1.3*l2
    p_rect = pc - np.array([l1_,l2_])/2

    ax.add_patch(Rectangle(p_rect, l1_, l2_, 
                           fill = False, lw=2, linestyle="--"))
    ax.text(p_rect[0] + 1.05*l1_, p_rect[1], r"$\mathcal{S}$")

"""
Funcion to distribute the agents following a geometric series
"""
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

def regpoly_formation(N,r,thetha0=0):
    d_theta = 2*np.pi/N
    theta = []

    for i in range(N):
        if N == 2:
            theta.append(d_theta*i + thetha0)
        elif N%2 == 0:
            theta.append(d_theta*i + d_theta/2 + thetha0)
        else:
            theta.append(d_theta*i + d_theta/4 + thetha0)
    
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

# ----------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------

"""
Cluster class 
-------------
P: N x 2 ndarray
"""
class cluster:
    def __init__ (self, P):
        self.P = P
        self.Pc = np.mean(P, axis=0)
        self.N = P.shape[0]
        self.x = self.P - self.Pc
        self.D = np.sqrt(np.max(self.x.T @ self.x))

    def l_sigma(self,sigma):
        self.sigma_vals = sigma.value(self.P)
        l_sigma =  self.sigma_vals[:,None].T @ self.x / (self.N * self.D**2)
        return l_sigma.flatten()

"""
Function to verify the Collorary 1
"""
def clusters_plot(clusters, sigma_class, ag_rad = 0.2, c_rad = 0.3):
    Nc = len(clusters)

    fig     = plt.figure(figsize=(16, 8), dpi=100)
    main_ax = fig.subplots()

    # Draw the scalar field
    sigma_class.draw(fig=fig, ax=main_ax, xlim=60, ylim=40, n=300, contour_levels=20)

    # Axis configuration
    main_ax.set_xlim([-30,70])
    main_ax.set_ylim([-20,30])
    main_ax.set_xlabel(r"$P_x$ [L]")
    main_ax.set_ylabel(r"$P_y$ [L]")
    main_ax.grid(True)
    
    # Drawing agents, centroid and vectors --

    # Centroid of agents for each cluster
    pc_tilde = 0
    l_tilde = np.zeros(2)
    l1_tilde  = np.zeros(2)
    for k in range(Nc):
        D  = clusters[k].D
        pc = clusters[k].Pc

        l_sigma = clusters[k].l_sigma(sigma_class)
        l_sigma_hat = l_sigma/np.sqrt(l_sigma[0]**2 + l_sigma[1]**2)
        l1_vec = sigma_class.draw_L1(pc, clusters[k].P)
        l1_vec_hat = l1_vec/np.sqrt(l1_vec[0]**2 + l1_vec[1]**2)

        pc_tilde = pc_tilde + pc
        l_tilde = l_tilde + l_sigma
        l1_tilde = l1_tilde + l1_vec
        
        vector2d(main_ax, pc, l_sigma_hat*3, c="red", **KW_ARROW)
        vector2d(main_ax, pc, l1_vec_hat*3, c="green", **KW_ARROW)
        icon_centroid = plt.Circle(pc, ag_rad, color="k")
        icon_centroid.set_zorder(2)
        main_ax.add_patch(icon_centroid)
        main_ax.add_patch(plt.Circle(pc, D, color="lightgrey", alpha=0.5))

    # Centroid of clusters
    pc_tilde = pc_tilde / Nc
    l_tilde = l_tilde/Nc
    l1_tilde = l1_tilde/Nc

    l_tilde_hat = l_tilde/np.sqrt(l_tilde[0]**2 + l_tilde[1]**2)
    l1_tilde_hat = l1_tilde/np.sqrt(l1_tilde[0]**2 + l1_tilde[1]**2)

    vector2d(main_ax, pc_tilde, l_tilde_hat*6, c="darkred", **KW_ARROW)
    vector2d(main_ax, pc_tilde, l1_tilde_hat*6, c="darkgreen", **KW_ARROW)
    icon_centroid = plt.Circle(pc_tilde, c_rad, color="k")
    icon_centroid.set_zorder(2)
    main_ax.add_patch(icon_centroid)

    # Draw the gradient
    sigma_class.draw_grad(pc_tilde, main_ax, width=0.003, scale=15)

    # Drawing patches
    for k in range(Nc):
        k_random = np.random.rand(1)[0]
        cluster  = clusters[k]
        for i in range(cluster.N):
            kw_patch = {"color": "blue", "size":0.2, "lw":0.4} 
            ang = 0.1 + 0.1*k_random + 0.05*np.random.rand(1)[0]
            p = cluster.P[i,:]
            icon = unicycle_patch(list(p), ang*2*np.pi, "royalblue", **KW_PATCH)
            icon.set_zorder(2)
            main_ax.add_patch(icon)
    
    # Generate the legends
    arr1 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
    arr2 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
    arr3 = plt.scatter([],[],c='green',marker=r'$\uparrow$',s=60)
    arr4 = plt.scatter([],[],c='darkred',marker=r'$\uparrow$',s=60)
    arr5 = plt.scatter([],[],c='darkgreen',marker=r'$\uparrow$',s=60)

    leg1 = Legend(main_ax, [arr2, arr3], 
                [r"$L_{\sigma_k}$: Actual computed ascending direction from cluster $k$",
                 r"$L_{1_k}$ (Non-computed)"],
                loc="upper center", prop={'size': 10})
    

    leg2 = Legend(main_ax, [arr1, arr4, arr5], 
                [r"$\nabla \sigma (\tilde{p_c})$ (Non-computed)",
                 r"$\tilde{L_{\sigma}}$: Actual computed ascending direction",
                 r"$\tilde{L_1}$ (Non-computed)"],
                loc="upper left", prop={'size': 10})
    
    main_ax.add_artist(leg1)
    main_ax.add_artist(leg2)

    # Show the plot!
    plt.show()

"""
Funtion to verify lemma 4
"""
def plot_polyreg(ax, N, r, theta0=0, legend=False, xlab=False, ylab=False, title_full=False):
    mu0 = np.array([30,20])
    p0 = np.array([-10,-5])

    # Generating the scalar field -------------
    sigma_func = sigma_gauss(mu=mu0, max_intensity=100, dev=20)
    sigma_test = sigma(sigma_func)

    # Generate the formation -------------
    phi = np.pi/4

    X = regpoly_formation(N,r,theta0)
    xc = np.sum(X, axis=0)/N

    P = p0 + X
    pc = p0 + xc

    sigma_values = sigma_test.value(P)

    # Compute gradient
    grad = sigma_test.grad(pc)[0]
    grad = grad/np.sqrt(grad[0]**2 + grad[1]**2)

    # Compute L_sigma
    l_sigma = L_sigma(P - pc, sigma_values)
    l_sigma = l_sigma/np.sqrt(l_sigma[0]**2 + l_sigma[1]**2)

    # Compute L_sigma^1
    l1_vec = sigma_test.draw_L1(pc, P)
    l1_vec = l1_vec/np.sqrt(l1_vec[0]**2 + l1_vec[1]**2)

    # Plotting -------------
    # Axis configuration
    dr = r + r/6
    ax.axis([-dr, dr, -dr, dr])
    ax.set_aspect("equal")
    ax.grid(True)
    
    if title_full:
        ax.set_title(r"N = {0:d}, $\theta$ = {1:.0f} deg.".format(N, theta0*180/np.pi))
    else:
        ax.set_title(r"N = {0:d}".format(N))

    if xlab:
       ax.set_ylabel("$X$ [L]")
    if ylab:
        ax.set_xlabel("$Y$ [L]")

    # Lines
    ax.axhline(0, c="k", ls="-", lw=1.1)
    ax.axvline(0, c="k", ls="-", lw=1.1)

    for n in range(N):
        ax.plot([X[n-1,0], X[n,0]], [X[n-1,1], X[n,1]], "k--", alpha=0.6)

    # Agents
    for n in range(N):
        icon = unicycle_patch(X[n,:], phi, "royalblue", **kw_patch_dyn(r))
        ax.add_patch(icon)

    # Arrows
    kw_arrow = kw_arrow_dyn(r**(1/1.5))
    vector2d(ax, [0,0], l_sigma*r/1.2, c="red", **kw_arrow)
    vector2d(ax, [0,0], l1_vec*r/1.3, c="green", **kw_arrow)
    vector2d(ax, [0,0], grad*r/1.4, c="k", **kw_arrow)

    # Generate the legend
    if legend:
        arr1 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
        arr2 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
        arr3 = plt.scatter([],[],c='green',marker=r'$\uparrow$',s=60)

        leg = Legend(ax, [arr1, arr2, arr3], 
                    [r"$\nabla \sigma (p_c)$ (Non-computed)",
                    r"$L_{\sigma}$: Actual computed ascending direction",
                    r"$L_1$ (Non-computed)"],
                    loc="upper left", prop={'size': 10}, ncol=1)

        ax.add_artist(leg)