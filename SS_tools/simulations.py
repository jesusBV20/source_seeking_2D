import numpy as np
from numpy import linalg as la

from SS_tools.toolbox import angle_of_vectors
from SS_tools.simulator_frame import sim_frame

# ----------------------------------------------------------------------
# Data Collector Class
# ----------------------------------------------------------------------

"""
Clase para recolectar los datos de la simulación, para todos los frames y
clusters.
  * simulation: simulador que controla a los clusters.
  * data_labels: datos que se desean extraer.

Devuelve como resultado:
  * data: Diccionario con todos los datos de la simulación.
"""
class data_collector:
  def __init__(self, simulation, data_labels):
    self.simulation = simulation
    self.data_labels = data_labels

    # Generate the data dictionary
    self.data = {
      "N" : self.simulation.sim_telm["N"], 
      "dt": self.simulation.sim_telm["dt"], 
      "tf": self.simulation.sim_telm["tf"]
      }
    for label in self.data_labels:
      self.data.update({label:[]})

  """
  Collect and record data from the actual state of the simulation.
  """
  def collect(self):
    self.data["tf"] = self.simulation.sim_telm["tf"]
    for label in self.data_labels:
      self.data[label].append(self.simulation.sim_telm[label])
  
  """
  Extract the desired field.
  """
  def get(self, label):
    if label == "N":
      return self.data["N"]
    elif label == "dt":
      return self.data["dt"]
    elif label == "tf":
      return self.data["tf"]
    else:
      return np.array(self.data[label], dtype=object)

# ----------------------------------------------------------------------
# Simulation Class I
# ----------------------------------------------------------------------

"""
Simulation Class I
  * sigma_field: scalar field class.
  * n_agents: number of agents in the simulation.
  * x0: Initial parameter vector.
    - t0: Initial time of the simulation.
    - p0: Matrix with the initial position of each agent. (N x n_space)
    - v0: Constant velocity for all agents.
    - active: Active agents at the beginning of the simulation. (N vector)
  * mod_shape: Switch to activate the shape controller.
  * obstacles = [[x1,y1,r1], ..., [xn,yn,rn]]: Virtual obstacles to be drawn.

"""
class simulation_class1(sim_frame):
  def __init__(self, sigma_field, n_agents, x0, dt = 0.01, mod_shape = False, obstacles = [], ang_noise = 0):
    t0 = x0[0]
    p0 = x0[1]
    v0 = x0[2]

    # Initialise the simulation frame
    super().__init__(n_agents, [t0,p0], dt)

    # Activation vector of the agents
    if len(x0)>3:
      self.active = x0[3]
    else:
      self.active = np.ones(n_agents, dtype=bool)

    # States of this simulation
    self.vf = v0                                       # Constant common velocity
    self.sigma_field = sigma_field                     # Scalar field class
    self.sigma = self.sigma_field.value(p0)            # Measured sigmas (N x 1)

    self.rc = np.mean(p0[self.active,:], axis=0)
    self.rc_e = la.norm(self.rc - self.sigma_field.mu) # Distance to the source
    self.rc_sigma = self.sigma_field.value(self.rc)[0] # Sigma at the centroid
    self.X = p0 - self.rc                              # Swarm geometry
    self.d = la.norm(self.X, axis=1)                   # Distance to the centroid

    self.l_sigma_hat = np.zeros(2)
    self.rc_grad = self.sigma_field.grad(self.rc)[0]

    # Useful variables for this simulation
    self.mod_shape = mod_shape                         # Geometry control switch
    self.Xd = self.X                                   # Desired geometry
    self.obstacles = obstacles                         # Simulation obstacles
    self.ang_noise = ang_noise

    # Initialise the simulation telemetry dictionary
    self.update_telemetry()
  
  """
  Get state from simulator frame.
  """
  def get_pf(self):
    return self.states[0]

  """
  Update the telemetry dictionary.
  """
  def update_telemetry(self):
    self.sim_telm = {
      "N"         : self.N, 
      "dt"        : self.dt, 
      "tf"        : self.tf,
      "pf"        : self.get_pf(), 
      "rc"        : self.rc,
      "e"         : self.rc_e,
      "d"         : self.d,
      "sigma"     : self.sigma,
      "active"    : np.copy(self.active),
      "l_sigma"   : self.l_sigma_hat,
      "rc_grad"   : self.rc_grad,
      "field_rot" : self.sigma_field.rot
    }
  
  """
  Free dynamics following the ascent direction (constant speed).
    * X: Matrix of relative distances to the centroid (N x n) xi = [px, py, pz].
    * sigma: Vector with sigma_field measurements of each agent (N x 1).
  """
  def free_kinematics(self, X):
    # compute l_sigma
    l_sigma_hat = self.L_sigma(X, self.sigma)
    l_sigma_hat_norm = la.norm(l_sigma_hat)

    if l_sigma_hat_norm != 0:
      self.l_sigma_hat = l_sigma_hat[:] / l_sigma_hat_norm
    else:
      self.l_sigma_hat = np.zeros(2)
    
    # virtual noise in l_sigma_hat
    if self.ang_noise > 0:
      alfa = (np.random.rand(self.N) - 0.5) * 2 * self.ang_noise / 180 * np.pi
      R_alfa = np.array([[np.cos(alfa), -np.sin(alfa)], [np.sin(alfa), np.cos(alfa)]])
      l_sigma_hat = np.squeeze(self.l_sigma_hat[:,None].T @ R_alfa).T  

    # compute p_dot
    p_dot = self.vf * l_sigma_hat
    return p_dot

  def shape_control(self):
    return (self.Xd - self.X)/10

  """
  Euler integration (Step-wise).
  """
  def int_step(self):
    self.rc = np.mean(self.get_pf()[self.active,:], axis=0)
    self.rc_e = la.norm(self.rc - self.sigma_field.mu)
    self.rc_sigma = self.sigma_field.value(self.rc)[0]
    self.rc_grad = self.sigma_field.grad(self.rc)[0]
    self.rc_grad = self.rc_grad/la.norm(self.rc_grad)

    self.X = self.get_pf() - self.rc
    self.d = la.norm(self.X, axis=1)
    self.sigma = self.sigma_field.value(self.get_pf())

    # Compute p_dot and send it to sim_frame
    X_active = np.diag(self.active) @ self.X
    p_dot = self.free_kinematics(X_active)
    if self.mod_shape:
      p_dot = p_dot + self.shape_control()
    p_dot_active = np.diag(self.active) @ p_dot
    self.int_euler([p_dot_active])

    # Update telemetry
    self.update_telemetry()


# ----------------------------------------------------------------------
# Simulation Class II
# ----------------------------------------------------------------------

"""
Simulation Class II
  * x0: Initial parameter vector [t0,p0,v0,phi0]
    - t0: Initial time of the simulation.
    - p0: Matrix with the initial position of each agent. (N x n_space)
    - v0: Constant velocity for all agents.
    - phi0: Initial heading of each unicycle.
"""
class simulation_class2(sim_frame):
  def __init__(self, sigma_field, n_agents, x0, dt = 0.01, kd = 0.5):
    # Initialise the simulation frame
    super().__init__(n_agents, [x0[0],x0[1],x0[3]], dt)

    # Simulation variables
    self.vf = x0[2]                                    # Constant common velocity
    self.omega_i = np.zeros(self.N)                    # Control action
    self.sigma_field = sigma_field                     # Scalar field class
    self.sigma = self.sigma_field.value(x0[1])         # Measured sigmas (N x 1)

    self.rc = np.mean(x0[1], axis=0)
    self.rc_e = la.norm(self.rc - self.sigma_field.mu) # Distance to the source
    self.rc_sigma = self.sigma_field.value(self.rc)[0] # Sigma at the centroid
    self.X = x0[1] - self.rc                           # Swarm geometry
    self.d = la.norm(self.X, axis=1)                   # Distance to the centroid

    self.l_sigma_hat = np.zeros(2)
    self.rc_grad = self.sigma_field.grad(self.rc)[0]

    # GVF controller parameters
    self.kd = kd

    # Initialise the simulation telemetry dictionary
    self.update_telemetry()

  """
  Get state from simulator frame.
  """
  def get_pf(self):
    return self.states[0]
  
  def get_phif(self):
    return self.states[1]
  
  """
  Update the telemetry dictionary.
  """
  def update_telemetry(self):
    self.sim_telm = {
      "N"       : self.N, 
      "dt"      : self.dt, 
      "tf"      : self.tf,
      "pf"      : self.states[0],
      "phif"    : self.states[1],
      "rc"      : self.rc,
      "e"       : self.rc_e,
      "d"       : self.d,
      "sigma"   : self.sigma,
      "omega"   : self.omega_i,
      "l_sigma" : self.l_sigma_hat,
      "rc_grad" : self.rc_grad
    }
  
  def get_pf(self):
    return self.states[0]
  """
  Calculate l_sigma_hat.
  """
  def LsigmaHat(self, X):
    l_sigma_hat = self.L_sigma(X, self.sigma)
    l_sigma_hat_norm = la.norm(l_sigma_hat)
    if l_sigma_hat_norm != 0:
      return l_sigma_hat / l_sigma_hat_norm
    else:
      return np.zeros(2)

  """
  Controlador GVF classic asumiendo omega_d = 0
  """
  def gvf_control(self):
    phi = self.get_phif()
    vel_hat = np.array([np.cos(phi), np.sin(phi)]).T
    omega = - self.kd * angle_of_vectors(self.l_sigma_hat * np.ones((self.N,2)), vel_hat)
    return omega

  """
  Free dynamics following the ascent direction (constant speed)
    * X: Matrix of relative distances to the centroid (N x n) xi = [px, py, pz]
    * sigma: Vector with sigma_field measurements of each agent (N x 1)
  """
  def unicycle_kinematics(self):
    phi = self.get_phif()
    p_dot = self.vf * np.array([np.cos(phi), np.sin(phi)]).T
    phi_dot = self.omega_i
    return p_dot, phi_dot

  def shape_control(self):
    return (self.Xd - self.X)/10

  """
  Euler integration (Step-wise).
  """
  def int_step(self):
    self.rc = np.mean(self.get_pf(), axis=0)
    self.rc_e = la.norm(self.rc - self.sigma_field.mu)
    self.rc_sigma = self.sigma_field.value(self.rc)[0]
    self.rc_grad = self.sigma_field.grad(self.rc)[0]
    self.rc_grad = self.rc_grad/la.norm(self.rc_grad)

    self.X = self.get_pf() - self.rc
    self.d = la.norm(self.X, axis=1)
    self.sigma = self.sigma_field.value(self.get_pf())

    # Calculate x_dot and pass it to sim_frame
    self.l_sigma_hat = self.LsigmaHat(self.X)
    self.omega_i = self.gvf_control()
    x_dot = self.unicycle_kinematics()

    self.int_euler(x_dot)

    # Update telemetry
    self.update_telemetry()