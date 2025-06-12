import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend # two legends

# Animation tools from matplotlib
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Source Seeking tools
from .toolbox import _zoom_range, unicycle_patch

# ----------------------------------------------------------------------
# Global plotting parameters
# ----------------------------------------------------------------------

# Figsize adjusted to 16:9
FIGSIZE = (16,9)

# DPI resolution dictionary
RES_DIC = {
    "480p"   : 640,
    "HD"     : 1280,
    "FullHD" : 1920,
    "2K"     : 2560,
    "4K"     : 3880
    }

PALETTE = ["b", "green", "r"]
LEGEND_LABELS = ["active", "computer unit", "non-active", "centroid"]

# Agents parameters
AGENTS_RAD = 0.05
AGENTS_DATA_LW = 0.7
AGENTS_ACTIVE_LW = 0.08
AGENTS_DEAD_LW = 0.05
ALPHA_INIT = 0.02

# Size of the unicycle patches
KW_PATCH = {"size":1.5, "lw":0.5}

# Arrow parameters
arr_kw = {"width":0.003, "scale":25, "zorder":10}
thr_arrows = -1

# Limits of the main axis
xlim, ylim = 75, 75
kw_draw_field = {"xlim":2*xlim, "ylim":2*ylim, "n":400, "contour_levels":0}

# ----------------------------------------------------------------------
# Visualizer Class I
# ----------------------------------------------------------------------

#######################
# Static Plot Class I #
#######################

def plot_class1(data_col, sim, t_list = None, field_rot_sw = False, tail_time=None, d_max=0):
    # Init and final elements from data
    if t_list is None:
        li_list = np.array([0, data_col.get("pf").shape[0]-1], dtype=int)
    else:
        li_list = np.array(np.array(t_list)/sim.dt, dtype=int)

    # Set tail time
    if tail_time is None:
        if len(t_list) > 1:
            tail_time = t_list[-1] - t_list[0] 
        else:
            tail_time = sim.tf

    # Get data from the Data Collector
    xdata = data_col.get("pf")[..., 0]
    ydata = data_col.get("pf")[..., 1]
    rc_xdata = data_col.get("rc")[:, 0]
    rc_ydata = data_col.get("rc")[:, 1]
    edata = data_col.get("e")
    ddata = data_col.get("d")
    sigma_data = data_col.get("sigma")
    active_status = data_col.get("active")
    l_sigma = data_col.get("l_sigma")
    rc_grad = data_col.get("rc_grad")
    field_rot = data_col.get("field_rot")

    # Initilise visualizer parameters
    tail_frames = int(tail_time/sim.dt)

    color_lmb = lambda i: [PALETTE[0] if active_status[i,n] else PALETTE[2] for n in range(sim.N)]
    color_init = color_lmb(0)
    color = color_lmb(li_list[-1])

    lw = [AGENTS_ACTIVE_LW if sim.active[n] else AGENTS_DEAD_LW for n in range(sim.N)]
    z_order = [3 if sim.active[n] else 4 for n in range(sim.N)]
    alpha_init = ALPHA_INIT

    legend_flags = [False, False, False]

    #############
    # FIGURE init
    #############
    fig = plt.figure(figsize=FIGSIZE, dpi=100)
    grid = plt.GridSpec(3, 5, hspace=0.1, wspace=1)
    main_ax       = fig.add_subplot(grid[:, 0:3])
    sigma_data_ax = fig.add_subplot(grid[0, 3:5], xticklabels=[])
    ddata_ax      = fig.add_subplot(grid[1, 3:5], xticklabels=[])
    edata_ax      = fig.add_subplot(grid[2, 3:5])

    # Axis configuration
    main_ax.set_xlim([-xlim,xlim])
    main_ax.set_ylim([-ylim,ylim])
    main_ax.set_ylabel(r"$p_y$ [L]")
    main_ax.set_xlabel(r"$p_x$ [L]")
    main_ax.grid(True)

    sigma_data_ax.set_ylabel(r"$\sigma_i$ [u]")
    sigma_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    sigma_data_ax.yaxis.tick_right()
    sigma_data_ax.grid(True)
    ddata_ax.set_ylabel(r"$||x_i||$ [L]")
    ddata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ddata_ax.yaxis.tick_right()
    ddata_ax.grid(True)
    edata_ax.set_xlabel(r"$t$ [T]")
    edata_ax.set_ylabel(r"$||p_\sigma - p_c||$ [L]")
    edata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    edata_ax.yaxis.tick_right()
    edata_ax.grid(True)

    #############
    # MAIN axis
    #############
    if field_rot_sw:
        li = li_list[1]
        sim.sigma_field.rot = field_rot[li]
    sim.sigma_field.draw(fig, main_ax, **kw_draw_field)

    main_ax.set_title("N = {1:d} robots".format(sim.tf, sim.N))
    main_ax.grid(True)

    # Plot agents
    for li in li_list:
        color = color_lmb(li)
        for n in range(sim.N):
            icon_init = plt.Circle((xdata[0,n], ydata[0,n]), AGENTS_RAD, color=color_init[n])
            icon_init.set_alpha(alpha_init)
            icon = plt.Circle((xdata[li,n], ydata[li,n]), AGENTS_RAD, color=color[n])

            icon_init.set_zorder(z_order[n])
            icon.set_zorder(z_order[n])

            for i in range(len(PALETTE)):
                if not legend_flags[i] and color[n] == PALETTE[i]:
                    icon.set_label(LEGEND_LABELS[i])
                    legend_flags[i] = True

            main_ax.add_patch(icon_init)
            main_ax.add_patch(icon)
            main_ax.plot(xdata[li-tail_frames:li,n],ydata[li-tail_frames:li,n],
                         c=color[n], ls="-", lw=lw[n], zorder=z_order[n])

        # Gradient arrow
        sim.sigma_field.draw_grad([rc_xdata[li], rc_ydata[li]], main_ax, **arr_kw)
        main_ax.quiver(rc_xdata[li], rc_ydata[li], l_sigma[li,0], l_sigma[li,1],
                        color="red", **arr_kw)

        # Time texts
        main_ax.text(rc_xdata[li] + 3, rc_ydata[li] - 10, "t = {0:.0f} s".format(li*sim.dt))

    li = li_list[-1]
    color = color_lmb(li)

    # Plot centroid
    icon_centroid = plt.Circle((rc_xdata[li], rc_ydata[li]), AGENTS_RAD*2, color="k")
    main_ax.add_patch(icon_centroid)
    icon_centroid.set_label(LEGEND_LABELS[3])

    main_ax.plot(rc_xdata[0],rc_ydata[0], "x", c="k")
    main_ax.plot(rc_xdata[:],rc_ydata[:], c="k", ls="--", lw=1.2)

    # Legends
    main_ax.legend(loc="upper right", ncol=sim.N,
                fancybox=True, framealpha=1, prop={'size': 9})

    arr1 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
    arr2 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
    leg = Legend(main_ax, [arr1, arr2], ["Actual computed \nascending direction", "(Non-computed) \ngradient"],
                loc="upper left", prop={'size': 9})
    main_ax.add_artist(leg)

    # Obstacles
    for obs in sim.obstacles:
        main_ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color="white"))

    #############
    # DATA axis
    #############
    time_vec = np.linspace(0, sim.tf, int((sim.tf + 11/10*sim.dt)/sim.dt))

    sigma_data_ax.axhline(sim.sigma_field.value(np.array(sim.sigma_field.mu)), c="k", ls="--", lw=1.2)
    ddata_ax.axhline(0, c="k", ls="--", lw=1.2)
    edata_ax.axhline(0, c="k", ls="--", lw=1.2)

    if d_max > 0:
        ddata_ax.axhline(d_max, c="k", ls="--", lw=0.8, alpha=0.6)

    for n in range(sim.N):
        sigma_data_ax.plot(time_vec, sigma_data[:,n], c=color[n], lw=lw[n], zorder=z_order[n])
        ddata_ax.plot(time_vec, ddata[:,n], c=color[n], lw=lw[n], zorder=z_order[n])
    edata_ax.plot(time_vec, edata[:], c="k", lw=1.2)

    # Visualise the final plot
    plt.show()


##########################
# Animation Plot Class I #
##########################

def anim_class1(data_col, sim, anim_tf=None, tail_frames=100, res_label="HD", field_rot_sw=False, d_max=0):

    # Get data from the Data Collector
    xdata = data_col.get("pf")[..., 0]
    ydata = data_col.get("pf")[..., 1]
    rc_xdata = data_col.get("rc")[:, 0]
    rc_ydata = data_col.get("rc")[:, 1]
    edata = data_col.get("e")
    ddata = data_col.get("d")
    sigma_data = data_col.get("sigma")
    active_status = data_col.get("active")
    l_sigma = data_col.get("l_sigma")
    rc_grad = data_col.get("rc_grad")
    field_rot = data_col.get("field_rot")

    # Initialize some animation variables
    res = RES_DIC[res_label]
    figsize = FIGSIZE

    dt = sim.dt
    if anim_tf is None:
        anim_tf = sim.tf
    anim_frames = int((anim_tf + 11/10*dt) / dt)

    color_lmb = lambda i: [PALETTE[0] if active_status[i,n] else PALETTE[2] for n in range(sim.N)]
    lw_lmb = lambda i: [AGENTS_ACTIVE_LW if active_status[i,n] else AGENTS_DEAD_LW for n in range(sim.N)]
    z_lmb = lambda i: [3 if active_status[i,n] else 4 for n in range(sim.N)]
    
    color = color_lmb(0)
    lw = lw_lmb(0)
    z_order = z_lmb(0)

    alpha_init = ALPHA_INIT

    # Zoom effect
    min_zoom, max_zoom = 1.0, 1.0
    min_zoom_t, max_zoom_t = 10, 40

    if min_zoom_t <= max_zoom_t:
        _min_z, _max_z = min_zoom, max_zoom
        _min_t, _max_t = min_zoom_t, max_zoom_t
    else:
        _min_z, _max_z = max_zoom, min_zoom
        _min_t, _max_t = max_zoom_t, min_zoom_t


    #############
    # FIGURE init
    #############
    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    grid = plt.GridSpec(3, 5, hspace=0.1, wspace=1)
    main_ax       = fig.add_subplot(grid[:, 0:3])
    sigma_data_ax = fig.add_subplot(grid[0, 3:5], xticklabels=[])
    ddata_ax      = fig.add_subplot(grid[1, 3:5], xticklabels=[])
    edata_ax      = fig.add_subplot(grid[2, 3:5])

    # Axis configuration
    main_ax.set_xlim(rc_xdata[0] + np.array(_zoom_range(-xlim, xlim, 0, _min_z)))
    main_ax.set_ylim(rc_ydata[0] + np.array(_zoom_range(-ylim, ylim, 0, _min_z)))
    main_ax.set_xlim([-xlim,xlim])
    main_ax.set_ylim([-ylim,ylim])
    main_ax.set_ylabel(r"$p_y$ [L]")
    main_ax.set_xlabel(r"$p_x$ [L]")

    sigma_data_ax.set_ylabel(r"$\sigma_i$ [u]")
    sigma_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    sigma_data_ax.yaxis.tick_right()
    sigma_data_ax.grid(True)
    ddata_ax.set_ylabel(r"$||x_i||$ [L]")
    ddata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ddata_ax.yaxis.tick_right()
    ddata_ax.grid(True)
    edata_ax.set_xlabel(r"$t$ [T]")
    edata_ax.set_ylabel(r"$||p_\sigma - p_c||$ [L]")
    edata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    edata_ax.yaxis.tick_right()
    edata_ax.grid(True)

    #############
    # MAIN axis
    #############
    if field_rot_sw:
        kw_draw_img = {key: kw_draw_field[key] for key in kw_draw_field.keys() if key != "contour_levels"}
        sim.sigma_field.rot = field_rot[0]
        field_img = sim.sigma_field.draw_imshow(fig, main_ax, **kw_draw_img)
    else:
        sim.sigma_field.draw(fig, main_ax, **kw_draw_field)
        main_ax.grid(True)


    # Agents
    lines_agents = []
    icons_agents = []
    legend_flags = [False, False, False]
    for n in range(sim.N):
        icon_init = plt.Circle((xdata[0,n], ydata[0,n]), AGENTS_RAD, color=color[n])
        icon_init.set_alpha(alpha_init)
        icon = plt.Circle((xdata[0,n], ydata[0,n]), AGENTS_RAD, color=color[n])

        main_ax.add_patch(icon_init)
        main_ax.add_patch(icon)
        line, = main_ax.plot(xdata[0,n], ydata[0,n], c=color[n], ls="-", lw=lw[n])

        for i in range(len(PALETTE)):
            if not legend_flags[i] and color[n] == PALETTE[i]:
                icon.set_label(LEGEND_LABELS[i])
                legend_flags[i] = True

        lines_agents.append(line)
        icons_agents.append(icon)

    # Centroids
    lines_centroids = []
    icons_centroids = []

    line, = main_ax.plot(rc_xdata[0], rc_ydata[0], c="k", ls="--", lw=1.2)
    icon_centroid = plt.Circle((rc_xdata[0], rc_ydata[0]), AGENTS_RAD*2, color="k")
    icon_centroid.set_label(LEGEND_LABELS[3])

    main_ax.add_patch(icon_centroid)
    main_ax.plot(rc_xdata[0],rc_ydata[0], "x", c="k")

    lines_centroids.append(line)
    icons_centroids.append(icon_centroid)

    # Gradient arrow
    q_grad = sim.sigma_field.draw_grad([rc_xdata[0], rc_ydata[0]], main_ax, **arr_kw)
    q_Lsig = main_ax.quiver(rc_xdata[0], rc_ydata[0], l_sigma[0,0], l_sigma[0,1],
                            color="red", **arr_kw)

    # Legends and title
    main_ax.legend(loc="upper right", ncol=sim.N,
                fancybox=True, framealpha=1, prop={'size': 9})

    arr1 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
    arr2 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
    leg = Legend(main_ax, [arr1, arr2], ["Actual computed \nascending direction", "(Non-computed) \ngradient"],
                loc="upper left", prop={'size': 9})
    main_ax.add_artist(leg)

    txt_title = main_ax.set_title("")

    # Obstacles
    for obs in sim.obstacles:
        main_ax.add_patch(plt.Circle((obs[0], obs[1]), obs[2], color="white"))


    #############
    # DATA axis
    #############
    time_vec = np.linspace(0, anim_tf, anim_frames)

    data_lines_plt = []
    sigma_lines_data = []
    ddata_lines_data = []
    for n in range(sim.N):
        sigma_line_data, = sigma_data_ax.plot(time_vec, sigma_data[0:len(time_vec),n], c=color[n], lw=lw[n])
        ddata_line_data, = ddata_ax.plot(time_vec, ddata[0:len(time_vec),n], c=color[n], lw=lw[n])
        sigma_lines_data.append(sigma_line_data)
        ddata_lines_data.append(ddata_line_data)
    edata_ax.plot(time_vec, edata[0:len(time_vec)], c="k", lw=1.2)
    ddata_ax.axhline(0, c="k", ls="--", lw=1.2, alpha=0.6)

    if d_max > 0:
        ddata_ax.axhline(d_max, c="k", ls="--", lw=0.8, alpha=0.6)

    sigma_data_ax.axhline(sim.sigma_field.value(np.array(sim.sigma_field.mu)), c="k", ls="--", lw=1.2)
    sigma_line = sigma_data_ax.axvline(0, c="k", ls="--", lw=1.2)
    dline = ddata_ax.axvline(0, c="k", ls="--", lw=1.2)
    eline = edata_ax.axvline(0, c="k", ls="--", lw=1.2)
    

    #########################
    # Building the animation
    #########################

    # Function to update the animation
    def animate(i):
        color = color_lmb(i)
        lw = lw_lmb(i)
        z_order = z_lmb(i)

        # Update the vector field
        if field_rot_sw:
            sim.sigma_field.rot = field_rot[i]
            Z = sim.sigma_field.draw_imshow(**kw_draw_img, make_im=False)
            field_img.set_data(Z)

        # Update the agents
        for n in range(sim.N):
            icons_agents[n].remove()
            icons_agents[n] = plt.Circle((xdata[i,n], ydata[i,n]), AGENTS_RAD, color=color[n])

            icons_agents[n].set_zorder(z_order[n])
            icon_init.set_zorder(z_order[n])
            lines_agents[n].set_zorder(z_order[n])
            lines_agents[n].set_color(color[n])
            sigma_lines_data[n].set_color(color[n])
            ddata_lines_data[n].set_color(color[n])

            if i > tail_frames:
                lines_agents[n].set_data(xdata[i-tail_frames:i,n], ydata[i-tail_frames:i,n])
            else:
                lines_agents[n].set_data(xdata[0:i,n], ydata[0:i,n])
            main_ax.add_patch(icons_agents[n])

        # Update icon and lines of the centroid
        icons_centroids[0].remove()
        icons_centroids[0] = plt.Circle((rc_xdata[i], rc_ydata[i]), 0.1, color="k")
        icons_centroids[0].set_zorder(6)

        lines_centroids[0].set_data(rc_xdata[0:i], rc_ydata[0:i])
        main_ax.add_patch(icons_centroids[0])

        # Update arrows
        if edata[i] > thr_arrows:
            q_grad.set_offsets(np.array([rc_xdata[i], rc_ydata[i]]).T)
            q_Lsig.set_offsets(np.array([rc_xdata[i], rc_ydata[i]]).T)
            q_grad.set_UVC(rc_grad[i,0], rc_grad[i,1])
            q_Lsig.set_UVC(l_sigma[i,0], l_sigma[i,1])
        else:
            q_grad.set_offsets(np.array([2*xlim, 2*ylim]).T)
            q_Lsig.set_offsets(np.array([2*xlim, 2*ylim]).T)

        # string format: https://www.w3schools.com/python/ref_string_format.asp
        txt_title.set_text('Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>4} robots'.format(i, i*dt, sim.N))

        sigma_line.set_xdata([i*dt])
        dline.set_xdata([i*dt])
        eline.set_xdata([i*dt])

        # Efecto zoom
        # if (i*dt < _min_t):
        #   zoom = _min_z
        # elif (i*dt >= _min_t) and (i*dt <= _max_t):
        #   zoom = _min_z + (_max_z - _min_z) * (i*dt - _min_t)/(_max_t - _min_t)
        # else:
        #   zoom = _min_z + (_max_z - _min_z)
        # main_ax.set_xlim(0 + sim.sigma_field.mu[0]/2 * i/anim_frames +
        #                  np.array(_zoom_range(-ylim, ylim, 0, zoom)))
        # main_ax.set_ylim(0 + sim.sigma_field.mu[1]/2 * i/anim_frames +
        #                  np.array(_zoom_range(-ylim, ylim, 0, zoom)))

        # Show the completion percentage of the simulation
        if (i % int((anim_frames-1)/10) == 0):
            print("tf = {0:>5.2f} | {1:.2%}".format(i*dt, i/(anim_frames-1)))

    # Generate the animation
    print("Simulating {0:d} frames... \nProgress:".format(anim_frames))
    anim = FuncAnimation(fig, animate, frames=anim_frames, interval=1000/60)
    anim.embed_limit = 40
    
    # Close plots and return the animation class to be compiled
    plt.close()
    return anim


# ----------------------------------------------------------------------
# Visualizer Class II
# ----------------------------------------------------------------------

########################
# Static Plot Class II #
########################

def plot_class2(data_col, sim, t_list = None, tail_time=None, dpi=100, figsize=None, alpha=None):

    # Init and final elements from data
    if t_list is None:
        li_list = np.array([0, data_col.get("pf").shape[0]-1], dtype=int)
    else:
        li_list = np.array(np.array(t_list)/sim.dt, dtype=int)

    # Set tail time
    if tail_time is None:
        if t_list is None:
            tail_time = sim.tf
        elif len(t_list) > 1:
            tail_time = t_list[-1] - t_list[0] 
        else:
            tail_time = sim.tf

    # Collect data from simulation
    xdata = data_col.get("pf")[..., 0]
    ydata = data_col.get("pf")[..., 1]
    phidata = data_col.get("phif")
    rc_xdata = data_col.get("rc")[:, 0]
    rc_ydata = data_col.get("rc")[:, 1]
    edata = data_col.get("e")
    ddata = data_col.get("d")
    sigma_data = data_col.get("sigma")
    omega_data = data_col.get("omega")
    l_sigma = data_col.get("l_sigma")
    rc_grad = data_col.get("rc_grad")

    # Initilise visualizer parameters
    tail_frames = int(tail_time/sim.dt)

    kw_patch = KW_PATCH

    color_init = [PALETTE[0] for n in range(sim.N)]
    color = [PALETTE[0] for n in range(sim.N)]
    lw = [AGENTS_ACTIVE_LW for n in range(sim.N)]
    z_order = [3 for n in range(sim.N)]
    alpha_init = ALPHA_INIT

    legend_flags = [False, False, False]

    #############
    # FIGURE init
    #############
    if figsize is None:
        figsize = FIGSIZE
    if alpha is None:
        alpha = 1

    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = plt.GridSpec(4, 7, hspace=0.1, wspace=7)
    main_ax       = fig.add_subplot(grid[:, 0:4])
    sigma_data_ax = fig.add_subplot(grid[0, 4:8], xticklabels=[])
    omega_data_ax = fig.add_subplot(grid[1, 4:8], xticklabels=[])
    ddata_ax      = fig.add_subplot(grid[2, 4:8], xticklabels=[])
    edata_ax      = fig.add_subplot(grid[3, 4:8])

    # Axis configuration
    main_ax.set_xlim([-xlim,xlim])
    main_ax.set_ylim([-ylim,ylim])
    main_ax.set_ylabel(r"$Y$ [L]")
    main_ax.set_xlabel(r"$X$ [L]")
    main_ax.grid(True)
    main_ax.set_aspect("equal")

    sigma_data_ax.set_ylabel(r"$\sigma_i$ [u]")
    sigma_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    sigma_data_ax.yaxis.tick_right()
    sigma_data_ax.grid(True)
    omega_data_ax.set_ylabel(r"$\omega_i$ [rad/T]")
    omega_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    omega_data_ax.yaxis.tick_right()
    omega_data_ax.grid(True)
    ddata_ax.set_ylabel(r"$||x_i||$ [L]")
    ddata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ddata_ax.yaxis.tick_right()
    ddata_ax.grid(True)
    edata_ax.set_xlabel(r"$t$ [T]")
    edata_ax.set_ylabel(r"$||p_\sigma - p_c||$ [L]")
    edata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    edata_ax.yaxis.tick_right()
    edata_ax.grid(True)

    #############
    # MAIN axis
    #############
    sim.sigma_field.draw(fig, main_ax, **kw_draw_field)

    main_ax.set_title("t = {0:.2f} [T] | N = {1:d} robots".format(sim.tf, sim.N))
    main_ax.grid(True)

    # Agents
    for li in li_list:
        for n in range(sim.N):
            icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color_init[n], **kw_patch)
            icon_init.set_alpha(alpha_init)
            icon = unicycle_patch([xdata[li,n], ydata[li,n]], phidata[li,n], color[n], **kw_patch)

            icon_init.set_zorder(z_order[n])
            icon.set_zorder(z_order[n])

            for i in range(len(PALETTE)):
                if not legend_flags[i] and color[n] == PALETTE[i]:
                    icon.set_label(LEGEND_LABELS[i])
                    legend_flags[i] = True

            main_ax.add_patch(icon_init)
            main_ax.add_patch(icon)
            main_ax.plot(xdata[li-tail_frames:li,n],ydata[li-tail_frames:li,n],
                        c=color[n], ls="-", lw=lw[n], zorder=z_order[n])

        # Gradient arrow
        sim.sigma_field.draw_grad([rc_xdata[li], rc_ydata[li]], main_ax, **arr_kw)
        main_ax.quiver(rc_xdata[li], rc_ydata[li], l_sigma[li,0], l_sigma[li,1],
                        color="red", **arr_kw)

        # Time texts
        main_ax.text(rc_xdata[li] + 3, rc_ydata[li] - 15, "t = {0:.0f} s".format(li*sim.dt))

    li = li_list[-1]

    # Centroid
    icon_centroid = plt.Circle((rc_xdata[li], rc_ydata[li]), AGENTS_RAD*2, color="k")
    main_ax.add_patch(icon_centroid)
    icon_centroid.set_label(LEGEND_LABELS[3])

    main_ax.plot(rc_xdata[0],rc_ydata[0], "x", c="k")
    main_ax.plot(rc_xdata[:],rc_ydata[:], c="k", ls="--", lw=1.2)

    # Legend
    main_ax.legend(loc="upper right", ncol=sim.N,
                fancybox=True, framealpha=1, prop={'size': 9})

    arr1 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
    arr2 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
    leg = Legend(main_ax, [arr1, arr2], ["Actual computed ascending direction", "(Non-computed) gradient"],
                loc="upper left", prop={'size': 10})
    main_ax.add_artist(leg)

    #############
    # DATA axis
    #############
    time_vec = np.linspace(0, sim.tf, int((sim.tf + 11/10*sim.dt)/sim.dt))

    sigma_data_ax.axhline(sim.sigma_field.value(np.array(sim.sigma_field.mu)), c="k", ls="--", lw=1.2)
    edata_ax.axhline(0, c="k", ls="--", lw=1.2)
        
    for n in range(sim.N):
        kw_data = {"c":color[n], "lw":AGENTS_DATA_LW, "zorder":z_order[n], "alpha":alpha}
        sigma_data_ax.plot(time_vec, sigma_data[:,n], **kw_data)
        omega_data_ax.plot(time_vec, omega_data[:,n], **kw_data)
        ddata_ax.plot(time_vec, ddata[:,n], **kw_data)
    
    edata_ax.plot(time_vec, edata[:], c="k", lw=AGENTS_DATA_LW+1)

    # Visualise the final plot
    plt.show()


###########################
# Animation Plot Class II #
###########################

def anim_class2(data_col, sim, anim_tf=None, tail_frames=100, res_label="HD"):

    # Collect data from simulation
    xdata = data_col.get("pf")[..., 0]
    ydata = data_col.get("pf")[..., 1]
    phidata = data_col.get("phif")
    rc_xdata = data_col.get("rc")[:, 0]
    rc_ydata = data_col.get("rc")[:, 1]
    edata = data_col.get("e")
    ddata = data_col.get("d")
    sigma_data = data_col.get("sigma")
    omega_data = data_col.get("omega")
    l_sigma = data_col.get("l_sigma")
    rc_grad = data_col.get("rc_grad")

    # Initialize some animation variables
    res = RES_DIC[res_label]
    figsize = FIGSIZE

    dt = sim.dt
    if anim_tf is None:
        anim_tf = sim.tf
    anim_frames = int((anim_tf + 11/10*dt) / dt)

    kw_patch = KW_PATCH

    color_init = [PALETTE[0] for n in range(sim.N)]
    color = [PALETTE[0] for n in range(sim.N)]
    lw = [AGENTS_ACTIVE_LW for n in range(sim.N)]
    alpha_init = ALPHA_INIT

    # Zoom effect variables
    min_zoom, max_zoom = 1.0, 1.1
    min_zoom_t, max_zoom_t = 10, 40

    if min_zoom_t <= max_zoom_t:
        _min_z, _max_z = min_zoom, max_zoom
        _min_t, _max_t = min_zoom_t, max_zoom_t
    else:
        _min_z, _max_z = max_zoom, min_zoom
        _min_t, _max_t = max_zoom_t, min_zoom_t


    #############
    # FIGURE init
    #############
    fig = plt.figure(figsize=figsize, dpi=res/figsize[0])
    grid = plt.GridSpec(4, 7, hspace=0.1, wspace=7)
    main_ax       = fig.add_subplot(grid[:, 0:4])
    sigma_data_ax = fig.add_subplot(grid[0, 4:8], xticklabels=[])
    omega_data_ax = fig.add_subplot(grid[1, 4:8], xticklabels=[])
    ddata_ax      = fig.add_subplot(grid[2, 4:8], xticklabels=[])
    edata_ax      = fig.add_subplot(grid[3, 4:8])

    # Axis configuration
    main_ax.set_xlim([-xlim,xlim])
    main_ax.set_ylim([-ylim,ylim])
    main_ax.set_ylabel(r"$p_y$ [L]")
    main_ax.set_xlabel(r"$p_x$ [L]")
    main_ax.grid(True)

    sigma_data_ax.set_ylabel(r"$\sigma_i$ [u]")
    sigma_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    sigma_data_ax.yaxis.tick_right()
    sigma_data_ax.grid(True)
    omega_data_ax.set_ylabel(r"$w_i$ [rad/T]")
    omega_data_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    omega_data_ax.yaxis.tick_right()
    omega_data_ax.grid(True)
    ddata_ax.set_ylabel(r"$||x_i||$ [L]")
    ddata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ddata_ax.yaxis.tick_right()
    ddata_ax.grid(True)
    edata_ax.set_xlabel(r"$t$ [T]")
    edata_ax.set_ylabel(r"$||p_\sigma - p_c||$ [L]")
    edata_ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    edata_ax.yaxis.tick_right()
    edata_ax.grid(True)


    #############
    # MAIN axis
    #############
    sim.sigma_field.draw(fig, main_ax, **kw_draw_field)

    # Agents
    lines_agents = []
    icons_agents = []
    legend_flags = [False, False, False]
    for n in range(sim.N):
        icon_init = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color_init[n], **kw_patch)
        icon_init.set_alpha(alpha_init)
        icon = unicycle_patch([xdata[0,n], ydata[0,n]], phidata[0,n], color[n], **kw_patch)

        main_ax.add_patch(icon_init)
        main_ax.add_patch(icon)
        line, = main_ax.plot(xdata[0,n], ydata[0,n], c=color[n], ls="-", lw=lw[n])

        for i in range(len(PALETTE)):
            if not legend_flags[i] and color[n] == PALETTE[i]:
                icon.set_label(LEGEND_LABELS[i])
                legend_flags[i] = True

        lines_agents.append(line)
        icons_agents.append(icon)

    # Centroids
    lines_centroids = []
    icons_centroids = []

    line, = main_ax.plot(rc_xdata[0], rc_ydata[0], c="k", ls="--", lw=1.2)
    icon_centroid = plt.Circle((rc_xdata[0], rc_ydata[0]), AGENTS_RAD*2, color="k")
    icon_centroid.set_label(LEGEND_LABELS[3])

    main_ax.add_patch(icon_centroid)
    main_ax.plot(rc_xdata[0],rc_ydata[0], "x", c="k")

    lines_centroids.append(line)
    icons_centroids.append(icon_centroid)

    # Gradient arrow
    q_grad = sim.sigma_field.draw_grad([rc_xdata[0], rc_ydata[0]], main_ax, **arr_kw)
    q_Lsig = main_ax.quiver(rc_xdata[0], rc_ydata[0], l_sigma[0,0], l_sigma[0,1],
                            color="red", **arr_kw)

    # Legend and title
    main_ax.legend(loc="upper right", ncol=sim.N,
                fancybox=True, framealpha=1, prop={'size': 9})

    arr1 = plt.scatter([],[],c='red',marker=r'$\uparrow$',s=60)
    arr2 = plt.scatter([],[],c='k'  ,marker=r'$\uparrow$',s=60)
    leg = Legend(main_ax, [arr1, arr2], ["Actual computed ascending direction", "(Non-computed) gradient"],
                loc="upper left", prop={'size': 10})
    main_ax.add_artist(leg)

    txt_title = main_ax.set_title("")

    #############
    # DATA axis 
    #############
    time_vec = np.linspace(0, anim_tf, anim_frames)

    data_lines_plt = []
    sigma_lines_data = []
    omega_lines_data = []
    ddata_lines_data = []
    for n in range(sim.N):
        sigma_line_data, = sigma_data_ax.plot(time_vec, sigma_data[0:len(time_vec),n], c=color[n], lw=lw[n])
        ddata_line_data, = ddata_ax.plot(time_vec, ddata[0:len(time_vec),n], c=color[n], lw=lw[n])
        omega_line_data, = omega_data_ax.plot(time_vec, omega_data[0:len(time_vec),n], c=color[n], lw=lw[n])
        sigma_lines_data.append(sigma_line_data)
        omega_lines_data.append(omega_line_data)
        ddata_lines_data.append(ddata_line_data)
    edata_ax.plot(time_vec, edata[0:len(time_vec)], c="k", lw=1.2)

    sigma_data_ax.axhline(sim.sigma_field.value(np.array(sim.sigma_field.mu)), c="k", ls="--", lw=1.2)
    sigma_line = sigma_data_ax.axvline(0, c="k", ls="--", lw=1.2)
    omega_line = omega_data_ax.axvline(0, c="k", ls="--", lw=1.2)
    dline = ddata_ax.axvline(0, c="k", ls="--", lw=1.2)
    eline = edata_ax.axvline(0, c="k", ls="--", lw=1.2)

    #########################
    # Building the animation
    #########################

    # Function to update the animation
    def animate(i):
        # Agents
        for n in range(sim.N):
            icons_agents[n].remove()
            icons_agents[n] = unicycle_patch([xdata[i,n], ydata[i,n]], phidata[i,n], color_init[n])
            main_ax.add_patch(icons_agents[n])

            if i > tail_frames:
                lines_agents[n].set_data(xdata[i-tail_frames:i,n], ydata[i-tail_frames:i,n])
            else:
                lines_agents[n].set_data(xdata[0:i,n], ydata[0:i,n])

        # Icon and lines of the centroid
        icons_centroids[0].remove()
        icons_centroids[0] = plt.Circle((rc_xdata[i], rc_ydata[i]), 0.1, color="k")
        icons_centroids[0].set_zorder(6)

        lines_centroids[0].set_data(rc_xdata[0:i], rc_ydata[0:i])
        main_ax.add_patch(icons_centroids[0])

        # Arrows
        if edata[i] > thr_arrows:
            q_grad.set_offsets(np.array([rc_xdata[i], rc_ydata[i]]).T)
            q_Lsig.set_offsets(np.array([rc_xdata[i], rc_ydata[i]]).T)
            q_grad.set_UVC(rc_grad[i,0], rc_grad[i,1])
            q_Lsig.set_UVC(l_sigma[i,0], l_sigma[i,1])
        else:
            q_grad.set_offsets(np.array([2*xlim, 2*ylim]).T)
            q_Lsig.set_offsets(np.array([2*xlim, 2*ylim]).T)

        # string format: https://www.w3schools.com/python/ref_string_format.asp
        txt_title.set_text('Frame = {0:>4} | Tf = {1:>5.2f} [T] | N = {2:>4} robots'.format(i, i*dt, sim.N))

        sigma_line.set_xdata([i*dt])
        omega_line.set_xdata([i*dt])
        dline.set_xdata([i*dt])
        eline.set_xdata([i*dt])

        # Zoom effect
        if (i*dt < _min_t):
            zoom = _min_z
        elif (i*dt >= _min_t) and (i*dt <= _max_t):
            zoom = _min_z + (_max_z - _min_z) * (i*dt - _min_t)/(_max_t - _min_t)
        else:
            zoom = _min_z + (_max_z - _min_z)
        main_ax.set_xlim(0 + sim.sigma_field.mu[0]/2 * i/anim_frames +
                        np.array(_zoom_range(-ylim, ylim, 0, zoom)))
        main_ax.set_ylim(0 + sim.sigma_field.mu[1]/2 * i/anim_frames +
                        np.array(_zoom_range(-ylim, ylim, 0, zoom)))
        main_ax.grid(True)

        # Show the completion percentage of the simulation
        if (i % int((anim_frames-1)/10) == 0):
            print("tf = {0:>5.2f} | {1:.2%}".format(i*dt, i/(anim_frames-1)))

    # Generate the animation
    print("Simulating {0:d} frames... \nProgress:".format(anim_frames))
    anim = FuncAnimation(fig, animate, frames=anim_frames, interval=1000/60)
    anim.embed_limit = 40
    
    # Close plots and return the animation class to be compiled
    plt.close()
    return anim