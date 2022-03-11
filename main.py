import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from models import BicycleModel1WS, BicycleModel2WS
from controllers import StanleyController


def path_sin():
    """Generate sinusoidal path along global y axis."""
    s = np.linspace(0, 4 * np.pi, 1000)
    pos = np.array([2 * np.sin(0.5 * s), 2 * s]).T

    # Calculate yaw angle of path points
    segments = np.diff(pos, axis=0)
    yaw = np.arctan2(segments[:, 1], segments[:, 0])
    path = np.hstack((pos[:-1], yaw.reshape(-1, 1)))

    return path

def plot_trajectory(state, path_ref, L):
    x, y, yaw = state[:3]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8)) 
    ax.plot(x, y, 'k', label="rear wheel")
    ax.plot(x + L * np.cos(yaw), y + L * np.sin(yaw), 'k-.', label="front wheel")
    ax.plot(path_ref[:, 0], path_ref[:, 1], 'r--', label="path ref")
    ax.legend()
    ax.axis('equal')
    ax.grid()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    return fig

def plot_state(t, state, x_ref=None, y_ref=None, v_ref=None):
    fig, axs = plt.subplots(1, 4, figsize=(15, 5)) 
    axs[0].plot(t, state[0], 'k')
    axs[0].grid()
    axs[0].set_xlabel('t [s]')
    axs[0].set_ylabel('x [m]')

    axs[1].plot(t, state[1], 'k')
    axs[1].grid()
    axs[1].set_xlabel('t [s]')
    axs[1].set_ylabel('y [m/s]')

    axs[2].plot(t, np.rad2deg(state[2]) % (360), 'k')
    axs[2].grid()
    axs[2].set_xlabel('t [s]')
    axs[2].set_ylabel('yaw [deg]')

    v_ref_vec = v_ref * np.ones_like(t)
    axs[3].plot(t, state[3], 'k')
    axs[3].plot(t, v_ref_vec, 'r--')
    axs[3].grid()
    axs[3].set_xlabel('t [s]')
    axs[3].set_ylabel('speed [m/s]')

    fig.tight_layout()
    return fig


def main():
    path_ref = path_sin()
    v_ref = 0.2

    bicycle_1ws = BicycleModel1WS(delta_max=np.radians(30), L=2)
    bicycle_2ws = BicycleModel2WS(delta_max=np.radians(30), L=2)
    controller = StanleyController(bicycle_2ws, path_ref, v_ref, 0.8, 0.5)

    n_timesteps = 1000
    t_span = (0.0, 200.0)
    t_eval = np.linspace(*t_span, n_timesteps)
    state_init = np.array([-4, -3, np.radians(-90), 0.0])
    sol = solve_ivp(controller.dynamics, t_span, state_init, t_eval=t_eval)

    plot_trajectory(sol.y, path_ref, bicycle_1ws.L)
    # plot_state(sol.t, sol.y, v_ref=v_ref)
    plt.show()



if __name__ == "__main__":
    main()