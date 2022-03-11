import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from models import BicycleModel1WS, BicycleModel2WS
from controllers import StanleyController


def path_sin():
    """Generate sinusoidal path along global y axis."""
    s = np.linspace(0, 4 * np.pi, 1000)
    pos = np.array([2 * np.sin(1 * s), 2 * s]).T

    # Calculate yaw angle of path points
    segments = np.diff(pos, axis=0)
    yaw = np.arctan2(segments[:, 1], segments[:, 0])
    path = np.hstack((pos[:-1], yaw.reshape(-1, 1)))

    return path

def simulate():
    dt = 0.5  # Controller frequency
    t_max = 200  # Max simulation time
    n_steps = int(t_max / dt)
    t_vec = np.linspace(0, t_max, n_steps)

    # Controller input
    path_ref = path_sin()
    v_ref = 0.2

    # Bicycle model
    wheelbase = 2
    delta_max = np.radians(30)
    model = BicycleModel2WS(delta_max, wheelbase)

    # Controller
    params = {"wheelbase": wheelbase,
              "k": 0.8,
              "k_soft": 2,
              "k_p": 2}
    controller = StanleyController(path_ref, v_ref, params)

    # Initialize histories for time, state and inputs
    t_hist = []
    state_hist = []
    inputs_hist = []

    # Initial state and input
    state = np.array([-4, -2, np.radians(-90), 0.0])
    inputs = controller.compute_controls(state)

    # Simulate
    for t in t_vec:
        t_span = (t, t + dt)
        t_eval = np.linspace(*t_span, 5)

        sol = solve_ivp(model.kinematics, t_span, state, t_eval=t_eval, args=(inputs, ))
        state = sol.y[:, -1]
        inputs = controller.compute_controls(state)

        # Store state, inputs and time for analysis
        state_hist.append(sol.y)
        inputs_hist.append(inputs)
        t_hist.append(sol.t)

        if reached_target(state, path_ref[-1, :2], wheelbase):
            print("Reached end of path.")
            break

    state_hist = np.concatenate(state_hist, axis=1)
    inputs_hist = np.vstack(inputs_hist).T
    t_hist = np.concatenate(t_hist)

    plot_trajectory(state_hist, path_ref, wheelbase)
    # plot_state(t_hist, state_hist, v_ref=v_ref)

def reached_target(state, target, wheelbase):
    pos = state[:2]
    yaw = state[2]
    pos_fw = pos + wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel
    dist_to_target = np.sqrt(np.sum((pos_fw - target)**2))

    return dist_to_target < 0.05

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


if __name__ == "__main__":
    simulate()
    plt.show()