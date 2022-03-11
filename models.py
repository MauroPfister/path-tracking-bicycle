import numpy as np

# Note: Position of vehicle is defined as the position of the rear wheel
class BicycleModel1WS:
    """
    Class representing bicycle model with front wheel steering.
    """

    def __init__(self, delta_max=np.radians(90), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, t, state, inputs):
        """
        Kinematic model for Scipy's solve_ivp function.
        Note that the position [x, y] and velocity v of the bicycle correspond 
        to the position and velocity of the rear wheel.
        :param t: continuous time
        :param state: [x, y, yaw, v] state
        :param inputs: [a, delta] input
        """
        yaw, v = state[2:4]
        a, delta = inputs
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        x_dot = v * np.cos(yaw)
        y_dot = v * np.sin(yaw)
        yaw_dot = v / self.L * np.tan(delta)
        v_dot = a

        return np.array([x_dot, y_dot, yaw_dot, v_dot])


class BicycleModel2WS:
    """
    Class representing bicycle model with front and back wheel steering.
    """

    def __init__(self, delta_max=np.radians(90), L=2):
        self.delta_max = delta_max # [rad] max steering angle
        self.L = L # [m] Wheel base of vehicle

    def kinematics(self, t, state, inputs):
        """
        Kinematic model for Scipy's solve_ivp function.
        Note that the position [x, y] and velocity v of the bicycle correspond 
        to the position and velocity of the rear wheel.
        :param t: continuous time
        :param state: [x, y, yaw, v] state
        :param inputs: [a, delta] input
        """
        yaw, v = state[2:4]
        a, delta = inputs
        delta = np.clip(delta, -self.delta_max, self.delta_max)

        x_dot = v * np.cos(yaw - delta)
        y_dot = v * np.sin(yaw - delta)
        yaw_dot = 2 * v / self.L * np.sin(delta)
        v_dot = a

        return np.array([x_dot, y_dot, yaw_dot, v_dot])