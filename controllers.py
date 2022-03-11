import numpy as np

class StanleyController:
    """
    Path tracking controller using Stanley controller for lateral control and simple proportional
    controller for lateral longitudinal control.
    See: https://ieeexplore.ieee.org/document/4282788
    """

    def __init__(self, path_ref, v_ref, params):
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.wheelbase = params["wheelbase"]
        self.k = params["k"]
        self.k_soft = params["k_soft"]
        self.k_p = params["k_p"]

    def steering_angle(self, state):
        """
        Calculate control action for steering angle.
        """
        pos = state[:2]
        yaw, v = state[2:4]
        pos_fw = pos + self.wheelbase * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel

        # Find point on path nearest to front wheel
        dists = np.sum((pos_fw - self.path_ref[:, :2])**2, axis=1)
        id_nearest = np.argmin(dists)
        path_point_nearest = self.path_ref[id_nearest]  # [x, y, yaw] of nearest path point

        # Yaw error term
        yaw_error = path_point_nearest[2] - yaw # TODO: Normalize angles correctly

        # Cross-track error to nearest point on path
        e_ct = np.sqrt(dists[id_nearest])

        # Cross-track error term has to be negative if we are on left side
        # of path and positive if we are on right side of path
        vehicle_normal = np.array([np.sin(yaw), -np.cos(yaw)])
        nearest_p_to_front_wheel = pos_fw - path_point_nearest[:2]
        dir_ct = np.sign(np.dot(vehicle_normal, nearest_p_to_front_wheel))

        # Final steering angle output
        steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_soft))
        # print(f"yaw error: {yaw_error}")
        # print(f"Cross-track error: {np.arctan2(self.k * e_ct, (v + self.k_soft))}")

        return steering_angle

    def acceleration(self, state):
        """
        Calculate control action for acceleration.
        """
        v = state[3]
        acceleration = (self.v_ref - v) * self.k_p

        return acceleration

    def compute_controls(self, state):
        return [self.acceleration(state), self.steering_angle(state)]