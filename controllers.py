import numpy as np

class StanleyController:

    def __init__(self, model, path_ref, v_ref, k, k_v):
        self.model = model
        self.path_ref = path_ref
        self.v_ref = v_ref
        self.k = k
        self.k_v = k_v

    def steering_angle(self, state):
        pos = state[:2]
        yaw, v = state[2:4]
        pos_fw = pos + self.model.L * np.array([np.cos(yaw), np.sin(yaw)])  # Position front wheel

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
        steering_angle = yaw_error + dir_ct * np.arctan2(self.k * e_ct, (v + self.k_v))
        # print(f"yaw error: {yaw_error}")
        # print(f"Cross-track error: {np.arctan2(self.k * e_ct, (v + self.k_v))}")

        return steering_angle

    def acceleration(self, state):
        """Proportional controller for speed."""
        k_p = 2  # Gain
        v = state[3]
        acceleration = (self.v_ref - v) * k_p

        return acceleration


    def dynamics(self, t, state):
        inputs = self.acceleration(state), self.steering_angle(state)

        return self.model.kinematics(t, state, inputs)



        



