"""
Nelder-Mead optimization Algorithm

"""


import numpy as np
from scipy.optimize import minimize

class PIDController_NelderMead :
    def __init__(self, kp, ki, kd):
        """
        Automatically optimize PID gains when the object is created.
        kp, ki, kd passed are ignored.
        """
        # Only import when inside __init__ to avoid circular import
        from grade import read_waypoints, simulate, grade_performance

        self.dt = 0.1
        self.total_time = 179
        self.initial_speed = 0

        # Load target path
        self.times, self.speeds = read_waypoints('waypointsNew.csv')

        # Store functions locally (not globally)
        self.simulate_fn = simulate
        self.grade_fn = grade_performance

        # Run PID optimization
        result = minimize(self._mse_cost, [3.80, 0, 4.00], method='Nelder-Mead',
                          options={'maxiter': 30, 'disp': False})

        self.kp, self.ki, self.kd = result.x
        print(f"\n Optimized PID: Kp={self.kp:.4f}, Ki={self.ki:.4f}, Kd={self.kd:.4f}, MSE={result.fun:.6f}")

        # PID controller state
        self.integral = 0.0
        self.prev_error = 0.0


    def _mse_cost(self, params):
        kp, ki, kd = params
        temp_controller = PIDController_NelderMead._create_temp(kp, ki, kd)
        sim_times, sim_speeds = self.simulate_fn(
            self.times, self.speeds, temp_controller, self.dt, self.total_time, self.initial_speed
        )
        target_speeds = np.interp(sim_times, self.times, self.speeds)
        return self.grade_fn(sim_speeds, target_speeds)

    @staticmethod
    def _create_temp(kp, ki, kd):
        obj = PIDController_NelderMead.__new__(PIDController_NelderMead)
        obj.kp, obj.ki, obj.kd = kp, ki, kd
        obj.integral = 0.0
        obj.prev_error = 0.0
        return obj



    def control(self, target_speed, current_speed, dt):
        """
        Normal PID control function.
        """
        error = target_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
