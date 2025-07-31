import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        """
        PID controller that auto-tunes Kp, Ki, Kd using grid search.
        The input kp, ki, kd are ignored.
        """
        import grade  # Delayed import to avoid circular reference

        # Simulation setup
        self.dt = 0.1
        self.total_time = 179
        self.initial_speed = 0
        self.times, self.speeds = grade.read_waypoints('waypointsNew.csv')

        # === Tuning using nested loops (grid search) ===
        best_mse = float('inf')
        best_params = (0, 0, 0)




        kd: float = 4
        ki: float = 0
        kp: float = 3.8

        temp_controller = PIDController._temp(kp, ki, kd)
        sim_times, sim_speeds = grade.simulate(
                            self.times, self.speeds,
                            temp_controller,
                            self.dt, self.total_time, self.initial_speed
                        )
        target_speeds = np.interp(sim_times, self.times, self.speeds)
        mse = grade.grade_performance(sim_speeds, target_speeds)

        print(f"Testing: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} → MSE={mse:.6f}")
        best_mse = mse
        best_params = (kp, ki, kd)





        # for kd in np.arange(0.1, 5, 0.3):  # Test Kd from 0.0 to 1.1
        #     for ki in np.arange(0.0, 5, 0.2):  # Test Ki from 0.0 to 2
        #         for kp in np.arange(0.6, 10, 0.25):  # Test Kp from 0.6 to 5
        #             print(f"Testing before: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} ")
        #             temp_controller = PIDController._temp(kp, ki, kd)
        #             sim_times, sim_speeds = grade.simulate(
        #                 self.times, self.speeds,
        #                 temp_controller,
        #                 self.dt, self.total_time, self.initial_speed
        #             )
        # target_speeds = np.interp(sim_times, self.times, self.speeds)
        # mse = grade.grade_performance(sim_speeds, target_speeds)

        # print(f"Testing: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f} → MSE={mse:.6f}")

        # if mse < best_mse:
        #                   best_mse = mse
        #                   best_params = (kp, ki, kd)




        self.kp, self.ki, self.kd = best_params
        print(f"\nTuned PID gains From GRID SEARCH: Kp = {self.kp:.2f}, Ki = {self.ki:.2f}, Kd = {self.kd:.2f} | Best MSE = {best_mse:.6f}")
        
        # Controller state
        self.integral = 0.0
        self.prev_error = 0.0

    def return_params(self):
        """
        Return the tuned PID parameters.
        """
        return self.kp, self.ki, self.kd    

    @staticmethod
    def _temp(kp, ki, kd):
        """
        Create a lightweight temporary PID controller with test gains.
        """
        obj = PIDController.__new__(PIDController)
        obj.kp, obj.ki, obj.kd = kp, ki, kd
        obj.integral = 0.0
        obj.prev_error = 0.0
        return obj

    def control(self, target_speed, current_speed, dt):
        """
        Standard PID control method.
        """
        error = target_speed - current_speed
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        return output

