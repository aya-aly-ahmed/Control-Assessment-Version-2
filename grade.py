import csv
import matplotlib.pyplot as plt
import numpy as np
from controller_NelderMead import PIDController_NelderMead
from controller_GridSearch import PIDController

def read_waypoints(file_path):
    times = []
    speeds = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            speeds.append(float(row[0]))
            times.append(float(row[1]))
    return times, speeds

def simulate(times, speeds, pid_controller, dt, total_time, initial_speed=0):
    time_steps = int(total_time / dt)
    sim_times = np.linspace(0, total_time, time_steps)
    sim_speeds = np.zeros(time_steps)
    current_speed = initial_speed
    m = 1750
    c = 40         # I know this is exaggerated
    p_max = 90000
    f_traction_max = 6000
    jerk_max = 1.5 # This is only modeled to prevent unrealistic spikes.
    a_prev = 0
    for i in range(1, time_steps):
        target_speed = np.interp(sim_times[i], times, speeds)
        control_signal = pid_controller.control(target_speed, current_speed, dt)
        u_throttle = max(0.0, min(1.0, abs(control_signal)))
 
        # For simplicity negative throttle (braking) is calculated the same way positive throttle is.
        u_throttle = np.sign(control_signal) * u_throttle

        v_safe = max(current_speed, 0.1)
        f_max = min(f_traction_max, p_max / v_safe)
        f_throttle = u_throttle * f_max
        a_raw = (1 / m) * (f_throttle - c * current_speed)
        max_da = jerk_max * dt
        da = a_raw - a_prev
        if da > max_da:
            a_limited = a_prev + max_da
        elif da < -max_da:
            a_limited = a_prev - max_da
        else:
            a_limited = a_raw

        current_speed += a_limited * dt
        a_prev = a_limited
        sim_speeds[i] = current_speed

    return sim_times, sim_speeds

def grade_performance(actual_speeds, target_speeds):
    error = np.array(actual_speeds) - np.array(target_speeds)
    mse = np.mean(np.square(error))
    return mse

def main():
    times, speeds = read_waypoints('waypointsNew.csv')
    
    dt = 0.1
    total_time = 179
    initial_speed = 0


    #######################################################################################################
    ######################### Apply your tuned parameters to the controller class #########################
    #######################################################################################################

    # Your control output is now a throttle output from 0 to 1. The system model handles the simulation of this throttle output to throttle_force and then to acceleration then finally to velocity.

    # You don't need to match the waypoints exactly. Overshoot, stability, and steady state error are more important criteria to take into consideration.

    kp=0 # Tune these parameters to achieve the most stable control. (MSE isn't the only deciding criterion)
    ki=0 # Tune these parameters to achieve the most stable control. (MSE isn't the only deciding criterion)
    kd=0 # Tune these parameters to achieve the most stable control. (MSE isn't the only deciding criterion)
    # pid_controller 

    First_Parameters = PIDController(kp,ki,kd) #to get the first suggestion of parameters from Grid search
    First_Parameters.return_params()
    pid_controller = PIDController_NelderMead(First_Parameters.kp, First_Parameters.ki, First_Parameters.kd) #to get the optimized parameters from Nelder-Mead
    
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    sim_times, sim_speeds = simulate(times, speeds, pid_controller, dt, total_time, initial_speed)
    
    target_speeds = np.interp(sim_times, times, speeds)
    mse = grade_performance(sim_speeds, target_speeds)
    print(f'Mean Squared Error: {mse}')

    
    """
    here is the overshoot and rise-time calculation.//////////////////////////////////////////////////////////
    """
    min_plateau_duration = 3.0  # Minimum duration to consider a plateau (in seconds)
    sim_speeds = np.array(sim_speeds)
    target_speeds = np.array(target_speeds)

    plateau_threshold = 0.01
    min_samples = int(min_plateau_duration / dt)

    max_rise_time = 0
    max_overshoot = 0

    i = 0
    segment_number = 1

    while i < len(target_speeds) - min_samples:
        segment = target_speeds[i:i + min_samples]
        if np.max(segment) - np.min(segment) < plateau_threshold:
            plateau_value = np.mean(segment)
            start_idx = i

            # Extend plateau
            while i < len(target_speeds) and abs(target_speeds[i] - plateau_value) < plateau_threshold:
                i += 1
            end_idx = i

            actual = sim_speeds[start_idx:end_idx]
            time_window = sim_times[start_idx:end_idx]

            # Determine trend: look a bit before plateau to compare
            trend = "flat"
            if start_idx > 5:
                recent_avg = np.mean(target_speeds[start_idx-5:start_idx])
                if plateau_value > recent_avg:
                    trend = "increasing"
                elif plateau_value < recent_avg:
                    trend = "decreasing"

            # Rise time logic
            v10 = 0.1 * plateau_value
            v90 = 0.9 * plateau_value
            t_rise_start, t_rise_end = None, None

            for t, v in zip(time_window, actual):
                if t_rise_start is None and v >= v10:
                    t_rise_start = t
                if t_rise_end is None and v >= v90:
                    t_rise_end = t
                    break

            rise_time = (t_rise_end - t_rise_start) if t_rise_start and t_rise_end else None

            # Overshoot logic depending on trend
            if trend == "increasing":
                peak_value = np.max(actual)
                overshoot = max(0, (peak_value - plateau_value) / plateau_value * 100)
            elif trend == "decreasing":
                min_value = np.min(actual)
                overshoot = max(0, (plateau_value - min_value) / plateau_value * 100)
            else:
                overshoot = 0

            # Print details
            print(f"\n📌 Segment {segment_number} ({trend.upper()})")
            print(f"→ Time Range: {sim_times[start_idx]:.2f}s to {sim_times[end_idx-1]:.2f}s")
            print(f"→ Target Speed: {plateau_value:.2f} m/s")
            if trend == "increasing":
                print(f"→ Peak Speed: {peak_value:.2f} m/s → Overshoot: {overshoot:.2f}%")
            elif trend == "decreasing":
                print(f"→ Lowest Speed: {min_value:.2f} m/s → Overshoot: {overshoot:.2f}%")
            else:
                print("→ Overshoot: N/A")

            if rise_time:
                print(f"→ Rise Time: {rise_time:.2f}s (from {v10:.2f} to {v90:.2f}) between {t_rise_start:.2f}s and {t_rise_end:.2f}s")
            else:
                print("→ Rise Time: not detected")

            max_rise_time = max(max_rise_time, rise_time or 0)
            max_overshoot = max(max_overshoot, overshoot)
            segment_number += 1
        else:
            i += 1

    print(f"\n✅ Max Overshoot Across All Segments: {max_overshoot:.2f}%")
    print(f"✅ Max Rise Time Across All Segments: {max_rise_time:.2f}s")
    # Plotting the results
    
    plt.plot(sim_times, sim_speeds, label='Speed')
    plt.plot(sim_times, target_speeds, label='Target Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()