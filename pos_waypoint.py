import time
import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('robots/summit_xl_description/summit_xls.xml')
d = mujoco.MjData(m)

# PID parameters for wheel velocity
Kp = 200.0
Ki = 10.0
Kd = 10.0
error_prev = np.zeros(4)
integral = np.zeros(4)

# Robot geometry
L = 0.3  # half-length or width between wheels [m]
R = 0.1  # wheel radius [m]
IK_matrix = (1 / R) * np.array([
    [1, -1, -L],
    [1,  1,  L],
    [1,  1, -L],
    [1, -1,  L]
])

# Define multiple waypoints
waypoints = [
    np.array([3.0, 0.0]),
    np.array([3.0, 3.0]),
    np.array([0.0, 0.0])
]
current_idx = 0
position_threshold = 0.1  # meters

# Wheel and command placeholders
mobile_dot = np.zeros(4)
command = np.zeros(4)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)

        # Get robot base position
        robot_xy = d.body('base').xpos[:2]
        target_xy = waypoints[current_idx]
        error_xy = target_xy - robot_xy
        distance = np.linalg.norm(error_xy)

        # Advance to next waypoint if close
        if distance < position_threshold:
            if current_idx < len(waypoints) - 1:
                current_idx += 1
                target_xy = waypoints[current_idx]
            else:
                v_body = np.zeros(3)  # Stop at final target
        else:
            # Position PID control for body twist
            Kp_pos = 1.5
            vx = Kp_pos * error_xy[0]
            vy = Kp_pos * error_xy[1]
            wz = 0.0
            v_body = np.array([vx, vy, wz])

        # Limit speed to avoid jumps
        max_speed = min(1.0, distance * 2)
        v_body = np.clip(v_body, -max_speed, max_speed)

        # Convert desired twist to wheel velocities
        target_vel = IK_matrix @ v_body

        # Read current wheel speeds
        mobile_dot[0] = d.qvel[19]  # Front Left
        mobile_dot[1] = d.qvel[6]   # Front Right
        mobile_dot[2] = d.qvel[45]  # Back Left
        mobile_dot[3] = d.qvel[32]  # Back Right

        # Wheel velocity PID
        error = target_vel - mobile_dot
        integral += error * m.opt.timestep
        integral = np.clip(integral, -1.0, 1.0)
        derivative = (error - error_prev) / m.opt.timestep
        error_prev = error.copy()
        command = Kp * error + Ki * integral + Kd * derivative

        # Send to actuators
        d.ctrl[0] = command[1]  # Front Right
        d.ctrl[1] = command[0]  # Front Left
        d.ctrl[2] = command[3]  # Back Right
        d.ctrl[3] = command[2]  # Back Left

        # Debug info
        if int(d.time * 1000) % 500 == 0:
            print(f"[{d.time:.2f}s] → Waypoint {current_idx+1}/{len(waypoints)} | Pos: {robot_xy}, Target: {target_xy}, Dist: {distance:.3f}")

        viewer.sync()

        # Real-time step sync
        dt = time.time() - step_start
        remaining = m.opt.timestep - dt
        if remaining > 0:
            time.sleep(remaining)
