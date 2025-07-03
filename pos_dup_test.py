import time
import mujoco
import mujoco.viewer
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('robots/summit_xl_description/summit_xls.xml')
d = mujoco.MjData(m)

# PID parameters for wheel speed control
Kp = 80.0
Ki = 2.0
Kd = 25.0
error_prev = np.zeros(4)
integral = np.zeros(4)

# Robot geometry
L = 0.3  # half-length of chassis
R = 0.1  # wheel radius
IK_matrix = (1 / R) * np.array([
    [1, -1, -L],
    [1,  1,  L],
    [1,  1, -L],
    [1, -1,  L]
])

# Target position (in meters)
target_xy = np.array([10.0, 0])
position_threshold = 0.05

# Wheel velocity placeholders
mobile_dot = np.zeros(4)
command = np.zeros(4)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)

        # Get robot world position
        robot_xy = d.body('base').xpos[:2]
        error_xy = target_xy - robot_xy
        distance = np.linalg.norm(error_xy)

        # Stop if close enough
        if distance < position_threshold:
            v_body = np.zeros(3)
        else:
            # Position PID → linear velocities
            Kp_pos = 1.5
            vx = Kp_pos * error_xy[0]
            vy = Kp_pos * error_xy[1]
            wz = 0.0
            v_body = np.array([vx, vy, wz])

        # Convert body twist to wheel velocities
        target_vel = IK_matrix @ v_body

        # Read wheel angular velocities from joints
        mobile_dot[0] = d.qvel[19]  # Front Left
        mobile_dot[1] = d.qvel[6]   # Front Right
        mobile_dot[2] = d.qvel[45]  # Back Left
        mobile_dot[3] = d.qvel[32]  # Back Right

        # PID for wheel velocity
        error = target_vel - mobile_dot
        integral += error * m.opt.timestep
        integral = np.clip(integral, -1.0, 1.0)
        derivative = (error - error_prev) / m.opt.timestep
        error_prev = error.copy()

        command = Kp * error + Ki * integral + Kd * derivative

        # Apply to actuators
        d.ctrl[0] = command[1]  # Front Right
        d.ctrl[1] = command[0]  # Front Left
        d.ctrl[2] = command[3]  # Back Right
        d.ctrl[3] = command[2]  # Back Left

        viewer.sync()

        # Real-time pacing
        remaining = m.opt.timestep - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)

        # Debug output
        if int(d.time * 1000) % 500 == 0:
            print(f"[{d.time:.2f}s] Position: {robot_xy}, Target: {target_xy}, Distance: {distance:.3f}")
