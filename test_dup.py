import time
import mujoco
import mujoco.viewer as viewer
import numpy as np

# Load model and data
m = mujoco.MjModel.from_xml_path('robots/summit_xl_description/summit_xls.xml')
d = mujoco.MjData(m)

# PID parameters
Kp = 100.0
Ki = 5.0
Kd = 10.0

# Initialize PID memory
error_prev = np.zeros(4)
integral = np.zeros(4)

# Wheel order: [FL, FR, BL, BR]
mobile_dot = np.zeros(4)
target_vel = np.zeros(4)
command = np.zeros(4)

with viewer.launch(m, d) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)

        # Get wheel velocities
        mobile_dot[0] = d.qvel[19]  # Front Left
        mobile_dot[1] = d.qvel[6]   # Front Right
        mobile_dot[2] = d.qvel[45]  # Back Left
        mobile_dot[3] = d.qvel[32]  # Back Right

        # Handle keyboard input
        key_events = viewer.get_key_events()
        key_map = {k.key for k in key_events}

        if viewer.KEY_UP in key_map:
            # Forward
            target_vel = np.array([1.5, 1.5, 1.5, 1.5])
        elif viewer.KEY_DOWN in key_map:
            # Backward
            target_vel = np.array([-1.5, -1.5, -1.5, -1.5])
        elif viewer.KEY_RIGHT in key_map:
            # Right strafe
            target_vel = np.array([1.5, -1.5, -1.5, 1.5])
        elif viewer.KEY_LEFT in key_map:
            # Left strafe
            target_vel = np.array([-1.5, 1.5, 1.5, -1.5])
        else:
            # No key: stop
            target_vel = np.zeros(4)

        # PID control
        error = target_vel - mobile_dot
        integral += error * m.opt.timestep
        derivative = (error - error_prev) / m.opt.timestep
        error_prev = error.copy()

        command = Kp * error + Ki * integral + Kd * derivative

        # Apply to actuators
        d.ctrl[0] = command[1]  # Front Right
        d.ctrl[1] = command[0]  # Front Left
        d.ctrl[2] = command[3]  # Back Right
        d.ctrl[3] = command[2]  # Back Left

        viewer.sync()

        # Sleep to maintain real-time
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
