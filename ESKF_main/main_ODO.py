import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt
from pathlib import Path

from dataLoader import DataLoader
from eskf import ErrorStateKalmanFilter
from imuDynamics import IMU_MotionModel
from OdometrySensor import Odometry_Sensor

# Load data from rosbag
loader = DataLoader("")

# Output file in TUM trajectory format
TXT_OUTPUT = Path("")

# === Create Models ===
imuModel = IMU_MotionModel()     # Motion model (IMU-based propagation)
odoSensor = Odometry_Sensor()    # Wheel odometry sensor model
eskf = ErrorStateKalmanFilter(imuModel, odoSensor)

# === Set IMU Noise (from experimental calibration) ===
imuModel.accNoiseVar = 3e-03    # Accelerometer noise variance
imuModel.gyroNoiseVar = 3e-06   # Gyroscope noise variance

# === Set Odometry Sensor Noise ===
odoSensor.R = np.eye(7) * 1e-3**2

# === Set initial uncertainty (error state covariance) ===
eskf.errStateCovParts("dPos", np.ones(3) * 1e2)
eskf.errStateCovParts("dVel", np.ones(3) * 1e0)
eskf.errStateCovParts("dTheta", np.ones(3) * 1e0)
eskf.errStateCovParts("dAccBias",np.ones(3) * 1e-2**2)
eskf.errStateCovParts("dGyroBias",np.ones(3) * 1e-2**2)

# Optional: set initial bias estimates manually
# eskf.stateParts("AccBias",np.array([ 0.13153156,  0.12914237, -0.18990788]))
eskf.stateParts("GyroBias",np.array([-0.00271475, -0.00240784, -0.00176051]))

# === Logs ===
events = list(loader)   # Chronological list of all sensor measurements
N = len(events)
logT    = np.zeros(N)   # Time log
logX    = np.zeros((N, eskf.nStates))       # State log
logCovX = np.zeros((N, eskf.nErrStates))    # Covariance log

# Initialize previous timestamps
t_prev_imu   = None
t_prev_wheel = None

# === Run ESKF ===
logT[0] = 0.0
logX[0] = eskf.stateParts().T
logCovX[0] = np.diag(eskf.errStateCovParts())
t0 = events[0][0]

for k, (ts, key, data) in enumerate(events, start=0):
    dt = ts - events[k-1][0]
    
    # Always propagate with IMU data
    if key == 'imu':
        if t_prev_imu is None:
            dt_imu = 0.0
        else:
            dt_imu = ts - t_prev_imu
        # Prediction step with IMU
        eskf.predict(dt_imu, acc=data['acc'], gyro=data['gyro'])
        t_prev_imu = ts

    elif key == 'wheel':
        if t_prev_wheel is None:
            dt_wheel = 0.0
        else:
            dt_wheel = ts - t_prev_wheel
        # Update step with wheel odometry (linear + angular velocity)
        eskf.update('Odometry Sensor', v=data['v'], w=data['w'], dt=dt_wheel)
        t_prev_wheel = ts
    
    # Store logs
    logT[k]    = ts - t0
    logX[k]    = eskf.stateParts().ravel()
    logCovX[k] = np.diag(eskf.errStateCovParts())

# === Save trajectory in TUM format ===
with open(TXT_OUTPUT, 'w') as f:
    for t, x in zip(logT, logX):
        px, py, pz = x[eskf.stateIdxs("Position")]
        qw, qx, qy, qz = x[eskf.stateIdxs("Orientation")]  # Pay attention to quaternion order
        f.write(f"{t:.6f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

print(f"[TUM] ESKF trajectory exported to: {TXT_OUTPUT}")

# === Plots ===
plt.figure()

# Position
data = logX[:,eskf.stateIdxs("Position")]
plt.subplot(2, 3, 1)
plt.plot(logT, data[:,0:3], label=["px","py","pz"])
plt.legend()
plt.grid(True)

# Velocity
data = logX[:,eskf.stateIdxs("Velocity")]
plt.subplot(2, 3, 2)
plt.plot(logT, data[:,0:3], label=["vx","vy","vz"])
plt.legend()
plt.grid(True)

# Orientation (quaternion)
data = logX[:,eskf.stateIdxs("Orientation")]
plt.subplot(2, 3, 3)
plt.plot(logT, data[:,0:4], label=["qw","qx","qy","qz"])
plt.legend()
plt.grid(True)

# Accelerometer bias
data = logX[:,eskf.stateIdxs("AccBias")]
plt.subplot(2, 3, 4)
plt.plot(logT, data[:,0:3], label=["Ab x","Ab y","Ab z"])
plt.legend()
plt.grid(True)

# Gyroscope bias
data = logX[:,eskf.stateIdxs("GyroBias")]
plt.subplot(2, 3, 5)
plt.plot(logT, data[:,0:3], label=["wb x","wb y","wb z"])
plt.legend()
plt.grid(True)

# --- 3D Trajectory (debug top view) ---
pos = logX[:, eskf.stateIdxs("Position")]
ax = plt.subplot(2, 3, 6, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], '-')
ax.view_init(elev=90, azim=-90)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('3D Trajectory (debug)')
ax.grid(True)

plt.tight_layout()
plt.show()

# --- 3D Trajectory with equal axis scale ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], '-', lw=1.5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('IMU+ODO')
ax.grid(True)

# Force equal scale on all axes
max_range = np.array([pos[:,0].max() - pos[:,0].min(),
                      pos[:,1].max() - pos[:,1].min(),
                      pos[:,2].max() - pos[:,2].min()]).max() / 2.0

mid_x = (pos[:,0].max() + pos[:,0].min()) * 0.5
mid_y = (pos[:,1].max() + pos[:,1].min()) * 0.5
mid_z = (pos[:,2].max() + pos[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1]) 

plt.tight_layout()
plt.show()
