import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt
from pathlib import Path

from dataLoader import DataLoader
from eskf import ErrorStateKalmanFilter
from imuDynamics import IMU_MotionModel
from VOSensor import VOSensor
from OdometrySensor import Odometry_Sensor
from GnssSensor import GNSS_Sensor

# Load data (rosbag + VO CSV)
loader = DataLoader(
    bag_path="",
    vo_csv_path=""
)

# Output file in TUM trajectory format
TXT_OUTPUT = Path("##")

# === Create Models ===
imuModel   = IMU_MotionModel()  # IMU motion model (propagation)
odoSensor  = Odometry_Sensor()  # Wheel odometry sensor
voSensor   = VOSensor()         # Visual Odometry sensor
gnssSensor = GNSS_Sensor()      # GNSS sensor

# Instantiate ESKF with IMU + VO + Odometry + GNSS
eskf = ErrorStateKalmanFilter(imuModel, voSensor, odoSensor, gnssSensor)

# === Set IMU Noise (from experimental calibration) ===
imuModel.accNoiseVar  = 3e-03   # Accelerometer noise variance
imuModel.gyroNoiseVar = 3e-06   # Gyroscope noise variance

# === Set Odometry Sensor Noise ===
odoSensor.R = np.eye(7) * 1e-2**2

# === Set GNSS Sensor Noise ===
gnssSensor.R = np.eye(3) * 1e-2**2  # "Super-GPS"

# === VO Sensor Noise (optional override) ===
# voSensor.R = np.eye(7) * 1e-4**2

# === Set initial uncertainty (error state covariance) ===
eskf.errStateCovParts("dPos",     np.ones(3) * 1e2)
eskf.errStateCovParts("dVel",     np.ones(3) * 1e0)
eskf.errStateCovParts("dTheta",   np.ones(3) * 1e0)
eskf.errStateCovParts("dAccBias", np.ones(3) * 1e-2**2)
eskf.errStateCovParts("dGyroBias",np.ones(3) * 1e-2**2)

# Example of setting sensor bias manually
# eskf.stateParts("AccBias", np.array([0.1315, 0.1291, -0.1899]))
eskf.stateParts("GyroBias", np.array([-0.00271475, -0.00240784, -0.00176051]))

# === Logs ===
events = list(loader)   # Chronological list of all sensor measurements
N = len(events)
logT    = np.zeros(N)   # Time log
logX    = np.zeros((N, eskf.nStates))       # State log
logCovX = np.zeros((N, eskf.nErrStates))    # Covariance log

# Initialize previous timestamps for each sensor
t_prev_imu   = None
t_prev_vo    = None
t_prev_wheel = None

# === GNSS time window ===
X_SECONDS = 20.0  # window size (first and last 20 s)

# Last GNSS timestamp (relative to dataset end)
gnss_ts = [ts for ts, k, _ in events if k == 'gnss']
t_last_gnss = gnss_ts[-1] if gnss_ts else None

# === Run ESKF ===
logT[0]    = 0.0
logX[0]    = eskf.stateParts().T
logCovX[0] = np.diag(eskf.errStateCovParts())
t0 = events[0][0]

for k, (ts, key, data) in enumerate(events, start=0):
    dt = ts - events[k-1][0]
    
    # --- IMU propagation ---
    if key == 'imu':
        if t_prev_imu is None:
            dt_imu = 0.0
        else:
            dt_imu = ts - t_prev_imu
        eskf.predict(dt_imu, acc=data['acc'], gyro=data['gyro'])
        t_prev_imu = ts

    # --- Wheel odometry update ---
    elif key == 'wheel':
        if t_prev_wheel is None:
            dt_wheel = 0.0
        else:
            dt_wheel = ts - t_prev_wheel
        eskf.update('Odometry Sensor', v=data['v'], w=data['w'], dt=dt_wheel)
        t_prev_wheel = ts
 
    # --- Visual odometry update ---
    elif key == 'vo':
        if t_prev_vo is None:
            dt_vo = 0.0
        else:
            dt_vo = ts - t_prev_vo
        eskf.update('VO Sensor',
                    dx=data['dx'], dy=data['dy'], dz=data['dz'], dyaw=data['dyaw'],
                    inliers_pose=data['inliers_pose'], inliers_plane=data['inliers_plane'],
                    dt=dt_vo)
        t_prev_vo = ts

    # --- GNSS update (only in first and last X_SECONDS) ---
    elif key == 'gnss':
        if ((ts - t0) <= X_SECONDS) or (t_last_gnss is not None and (t_last_gnss - ts) <= X_SECONDS):
            eskf.update('GNSS Sensor',
                        lat=data['pos'][0], lon=data['pos'][1], alt=data['pos'][2],
                        cov=data['cov'])
            t_prev_gps = ts

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

# === Plotting ===
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

# --- 3D Trajectory with equal axis scaling ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], '-', lw=1.5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('IMU+VO+ODO+GPS')
ax.grid(True)

# Force same scale on all axes
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
