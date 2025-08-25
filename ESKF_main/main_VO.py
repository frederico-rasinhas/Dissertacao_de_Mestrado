import numpy as np
import quaternion as qt
import matplotlib.pyplot as plt
from pathlib import Path

from dataLoader import DataLoader
from eskf import ErrorStateKalmanFilter
from imuDynamics import IMU_MotionModel
from VOSensor import VOSensor

# Load data from rosbag + Visual Odometry CSV
loader = DataLoader(
    bag_path="bags/4m.bag",
    vo_csv_path="VO/4m.csv"
)

# Output file in TUM trajectory format
TXT_OUTPUT = Path("evaluation/ESKF/Trajeto1/IMU+VO_4m.txt")

# === Create Models ===
imuModel = IMU_MotionModel()   # Motion model (IMU-based propagation)
voSensor = VOSensor()          # Visual Odometry sensor model
eskf = ErrorStateKalmanFilter(imuModel, voSensor)

# === Set IMU Noise (from experimental calibration) ===
imuModel.accNoiseVar = 3e-03    # Accelerometer noise variance
imuModel.gyroNoiseVar = 3e-06   # Gyroscope noise variance

# === Set initial uncertainty (error state covariance) ===
eskf.errStateCovParts("dAccBias", np.ones(3))
eskf.errStateCovParts("dGyroBias", np.ones(3))

# === Logs ===
events = list(loader)   # Chronological list of all sensor measurements
N = len(events)
logT    = np.zeros(N)   # Time log
logX    = np.zeros((N, eskf.nStates))       # State log
logCovX = np.zeros((N, eskf.nErrStates))    # Covariance log

# Initialize previous timestamps
t_prev_imu = None
t_prev_vo  = None

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
 
    elif key == 'vo':
        if t_prev_vo is None:
            dt_vo = 0.0
        else:
            dt_vo = ts - t_prev_vo

        # Update step with Visual Odometry (incremental pose + feature inliers)
        eskf.update('VO Sensor',
                    dx=data['dx'], dy=data['dy'], dz=data['dz'], dyaw=data['dyaw'],
                    inliers_pose=data['inliers_pose'], inliers_plane=data['inliers_plane'],
                    dt=dt_vo)

        t_prev_vo = ts
         
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

# --- 3D Trajectory with equal axis scale ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(pos[:,0], pos[:,1], pos[:,2], '-', lw=1.5)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('IMU+VO')
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
