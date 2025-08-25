from eskf import *
import numpy as np
import quaternion as qt

class LIDAR_Sensor(ErrorStateSensorModel):
    def __init__(self):
        super().__init__("LIDAR Sensor")

        # Initial measurement: [vx, vy, vz, qw, qx, qy, qz]
        self.z = np.array([0,0,0,1,0,0,0]).reshape(-1,1)

        # Measurement noise (tune as needed)
        self.R = np.zeros((7,7))

        # Measurement Jacobian
        self.H = None

        # Previous orientation
        self.qOld = None

    def measure(self, filter, *args, **kwargs):
        dx = kwargs['dx']
        dy = kwargs['dy']
        dyaw = kwargs['dyaw']
        dt = kwargs['dt']
        n_edges = kwargs['n_edges']
        n_flats = kwargs['n_flats']

        if dt <= 1e-8:
            print(f"[LIDAR Sensor] WARNING: dt = {dt:.8f} is too small. Replacing with current estimate.")
            q = filter.stateParts("Orientation").reshape(-1,1)
            vB = np.zeros((3,1))  # No velocity
            z = np.concatenate((vB, q))
            return z, 1e6 * np.eye(7)  # Very noisy → low confidence

        # === Velocities in the body frame ===
        vx = dx / dt
        vy = dy / dt
        vB = np.array([[vx, 0, 0]]).T
        
        # === Dynamic noise model based on number of features ===
        N_total = n_edges + n_flats
        sigma = 1.0 / ((N_total ** 2 )/2 + 1)
        print(f"[LIDAR] sigma = {sigma}")

        # === Update covariance matrix R ===
        self.R = np.eye(7) * sigma**2  # Diagonal covariance matrix

        # === First measurement ===
        if self.qOld is None:
            self.qOld = qt.as_quat_array(filter.stateParts("Orientation").ravel())
            z = np.concatenate((vB, filter.stateParts("Orientation")))
            return z, 1000 * self.R

        # === Orientation update ===
        phi = np.array([0, 0, dyaw])
        dq = qt.from_rotation_vector(phi)
        q = qt.as_float_array(self.qOld * dq).reshape(-1, 1)
        self.qOld = qt.as_quat_array(filter.stateParts("Orientation").ravel())

        z = np.concatenate((vB, q))  # Final measurement
        
        return z, self.R

    def estimate(self, filter):
        if self.H is None:
            self.H = np.zeros((7, filter.nErrStates))

        # Extract orientation and velocity in navigation frame
        q = filter.stateParts("Orientation").ravel()
        vNav = filter.stateParts("Velocity").reshape(-1,1)

        # Rotation matrix world → body (Rᵀ)
        qw, qx, qy, qz = q
        R = np.array([
            [qw**2+qx**2-qy**2-qz**2, 2*(qx*qy+qw*qz),         2*(qx*qz-qw*qy)],
            [2*(qx*qy-qw*qz),         qw**2-qx**2+qy**2-qz**2, 2*(qy*qz+qw*qx)],
            [2*(qx*qz+qw*qy),         2*(qy*qz-qw*qx),         qw**2-qx**2-qy**2+qz**2]
        ])

        # Body-frame velocity estimate
        vB = R @ vNav

        # Auxiliary derivatives for Jacobian
        Rw = np.array([[qw, qz, -qy], [-qz, qw, qx], [qy, -qx, qw]])
        Rx = np.array([[qx, qy, qz], [qy, -qx, qw], [qz, -qw, -qx]])
        Ry = np.array([[-qy, qx, -qw], [qx, qy, qz], [qw, qz, -qy]])
        Rz = np.array([[-qz, qw, qx], [-qw, -qz, qy], [qx, qy, qz]])

        d_vB_dq = np.empty((3,4))
        d_vB_dq[:,0:1] = 2*(Rw @ vNav - qw*vB)
        d_vB_dq[:,1:2] = 2*(Rx @ vNav - qx*vB)
        d_vB_dq[:,2:3] = 2*(Ry @ vNav - qy*vB)
        d_vB_dq[:,3:4] = 2*(Rz @ vNav - qz*vB)

        dq_dTh = 0.5 * np.array([[-qx, -qy, -qz],
                                 [ qw, -qz,  qy],
                                 [ qz,  qw, -qx],
                                 [-qy,  qx,  qw]])

        # Estimated measurement
        estZ = np.vstack((vB, q.reshape(-1,1)))

        # Measurement Jacobian
        self.H[0:3, filter.errStateIdxs("dTheta")] = d_vB_dq @ dq_dTh
        self.H[0:3, filter.errStateIdxs("dVel")]   = R
        self.H[3:7, filter.errStateIdxs("dTheta")] = dq_dTh

        return estZ, self.H
    
    def estimate(self, filter):
        if self.H is None:
            self.H = np.zeros((7,filter.nErrStates))

        # Get relevant state values
        q = filter.stateParts("Orientation").ravel()
        vNav = filter.stateParts("Velocity").reshape(-1,1)

        # Auxiliary stuff
        qw, qx, qy, qz = q
        # Rotation from *world to body* <- i.e., this is the *transpose* of R{q}
        R = np.array([
            [qw**2+qx**2-qy**2-qz**2, 2*(qx*qy+qw*qz),         2*(qx*qz-qw*qy)],
            [2*(qx*qy-qw*qz),         qw**2-qx**2+qy**2-qz**2, 2*(qy*qz+qw*qx)],
            [2*(qx*qz+qw*qy),         2*(qy*qz-qw*qx),         qw**2-qx**2-qy**2+qz**2]
        ])

        # Estimated body velocity:
        vB = R @ vNav

        # Auxiliary stuff
        Rw = np.array([[qw, qz, -qy], [-qz, qw, qx], [qy, -qx, qw]])
        Rx = np.array([[qx, qy, qz], [qy, -qx, qw], [qz, -qw, -qx]])
        Ry = np.array([[-qy, qx, -qw], [qx, qy, qz], [qw, qz, -qy]])
        Rz = np.array([[-qz, qw, qx], [-qw, -qz, qy], [qx, qy, qz]])

        d_vB_dq = np.empty((3,4))
        d_vB_dq[:,0:1] = 2*(Rw @ vNav - qw*vB)
        d_vB_dq[:,1:2] = 2*(Rx @ vNav - qx*vB)
        d_vB_dq[:,2:3] = 2*(Ry @ vNav - qy*vB)
        d_vB_dq[:,3:4] = 2*(Rz @ vNav - qz*vB)

        dq_dTh = 0.5 * np.array([[-qx, -qy, -qz],[qw, -qz, qy],[qz, qw, -qx],[-qy, qx, qw]])  # eq. 281

        # Estimated observation:
        estZ = np.concatenate((vB, q.reshape(-1,1)))  # [Body velocity; Orientation]

        # Error state Jacobian
        self.H[0:3,filter.errStateIdxs("dTheta")] = d_vB_dq @ dq_dTh
        self.H[0:3,filter.errStateIdxs("dVel")]   = R
        self.H[3:7,filter.errStateIdxs("dTheta")] = dq_dTh           # eq. 281

        return estZ, self.H
