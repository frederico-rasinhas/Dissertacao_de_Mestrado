from eskf import *
import quaternion as qt

# ----------- Class Odometry (Sensor Model) -----------
class Odometry_Sensor(ErrorStateSensorModel):
    def __init__(self):
        super().__init__("Odometry Sensor")

        # Some inits
        self.z = np.array([0,0,0,1,0,0,0]).reshape(-1,1) # Measurement
        
        self.R = np.zeros((7,7))                   # Measurement Noise

        self.H = None   # Just for computation optimization

        self.qOld = None


    def measure(self, filter, *args, **kwargs):
        # Extracting v and w from method arguments
        v = kwargs["v"]
        w = kwargs["w"]
        #w += 0.009  # This is a hack to correct the wheel speed offset
        dt = kwargs["dt"]

        # Body Velocity (with kinematic holonomic constraints)
        vB = np.array([[v, 0, 0]]).T

        if self.qOld is None:
            self.qOld = qt.as_quat_array(filter.stateParts("Orientation").ravel())
            z = np.concatenate((vB, filter.stateParts("Orientation")))

            return z, 1000*self.R  # Kind of hack...
        
        # Rotation quaternion
        # (from previous orientation, angular velocity and dt)
        phi = np.array([0, 0, w * dt])
        dq = qt.from_rotation_vector(phi)

        
        q = qt.as_float_array(self.qOld * dq)

        z = np.concatenate((vB, q.reshape(-1,1)))  # [Body velocity; Orientation]

        # Save current estimated orientation for next iteration
        self.qOld = qt.as_quat_array(filter.stateParts("Orientation").ravel())

        return z, self.R
    
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
