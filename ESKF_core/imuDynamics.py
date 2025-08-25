from eskf import *
import quaternion as qt

# ----------- class IMU (Motion Model) -----------
class IMU_MotionModel(ErrorStateDynamicalModel):
    def __init__(self):
        super().__init__("IMU Motion Model")

        self.__initial_state = {
            "Position": [0, 0, 0],
            "Velocity": [0, 0, 0],
            "Orientation": [1, 0, 0, 0],
            "AccBias": [0, 0, 0],
            "GyroBias": [0, 0, 0] }
        self.__initial_error_state_cov = {
            "dPos": [0, 0, 0],
            "dVel": [0, 0, 0],
            "dTheta": [0, 0, 0],
            "dAccBias": [0, 0, 0],
            "dGyroBias": [0, 0, 0] }

        # Acc and Gyro noise
        self.accNoiseVar = 0.0
        self.gyroNoiseVar = 0.0
        # Acc and Gyro Bias drift
        self.accBiasVar = 0.0
        self.gyroBiasVar = 0.0

        # Gravity
        self.g = np.array([0, 0, -9.80665]).reshape(-1,1)

        self.FirstTime = True   # Just for computation optimization

    # Getters
    @property
    def states(self): return self.__initial_state
    @property
    def errorStates(self): return self.__initial_error_state_cov

    def stateTransition(self, filter: ErrorStateKalmanFilter, dt: float, *args, **kwargs):
        # Measurements:
        acc_m = kwargs["acc"].reshape(-1,1)
        gyro_m = kwargs["gyro"].reshape(-1,1)

        # Get Nominal State
        p = filter.stateParts("Position")
        v = filter.stateParts("Velocity")
        q = qt.as_quat_array(filter.stateParts("Orientation").ravel())
        aB = filter.stateParts("AccBias")
        wB = filter.stateParts("GyroBias")

        # Some convenience variables
        R = qt.as_rotation_matrix(q)
        acc = acc_m - aB
        a_t = R @ acc + self.g
        acc = acc.ravel()*(-dt)
        acc_skew = np.array([[0.0, -acc[2], acc[1]], [acc[2], 0.0, -acc[0]], [-acc[1], acc[0], 0.0]])
        qRot = qt.from_rotation_vector((gyro_m - wB).ravel() * dt)
        R_w = qt.as_rotation_matrix(qRot)

        ## ------ Nominal State Kinematics ------
        # TODO: should return this stuff, but it is faster this way...
        filter.nextX[filter.stateIdxs("Position")] = p + v * dt #+ (0.5*dt**2) * a_t    # eq. 260a
        filter.nextX[filter.stateIdxs("Velocity")] = v + a_t * dt                       # eq. 260b
        filter.nextX[filter.stateIdxs("Orientation"),0] = qt.as_float_array(q * qRot)   # eq. 260c
        filter.nextX[filter.stateIdxs("AccBias")] = aB                                  # eq. 260d
        filter.nextX[filter.stateIdxs("GyroBias")] = wB                                 # eq. 260e

        ## ------ Error State Jacobian ------
        # TODO: Jacobian should be returned by the method, but it is way faster this way...
        if self.FirstTime:
            # Init some constant Jacobian Matrix blocs to speed up things, assuming it is a zero matrix
            # eq. 270
            np.fill_diagonal(filter.F[filter.errStateIdxs("dPos"),filter.errStateIdxs("dPos")], 1.0)
            np.fill_diagonal(filter.F[filter.errStateIdxs("dVel"),filter.errStateIdxs("dVel")], 1.0)
            np.fill_diagonal(filter.F[filter.errStateIdxs("dAccBias"),filter.errStateIdxs("dAccBias")], 1.0)
            np.fill_diagonal(filter.F[filter.errStateIdxs("dGyroBias"),filter.errStateIdxs("dGyroBias")], 1.0)
            self.FirstTime = False
        
        # eq. 270 (filling up the blocks):
        np.fill_diagonal(filter.F[filter.errStateIdxs("dPos"),filter.errStateIdxs("dVel")], dt)
        np.fill_diagonal(filter.F[filter.errStateIdxs("dTheta"),filter.errStateIdxs("dGyroBias")], -dt)
        filter.F[filter.errStateIdxs("dVel"),filter.errStateIdxs("dTheta")] = R @ acc_skew
        filter.F[filter.errStateIdxs("dVel"),filter.errStateIdxs("dAccBias")] = R*(-dt)
        filter.F[filter.errStateIdxs("dTheta"),filter.errStateIdxs("dTheta")] = R_w.T

        ## ------ Transition Noise ------
        # TODO: Transition Noise should be returned by the method, but it is way faster this way...
        # eq. 269 and eq. 271
        # This noise matrix is defined Q = F_i * Q_i * F_i^T
        np.fill_diagonal(filter.Q[filter.errStateIdxs("dVel"),filter.errStateIdxs("dVel")],           self.accNoiseVar * dt**2)
        np.fill_diagonal(filter.Q[filter.errStateIdxs("dTheta"),filter.errStateIdxs("dTheta")],       self.gyroNoiseVar * dt**2)
        np.fill_diagonal(filter.Q[filter.errStateIdxs("dAccBias"),filter.errStateIdxs("dAccBias")],   self.accBiasVar * dt)
        np.fill_diagonal(filter.Q[filter.errStateIdxs("dGyroBias"),filter.errStateIdxs("dGyroBias")], self.gyroBiasVar * dt)

    def injectError(self, filter: ErrorStateKalmanFilter, dX):
        filter.x[filter.stateIdxs("Position")] += dX[filter.errStateIdxs("dPos")]                                  # eq. 283a
        filter.x[filter.stateIdxs("Velocity")] += dX[filter.errStateIdxs("dVel")]                                  # eq. 283b
        # q = filter.x[filter.stateIdxs("Orientation")]
        # dq = qt.as_float_array(qt.from_rotation_vector(dX[filter.errStateIdxs("dTheta")].ravel())).reshape(-1,1)
        q = qt.as_quat_array(filter.stateParts("Orientation").ravel())
        dq = qt.from_rotation_vector(dX[filter.errStateIdxs("dTheta")].ravel())
        filter.x[filter.stateIdxs("Orientation")] = qt.as_float_array(q * dq).reshape(-1,1)                                                        # eq. 283c
        filter.x[filter.stateIdxs("AccBias")] += dX[filter.errStateIdxs("dAccBias")]                               # eq. 283d
        filter.x[filter.stateIdxs("GyroBias")] += dX[filter.errStateIdxs("dGyroBias")]                             # eq. 283e
