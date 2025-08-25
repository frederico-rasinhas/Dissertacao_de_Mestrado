import numpy as np

# Virtual class DynamicalModel
class DynamicalModel:
    def __init__(self,name: str):
        self.__name = name
    def name(self):
        return self.__name
    # Returns a dictionary with state names and corresponding initial values
    @property
    def states(self): raise NotImplementedError("You must implement this abstract method!")
    # Performs state transition: next state, state Jacobian, transition noise
    def stateTransition(self, filter, dt, *args, **kwargs): raise NotImplementedError("You must implement this abstract method!")

# Virtual class ErrorStateDynamicalModel
class ErrorStateDynamicalModel(DynamicalModel):
    # Returns a dictionary with error state names and corresponding initial covariance values
    @property
    def errorStates(self): raise NotImplementedError("You must implement this abstract method!")
    # Performs state transition: next state, error state Jacobian, transition noise
    def stateTransition(self, filter, dt, *args, **kwargs): raise NotImplementedError("You must implement this abstract method!")
    # injects observed error state into nominal state
    def injectError(self, filter, dX): raise NotImplementedError("You must implement this abstract method!")

# Virtual class SensorModel
class SensorModel(DynamicalModel):
    @property
    def states(self): return {}                                     # Overload this if you want to use sensor states
    def stateTransition(self, filter, dt, *args, **kwargs): pass    # Overload this if you want to use sensor states
    # MEASUREMENT MODEL:
    def measure(self, filter, *args, **kwargs): raise NotImplementedError("You must implement this abstract method!")
    # This function should return z, R, the measurement and corresponding noise covariance matrix
    def estimate(self, filter): raise NotImplementedError("You must implement this abstract method!")
    # This function should return estZ, H, the expected measurement according to the state and corresponding Jacobian

# Virtual class ErrorStateSensorModel
class ErrorStateSensorModel(SensorModel,ErrorStateDynamicalModel):
    @property
    def errorStates(self): return {}                                # Overload this if you want to use sensor states
    def injectError(self, filter, dX): pass                         # Overload this if you want to use sensor states

class KalmanFilter:
    pass

class ErrorStateKalmanFilter(KalmanFilter):
    def __init__(self, motionModel: ErrorStateDynamicalModel, *args: ErrorStateSensorModel):
        self.__filterStates = {}
        self.__nStates = 0
        self.__filterErrStates = {}
        self.__nErrStates = 0

        # Registering Motion Model:
        self.__motionModel = motionModel
        self.__registerStates( motionModel )
        print("Starting ESKF with motion model:\n\t", motionModel.name(), sep="")

        # Registering Sensor Models:
        print("and the following sensors:")
        self.__sensorModels = {}
        for arg in args:
            self.__sensorModels[arg.name()] = arg
            self.__registerStates( arg )
            print("\t",arg.name())
        print("---------"*4)
        # Print State Info
        self.printStateIdxs()

        # Initializing nominal state and error state covariance
        self.x = np.empty((self.__nStates, 1))
        self.P = np.empty((self.__nErrStates,self.__nErrStates))
        self.reset()
        print("\nInitial Nominal State:\n\t",self.x.ravel(),sep="")
        print("Initial Error State Covariance:\n\t",np.diag(self.P),sep="")

        # Initializing next state and Error State Jacobian
        self.nextX = self.x.copy()
        self.F = np.zeros((self.__nErrStates,self.__nErrStates))
        self.Q = np.zeros((self.__nErrStates,self.__nErrStates))

        # Aux variables
        self.eyeN = np.eye(self.__nErrStates)
    
    # Getters
    @property
    def nStates(self): return self.__nStates
    @property
    def nErrStates(self): return self.__nErrStates

    def stateIdxs(self, stateName: str = None):
        if stateName is None: return self.__filterStates
        return self.__filterStates[stateName]
    
    def errStateIdxs(self, stateName: str = None):
        if stateName is None: return self.__filterErrStates
        return self.__filterErrStates[stateName]

    def printStateIdxs(self):
        print("Nominal State:")
        max_len = len(max(self.__filterStates.keys(), key=len))
        for stateName, idx in self.__filterStates.items():
            print(f"\t{stateName:<{max_len+1}}: {np.array(range(idx.start,idx.stop))}")
        print("Error State:")
        max_len = len(max(self.__filterErrStates.keys(), key=len))
        for stateName, idx in self.__filterErrStates.items():
            print(f"\t{stateName:<{max_len+1}}: {np.array(range(idx.start,idx.stop))}")

    def stateParts(self, stateName: str = None, val: np.ndarray = None):
        if val is None: # Getter
            if stateName is None: return self.x
            return self.x[self.__filterStates[stateName]]
        else:           # Setter
            self.x[self.__filterStates[stateName],0] = val

    def errStateCovParts(self, stateName: str = None, val: np.ndarray = None):
        if val is None: # Getter
            if stateName is None: return self.P
            idx = self.__filterErrStates[stateName]
            return self.P[idx,idx]
        else:           # Setter
            idx = self.__filterErrStates[stateName]
            self.P[idx,idx] = np.diag(val)
    
    def reset(self):
        self.x.fill(0.0)
        self.P.fill(0.0)
        self.__resetStates(self.__motionModel)
        for sensor in self.__sensorModels.values():
            self.__resetStates(sensor)
    
    def predict(self, dt: float, *args, **kwargs):
        # State Transition
        self.__motionModel.stateTransition(self, dt, *args, **kwargs)
        for sensor in self.__sensorModels.values():
            sensor.stateTransition(self, dt, *args, **kwargs)
        self.x = self.nextX.copy()
        
        self.P = self.F @ self.P @ self.F.T + self.Q     # eq. 269

    def update(self, sensorName, *args, **kwargs):
        sensor = self.__sensorModels[sensorName]
        z, R = sensor.measure(self, *args, **kwargs)
        estZ, H = sensor.estimate(self)
        
        aux = self.P @ H.T
        K = aux @ np.linalg.inv(H @ aux + R)        # eq. 274
        # TODO: alternative calculation with K = a/b
        dX = K @ (z - estZ)                         # eq. 275

        # print((z - estZ).T, dX[0:3].T)
        # print(K.T)

        aux = self.eyeN - K @ H
        # self.P = aux @ self.P     # eq. 276
        self.P = aux @ self.P @ aux.T + K @ R @ K.T # eq. 276, Joseph form

        # Inject error into nominal state:
        self.__motionModel.injectError(self, dX)
        for sensor in self.__sensorModels.values():
            sensor.injectError(self, dX)
        
        # Reset Jacobian and Error State Covariance Update
        # TODO eq. 286

    # ------ PRIVATE METHODS -------
    def __registerStates(self, model):
        for stateName, init_val in model.states.items():
            if stateName in self.__filterStates:
                raise Exception(stateName, " key already exists!")
            d = len(init_val)
            self.__filterStates[stateName] = slice(self.__nStates,self.__nStates+d)
            self.__nStates += d
        for stateName, init_val in model.errorStates.items():
            if stateName in self.__filterErrStates:
                raise Exception(stateName, " key already exists!")
            d = len(init_val)
            self.__filterErrStates[stateName] = slice(self.__nErrStates,self.__nErrStates+d)
            self.__nErrStates += d
    
    def __resetStates(self, model):
        for stateName, init_val in model.states.items():
            self.stateParts(stateName,init_val)
        for stateName, init_val in model.errorStates.items():
            self.errStateCovParts(stateName,init_val)





