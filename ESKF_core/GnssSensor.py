import numpy as np
from pymap3d import geodetic2enu
from eskf import *

class GNSS_Sensor(ErrorStateSensorModel):
    def __init__(self):
        super().__init__("GNSS Sensor")
        self.R = np.zeros((3, 3))  
        self.H = None
        self._ref = None  # lat0, lon0, alt0 em metros

    def measure(self, filter, *args, **kwargs):
        """
        Recebe argumentos:
        - lat, lon, alt → posição em coordenadas geodésicas
        - cov → vetor de variâncias [σ²x, σ²y, σ²z]
        """
        lat = kwargs['lat']
        lon = kwargs['lon']
        alt_ft = kwargs['alt']     # Altitude vem em pés
        cov = kwargs['cov']

        # Converte altitude para metros
        alt = alt_ft * 0.3048

        # Inicializa a referência (lat0, lon0, alt0)
        if self._ref is None:
            self._ref = (lat, lon, alt)
        lat0, lon0, alt0 = self._ref

        # Conversão geodésica → ENU
        x_enu, y_enu, z_enu = geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        z_enu = np.array([x_enu, y_enu, z_enu]).reshape(3, 1)

        
        # === MATRIZ DE ROTAÇÃO ESTIMADA ENTRE REFERENCIAIS ===
        R_2D = np.array([
            [ 0.696657,  0.717405],
            [-0.717405,  0.696657]
        ])


        #Aplica rotação inversa (ENU → rover base_link)
        R_3D = np.eye(3)
        R_3D[0:2, 0:2] = R_2D.T  # transposta para inverter

        z_rover = R_3D @ z_enu
        
        #self.R = np.diag(cov)  # se quiseres usar covariância real

        return z_rover, self.R

    def estimate(self, filter):
        if self.H is None:
            self.H = np.zeros((3, filter.nErrStates))
            self.H[:, filter.errStateIdxs("dPos")] = np.eye(3)

        estZ = filter.stateParts("Position").reshape(3, 1)
        return estZ, self.H
