import rosbag
import os
import numpy as np

class DataLoader:

    def __init__(self, bag_path, lidar_csv_path=None, vo_csv_path=None):
        self.bag_path = bag_path
        self.lidar_csv_path = lidar_csv_path
        self.vo_csv_path = vo_csv_path

        # Automatic detection of topics inside the bag
        bag = rosbag.Bag(str(self.bag_path), 'r')
        info = bag.get_type_and_topic_info().topics
        self.topic_imu   = None
        self.topic_gps   = None
        self.topic_wheel = None

        for topic, topic_info in info.items():
            if topic_info.msg_type == 'sensor_msgs/Imu':
                self.topic_imu = topic
            elif topic_info.msg_type == 'sensor_msgs/NavSatFix':
                self.topic_gps = topic
            elif topic_info.msg_type == 'leo_msgs/WheelOdom':
                self.topic_wheel = topic
        bag.close()

        # Verify detected topics
        if self.topic_imu is None:
            raise RuntimeError('IMU not found')
        if self.topic_gps is None:
            print('Warning: NavSatFix topic not found. GNSS measurements will be skipped.')
        if self.topic_wheel is None:
            print('Warning: WheelOdom topic not found. Wheel odometry measurements will be skipped.')

        self._records = []  # Will store chronological sensor records

    def load_imu_data(self):
        """Load IMU data: timestamp, accelerometer (x,y,z), gyroscope (x,y,z)."""
        imu_data = []
        with rosbag.Bag(str(self.bag_path), 'r') as bag:
            for _, msg, _ in bag.read_messages(topics=[self.topic_imu]):
                ts_imu = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                acc  = msg.linear_acceleration
                gyro = msg.angular_velocity
                imu_data.append([ts_imu, acc.x, acc.y, acc.z,
                                          gyro.x, gyro.y, gyro.z])
        return np.array(imu_data)

    def load_gps_data(self):
        """Load GNSS data: timestamp, latitude, longitude, altitude, covariance."""
        if self.topic_gps is None:
            return np.empty((0, 7))  # timestamp, lat, lon, alt, cov_x, cov_y, cov_z

        gps_data = []
        min_var = 1e-6  # Minimum variance (avoid zeros)

        with rosbag.Bag(str(self.bag_path), 'r') as bag:
            for _, msg, _ in bag.read_messages(topics=[self.topic_gps]):
                ts_gps = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                lat = msg.latitude
                lon = msg.longitude
                alt = msg.altitude * 0.3048  # Convert altitude from feet to meters

                # Use provided covariance if available, otherwise assign high uncertainty
                if msg.position_covariance_type == type(msg).COVARIANCE_TYPE_DIAGONAL_KNOWN:
                    cov_x = max(msg.position_covariance[0], min_var)
                    cov_y = max(msg.position_covariance[4], min_var)
                    cov_z = max(msg.position_covariance[8], min_var)
                else:
                    cov_x, cov_y, cov_z = 100.0, 100.0, 100.0

                gps_data.append([ts_gps, lat, lon, alt, cov_x, cov_y, cov_z])

        return np.array(gps_data)

    def load_wheel_odom(self):
        """Load wheel odometry: timestamp, linear velocity, angular velocity."""
        if self.topic_wheel is None:
            return np.empty((0,3))
        wheel = []
        with rosbag.Bag(str(self.bag_path), 'r') as bag:
            for _, msg, t in bag.read_messages(topics=[self.topic_wheel]):
                ts_wheel = t.to_sec()
                v = msg.velocity_lin
                w = msg.velocity_ang
                wheel.append([ts_wheel, v, w])
        return np.array(wheel)

    def load_lidar_pose(self):
        """Load LiDAR poses (from CSV): timestamp, dx, dy, dyaw, n_edges, n_flats."""
        if not hasattr(self, 'lidar_csv_path') or self.lidar_csv_path is None:
            return np.empty((0, 6))  
        path = self.lidar_csv_path
        if not os.path.isfile(path):
            return np.empty((0, 6))
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        if data.ndim == 1 and data.shape[0] == 6:
            data = data.reshape(1, 6)
        return data
    
    def load_vo_pose(self):
        """Load Visual Odometry poses (from CSV): timestamp, dx, dy, dz, dyaw, inliers_pose, inliers_plane."""
        if not hasattr(self, 'vo_csv_path') or self.vo_csv_path is None:
            return np.empty((0, 7))
        path = self.vo_csv_path
        if not os.path.isfile(path):
            return np.empty((0, 7))
        data = np.genfromtxt(path, delimiter=',', skip_header=1)
        if data.ndim == 1 and data.shape[0] == 7:
            data = data.reshape(1, 7)
        return data

    def _build_records(self):
        """Build chronological list of all sensor measurements (IMU, GNSS, wheel, LiDAR, VO)."""
        self._records = []
        # IMU
        for row in self.load_imu_data():
            self._records.append((row[0], 'imu', {'acc': row[1:4], 'gyro': row[4:7]}))
        # GNSS
        for row in self.load_gps_data():
            self._records.append((row[0], 'gnss', {'pos': row[1:4],'cov': row[4:7]}))
        # Wheel odometry
        for row in self.load_wheel_odom():
            self._records.append((row[0], 'wheel', {'v': row[1], 'w': row[2]}))
        # LiDAR poses
        for row in self.load_lidar_pose():
            self._records.append((row[0], 'lidar', {'dx': row[1], 'dy': row[2], 'dyaw': row[3], "n_edges": row[4], "n_flats": row[5]}))
        # VO poses
        for row in self.load_vo_pose():
            self._records.append((row[0], 'vo', {'dx': row[1], 'dy': row[2], 'dz': row[3], 'dyaw': row[4], 'inliers_pose': row[5], 'inliers_plane': row[6]}))

        # Sort chronologically by timestamp
        self._records.sort(key=lambda x: x[0])

    def __iter__(self):
        """Iterate through all sensor records in chronological order."""
        if not self._records:
            self._build_records()
        return iter(self._records)
