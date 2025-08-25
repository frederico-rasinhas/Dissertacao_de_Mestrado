from pathlib import Path
import os, sys, rosbag, cv2, csv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# Configurações
BAG_PATH = ''
CSV_OUTPUT = ''
TXT_OUTPUT = Path("")

class VisualOdometry:
    def __init__(self, data_dir, cam_height=0.15, roi_fraction=0.4, ransac_iters=200, ransac_thresh=0.35):
        self.K = np.array([[340.125829, 0.0, 305.876356],
                           [0.0, 340.218367, 232.10263],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.D = np.array([-0.279817, 0.060321, 0.000487, 0.00031, 0.0], dtype=np.float64)
        self.P = np.hstack((self.K, np.zeros((3,1), dtype=np.float64)))
        self.cam_height = cam_height

        self.roi_fraction = roi_fraction
        self.ransac_iters = ransac_iters
        self.ransac_thresh = ransac_thresh

        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(31,31), maxLevel=5,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))

        self.images, self.timestamps = self._load_images_from_bag(data_dir)

        self.dx_list, self.dy_list, self.dz_list = [], [], []
        self.dtheta_list, self.ts_list = [], []
        self.inliers_pose_list = []
        self.inliers_plane_list = []
        self.escalas = []

    def _load_images_from_bag(self, bag_path, topic='/camera/image_raw/compressed'):
        bag = rosbag.Bag(bag_path, 'r')
        bridge = CvBridge()
        images, times = [], []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        first = True
        for _, msg, t in bag.read_messages(topics=[topic]):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if first:
                h, w = img.shape
                self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, (w, h), cv2.CV_32FC1)
                first = False
            img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
            img = clahe.apply(img)
            images.append(img)
            times.append(t.to_sec())
        bag.close()
        return images, np.array(times)

    def get_matches(self, i):
        prev, curr = self.images[i-1], self.images[i]
        h = prev.shape[0]

        mask_roi = np.zeros_like(prev, dtype=np.uint8)
        mask_roi[int((1.0 - self.roi_fraction) * h):, :] = 255
        p0_roi = cv2.goodFeaturesToTrack(prev, mask=mask_roi, **self.feature_params)
        q1_scale, q2_scale = self._track_features(prev, curr, p0_roi)

        p0_all = cv2.goodFeaturesToTrack(prev, mask=None, **self.feature_params)
        q1_all, q2_all = self._track_features(prev, curr, p0_all)

        return q1_scale, q2_scale, q1_all, q2_all

    def _track_features(self, img1, img2, points):
        if points is None:
            return np.empty((0,2)), np.empty((0,2))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, **self.lk_params)
        st = st.ravel().astype(bool)
        return points.reshape(-1,2)[st].astype(np.float64), p1.reshape(-1,2)[st].astype(np.float64)

    def estimate_scale_from_plane(self, R, t, q1, q2):
        P1 = self.P.copy()
        T = self._form_transf(R, t)
        P2 = self.K @ T[:3]
        pts4d = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        best_inliers, best_count = None, 0
        for _ in range(self.ransac_iters):
            idx = np.random.choice(len(pts3d), 3, replace=False)
            sample = pts3d[idx]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            if np.linalg.norm(normal) < 1e-6:
                continue
            normal /= np.linalg.norm(normal)
            d = -normal.dot(sample[0])
            inliers = np.abs(pts3d @ normal + d) < self.ransac_thresh
            if inliers.sum() > best_count:
                best_count, best_inliers = inliers.sum(), inliers

        pts_in = pts3d[best_inliers if best_inliers is not None else np.ones(len(pts3d), bool)]
        centroid = pts_in.mean(axis=0)
        _, _, vh = np.linalg.svd(pts_in - centroid)
        normal = vh[-1]; d = -normal.dot(centroid)
        dist_to_plane = abs(d)
        return self.cam_height / dist_to_plane if dist_to_plane > 1e-3 else 1.0, best_count

    def get_pose(self, i):
        q1_scale, q2_scale, q1, q2 = self.get_matches(i)
        if len(q1) < 6:
            return np.eye(4)

        E, maskE = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return np.eye(4)

        q1_in, q2_in = q1[maskE.ravel().astype(bool)], q2[maskE.ravel().astype(bool)]
        _, R, t, maskR = cv2.recoverPose(E, q1_in, q2_in, self.K)
        if np.count_nonzero(maskR) < 5:
            return np.eye(4)

        scale, inliers_plane = self.estimate_scale_from_plane(R, t.flatten(), q1_scale, q2_scale)
        self.escalas.append(scale)
        T_cam = self._form_transf(R, t.flatten() * scale)

        T_conv = np.eye(4)
        T_conv[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
        T_body = T_conv @ T_cam @ np.linalg.inv(T_conv)

        t_body, R_body = T_body[:3, 3], T_body[:3, :3]
        dx, dy, dz = t_body
        dtheta = np.arctan2(R_body[1, 0], R_body[0, 0])

        self.dx_list.append(dx)
        self.dy_list.append(dy)
        self.dz_list.append(dz)
        self.dtheta_list.append(dtheta)
        self.ts_list.append(self.timestamps[i])

        self.inliers_pose_list.append(int(np.count_nonzero(maskR)))
        self.inliers_plane_list.append(int(inliers_plane))

        return T_cam

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4); T[:3,:3], T[:3,3] = R, t
        return T

def main():
    if not os.path.isfile(BAG_PATH):
        raise FileNotFoundError(f"Rosbag não encontrado: {BAG_PATH}")
    vo = VisualOdometry(BAG_PATH)

    cur_pose = np.eye(4)
    estimated_path = []

    for i in tqdm(range(1, len(vo.images)), desc="Estimando poses"):
        T = vo.get_pose(i)
        T_conv = np.eye(4)
        T_conv[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
        cur_pose = cur_pose @ (T_conv @ T @ np.linalg.inv(T_conv))
        estimated_path.append(cur_pose[:3, 3])

    with open(CSV_OUTPUT, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'dx', 'dy', 'dz', 'dyaw', 'inliers_pose', 'inliers_plane'])
        for ts, dx, dy, dz, dth, n_pose, n_plane in zip(vo.ts_list, vo.dx_list, vo.dy_list, vo.dz_list, vo.dtheta_list,
                                                        vo.inliers_pose_list, vo.inliers_plane_list):
            writer.writerow([ts, dx, dy, dz, dth, n_pose, n_plane])

    if estimated_path:
        xs, ys, zs = zip(*estimated_path)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=range(len(xs)), cmap='plasma', s=15)

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.set_title('Trajetória estimada em 3D')

        max_range = np.array([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]).max() / 2.0
        mid_x = (max(xs) + min(xs)) * 0.5
        mid_y = (max(ys) + min(ys)) * 0.5
        mid_z = (max(zs) + min(zs)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.colorbar(sc, ax=ax, label='Índice temporal (frames)')
        plt.tight_layout()
        plt.show()
    else:
        print("[AVISO] Nenhuma pose estimada com sucesso.")

    print("\nIncrementos por frame (dx, dy, dz, dtheta):")
    for ts, dx, dy, dz, dth in zip(vo.ts_list, vo.dx_list, vo.dy_list, vo.dz_list, vo.dtheta_list):
        print(f"t={ts:.3f} | dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, dψ={np.rad2deg(dth):.2f}°")

    if vo.escalas:
        media_escala = np.mean(vo.escalas)
        print(f"\n[INFO] Escala relativa média estimada: {media_escala:.3f}")
    else:
        print("\n[INFO] Nenhuma escala estimada.")

    with open(TXT_OUTPUT, 'w') as f:
        for pose, ts in zip(estimated_path, vo.ts_list):
            px, py, pz = pose
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            f.write(f"{ts:.6f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"[TUM] Trajetória ESKF exportada para: {TXT_OUTPUT}")

if __name__ == "__main__":
    main()
