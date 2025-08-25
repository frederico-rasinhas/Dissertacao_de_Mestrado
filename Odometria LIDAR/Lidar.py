#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR trajectory reconstruction:
  - Voxel downsampling
  - Feature extraction (edges/flats) via curvature
  - Local normals via PCA (only on edges/flats)
  - Scan-to-scan ICP (Huber loss, multi-scale)

How to run: set BAG_PATH and LIDAR_TOPIC, then execute.
"""

import os
import rosbag
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ===== CONFIGURATION =====
BAG_PATH     = ''       # Path to input rosbag file
CSV_OUTPUT   = ''       # Path to save CSV file with LiDAR increments
TXT_OUTPUT   = ''       # Path to save trajectory in TUM format
LIDAR_TOPIC  = '/scan'  # LiDAR topic name in rosbag
VOXEL_SIZE   = 0.015    # Voxel size for downsampling [m]
EDGE_MAX     = 200      # Max number of edge features per scan
FLAT_MAX     = 200      # Max number of flat features per scan
MAX_CORR     = 0.3      # Maximum correspondence distance [m]
CURV_WINDOW  = 5        # Number of neighbors for curvature computation

# ===== UTILITIES =====
def extract_xy(scan_msg) -> np.ndarray:
    """Extract XY coordinates from a 2D LiDAR scan message."""
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
    r = np.asarray(scan_msg.ranges)
    mask = np.isfinite(r)  # Remove NaN/inf ranges
    return np.vstack((r[mask]*np.cos(angles[mask]), r[mask]*np.sin(angles[mask]))).T

def voxel_downsample_xy(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Apply voxel downsampling to reduce redundant points."""
    if pts.size == 0:
        return pts
    keys = np.floor(pts/voxel).astype(int)
    buckets = {}
    for p,k in zip(pts,keys):
        key = (k[0],k[1])
        buckets.setdefault(key, []).append(p)
    return np.array([np.mean(plist,axis=0) for plist in buckets.values()])

def compute_normals(pts: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Compute local normals using PCA.
    In 2D: normal = perpendicular to first principal component.
    pts: Nx2 point cloud
    k: number of neighbors for PCA
    return: Nx2 array of unit normals
    """
    if len(pts) < k + 1:
        return np.zeros_like(pts)
    tree = cKDTree(pts)
    normals = np.zeros_like(pts)
    for i, p in enumerate(pts):
        dists, idxs = tree.query(p, k=k+1)
        neighbors = pts[idxs[1:]]  # exclude the point itself
        C = neighbors - neighbors.mean(axis=0)
        # PCA: eigenvector of smallest eigenvalue is the normal
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        n = Vt[-1]
        n = np.array([n[1], -n[0]])  # perpendicular (2D normal)
        n /= np.linalg.norm(n) + 1e-12
        normals[i] = n
    return normals

class FeatureExtractor:
    def __init__(self):
        self.max_corr = MAX_CORR

    @staticmethod
    def rotmat(theta: float):
        """Return a 2D rotation matrix."""
        c,s = np.cos(theta), np.sin(theta)
        return np.array([[c,-s],[s, c]])

    def extract_features_with_normals(self, pts: np.ndarray, k_norm=8):
        """
        Extract edges and flats, along with their normals using PCA.
        """
        if pts.size == 0:
            return (np.empty((0,2)), np.empty((0,2)),
                    np.empty((0,2)), np.empty((0,2)))
        ang = np.arctan2(pts[:,1], pts[:,0])
        order = np.argsort(ang)
        pts_s = pts[order]

        N = len(pts_s)
        if N < 2*CURV_WINDOW+1:
            return (np.empty((0,2)), np.empty((0,2)),
                    np.empty((0,2)), np.empty((0,2)))
        # Curvature computation
        curv = np.zeros(N)
        for i in range(CURV_WINDOW, N-CURV_WINDOW):
            neigh = pts_s[i-CURV_WINDOW : i+CURV_WINDOW+1]
            curv[i] = np.linalg.norm(neigh.sum(0) - (2*CURV_WINDOW+1)*pts_s[i])
        interior = curv[CURV_WINDOW : N-CURV_WINDOW]
        h = np.percentile(interior, 90)  # High curvature threshold
        l = np.percentile(interior, 10)  # Low curvature threshold

        edges_idx = np.where(curv > h)[0][:EDGE_MAX]
        flats_idx = np.where(curv <= l)[0][:FLAT_MAX]
        edges = pts_s[edges_idx]
        flats = pts_s[flats_idx]

        # Compute normals only for selected points
        normals_all = compute_normals(pts_s, k=k_norm)
        edge_normals = normals_all[edges_idx]
        flat_normals = normals_all[flats_idx]
        return edges, flats, edge_normals, flat_normals

# ===== ICP WITH NORMALS =====
class RobustICP(FeatureExtractor):
    def scan2scan_icp(self, e_now, f_now, ne_now, nf_now,
                      e_prev, f_prev, ne_prev, nf_prev,
                      iters=4, scales=None, huber=0.1):
        """
        Multi-scale robust ICP between consecutive scans,
        using edge (point-to-line) and flat (point-to-point) features.
        """
        if e_prev is None or len(e_prev)==0:
            return np.zeros(3), 0.0
        if scales is None:
            scales = [VOXEL_SIZE*4, VOXEL_SIZE*2, VOXEL_SIZE]
        total = len(e_prev) + len(f_prev) + 1e-6
        w_e = len(e_prev)/total
        w_f = len(f_prev)/total
        pose = np.zeros(3)  # [dx, dy, dtheta]
        Hfin = np.eye(3)
        for scale in scales:
            if e_now.size==0 and f_now.size==0:
                continue
            kE, kF = cKDTree(e_prev), cKDTree(f_prev)
            for _ in range(iters):
                c,s = np.cos(pose[2]), np.sin(pose[2])
                Rth = np.array([[c,-s],[s,c]])
                t2 = pose[:2]
                Te = (Rth@e_now.T).T + t2
                Tf = (Rth@f_now.T).T + t2
                H = np.zeros((3,3)); b = np.zeros(3)
                # --- Edge: point-to-line using normals
                for idx, p in enumerate(Te):
                    d, j = kE.query(p, k=1)
                    if d > self.max_corr: continue
                    p_ref = e_prev[j]
                    n_ref = ne_prev[j]
                    r = n_ref.dot(p - p_ref)
                    w = (1 if abs(r)<=huber else huber/abs(r)) * w_e
                    J = n_ref @ np.array([[1,0,-p[1]], [0,1,p[0]]])
                    H += w * np.outer(J, J)
                    b += w * J * r
                # --- Flat: point-to-point
                for idx, p in enumerate(Tf):
                    d, j = kF.query(p, k=1)
                    if d > self.max_corr: continue
                    p_ref = f_prev[j]
                    r = p - p_ref
                    w = (1 if np.linalg.norm(r)<=huber else huber/np.linalg.norm(r)) * w_f
                    J = np.array([[1,0,-p[1]], [0,1,p[0]]])
                    H += w * (J.T @ J)
                    b += w * (J.T @ r)
                if np.linalg.det(H) < 1e-6:
                    break
                dp = -np.linalg.solve(H, b)
                pose += dp
                if np.linalg.norm(dp) < 1e-4:
                    break
            Hfin = H
        try:
            cov = np.linalg.inv(Hfin)
            var = float(cov[0,0] + cov[1,1])
        except np.linalg.LinAlgError:
            var = 0.0
        return pose, var

# ===== MAIN =====
def main():
    if not os.path.isfile(BAG_PATH):
        raise FileNotFoundError(f"Rosbag not found: {BAG_PATH}")
    bag   = rosbag.Bag(BAG_PATH, 'r')
    scans = bag.read_messages([LIDAR_TOPIC])

    loam = RobustICP()
    prev_e = prev_f = prev_ne = prev_nf = None

    lidar_poses = []  # [timestamp, dx, dy, dyaw, n_edges, n_flats]

    for i, (_, msg, t) in enumerate(scans):
        pts    = extract_xy(msg)
        pts_ds = voxel_downsample_xy(pts, VOXEL_SIZE)
        if pts_ds.size == 0:
            continue
        e, f, ne, nf = loam.extract_features_with_normals(pts_ds)
        if prev_e is None or len(prev_e)==0:
            prev_e, prev_f, prev_ne, prev_nf = e, f, ne, nf
            continue

        inc, _ = loam.scan2scan_icp(e, f, ne, nf, prev_e, prev_f, prev_ne, prev_nf)
        
        print(f"[{i:04d}] Edges: {len(e):3d}, Flats: {len(f):3d}")

        dx, dy, dtheta = inc

        # --- FRAME TRANSFORMATION ---
        dx_rover =  dy      # "y" in LiDAR frame = "x" in rover frame
        dy_rover = -dx      # "x" in LiDAR frame = "-y" in rover frame

        ts = t.to_sec()
        n_edges = len(e)
        n_flats = len(f)
        lidar_poses.append([ts, dx_rover, dy_rover, dtheta, n_edges, n_flats])

        prev_e, prev_f, prev_ne, prev_nf = e, f, ne, nf

    bag.close()

    # Save CSV with LiDAR pose increments
    with open(CSV_OUTPUT, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "dx", "dy", "dyaw", "n_edges", "n_flats"])
        writer.writerows(lidar_poses)

    # Export trajectory in TUM format
    with open(TXT_OUTPUT, "w") as f:
        x, y, theta = 0.0, 0.0, 0.0  # Initialize global pose
        for entry in lidar_poses:
            ts, dx, dy, dtheta, *_ = entry

            # Incremental pose update
            c, s = np.cos(theta), np.sin(theta)
            x += c * dx - s * dy
            y += s * dx + c * dy
            theta += dtheta

            # Convert yaw (theta) to quaternion (assume motion in XY plane)
            qx = 0.0
            qy = 0.0
            qz = np.sin(theta / 2)
            qw = np.cos(theta / 2)

            # Write in TUM format: timestamp x y z qx qy qz qw
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} 0.000000 {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

if __name__ == '__main__':
    main()
