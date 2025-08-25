#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstrução de trajetória LiDAR 2D:
  - Downsampling voxel
  - Extração de features (edges/flats) via curvatura
  - Normais locais via PCA (apenas em edges/flats)
  - ICP scan-to-scan (Huber, multi-escala)

Para correr: ajustar BAG_PATH e LIDAR_TOPIC, depois executar.
"""

import os
import rosbag
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ===== CONFIGURAÇÃO =====
BAG_PATH     = ''
CSV_OUTPUT   = ''
TXT_OUTPUT   = ''
LIDAR_TOPIC  = '/scan'
VOXEL_SIZE   = 0.015   # m
EDGE_MAX     = 200
FLAT_MAX     = 200
MAX_CORR     = 0.3    # m
CURV_WINDOW  = 5      # pontos de cada lado na curvatura

# ===== UTILITÁRIOS =====
def extract_xy(scan_msg) -> np.ndarray:
    """Extrai coordenadas XY do scan."""
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
    r = np.asarray(scan_msg.ranges)
    mask = np.isfinite(r)
    return np.vstack((r[mask]*np.cos(angles[mask]), r[mask]*np.sin(angles[mask]))).T

def voxel_downsample_xy(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Downsampling voxel para reduzir pontos redundantes."""
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
    Calcula normais locais por PCA (em 2D: normal = perpendicular ao 1º vector próprio).
    pts: pontos Nx2
    k: vizinhos para PCA
    return: normais Nx2, normalizada
    """
    if len(pts) < k + 1:
        return np.zeros_like(pts)
    tree = cKDTree(pts)
    normals = np.zeros_like(pts)
    for i, p in enumerate(pts):
        dists, idxs = tree.query(p, k=k+1)
        neighbors = pts[idxs[1:]]  # ignora o próprio ponto
        C = neighbors - neighbors.mean(axis=0)
        # PCA: vetor próprio do menor autovalor é a normal (2D)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        n = Vt[-1]
        n = np.array([n[1], -n[0]])  # perpendicular (em 2D)
        n /= np.linalg.norm(n) + 1e-12
        normals[i] = n
    return normals

class FeatureExtractor:
    def __init__(self):
        self.max_corr = MAX_CORR

    @staticmethod
    def rotmat(theta: float):
        c,s = np.cos(theta), np.sin(theta)
        return np.array([[c,-s],[s, c]])

    def extract_features_with_normals(self, pts: np.ndarray, k_norm=8):
        """
        Extrai edges, flats e respetivas normais via PCA.
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
        # Curvatura
        curv = np.zeros(N)
        for i in range(CURV_WINDOW, N-CURV_WINDOW):
            neigh = pts_s[i-CURV_WINDOW : i+CURV_WINDOW+1]
            curv[i] = np.linalg.norm(neigh.sum(0) - (2*CURV_WINDOW+1)*pts_s[i])
        interior = curv[CURV_WINDOW : N-CURV_WINDOW]
        h = np.percentile(interior, 90)
        l = np.percentile(interior, 10)

        edges_idx = np.where(curv > h)[0][:EDGE_MAX]
        flats_idx = np.where(curv <= l)[0][:FLAT_MAX]
        edges = pts_s[edges_idx]
        flats = pts_s[flats_idx]

        # Normais calculadas só para os points de interesse
        normals_all = compute_normals(pts_s, k=k_norm)
        edge_normals = normals_all[edges_idx]
        flat_normals = normals_all[flats_idx]
        return edges, flats, edge_normals, flat_normals

# ===== ICP MELHORADO COM NORMAIS =====
class RobustICP(FeatureExtractor):
    def scan2scan_icp(self, e_now, f_now, ne_now, nf_now,
                      e_prev, f_prev, ne_prev, nf_prev,
                      iters=4, scales=None, huber=0.1):
        if e_prev is None or len(e_prev)==0:
            return np.zeros(3), 0.0
        if scales is None:
            scales = [VOXEL_SIZE*4, VOXEL_SIZE*2, VOXEL_SIZE]
        total = len(e_prev) + len(f_prev) + 1e-6
        w_e = len(e_prev)/total
        w_f = len(f_prev)/total
        pose = np.zeros(3)
        Hfin = np.eye(3)
        for scale in scales:
            # Opcional: recalcular features/normais em pts escalados 
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
                # --- Edge: point-to-line usando normais
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
        raise FileNotFoundError(f"Rosbag não encontrado: {BAG_PATH}")
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

        # --- CORREÇÃO DE REFERENCIAL ---
        dx_rover =  dy      # "y" LiDAR = "x" rover
        dy_rover = -dx      # "x" LiDAR = "-y" rover

        ts = t.to_sec()
        n_edges = len(e)
        n_flats = len(f)
        lidar_poses.append([ts, dx_rover, dy_rover, dtheta, n_edges, n_flats])



        prev_e, prev_f, prev_ne, prev_nf = e, f, ne, nf

    bag.close()

        # Guardar CSV das poses LiDAR
    
    with open(CSV_OUTPUT, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "dx", "dy", "dyaw", "n_edges", "n_flats"])
        writer.writerows(lidar_poses)

        x, y, theta = 0.0, 0.0, 0.0
        
    with open(TXT_OUTPUT, "w") as f:
        x, y, theta = 0.0, 0.0, 0.0  # Inicializar pose
        for entry in lidar_poses:
            ts, dx, dy, dtheta, *_ = entry

            # Atualizar pose incrementalmente
            c, s = np.cos(theta), np.sin(theta)
            x += c * dx - s * dy
            y += s * dx + c * dy
            theta += dtheta

            # Converter theta (yaw) em quaternion (assume movimento no plano XY)
            qx = 0.0
            qy = 0.0
            qz = np.sin(theta / 2)
            qw = np.cos(theta / 2)

            # Escrever no formato TUM: timestamp x y z qx qy qz qw
            f.write(f"{ts:.6f} {x:.6f} {y:.6f} 0.000000 {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


if __name__ == '__main__':
    main()