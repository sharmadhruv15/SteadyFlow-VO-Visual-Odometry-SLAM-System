import numpy as np
import cv2

class ScaleEstimator:
    def __init__(self, camera_height=1.65):
        self.h = camera_height  
        self.scale_buffer = []  

    def estimate_scale(self, pts_prev, pts_curr, K, R, t):
        P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(K, np.hstack((R, t.reshape(3,1))))
        pts4D = cv2.triangulatePoints(P1, P2, pts_prev.T, pts_curr.T)
        pts3D = pts4D[:3, :] / (pts4D[3, :] + 1e-6)
        
        y_coords = pts3D[1, :]
        z_coords = pts3D[2, :]
        
        mask = (z_coords > 3) & (z_coords < 50) & (y_coords > 1.0)
        valid_y = y_coords[mask]
        
        if len(valid_y) < 3:
            return self.scale_buffer[-1] if self.scale_buffer else 1.0
            
        raw_scale = abs(self.h / np.median(valid_y))
        raw_scale = np.clip(raw_scale, 0.1, 4.0) 
        
        self.scale_buffer.append(raw_scale)
        if len(self.scale_buffer) > 10: self.scale_buffer.pop(0)
        return np.mean(self.scale_buffer)