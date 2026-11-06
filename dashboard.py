import numpy as np
import cv2

class DashboardVisualizer:
    def __init__(self):
        # Increased map canvas size to be consistent
        self.map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        self.dashboard = None

    def update(self, frame, est_t, gt_pose, frame_id, speed, rmse, distance, pts3D=None):
        # 1. Update Map (Bird's Eye View)
        ex, ez = int(est_t[0] * 0.5) + 300, 500 - int(est_t[2] * 0.5)
        gx, gz = int(gt_pose[0, 3] * 0.5) + 300, 500 - int(gt_pose[2, 3] * 0.5)
        
        cv2.circle(self.map_canvas, (ex, ez), 1, (0, 255, 0), -1) # Green = VO
        cv2.circle(self.map_canvas, (gx, gz), 1, (0, 0, 255), -1) # Red = GT

        # 2. Fix the Dashboard size (Ensure it is at least 600px high)
        h, w = frame.shape[:2]
        dash_h = max(h + 250, 600) # Ensure we have enough room for map and text
        dash_w = w + 600
        
        self.dashboard = np.zeros((dash_h, dash_w, 3), dtype=np.uint8)
        
        # Place Camera Feed
        self.dashboard[:h, :w] = frame
        
        # 🚀 THE FIX: Place the 600x600 map correctly
        self.dashboard[:600, w:] = self.map_canvas
        
        # 3. Add Telemetry Text below the camera feed
        stats = [
            (f"FRAME: {frame_id}", (50, 255, 50)),
            (f"SPEED: {speed*36:.1f} km/h", (255, 255, 255)),
            (f"DIST: {distance:.1f} m", (255, 255, 255)),
            (f"RMSE: {rmse:.4f} m", (50, 50, 255))
        ]
        
        for idx, (text, color) in enumerate(stats):
            cv2.putText(self.dashboard, text, (20, h + 50 + (idx*45)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        cv2.imshow("Semantic SLAM Engine - Local Dashboard", self.dashboard)