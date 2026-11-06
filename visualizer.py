import numpy as np
import cv2

class Visualizer:
    def __init__(self, width=800, height=800):
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.center_x, self.center_y = width // 2, height // 2

    def update(self, est_pos, gt_pos, frame_id, pts3D=None):
        # Scale for display (0.5x zoom)
        ex, ez = int(est_pos[0] * 0.5) + self.center_x, self.center_y - int(est_pos[2] * 0.5)
        gx, gz = int(gt_pos[0, 3] * 0.5) + self.center_x, self.center_y - int(gt_pos[2, 3] * 0.5)

        # Draw the environment (3D Points) in Grey
        if pts3D is not None:
            for pt in pts3D:
                # Project 3D point to our 2D bird's eye map
                px = int(pt[0] * 0.5) + self.center_x
                pz = self.center_y - int(pt[2] * 0.5)
                if 0 <= px < 800 and 0 <= pz < 800:
                    self.canvas[pz, px] = [100, 100, 100] # Grey dots

        # Draw Trajectories: Green (VO), Red (GT)
        cv2.circle(self.canvas, (ex, ez), 1, (0, 255, 0), 1)
        cv2.circle(self.canvas, (gx, gz), 1, (0, 0, 255), 1)

        display = self.canvas.copy()
        cv2.putText(display, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("SLAM Map (Points + Trajectory)", display)