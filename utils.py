import numpy as np
import cv2
import os

class KITTILoader:
    def __init__(self, sequence_dir, ground_truth_file):
        self.image_dir = os.path.join(sequence_dir, "image_0")
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Cannot find images at: {os.path.abspath(self.image_dir)}")
            
        self.image_files = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')])
        self.K = self._load_calib(os.path.join(sequence_dir, "calib.txt"))
        self.gt_poses = self._load_gt(ground_truth_file)

    def _load_calib(self, filepath):
        if not os.path.exists(filepath):
            filepath = os.path.join(os.path.dirname(filepath), "..", "calib.txt")
        with open(filepath, 'r') as f:
            line = f.readline()
            data = np.fromstring(line.split(':')[-1], sep=' ')
            P = data.reshape(3, 4)
            return P[0:3, 0:3]

    def _load_gt(self, filepath):
        with open(filepath, 'r') as f:
            return [np.fromstring(line, sep=' ').reshape(3, 4) for line in f]