import cv2
import numpy as np
import os
from utils import KITTILoader
from dashboard import DashboardVisualizer
from yolo_utils import DynamicMasker 

# CONFIG
SEQ_ID = "00" 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_PATH = os.path.join(SCRIPT_DIR, "data", "sequences", SEQ_ID)
POSE_PATH = os.path.join(SCRIPT_DIR, "data", "poses", f"{SEQ_ID}.txt")

def calculate_rmse(estimated_path, ground_truth_poses):
    """
    Calculates the Root Mean Square Error between estimated and ground truth translations.
    """
    if not estimated_path: 
        return 0.0
    
    errors = []
    for est, gt in zip(estimated_path, ground_truth_poses[:len(estimated_path)]):
        # gt[:3, 3] extracts the x, y, z translation from the 3x4 pose matrix
        # est.flatten() ensures the estimated translation is a 1D array of 3 elements
        err = np.linalg.norm(est.flatten() - gt[:3, 3])
        errors.append(err)
        
    return float(np.sqrt(np.mean(np.square(errors))))

def main():
    data = KITTILoader(SEQUENCE_PATH, POSE_PATH)
    viz = DashboardVisualizer()
    masker = DynamicMasker()

    # Load initial frames
    img_prev = cv2.imread(data.image_files[0], 0)
    img_prev_bgr = cv2.imread(data.image_files[0], 1)
    
    # Initialize detector
    detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
    
    # Initial dynamic masking
    initial_mask = masker.get_static_mask(img_prev_bgr)
    kp_prev = detector.detect(img_prev, mask=initial_mask)
    pts_prev = np.array([x.pt for x in kp_prev], dtype=np.float32)

    # Initialize Pose: cur_t is initialized as a 1D array to avoid conversion errors
    cur_R = np.eye(3)
    cur_t = np.zeros(3) 
    estimated_path = [cur_t.copy()]
    total_distance = 0.0

    for i in range(1, len(data.image_files)):
        img_curr = cv2.imread(data.image_files[i], 0)
        img_curr_bgr = cv2.imread(data.image_files[i], 1)
        static_mask = masker.get_static_mask(img_curr_bgr)

        # KLT Tracker
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, pts_prev, None)
        status = status.reshape(-1)
        
        # Filter valid points
        valid_p, valid_c = pts_prev[status==1], pts_curr[status==1]
        
        # Calculate True Absolute Scale from Ground Truth
        p_prev_gt = data.gt_poses[i - 1]
        p_curr_gt = data.gt_poses[i]
        true_scale = np.linalg.norm(p_curr_gt[:3, 3] - p_prev_gt[:3, 3])
        
        if len(valid_c) >= 8:
            E, mask = cv2.findEssentialMat(valid_c, valid_p, data.K, method=cv2.RANSAC, prob=0.999, threshold=0.5)
            
            if E is not None and E.shape == (3, 3):
                # Recover relative rotation and translation direction
                _, R, t, _ = cv2.recoverPose(E, valid_c, valid_p, data.K, mask=mask)
                
                # Integration (The car must be moving to update)
                if true_scale > 0.1:
                    # t.flatten() converts the (3,1) pose result to a (3,) 1D array
                    # This prevents the TypeError in dashboard.py
                    cur_t = cur_t + true_scale * cur_R.dot(t.flatten())
                    cur_R = R.dot(cur_R)
                    total_distance += true_scale
                
        estimated_path.append(cur_t.copy())
        rmse = calculate_rmse(estimated_path, data.gt_poses)
        
        # Update Dashboard
        viz.update(img_curr_bgr, cur_t, data.gt_poses[i], i, true_scale, rmse, total_distance)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
        # Refresh Features if tracked points drop or every 10 frames
        if len(valid_c) < 1500 or i % 10 == 0:
            kp_curr = detector.detect(img_curr, mask=static_mask)
            pts_new = np.array([x.pt for x in kp_curr], dtype=np.float32)
            if len(pts_new) > 0:
                valid_c = cv2.cornerSubPix(img_curr, pts_new, (5,5), (-1,-1), 
                                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1))
        
        img_prev, pts_prev = img_curr, valid_c

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()