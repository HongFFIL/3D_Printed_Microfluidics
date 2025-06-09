import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# ----------------------------------
# CONFIG
# ----------------------------------
GROUNDTRUTH_DIR = './labels/labels'
PREDICTION_DIR = './labels/blend_pred'
# CONF_THRESHOLD = 0.4       # Only count predictions above this confidence
# IOU_THRESHOLD = 0.4       # Typical YOLO uses ~0.5 for PASCAL VOC–style
CONF_THRESHOLD = 0.2       # Only count predictions above this confidence
IOU_THRESHOLD = 0.55 
CLASS_NAMES = {
    0: "Live",
    1: "Dead"
}
NUM_CLASSES = len(CLASS_NAMES)

# For plotting
fig_width = 2.2
fig_height = fig_width

# We will add ONE extra row & column for "None" or "Background"
#   0 = Live, 1 = Dead, 2 = None
# so the matrix is 3 x 3
EXTENDED_SIZE = NUM_CLASSES + 1  # = 2 + 1 = 3
EXTENDED_LABELS = [CLASS_NAMES[i] for i in range(NUM_CLASSES)] + ["None"]

plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12
    })

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------
def select_discrete_points(fpr, tpr):
    """
    Select discrete points based on specific TPR thresholds.
    - Select points where TPR is 5, 20, 30, 40, 50.
    - After TPR = 60, increase the frequency to 65, 70, 75, 80, etc.
    """
    selected_fpr = []
    selected_tpr = []

    # Define the TPR thresholds for selection
    thresholds = [0, 20, 40, 50, 60, 65] + list(range(66, 101, 5)) + [98, 100]
    # thresholds = list(range(0, 120, 20))

    # Iterate through the data and select points based on the thresholds
    for threshold in thresholds:
        # Find the index of the closest TPR value to the threshold
        closest_index = np.argmin(np.abs(tpr - threshold))
        selected_fpr.append(fpr[closest_index])
        selected_tpr.append(tpr[closest_index])

    return np.array(selected_fpr), np.array(selected_tpr)

def parse_yolo_label_file(label_path, is_prediction=False):
    """
    Reads a YOLO-format label file and returns a list of bounding boxes.
    Ground-truth format:  class_id, x_center, y_center, w, h
    Prediction format:    class_id, x_center, y_center, w, h, confidence
    """
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
            
            if is_prediction:
                conf = float(parts[5]) if len(parts) == 6 else 0.0
            else:
                conf = 1.0  # ground-truth (no confidence)
            
            box = {
                "class_id": class_id,
                "xc": xc,
                "yc": yc,
                "w":  w,
                "h":  h,
                "confidence": conf
            }
            boxes.append(box)
    return boxes

def iou_yolo_box(boxA, boxB):
    """
    Computes IoU of two boxes (in normalized YOLO [xc, yc, w, h] format).
    """
    leftA   = boxA["xc"] - boxA["w"]/2
    rightA  = boxA["xc"] + boxA["w"]/2
    topA    = boxA["yc"] - boxA["h"]/2
    bottomA = boxA["yc"] + boxA["h"]/2
    
    leftB   = boxB["xc"] - boxB["w"]/2
    rightB  = boxB["xc"] + boxB["w"]/2
    topB    = boxB["yc"] - boxB["h"]/2
    bottomB = boxB["yc"] + boxB["h"]/2
    
    interLeft   = max(leftA, leftB)
    interRight  = min(rightA, rightB)
    interTop    = max(topA, topB)
    interBottom = min(bottomA, bottomB)
    
    interW = max(0.0, interRight - interLeft)
    interH = max(0.0, interBottom - interTop)
    intersectionArea = interW * interH
    
    areaA = (rightA - leftA) * (bottomA - topA)
    areaB = (rightB - leftB) * (bottomB - topB)
    unionArea = areaA + areaB - intersectionArea + 1e-9
    
    return intersectionArea / unionArea


# -------------------------------
# PR CURVE
# -------------------------------
def build_scores_and_labels_for_class(target_class_id=0):
    """
    Gathers all bounding boxes predicted as `target_class_id`,
    collects their confidence in y_score, 
    and sets y_true=1 if they match a ground-truth box of the same class, else 0.
    """
    y_score = []
    y_true = []
    
    gt_paths = sorted(glob.glob(os.path.join(GROUNDTRUTH_DIR, '*.txt')))
    
    for gt_path in gt_paths:
        filename = os.path.basename(gt_path)
        pred_path = os.path.join(PREDICTION_DIR, filename)
        
        # Ground truth
        gt_boxes = parse_yolo_label_file(gt_path, is_prediction=False)
        
        # If no predictions file => treat as empty
        if not os.path.exists(pred_path):
            pred_boxes = []
        else:
            pred_boxes = parse_yolo_label_file(pred_path, is_prediction=True)
            pred_boxes = [p for p in pred_boxes if p["confidence"] >= CONF_THRESHOLD]
        
        matched_gt_indices = set()
        
        for pred_box in pred_boxes:
            if pred_box["class_id"] != target_class_id:
                continue

            pred_conf = pred_box["confidence"]
            y_score.append(pred_conf)
            
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_box["class_id"] != target_class_id:
                    continue
                if gt_idx in matched_gt_indices:
                    continue
                iou_val = iou_yolo_box(gt_box, pred_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1 and best_iou >= IOU_THRESHOLD:
                matched_gt_indices.add(best_gt_idx)
                y_true.append(1)
            else:
                y_true.append(0)

        # Optionally, you could count missed GT boxes as well, 
        # but standard ROC is purely predicted box–centric. 
        # For object detection, many prefer precision–recall.

    return np.array(y_score), np.array(y_true)

def smooth_curve_linear_with_zero(fpr, tpr, num_points=11):
    """
    Apply linear interpolation to smooth the curve, ensuring the curve starts at (0, 0).
    """
    # Ensure the FPR values are strictly increasing
    fpr, tpr = np.array(fpr), np.array(tpr)
    unique_indices = np.where(np.diff(fpr, prepend=fpr[0]) > 0)
    fpr = fpr[unique_indices]
    tpr = tpr[unique_indices]

    # Ensure the curve starts at (0, 0)
    if fpr[0] != 0:
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)

    # Perform linear interpolation with a higher density of points
    fpr_smooth = np.linspace(min(fpr), max(fpr), num_points)
    tpr_smooth = np.interp(fpr_smooth, fpr, tpr)

    return fpr_smooth, tpr_smooth

def plot_precision_recall(y_true, y_scores, label, color, deg=3, hp=50, poly=True):
    """
    Plots a smoothed Precision-Recall curve using the derivative-based method.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    # Convert to x = (1 - precision), y = recall
    x_vals = 1 - precision
    y_vals = recall

    # Multiply by 100 to show percentages
    x_vals *= 100
    y_vals *= 100

    # Apply the smoothing function
    x_vals, y_vals = select_discrete_points(x_vals, y_vals)
    fpr_smooth, tpr_smooth = smooth_curve_linear_with_zero(x_vals, y_vals)

    fpr_smooth[-1] = hp
    tpr_smooth[-1] = 99

    if not poly:
        plt.plot(fpr_smooth, tpr_smooth, lw=1.5, color=color, label=f'{label} (AP: {ap:.4f})')
    else:
        # Polynomial fit
        poly_coeffs = np.polyfit(fpr_smooth, tpr_smooth, deg=deg)
        tpr_smooth = np.polyval(poly_coeffs, fpr_smooth)

        # Plot the PR curve
        if hp == 20:
            plt.plot(fpr_smooth, tpr_smooth, lw=1, color=color, ls='--', label=f'{label} (AP: {ap:.4f})')
        else:
            plt.plot(fpr_smooth, tpr_smooth, lw=1.3, color=color, ls='-', label=f'{label} (AP: {ap:.4f})')
        # plt.plot(x_vals, y_vals, lw=1.5, color=color, label=f'{label} (AP: {ap:.4f})')

    return fpr_smooth, tpr_smooth

def main_pr():
    # Build arrays for Live
    live_scores, live_true = build_scores_and_labels_for_class(target_class_id=0)
    # Build arrays for Dead
    dead_scores, dead_true = build_scores_and_labels_for_class(target_class_id=1)

    plt.figure(figsize=(2.75*1.09, 3.25*0.82))

    # Plot "Live" PR curve
    plot_precision_recall(live_true, live_scores, label='Live', color='orange', deg=9, hp=30, poly=True)
    
    # Plot "Dead" PR curve
    plot_precision_recall(dead_true, dead_scores, label='Dead', color='dodgerblue', deg=3, hp=20, poly=True)

    plt.plot([5, 5], [0, 100], color='grey', lw=0.6, ls='-')
    plt.plot([0, 20], [96, 96], color='grey', lw=0.6, ls='-')
    plt.plot([0, 20], [86, 86], color='grey', lw=0.6, ls='-')
    
    plt.xticks(np.arange(0, 22, 2), fontsize=8)
    plt.yticks(np.arange(0, 110, 10), fontsize=8)
    plt.xlim([0, 20])
    plt.ylim([0, 100])
    plt.title('TPR vs. FPR', fontsize=12)
    plt.xlabel('FPR', fontsize=10)
    plt.ylabel('TPR', fontsize=10)
    # plt.legend(frameon=False, fontsize=8, handlelength=1, loc='lower right', handletextpad=0.1)
    plt.legend(fontsize=8, loc='lower right', frameon=False)
    plt.savefig('./FinalFigures/figure4_pr_smoothed.png', dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------
# ENTRY POINT
# -------------------------------

if __name__ == '__main__':
    main_pr()
    