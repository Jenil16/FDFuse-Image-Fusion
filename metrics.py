import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import piq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set paths
vi_dir = "C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/test/vi"
ir_dir = "C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/test/ir"
fused_dir = "./output_model_2"

# --- Metric functions ---

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) if img is not None else None

def calc_entropy(img):
    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calc_std(img):
    return np.std(img)

def calc_spatial_freq(img):
    RF = np.sqrt(np.mean(np.diff(img, axis=0)**2))
    CF = np.sqrt(np.mean(np.diff(img, axis=1)**2))
    return np.sqrt(RF**2 + CF**2)

def calc_scd(img1, img2, fused):
    diff1 = np.abs(img1 - fused)
    diff2 = np.abs(img2 - fused)
    scd1 = np.corrcoef(img1.ravel(), diff1.ravel())[0, 1]
    scd2 = np.corrcoef(img2.ravel(), diff2.ravel())[0, 1]
    return scd1 + scd2

def gradient_maps(img):
    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(dx**2 + dy**2)
    phase = np.arctan2(dy, dx + 1e-8)
    return mag, phase

def compute_qabf(img1, img2, fused):
    mag1, phase1 = gradient_maps(img1)
    mag2, phase2 = gradient_maps(img2)
    magf, phasef = gradient_maps(fused)

    mag1 /= (np.max(mag1) + 1e-8)
    mag2 /= (np.max(mag2) + 1e-8)
    magf /= (np.max(magf) + 1e-8)

    QG1 = (2 * mag1 * magf + 1e-8) / (mag1**2 + magf**2 + 1e-8)
    QG2 = (2 * mag2 * magf + 1e-8) / (mag2**2 + magf**2 + 1e-8)

    QP1 = (2 * np.cos(phase1 - phasef) + 1) / 3
    QP2 = (2 * np.cos(phase2 - phasef) + 1) / 3

    QG = QG1 * QG2
    QP = QP1 * QP2

    return np.mean(QG * QP)

def compute_vif_piq(ref, fused):
    try:
        # Normalize and convert to torch tensor with shape (1, 1, H, W)
        ref_tensor = torch.tensor(ref / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        fused_tensor = torch.tensor(fused / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return piq.vif_p(fused_tensor, ref_tensor).item()
    except Exception as e:
        logging.warning(f"VIF error: {e}")
        return np.nan

# --- Main execution ---

metrics = {
    "entropy": [],
    "std": [],
    "spatial_freq": [],
    "scd": [],
    "qabf": [],
    "vif": []
}

image_files = sorted([f for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

for img_name in tqdm(image_files, desc="Processing Images"):
    vi_path = os.path.join(vi_dir, img_name)
    ir_path = os.path.join(ir_dir, img_name)
    fused_path = os.path.join(fused_dir, f"fused_{img_name}")

    vi = load_image(vi_path)
    ir = load_image(ir_path)
    fused = load_image(fused_path)

    if vi is None or ir is None or fused is None:
        logging.warning(f"Skipping {img_name} due to missing image.")
        continue

    metrics["entropy"].append(calc_entropy(fused))
    metrics["std"].append(calc_std(fused))
    metrics["spatial_freq"].append(calc_spatial_freq(fused))
    metrics["scd"].append(calc_scd(vi, ir, fused))
    metrics["qabf"].append(compute_qabf(vi, ir, fused))

    vif_vi = compute_vif_piq(vi, fused)
    vif_ir = compute_vif_piq(ir, fused)
    metrics["vif"].append((vif_vi + vif_ir) / 2)

# --- Final Results ---
print("\n======== MEAN METRICS FOR FUSED IMAGES ========")
for key in metrics:
    values = np.array(metrics[key])
    print(f"{key.upper():15}: {np.nanmean(values):.4f}")









# For VIFB

# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# import torch
# import piq
# import logging

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Set paths
# vi_dir = "C:/Users/Jenil/OneDrive/Desktop/Thesis Project/Datasets/VIFB/vi"
# ir_dir = "C:/Users/Jenil/OneDrive/Desktop/Thesis Project/Datasets/VIFB/ir"
# fused_dir = "./output_VIFB"

# # --- Metric Functions ---

# def load_image(path):
#     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     return img.astype(np.float32) if img is not None else None

# def calc_entropy(img):
#     hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
#     hist = hist.ravel() / hist.sum()
#     hist = hist[hist > 0]
#     return -np.sum(hist * np.log2(hist))

# def calc_std(img):
#     return np.std(img)

# def calc_spatial_freq(img):
#     RF = np.sqrt(np.mean(np.diff(img, axis=0)**2))
#     CF = np.sqrt(np.mean(np.diff(img, axis=1)**2))
#     return np.sqrt(RF**2 + CF**2)

# def calc_scd(img1, img2, fused):
#     diff1 = np.abs(img1 - fused)
#     diff2 = np.abs(img2 - fused)
#     scd1 = np.corrcoef(img1.ravel(), diff1.ravel())[0, 1]
#     scd2 = np.corrcoef(img2.ravel(), diff2.ravel())[0, 1]
#     return scd1 + scd2

# def gradient_maps(img):
#     dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
#     mag = np.sqrt(dx**2 + dy**2)
#     phase = np.arctan2(dy, dx + 1e-8)
#     return mag, phase

# def compute_qabf(img1, img2, fused):
#     mag1, phase1 = gradient_maps(img1)
#     mag2, phase2 = gradient_maps(img2)
#     magf, phasef = gradient_maps(fused)

#     mag1 /= (np.max(mag1) + 1e-8)
#     mag2 /= (np.max(mag2) + 1e-8)
#     magf /= (np.max(magf) + 1e-8)

#     QG1 = (2 * mag1 * magf + 1e-8) / (mag1**2 + magf**2 + 1e-8)
#     QG2 = (2 * mag2 * magf + 1e-8) / (mag2**2 + magf**2 + 1e-8)

#     QP1 = (2 * np.cos(phase1 - phasef) + 1) / 3
#     QP2 = (2 * np.cos(phase2 - phasef) + 1) / 3

#     QG = QG1 * QG2
#     QP = QP1 * QP2

#     return np.mean(QG * QP)

# def compute_vif_piq(ref, fused):
#     try:
#         ref_tensor = torch.tensor(ref / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         fused_tensor = torch.tensor(fused / 255., dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         return piq.vif_p(fused_tensor, ref_tensor).item()
#     except Exception as e:
#         logging.warning(f"VIF error: {e}")
#         return np.nan

# # --- Main Execution ---

# metrics = {
#     "entropy": [],
#     "std": [],
#     "spatial_freq": [],
#     "scd": [],
#     "qabf": [],
#     "vif": []
# }

# image_files = sorted([f for f in os.listdir(vi_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# for img_name in tqdm(image_files, desc="Processing Images"):
#     name_no_ext = os.path.splitext(img_name)[0]

#     vi_path = os.path.join(vi_dir, img_name)
#     ir_path = os.path.join(ir_dir, img_name)
#     fused_path = os.path.join(fused_dir, f"fused_{name_no_ext}.png")

#     vi = load_image(vi_path)
#     ir = load_image(ir_path)
#     fused = load_image(fused_path)

#     if vi is None or ir is None or fused is None:
#         logging.warning(f"Skipping {img_name} due to missing image.")
#         continue

#     metrics["entropy"].append(calc_entropy(fused))
#     metrics["std"].append(calc_std(fused))
#     metrics["spatial_freq"].append(calc_spatial_freq(fused))
#     metrics["scd"].append(calc_scd(vi, ir, fused))
#     metrics["qabf"].append(compute_qabf(vi, ir, fused))

#     vif_vi = compute_vif_piq(vi, fused)
#     vif_ir = compute_vif_piq(ir, fused)
#     metrics["vif"].append((vif_vi + vif_ir) / 2)

# # --- Final Results ---

# print("\n======== MEAN METRICS FOR FUSED IMAGES ========")
# for key in metrics:
#     values = np.array(metrics[key])
#     print(f"{key.upper():15}: {np.nanmean(values):.4f}")
