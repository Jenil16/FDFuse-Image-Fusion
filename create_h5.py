# import os
# import h5py
# import numpy as np
# import cv2
# from tqdm import tqdm

# def load_image(path):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise ValueError(f"Failed to read {path}")
#     if len(img.shape) == 2:
#         img = img[:, :, np.newaxis]  # single-channel IR
#     else:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # VIS to YCrCb
#     img = img.astype(np.float32) / 255.0
#     return img.transpose(2, 0, 1)  # (C, H, W)

# def create_h5(vi_folder, ir_folder, out_path):
#     vi_files = sorted(os.listdir(vi_folder))
#     ir_files = sorted(os.listdir(ir_folder))
#     assert len(vi_files) == len(ir_files), "Mismatched image counts"

#     with h5py.File(out_path, 'w') as h5f:
#         vis_grp = h5f.create_group('vis_patchs')
#         ir_grp = h5f.create_group('ir_patchs')

#         for i, (v, r) in enumerate(tqdm(zip(vi_files, ir_files), total=len(vi_files))):
#             vi_img = load_image(os.path.join(vi_folder, v))
#             ir_img = load_image(os.path.join(ir_folder, r))

#             key = f'{i:05d}'
#             vis_grp.create_dataset(key, data=vi_img, compression="gzip")
#             ir_grp.create_dataset(key, data=ir_img, compression="gzip")

# if __name__ == '__main__':
#     vi_dir = 'C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/train/vi'
#     ir_dir = 'C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/train/ir'
#     out_file = 'train_data.h5'
#     create_h5(vi_dir, ir_dir, out_file)






import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import logging # Import logging module for better error messages

# Configure logging to show information, warnings, and errors in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(path):
    """
    Loads an image using OpenCV, performs basic validation,
    converts to YCrCb (for color) or adds channel (for grayscale),
    normalizes, and transposes to (C, H, W) format.
    Returns None if the image is unreadable or malformed.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        logging.error(f"Failed to read image file (cv2.imread returned None) for: {path}")
        return None # Indicate failure

    # Basic check for empty or zero-dimension arrays after reading
    if img.size == 0 or any(dim == 0 for dim in img.shape):
        logging.warning(f"Detected empty or zero-dimension image for: {path}. Shape: {img.shape}. Skipping.")
        return None # Indicate failure

    original_shape = img.shape
    processed_img = None

    try:
        if len(original_shape) == 2: # Grayscale image (e.g., IR) - shape is (H, W)
            processed_img = img[:, :, np.newaxis] # Convert to (H, W, 1)
            logging.info(f"Loaded grayscale image from {path}. Original shape: {original_shape}, Processed shape: {processed_img.shape}")
        elif len(original_shape) == 3: # Color image (e.g., VIS) - shape is (H, W, C)
            if original_shape[2] == 3: # Expected BGR 3-channel image
                processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # Convert BGR to YCrCb
                logging.info(f"Loaded color image from {path}. Original shape: {original_shape}, Processed shape: {processed_img.shape} (YCrCb)")
            elif original_shape[2] == 4: # Handle 4-channel (RGBA) images by dropping alpha
                processed_img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2YCrCb) # Drop alpha, then convert
                logging.info(f"Loaded RGBA image from {path}. Original shape: {original_shape}, Processed shape: {processed_img.shape} (YCrCb after dropping alpha)")
            elif original_shape[2] == 1: # Already (H, W, 1), treat as grayscale
                processed_img = img # Keep as is
                logging.info(f"Loaded single-channel 3D image from {path}. Original shape: {original_shape}, Processed shape: {processed_img.shape}")
            else:
                logging.error(f"Unexpected number of channels ({original_shape[2]}) for image: {path}. Original shape: {original_shape}. Skipping.")
                return None # Indicate failure
        else:
            logging.error(f"Unexpected number of dimensions ({len(original_shape)}) for image: {path}. Original shape: {original_shape}. Skipping.")
            return None # Indicate failure

        # Final check for zero dimensions after processing (before normalization and transpose)
        if processed_img is None or any(dim == 0 for dim in processed_img.shape):
            logging.error(f"Processed image is invalid or has a zero-dimension for: {path}. Final shape: {processed_img.shape if processed_img is not None else 'None'}. Skipping.")
            return None # Indicate failure

        # Normalize to float32 and transpose to (C, H, W)
        final_img = processed_img.astype(np.float32) / 255.0
        final_img = final_img.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)

        # Critical final check for 0 channels after transpose
        if final_img.shape[0] == 0:
            logging.error(f"Image {path} has 0 channels after transpose. Final shape: {final_img.shape}. This is critical. Skipping.")
            return None # Indicate failure
        
        return final_img

    except cv2.error as e:
        logging.error(f"OpenCV error processing image {path}: {e}")
        return None
    except Exception as e:
        logging.exception(f"General error occurred while processing image {path}.") # log full traceback
        return None

def create_h5(vi_folder, ir_folder, out_path):
    """
    Creates an H5 dataset from visible and infrared image folders.
    It skips problematic image files gracefully.
    """
    vi_files = sorted(os.listdir(vi_folder))
    ir_files = sorted(os.listdir(ir_folder))
    
    if len(vi_files) == 0 or len(ir_files) == 0:
        logging.error(f"No files found in '{vi_folder}' or '{ir_folder}'. Aborting H5 creation.")
        return

    if len(vi_files) != len(ir_files):
        logging.error(f"Mismatched image counts: VIS folder has {len(vi_files)} files, IR folder has {len(ir_files)} files. Aborting.")
        return

    # Pre-check image files for basic existence and non-zero size
    valid_pairs = []
    logging.info("Pre-checking image files for existence and size...")
    for v_file, r_file in tqdm(zip(vi_files, ir_files), total=len(vi_files), desc="Pre-checking"):
        v_path = os.path.join(vi_folder, v_file)
        r_path = os.path.join(ir_folder, r_file)

        if not os.path.exists(v_path) or os.path.getsize(v_path) == 0:
            logging.warning(f"Skipping VIS file '{v_file}' due to non-existence or zero size.")
            continue
        if not os.path.exists(r_path) or os.path.getsize(r_path) == 0:
            logging.warning(f"Skipping IR file '{r_file}' due to non-existence or zero size.")
            continue
        
        valid_pairs.append((v_file, r_file))

    if not valid_pairs:
        logging.error("No valid image pairs found after pre-checking. H5 file will be empty if created.")
        return

    logging.info(f"Found {len(valid_pairs)} valid image pairs to process.")

    with h5py.File(out_path, 'w') as h5f: # 'w' mode will create/overwrite the file
        vis_grp = h5f.create_group('vis_patchs')
        ir_grp = h5f.create_group('ir_patchs')

        skipped_count_during_creation = 0
        for i, (v_filename, r_filename) in enumerate(tqdm(valid_pairs, total=len(valid_pairs), desc="Creating H5")):
            vi_path = os.path.join(vi_folder, v_filename)
            ir_path = os.path.join(ir_folder, r_filename)

            vi_img = load_image(vi_path)
            ir_img = load_image(ir_path)

            if vi_img is None or ir_img is None:
                skipped_count_during_creation += 1
                logging.warning(f"Skipping H5 entry for pair '{v_filename}'/'{r_filename}' due to 'load_image' returning None.")
                continue # Skip this pair if load_image failed

            key = f'{i:05d}'
            try:
                # Add compression_opts=9 for maximum gzip compression
                vis_grp.create_dataset(key, data=vi_img, compression="gzip", compression_opts=9)
                ir_grp.create_dataset(key, data=ir_img, compression="gzip", compression_opts=9)
            except Exception as e:
                logging.error(f"Failed to create H5 dataset for key '{key}' ({v_filename}, {r_filename}): {e}")
                skipped_count_during_creation += 1
                # Decide if you want to abort or continue. Continuing allows creation of a partial H5.
                continue

        logging.info(f"H5 creation process completed. Total entries attempted: {len(valid_pairs)}. Skipped during H5 writing: {skipped_count_during_creation}.")

if __name__ == '__main__':
    # Define your image directories and output H5 file name
    vi_dir = 'C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/train/vi'
    ir_dir = 'C:/Users/Jenil/OneDrive/Desktop/Thesis Project/FDFuse/MSRS/train/ir'
    out_file = 'train_data.h5'
    
    # --- IMPORTANT: Delete existing H5 file to ensure a clean start ---
    if os.path.exists(out_file):
        os.remove(out_file)
        logging.info(f"Removed existing H5 file: {out_file}")
        
    # Check if input directories exist before starting
    if not os.path.exists(vi_dir):
        logging.critical(f"Error: Visible image directory not found: {vi_dir}")
        exit() # Exit if directory is not found
    if not os.path.exists(ir_dir):
        logging.critical(f"Error: Infrared image directory not found: {ir_dir}")
        exit() # Exit if directory is not found

    create_h5(vi_dir, ir_dir, out_file)
    logging.info(f"H5 file creation script finished. Check '{out_file}' for contents and console for any warnings/errors.")