# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from skimage.io import imsave
# from modules.build import Encoder,Decoder,FusionMoudle
# from utils.image_utils import image_read_cv2
# from options import TrainOptions 


# def test(ckpt,vi_path,ir_path,out_path):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     encoder = nn.DataParallel(Encoder()).to(device)
#     encoder.load_state_dict(torch.load(ckpt,weights_only=True)['encoder'])
#     encoder.eval()
    
#     decoder = nn.DataParallel(Decoder()).to(device) 
#     decoder.load_state_dict(torch.load(ckpt,weights_only=True)['decoder'])
#     decoder.eval()
    
#     fuser  = nn.DataParallel(FusionMoudle()).to(device)
#     fuser.load_state_dict(torch.load(ckpt,weights_only=True)['fuse'])
#     fuser.eval()

#     with torch.no_grad():
#         for img_name in os.listdir(vi_path):
#             Img_vi = image_read_cv2(os.path.join(vi_path, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
#             Img_ir = image_read_cv2(os.path.join(ir_path, img_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
#             Img_vi = torch.FloatTensor(Img_vi)
#             Img_ir = torch.FloatTensor(Img_ir)
#             Img_vi, Img_ir = Img_vi.cuda(), Img_ir.cuda()
#             vi_share, vi_private, ir_share, ir_private = encoder(Img_vi,Img_ir)
#             feats_share, feats_private = fuser(vi_share, vi_private, ir_share, ir_private)
#             out = decoder(feats_share, feats_private)
#             out = (out - torch.min(out)) / (torch.max(out) - torch.min(out))
#             out = (out * 255.0).cpu().numpy().squeeze(0).squeeze(0).astype('uint8') 
#             imsave(os.path.join(out_path, "{}.png".format(img_name.split(sep='.')[0])),out)



# if __name__ == "__main__":
#     parser = TrainOptions()
#     opts = parser.parse()
#     test(opts.ckpt_path,opts.vi_path,opts.ir_path,opts.out_path)    












# import os
# import numpy as np
# import torch
# import torch.nn as nn
# from skimage.io import imsave # Used for saving images
# import sys
# from tqdm import tqdm # For progress bar

# # Add the project root to sys.path to allow importing from utils and modules
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import your model components (adjust if your modules/build.py has different class names or structure)
# # Assuming Encoder, Decoder, FusionModule are directly available from modules.build
# from modules.build import Encoder, Decoder, FusionMoudle # Corrected FusionModule spelling if it's 'FusionMoudle'
# # If your model is in models/FDFuse.py, you might need:
# # from models.FDFuse import Encoder, Decoder, FusionModule

# # Import image utility functions (ensure image_utils.py has RGB2YCbCr and YCbCr2RGB)
# from utils.image_utils import RGB2YCbCr, YCbCr2RGB, clamp, image_read_cv2 # Also import clamp for safety if needed

# # Import TrainOptions (ensure options.py is correctly set up)
# from options import TrainOptions

# # --- Image Loading and Preprocessing Utility ---
# def load_and_preprocess_image_for_test(image_path, is_visible=True, device='cpu'):
#     """
#     Loads an image, preprocesses it for model input, and returns
#     the Y-channel tensor along with original Cb, Cr for visible images.
#     """
#     # Load image using OpenCV (image_read_cv2 handles BGR to RGB/GRAY/YCrCb)
#     # For visible, we need full color information to get Y, Cb, Cr
#     # For IR, we need single channel (Y or grayscale)
    
#     if is_visible:
#         # Load as RGB to use RGB2YCbCr from image_utils
#         img_rgb_np = image_read_cv2(image_path, mode='RGB') # (H, W, 3) numpy array, 0-255 range
#         img_rgb_tensor = torch.from_numpy(img_rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0 # (1, 3, H, W) tensor, 0-1 range
        
#         # Convert to YCbCr and store Cb, Cr
#         Y, Cb, Cr = RGB2YCbCr(img_rgb_tensor) # All are (1, 1, H, W)
        
#         # Ensure they are on the correct device
#         return Y.to(device), Cb.to(device), Cr.to(device)
#     else:
#         # Load infrared as grayscale (1 channel)
#         img_ir_np = image_read_cv2(image_path, mode='GRAY') # (H, W) numpy array, 0-255 range
#         img_ir_tensor = torch.from_numpy(img_ir_np).unsqueeze(0).unsqueeze(0).float() / 255.0 # (1, 1, H, W) tensor, 0-1 range
        
#         # Ensure it is on the correct device
#         return img_ir_tensor.to(device), None, None # No Cb/Cr for IR

# # --- Test Function ---
# def test(ckpt, vi_path, ir_path, out_path):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Initialize models
#     # Use FusionMoudle if that's the correct class name from modules.build
#     encoder = nn.DataParallel(Encoder()).to(device)
#     decoder = nn.DataParallel(Decoder()).to(device)
#     fuser = nn.DataParallel(FusionMoudle()).to(device) 

#     # Check if checkpoint exists
#     if not os.path.exists(ckpt):
#         print(f"Error: Checkpoint file not found at {ckpt}")
#         return

#     # Load state_dict (corrected: removed weights_only=True)
#     print(f"Loading model checkpoint from: {ckpt}")
#     checkpoint = torch.load(ckpt, map_location=device)
    
#     encoder.load_state_dict(checkpoint['encoder'])
#     decoder.load_state_dict(checkpoint['decoder'])
#     fuser.load_state_dict(checkpoint['fuse'])

#     # Set models to evaluation mode
#     encoder.eval()
#     decoder.eval()
#     fuser.eval()
    
#     print("Models loaded successfully and set to evaluation mode.")

#     # Create output directory if it doesn't exist
#     os.makedirs(out_path, exist_ok=True)
#     print(f"Fused images will be saved to: {out_path}")

#     # Get list of image names (assuming they match between vi and ir folders)
#     image_names = sorted([f for f in os.listdir(vi_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
#     if not image_names:
#         print(f"No image files found in {vi_path}. Please check the path.")
#         return

#     print(f"Found {len(image_names)} image pairs to process.")

#     with torch.no_grad(): # Disable gradient calculations for inference
#         for img_name in tqdm(image_names, desc="Fusing Images"):
#             try:
#                 # Construct full paths
#                 vi_image_path = os.path.join(vi_path, img_name)
#                 ir_image_path = os.path.join(ir_path, img_name)

#                 # Load and preprocess images
#                 # vi_y, vi_cb, vi_cr are (1, 1, H, W) tensors
#                 vi_y, vi_cb, vi_cr = load_and_preprocess_image_for_test(vi_image_path, is_visible=True, device=device)
#                 ir_y, _, _ = load_and_preprocess_image_for_test(ir_image_path, is_visible=False, device=device)

#                 # Ensure dimensions match for model input (should be 4D: Batch, Channel, H, W)
#                 # The load_and_preprocess_image_for_test already returns (1, C, H, W)
                
#                 # Forward pass through the model
#                 vi_share, vi_private, ir_share, ir_private = encoder(vi_y, ir_y)
#                 feats_share, feats_private = fuser(vi_share, vi_private, ir_share, ir_private)
#                 fused_y = decoder(feats_share, feats_private) # Output is the fused Y-channel

#                 # --- Post-processing: Reconstruct COLOR RGB image ---
#                 # fused_y is (1, 1, H, W), vi_cb, vi_cr are (1, 1, H, W)
                
#                 # Use YCbCr2RGB from image_utils to get back RGB
#                 reconstructed_rgb_tensor = YCbCr2RGB(fused_y, vi_cb, vi_cr)
                
#                 # Clamp values to [0, 1] and convert to numpy for saving
#                 # Permute to (H, W, C) for skimage.io.imsave (or (H, W) for grayscale)
#                 fused_rgb_np = reconstructed_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
#                 # Denormalize to 0-255 range and convert to uint8
#                 fused_rgb_np = (fused_rgb_np * 255.0).astype(np.uint8)

#                 # Save the fused image
#                 output_filename = f"fused_{os.path.splitext(img_name)[0]}.png"
#                 output_path_full = os.path.join(out_path, output_filename)
                
#                 imsave(output_path_full, fused_rgb_np)
                
#             except Exception as e:
#                 print(f"Error processing image pair {img_name}: {e}")
#                 continue

#     print(f"\nFusion complete. Fused images saved to: {out_path}")


# if __name__ == "__main__":
#     parser = TrainOptions() # Assumes TrainOptions in options.py handles arguments
#     opts = parser.parse()
    
#     # Call the test function with arguments parsed from options.py
#     test(opts.ckpt_path, opts.vi_path, opts.ir_path, opts.out_path)











import os
import numpy as np
import torch
import torch.nn as nn
from skimage.io import imsave # Used for saving images
import sys
from tqdm import tqdm # For progress bar

# Add the project root to sys.path to allow importing from utils and modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.build import Encoder, Decoder, FusionMoudle # Corrected FusionModule spelling if it's 'FusionMoudle'
from utils.image_utils import RGB2YCbCr, YCbCr2RGB, clamp, image_read_cv2 # Also import clamp for safety if needed
from options import TrainOptions

# --- Image Loading and Preprocessing Utility ---
def load_and_preprocess_image_for_test(image_path, is_visible=True, device='cpu'):
    """
    Loads an image, preprocesses it for model input, and returns
    the Y-channel tensor along with original Cb, Cr for visible images.
    """
    # Load image using OpenCV (image_read_cv2 handles BGR to RGB/GRAY/YCrCb)
    # For visible, we need full color information to get Y, Cb, Cr
    # For IR, we need single channel (Y or grayscale)
    
    if is_visible:
        # Load as RGB to use RGB2YCbCr from image_utils
        img_rgb_np = image_read_cv2(image_path, mode='RGB') # (H, W, 3) numpy array, 0-255 range
        img_rgb_tensor = torch.from_numpy(img_rgb_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0 # (1, 3, H, W) tensor, 0-1 range
        
        # Convert to YCbCr and store Cb, Cr
        Y, Cb, Cr = RGB2YCbCr(img_rgb_tensor) # All are (1, 1, H, W)
        
        # Ensure they are on the correct device
        return Y.to(device), Cb.to(device), Cr.to(device)
    else:
        # Load infrared as grayscale (1 channel)
        img_ir_np = image_read_cv2(image_path, mode='GRAY') # (H, W) numpy array, 0-255 range
        img_ir_tensor = torch.from_numpy(img_ir_np).unsqueeze(0).unsqueeze(0).float() / 255.0 # (1, 1, H, W) tensor, 0-1 range
        
        # Ensure it is on the correct device
        return img_ir_tensor.to(device), None, None # No Cb/Cr for IR

# --- Test Function ---
def test(ckpt, vi_path, ir_path, out_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize models
    # Use FusionMoudle if that's the correct class name from modules.build
    encoder = nn.DataParallel(Encoder()).to(device)
    decoder = nn.DataParallel(Decoder()).to(device)
    fuser = nn.DataParallel(FusionMoudle()).to(device) 

    # Check if checkpoint exists
    if not os.path.exists(ckpt):
        print(f"Error: Checkpoint file not found at {ckpt}")
        return

    # Load state_dict (corrected: removed weights_only=True)
    print(f"Loading model checkpoint from: {ckpt}")
    checkpoint = torch.load(ckpt, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    fuser.load_state_dict(checkpoint['fuse'])

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    fuser.eval()
    
    print("Models loaded successfully and set to evaluation mode.")

    # Create output directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    print(f"Fused images will be saved to: {out_path}")

    # Get list of image names (assuming they match between vi and ir folders)
    image_names = sorted([f for f in os.listdir(vi_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if not image_names:
        print(f"No image files found in {vi_path}. Please check the path.")
        return

    print(f"Found {len(image_names)} image pairs to process.")

    with torch.no_grad(): # Disable gradient calculations for inference
        for img_name in tqdm(image_names, desc="Fusing Images"):
            try:
                # Construct full paths
                vi_image_path = os.path.join(vi_path, img_name)
                ir_image_path = os.path.join(ir_path, img_name)

                # Load and preprocess images
                # vi_y, vi_cb, vi_cr are (1, 1, H, W) tensors
                vi_y, vi_cb, vi_cr = load_and_preprocess_image_for_test(vi_image_path, is_visible=True, device=device)
                ir_y, _, _ = load_and_preprocess_image_for_test(ir_image_path, is_visible=False, device=device)

                # Ensure dimensions match for model input (should be 4D: Batch, Channel, H, W)
                # The load_and_preprocess_image_for_test already returns (1, C, H, W)
                
                # Forward pass through the model
                vi_share, vi_private, ir_share, ir_private = encoder(vi_y, ir_y)
                feats_share, feats_private = fuser(vi_share, vi_private, ir_share, ir_private)
                fused_y = decoder(feats_share, feats_private) # Output is the fused Y-channel

                # --- Post-processing: Reconstruct COLOR RGB image ---
                # fused_y is (1, 1, H, W), vi_cb, vi_cr are (1, 1, H, W)
                
                # Use YCbCr2RGB from image_utils to get back RGB
                reconstructed_rgb_tensor = YCbCr2RGB(fused_y, vi_cb, vi_cr)
                
                # Clamp values to [0, 1] and convert to numpy for saving
                # Permute to (H, W, C) for skimage.io.imsave (or (H, W) for grayscale)
                fused_rgb_np = reconstructed_rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Denormalize to 0-255 range and convert to uint8
                fused_rgb_np = (fused_rgb_np * 255.0).astype(np.uint8)

                # Save the fused image
                output_filename = f"fused_{os.path.splitext(img_name)[0]}.png"
                output_path_full = os.path.join(out_path, output_filename)
                
                imsave(output_path_full, fused_rgb_np)
                
            except Exception as e:
                print(f"Error processing image pair {img_name}: {e}")
                continue

    print(f"\nFusion complete. Fused images saved to: {out_path}")


if __name__ == "__main__":
    parser = TrainOptions() # Assumes TrainOptions in options.py handles arguments
    opts = parser.parse()
    
    # Call the test function with arguments parsed from options.py
    test(opts.ckpt_path, opts.vi_path, opts.ir_path, opts.out_path)