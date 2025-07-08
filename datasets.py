# import h5py
# import torch
# import numpy as np
# import torch.utils.data as Data

# class H5Datasets(Data.Dataset):
#     def __init__(self, h5file_path):
#         self.h5file_path = h5file_path
#         h5f = h5py.File(h5file_path, 'r')
#         self.keys = list(h5f['vis_patchs'].keys())
#         h5f.close()

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, index):
#         h5f = h5py.File(self.h5file_path, 'r')
#         key = self.keys[index]
#         vis = np.array(h5f['vis_patchs'][key])
#         ir = np.array(h5f['ir_patchs'][key])
#         h5f.close()
#         return torch.Tensor(vis).cuda(), torch.Tensor(ir).cuda()






# utils/datasets.py

import h5py
import torch
import numpy as np
import torch.utils.data as Data
import torchvision.transforms as transforms

class H5Datasets(Data.Dataset):
    def __init__(self, h5file_path, image_size=(128, 128)):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['vis_patchs'].keys())
        h5f.close()
        self.resize_transform = transforms.Resize(image_size)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        vis = np.array(h5f['vis_patchs'][key])
        ir = np.array(h5f['ir_patchs'][key])
        h5f.close()

        # --- DIAGNOSTIC PRINT STATEMENTS ---
        # print(f"DEBUG: Key: {key}")
        # print(f"DEBUG: Raw vis numpy shape: {vis.shape}, dtype: {vis.dtype}")
        # print(f"DEBUG: Raw ir numpy shape: {ir.shape}, dtype: {ir.dtype}")
        # --- END DIAGNOSTIC ---

        vis_tensor = torch.Tensor(vis)
        ir_tensor = torch.Tensor(ir)

        # --- DIAGNOSTIC PRINT STATEMENTS ---
        # print(f"DEBUG: vis_tensor shape after torch.Tensor(): {vis_tensor.shape}")
        # print(f"DEBUG: ir_tensor shape after torch.Tensor(): {ir_tensor.shape}")
        # --- END DIAGNOSTIC ---

        # Important: Ensure tensor is (C, H, W) before resize.
        # Your previous error was 3 channels, now 0. This suggests a problem in the H5 data itself
        # or how it's being interpreted when converting to torch.Tensor.

        # If vis_tensor was (H, W, C), it needs permute:
        # if vis_tensor.ndim == 3 and vis_tensor.shape[2] in [1, 3]: # Assuming channels last
        #     vis_tensor = vis_tensor.permute(2, 0, 1) # HWC to CHW
        # elif vis_tensor.ndim == 2: # Grayscale (H, W) -> (1, H, W)
        #     vis_tensor = vis_tensor.unsqueeze(0)

        # Same for ir_tensor
        # if ir_tensor.ndim == 3 and ir_tensor.shape[2] in [1, 3]: # Assuming channels last
        #     ir_tensor = ir_tensor.permute(2, 0, 1) # HWC to CHW
        # elif ir_tensor.ndim == 2: # Grayscale (H, W) -> (1, H, W)
        #     ir_tensor = ir_tensor.unsqueeze(0)


        vis_resized = self.resize_transform(vis_tensor)
        ir_resized = self.resize_transform(ir_tensor)

        # --- DIAGNOSTIC PRINT STATEMENTS ---
        # print(f"DEBUG: vis_resized shape after transform: {vis_resized.shape}")
        # print(f"DEBUG: ir_resized shape after transform: {ir_resized.shape}")
        # --- END DIAGNOSTIC ---

        return vis_resized.cuda(), ir_resized.cuda()