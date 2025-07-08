# import  random
# import  numpy as np
# import os
# import sys
# import time
# import datetime
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import kornia





# from modules.build import Encoder,Decoder,FusionMoudle,A2B
# from utils.train_utils import Fusionloss
# from options import  TrainOptions 
# from datasets import H5Datasets
# from utils.image_utils import RGB2YCbCr # Import RGB2YCbCr

# parser = TrainOptions()
# opts = parser.parse()

# if torch.cuda.is_available():
#     print('cuda')
# else:
#     print('cpu')
# # Model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # loss
# criteria_fusion = Fusionloss()


# encoder_model = nn.DataParallel(Encoder()).to(device)
# decoder_model = nn.DataParallel(Decoder()).to(device)
# fusion_model = nn.DataParallel(FusionMoudle()).to(device)
# Temporary_model = nn.DataParallel(A2B()).to(device)
# encoder_model.train()
# decoder_model.train()  
# fusion_model.train()
# Temporary_model.train()

# optimizer1 = torch.optim.Adam(encoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
# optimizer2 = torch.optim.Adam(decoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
# optimizer3 = torch.optim.Adam(fusion_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
# optimizer4 = torch.optim.Adam(Temporary_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

# scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opts.step_size, gamma=opts.gamma)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opts.step_size, gamma=opts.gamma)
# scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opts.step_size, gamma=opts.gamma)
# scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opts.step_size, gamma=opts.gamma)



# MSELoss = nn.MSELoss()
# Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

# trainloader = DataLoader(H5Datasets(opts.train_data),batch_size=opts.batch_size,shuffle=True,num_workers=0)
# loader = {'train': trainloader, }
# timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
# prev_time = time.time()
# for epoch in range(opts.total_epoch):
#     for i, (vi, ir) in enumerate(loader['train']):
#         vi, ir = vi.cuda(), ir.cuda()
#         encoder_model.zero_grad()
#         decoder_model.zero_grad()
#         fusion_model.zero_grad()
#         Temporary_model.zero_grad()
#         optimizer1.zero_grad()
#         optimizer2.zero_grad()
#         optimizer3.zero_grad()
#         optimizer4.zero_grad()
#         if epoch < opts.gap_epoch: 
#             vi_share,vi_private,ir_share,ir_private = encoder_model(vi,ir)
#             loss_decomp = Temporary_model(vi_share,ir_share,vi,ir)
#             vi_hat,ir_hat = decoder_model(vi_share,vi_private),decoder_model(ir_share,ir_private)
#             loss_recon = 5 * Loss_ssim(vi, vi_hat) + MSELoss(vi, vi_hat) + 5 * Loss_ssim(ir, ir_hat) + MSELoss(ir, ir_hat)
#             loss =   5 * loss_decomp + 2 * loss_recon   
#             loss.backward()

#             nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=0.01, norm_type=2)
#             nn.utils.clip_grad_norm_(Temporary_model.parameters(), max_norm=0.01, norm_type=2)
#             nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=0.01, norm_type=2)

#             optimizer1.step()  
#             optimizer2.step()
#             optimizer4.step()
            
#         else:  
#             vi_share,vi_private,ir_share,ir_private = encoder_model(vi,ir)
#             fuse_share,fuse_private = fusion_model(vi_share,vi_private,ir_share,ir_private)
#             out = decoder_model(fuse_share,fuse_private)
#             loss,_,_  = criteria_fusion(vi, ir, out)
#             loss.backward()
                    
#             nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=0.01, norm_type=2)
#             nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=0.01, norm_type=2)
#             nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=0.01, norm_type=2)
#             nn.utils.clip_grad_norm_(Temporary_model.parameters(), max_norm=0.01, norm_type=2)
#             optimizer1.step()  
#             optimizer2.step()
#             optimizer3.step()
#             optimizer4.step()
#         batches_done = epoch * len(loader['train']) + i
#         batches_left = opts.total_epoch * len(loader['train']) - batches_done
#         time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
#         prev_time = time.time()
#         sys.stdout.write( 
#             "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f]  ETA: %.10s"
#             % (
#                 epoch,
#                 opts.total_epoch,
#                 i,
#                 len(loader['train']),
#                 loss.item(),
#                 time_left,
#             )
#         )
        
#     scheduler1.step()  
#     scheduler2.step()
#     scheduler4.step()
#     if not epoch < opts.gap_epoch:
#         scheduler3.step()
        

#     if optimizer1.param_groups[0]['lr'] <= 1e-6:
#         optimizer1.param_groups[0]['lr'] = 1e-6
#     if optimizer2.param_groups[0]['lr'] <= 1e-6:
#         optimizer2.param_groups[0]['lr'] = 1e-6
#     if optimizer3.param_groups[0]['lr'] <= 1e-6:
#         optimizer3.param_groups[0]['lr'] = 1e-6
#     if optimizer4.param_groups[0]['lr'] <= 1e-6:
#         optimizer4.param_groups[0]['lr'] = 1e-6

# checkpoint = {
#     'encoder': encoder_model.state_dict(),
#     'decoder': decoder_model.state_dict(),
#     'fuse': fusion_model.state_dict(),
# }
# torch.save(checkpoint,"FDFuse_2024.pth")





import random
import numpy as np
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import kornia
import torch.cuda.amp as amp
import logging

# Import project modules
from modules.build import Encoder, Decoder, FusionMoudle, A2B
from utils.train_utils import Fusionloss
from options import TrainOptions
from datasets import H5Datasets
from utils.image_utils import RGB2YCbCr, Sobelxy
import torch.nn.functional as F


# Setup at the top
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"training_log_{timestamp}.txt"
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
)


# --- Argument Parsing ---
parser = TrainOptions()
opts = parser.parse()

# --- Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



sobel = Sobelxy().to(device)




# --- Gradient Accumulation ---
accumulation_steps = 2

# --- Model Setup ---
encoder_model = nn.DataParallel(Encoder()).to(device)
decoder_model = nn.DataParallel(Decoder()).to(device)
fusion_model = nn.DataParallel(FusionMoudle()).to(device)
Temporary_model = nn.DataParallel(A2B()).to(device)

encoder_model.train()
decoder_model.train()
fusion_model.train()
Temporary_model.train()

# --- Losses and Optimizers ---
criteria_fusion = Fusionloss()
MSELoss = nn.MSELoss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

optimizer1 = torch.optim.Adam(encoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer2 = torch.optim.Adam(decoder_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer3 = torch.optim.Adam(fusion_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
optimizer4 = torch.optim.Adam(Temporary_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=opts.step_size, gamma=opts.gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=opts.step_size, gamma=opts.gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=opts.step_size, gamma=opts.gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=opts.step_size, gamma=opts.gamma)

# --- DataLoader ---
target_image_size = (128, 128)
trainloader = DataLoader(
    H5Datasets(opts.train_data, image_size=target_image_size),
    batch_size=opts.batch_size,
    shuffle=True,
    num_workers=0
)
loader = {'train': trainloader}

# --- Training Setup ---
scaler = torch.amp.GradScaler()
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
prev_time = time.time()

# --- Training Loop ---
for epoch in range(opts.total_epoch):
    torch.cuda.empty_cache()

    epoch_loss = 0.0
    batch_count = 0
    epoch_start_time = time.time()

    for i, (vi_rgb, ir_rgb) in enumerate(loader['train']):
        vi_y, _, _ = RGB2YCbCr(vi_rgb)
        ir_y = ir_rgb
        vi_y, ir_y = vi_y.to(device), ir_y.to(device)

        with torch.amp.autocast(device_type='cuda'):
            if epoch < opts.gap_epoch:
                vi_share, vi_private, ir_share, ir_private = encoder_model(vi_y, ir_y)
                loss_decomp = Temporary_model(vi_share, ir_share, vi_y, ir_y)
                vi_hat = decoder_model(vi_share, vi_private)
                ir_hat = decoder_model(ir_share, ir_private)
                loss_recon = (5 * Loss_ssim(vi_y, vi_hat) + MSELoss(vi_y, vi_hat)) + \
                             (5 * Loss_ssim(ir_y, ir_hat) + MSELoss(ir_y, ir_hat))
                loss = (5 * loss_decomp + 2 * loss_recon) / accumulation_steps
            else:
                vi_share, vi_private, ir_share, ir_private = encoder_model(vi_y, ir_y)
                fuse_share, fuse_private = fusion_model(vi_share, vi_private, ir_share, ir_private)
                out = decoder_model(fuse_share, fuse_private)
                loss, _, _ = criteria_fusion(vi_y, ir_y, out)

                # Edge-preserving loss (Sobel)
                edges_out = sobel(out)
                edges_vi = sobel(vi_y)
                edges_ir = sobel(ir_y)
                edge_loss = F.l1_loss(edges_out, edges_vi) + F.l1_loss(edges_out, edges_ir)
                # Optional: smoothness loss
                # smooth_loss = F.l1_loss(avg_filter(out, 5, 1.5), avg_filter(vi_y, 5, 1.5))
                # Combine losses
                lambda_edge = 0.1
                # lambda_smooth = 0.05
                loss = (loss + lambda_edge * edge_loss) / accumulation_steps

                # loss = loss / accumulation_steps   => Comment out to introduce this upper block of code of SobelXy

        scaler.scale(loss).backward()

        # Gradient Accumulation
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader['train']):
            if epoch < opts.gap_epoch:
                optimizers = [optimizer1, optimizer2, optimizer4]
                models = [encoder_model, decoder_model, Temporary_model]
            else:
                optimizers = [optimizer1, optimizer2, optimizer3]
                models = [encoder_model, decoder_model, fusion_model]

            # Unscale and clip
            for opt in optimizers:
                scaler.unscale_(opt)
            for model in models:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01, norm_type=2)

            # Step
            for opt in optimizers:
                scaler.step(opt)

            scaler.update()

            # Zero grad
            for opt in optimizers:
                opt.zero_grad()

        # --- Logging ---

        # Compute and accumulate loss
        epoch_loss += loss.item()
        batch_count += 1

        batches_done = epoch * len(loader['train']) + i
        batches_left = opts.total_epoch * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f]  ETA: %s"
            % (
                epoch+1,
                opts.total_epoch,
                i + 1,
                len(loader['train']),
                loss.item(),
                str(time_left).split('.')[0],
            )
        )
        # sys.stdout.flush()

    # --- After all batches in this epoch ---
    avg_loss = epoch_loss / batch_count
    epoch_duration = str(datetime.timedelta(seconds=time.time() - epoch_start_time)).split('.')[0]
    log_msg = f"[Epoch {epoch + 1}/{opts.total_epoch}] Avg Loss: {avg_loss:.6f} | Duration: {epoch_duration}"
    print(log_msg)
    logging.info(log_msg)


    # --- Scheduler Step ---
    scheduler1.step()
    scheduler2.step()
    if epoch < opts.gap_epoch:
        scheduler4.step()
    else:
        scheduler3.step()

    # --- LR Clamping ---
    for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
        for param_group in optimizer.param_groups:
            if param_group['lr'] < 1e-6:
                param_group['lr'] = 1e-6

# --- Save Final Model ---
checkpoint = {
    'encoder': encoder_model.state_dict(),
    'decoder': decoder_model.state_dict(),
    'fuse': fusion_model.state_dict(),
}
torch.save(checkpoint, "FDFuse_model_2.pth")
print("\nTraining completed. Model saved to FDFuse_model_2.pth")
