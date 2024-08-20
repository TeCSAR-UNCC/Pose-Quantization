from torchvision.models import inception_v3
from torchvision import transforms
from tqdm import tqdm
import torch
import numpy as np
from scipy.linalg import sqrtm
from piq import ssim

def compute_metrics(gt, pred):
    gt, pred = normalize_images(gt, pred)
    metrics = {}
    std = []

    psnr = calculate_psnr(gt, pred)
    std.append(psnr[-1])
    metrics['psnr'] = psnr[0]

    ssim = calculate_ssim(gt, pred)
    std.append(ssim[-1])
    metrics['ssim'] = ssim[0]

    l1 = calculate_l1(gt, pred)
    std.append(l1[-1])
    metrics['l1'] = l1[0]

    metrics['std'] = np.array(std).mean()

    return metrics

# def normalize_images(gt, pred, range=(0, 1)):
#     """
#     Normalize images based on the global min and max values of both images.
    
#     Args:
#         gt (torch.Tensor): Ground truth image tensor.
#         pred (torch.Tensor): Predicted image tensor.
#         range (tuple): Desired output range for normalization (default: (0, 1)).
    
#     Returns:
#         normalized_gt, normalized_pred: Normalized ground truth and predicted images.
#     """
#     # Concatenate both images along a new axis to find global min and max
#     stacked_images = torch.cat((gt, pred), dim=0)
    
#     # Compute global min and max across both images
#     global_min = stacked_images.min()
#     global_max = stacked_images.max()
    
#     # Normalize both images using the global min and max
#     normalized_gt = (gt - global_min) / (global_max - global_min)
#     normalized_pred = (pred - global_min) / (global_max - global_min)
    
#     # Scale to the desired range
#     if range != (0, 1):
#         normalized_gt = normalized_gt * (range[1] - range[0]) + range[0]
#         normalized_pred = normalized_pred * (range[1] - range[0]) + range[0]
    
#     return normalized_gt, normalized_pred

def normalize_images(gt, pred, range=(0, 1)):
    normalized_gt = torch.clip(gt, min=0.0, max=1.0)
    normalized_pred = torch.clip(pred, min=0.0, max=1.0)

    # normalized_gt = (gt - gt.min()) / (gt.max() - gt.min())
    # normalized_pred = (pred - pred.min()) / (pred.max() - pred.min())
    
    # # Scale to the desired range
    # if range != (0, 1):
    #     normalized_gt = normalized_gt * (range[1] - range[0]) + range[0]
    #     normalized_pred = normalized_pred * (range[1] - range[0]) + range[0]
    
    return normalized_gt, normalized_pred

def normalize_array(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def calculate_l1(gt, pred):
    l1_fn = torch.nn.L1Loss(reduction='none')
    l1_array = l1_fn(gt, pred).mean(dim=(0,1,3,4))

    l1_mean, l1_std = l1_array.mean(), normalize_array(l1_array).std()

    return [l1_mean.item(), l1_std.item()]

# from skimage.metrics import structural_similarity as ssim

def calculate_ssim(image1, image2):
    b, c, f, x, y = image1.shape
    image1 = image1.view(b*f*c, 1, x, y)
    image2 = image2.view(b*f*c, 1, x, y)

    scores = []
    score = ssim(image1, image2, data_range=1, reduction='none')
    
    ssim_array = score.view(b, f, c).mean(dim=(0,2))
    ssim_mean, ssim_std = ssim_array.mean(), normalize_array(ssim_array).std()

    return [ssim_mean.item(), ssim_std.item()]


from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(gt, pred, data_range=1):
    mse_fn = torch.nn.MSELoss(reduction='none')
    mse = mse_fn(gt, pred).mean(dim=(0, 1, 3, 4))
    psnr_array = 20 * torch.log10(data_range / torch.sqrt(mse))

    psnr_mean, psnr_std = psnr_array.mean(), normalize_array(psnr_array).std()

    return [psnr_mean.item(), psnr_std.item()]


if __name__ == "__main__":
    t_2d = torch.rand(1, 1, 64, 128, 128)
    t2_2d = torch.rand(1, 1, 64, 128, 128)

    m2d_m2d = torch.rand(1, 3, 64, 128, 128)
    m2d_m2d2 = torch.rand(1, 3, 64, 128, 128)

    t3_3d = torch.rand(1, 3, 64, 128, 128)
    t3_3d2 = torch.rand(1, 3, 64, 128, 128)

    print(compute_metrics(t_2d, t2_2d))
    print(compute_metrics(m2d_m2d, m2d_m2d2))
    print(compute_metrics(t3_3d, t3_3d2))

    print()