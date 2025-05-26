# coding=utf-8

import cv2
import math
import os
from PIL import Image
import numpy as np
import torch
import lpips
from torchvision import transforms
from tqdm import tqdm


def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为[0,1]
        transforms.Normalize((0.5,), (0.5,))  # 转为[-1,1]
    ])
    return transform(img).unsqueeze(0)  # 添加 batch 维度

# ----------
# PSNR
# ----------


def calculate_psnr(img1, img2, border_percent=0.05):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    border_h = int(h*border_percent)
    border_w = int(w*border_percent)
    img1 = img1[border_h:h-border_h, border_w:w-border_w]
    img2 = img2[border_h:h-border_h, border_w:w-border_w]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# ----------
# PSNR (on Y channel)
# ----------
def calculate_psnr_y(img1, img2, border_percent=0.05):
    # Convert RGB to Y channel
    img1_y = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    if not img1_y.shape == img2_y.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1_y.shape
    border_h = int(h * border_percent)
    border_w = int(w * border_percent)
    img1_y = img1_y[border_h:h-border_h, border_w:w-border_w]
    img2_y = img2_y[border_h:h-border_h, border_w:w-border_w]

    img1_y = img1_y.astype(np.float64)
    img2_y = img2_y.astype(np.float64)
    mse = np.mean((img1_y - img2_y) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ----------
# SSIM
# ----------


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# ----------
# SSIM (on Y channel)
# ----------
def calculate_ssim_y(img1, img2, border=0):
    # Convert RGB to Y channel
    img1_y = cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    img2_y = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)[:, :, 0]

    if not img1_y.shape == img2_y.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1_y.shape
    img1_y = img1_y[border:h-border, border:w-border]
    img2_y = img2_y[border:h-border, border:w-border]

    return ssim(img1_y, img2_y)


def ssim_y(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_psnr_ssim(img1, img2, border_percent=0.05):
    img1 = img1.convert("RGB")
    img1 = np.array(img1)
    img2 = np.array(img2)

    psnr_5 = calculate_psnr(img1, img2, border_percent=border_percent)
    psnr_0 = calculate_psnr(img1, img2, border_percent=0)
    ssim_ = calculate_ssim(img1, img2, border=0)
    return psnr_5, psnr_0, ssim_


def evaluate_metrics(gt_dir, pred_dir, output_txt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 LPIPS 模型
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    image_names = sorted(os.listdir(gt_dir))
    psnr_list = []
    ssim_list = []
    lpips_list = []

    with open(output_txt, 'w') as f:
        f.write("Image\tPSNR_Y\tSSIM_Y\tLPIPS\n")

        for name in tqdm(image_names):
            gt_path = os.path.join(gt_dir, name)
            pred_path = os.path.join(pred_dir, 'output_' + name)

            if not os.path.exists(pred_path):
                print(f"Warning: {name} not found in predicted dir.")
                continue

            # 读取图像并resize
            gt_img = Image.open(gt_path).convert('RGB')
            pred_img = Image.open(pred_path).convert('RGB')
            gt_img = gt_img.resize(pred_img.size, Image.LANCZOS)

            # --- 计算 PSNR_Y 和 SSIM_Y ---
            gt_np = np.array(gt_img)
            pred_np = np.array(pred_img)

            psnr_y = calculate_psnr_y(gt_np, pred_np)
            ssim_y = calculate_ssim_y(gt_np, pred_np)

            # --- 计算 LPIPS ---
            gt_tensor = image_to_tensor(gt_img).to(device)
            pred_tensor = image_to_tensor(pred_img).to(device)
            with torch.no_grad():
                lpips_val = loss_fn_alex(gt_tensor, pred_tensor).item()

            psnr_list.append(psnr_y)
            ssim_list.append(ssim_y)
            lpips_list.append(lpips_val)

            f.write(f"{name}\t{psnr_y:.4f}\t{ssim_y:.4f}\t{lpips_val:.4f}\n")

        # 平均值
        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        avg_lpips = sum(lpips_list) / len(lpips_list)

        f.write("\nAverage:\n")
        f.write(f"PSNR_Y: {avg_psnr:.4f}\n")
        f.write(f"SSIM_Y: {avg_ssim:.4f}\n")
        f.write(f"LPIPS: {avg_lpips:.4f}\n")

    print("Evaluation finished. Results saved to", output_txt)


# if __name__ == '__main__':
#     for num in range(600, 850, 100):
#         predicted_dir = os.path.join(
#             './results/train_with_prompt_x4/test_without_prompt_'+str(num))
#         groundtruth_dir = './datasets/DIV2K_valid_HR'
#         output_txt = os.path.join(
#             './results/train_with_prompt_x4/test_without_prompt_'+str(num)+'/psnr_ssim_lpips_y.txt')
#         evaluate_metrics(groundtruth_dir, predicted_dir, output_txt)
