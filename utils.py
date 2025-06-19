from torch.utils.data import Dataset
import torch
from torchvision import transforms
from PIL import Image
import os
import pydicom
import glob
import math, random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator

def trim_files(filenames):
    angles_temp = []
    for filename in filenames:
        angles_temp.append(pydicom.dcmread(filename).SliceLocation)
    start_idx = angles_temp.index(max(angles_temp))
    trimmed_files = filenames[start_idx:]
    remainder = len(trimmed_files) % 4
    if remainder != 0:
        trimmed_files = trimmed_files[remainder : ]
    return trimmed_files

def img_to_voxels(img, theta, theta_min, theta_max):
    theta, theta_min, theta_max = math.radians(theta), math.radians(theta_min), math.radians(theta_max)
    h, w = img.shape
    voxels = []
    orig_y = h * math.sin(abs(theta_min))
    vol_h, vol_w, vol_d = h, h * math.sin(abs(theta_min)) + h * math.sin(abs(theta_max)), w
    for i in range(h):
        for j in range(w):
            x = (vol_h - (h - i) * math.cos(theta)) / vol_h
            y = (orig_y + (h - i) * math.sin(theta)) / vol_w
            z = (w - j) / vol_d
            voxels.append(dict(coords = [x,y,z], intensity = img[i][j]))
    return voxels

def make_coord(h,w):
    """ Make coordinates at grid centers.
    """
    coord_seqs = [torch.arange(h).float(), torch.arange(w).float()]
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    ret = ret.view(-1, ret.shape[-1])
    return ret

def pixels_to_voxels(theta, r_minus_dist, index, h, w, theta_min, theta_max):
    theta, theta_min, theta_max = theta * torch.pi / 180, math.radians(theta_min), math.radians(theta_max)
    orig_y = h * math.sin(abs(theta_min))
    vol_h, vol_w, vol_d = h, h * math.sin(abs(theta_min)) + h * math.sin(abs(theta_max)), w
    x = (vol_h - (h - r_minus_dist) * torch.cos(theta)) / vol_h
    y = (orig_y + (h - r_minus_dist) * torch.sin(theta)) / vol_w
    z = (w - index - 1) / vol_d
    return torch.stack([x,y,z], dim = -1)


def random_downsample(img, h_hr, w_hr, h_lr, w_lr):
    rows = torch.randperm(h_hr)[ : h_lr].sort().values
    columns = torch.randperm(w_hr)[ : w_lr].sort().values
    return img[rows, :, :][:, columns, :]
    
def to_pixel_samples(img):
    coords = img[:, :, 1:].view(-1, 2)
    intensities = img[:, :, 0].view(-1, 1)
    return coords, intensities
    
def get_imaging_range(dicom_path):
    img = pydicom.dcmread(dicom_path).pixel_array
    h_orig, w_orig = img.shape
    h_s, h_e, w_s, w_e = 0, h_orig - 1, 0, w_orig - 1

    row_sums = img.sum(axis = 1)
    non_black_rows = np.nonzero(row_sums)[0]
    if non_black_rows.size > 0:
        h_s = non_black_rows[0]
        h_e = non_black_rows[-1] + 1

    column_sums = img.sum(axis = 0)
    non_black_columns = np.nonzero(column_sums)[0]
    if non_black_columns.size > 0:
        w_s = non_black_columns[0]
        w_e = non_black_columns[-1] + 1
    
    return (h_s, h_e), (w_s, w_e)

def extract_patches(img, lr_thetas_patch, hr_thetas_patch, radius_patch):
    h, w = img.shape
    theta_max, theta_min = lr_thetas_patch.max(), lr_thetas_patch.min()
    patch_length = len(radius_patch)
    r_base = radius_patch[-1]
    
    if theta_max < 0 and theta_min < 0:
        x_start_min = (r_base + patch_length) * torch.cos(theta_max)
        x_start_max = r_base * torch.cos(theta_min) + h - 1
        y_start_min = -(r_base + patch_length) * torch.sin(theta_min)
        y_start_max = w - 1 - (r_base) * torch.sin(theta_max)
    elif theta_max > 0 and theta_min > 0:
        x_start_min = (r_base + patch_length) * torch.cos(theta_min)
        x_start_max = r_base * torch.cos(theta_max) + h - 1
        y_start_min = -r_base * torch.sin(theta_min)
        y_start_max = w - 1 - (r_base + patch_length) * torch.sin(theta_max)
    elif theta_min <= 0 and theta_max >= 0:
        x_start_min = r_base + patch_length
        x_start_max = min(r_base * torch.cos(theta_max) + h - 1,
                      r_base * torch.cos(theta_min) + h - 1)
        y_start_min = -(r_base + patch_length) * torch.sin(theta_min)
        y_start_max = w - 1 - (r_base + patch_length) * torch.sin(theta_max)
    
    # if x_start_min > x_start_max or y_start_min > y_start_max:
    #     print(x_start_min, x_start_max)
    #     print()
    assert x_start_min <= x_start_max and y_start_min <= y_start_max
    x_center, y_center = random.uniform(x_start_min, x_start_max), random.uniform(y_start_min, y_start_max)
        
    lr_x_coords = x_center - torch.outer(radius_patch, torch.cos(lr_thetas_patch))
    lr_y_coords = y_center + torch.outer(radius_patch, torch.sin(lr_thetas_patch))
    hr_x_coords = x_center - torch.outer(radius_patch, torch.cos(hr_thetas_patch))
    hr_y_coords = y_center + torch.outer(radius_patch, torch.sin(hr_thetas_patch))
    
    lr_x_coords_norm = 2 * (lr_x_coords / (h - 1)) - 1
    lr_y_coords_norm = 2 * (lr_y_coords / (w - 1)) - 1
    hr_x_coords_norm = 2 * (hr_x_coords / (h - 1)) - 1
    hr_y_coords_norm = 2 * (hr_y_coords / (w - 1)) - 1
    
    assert lr_x_coords_norm.max() <= 1, f"Assertion failed: max value of lr_x_coords_norm({lr_x_coords_norm}) > 1"
    assert lr_x_coords_norm.min() >= -1, f"Assertion failed: min value of lr_x_coords_norm({lr_x_coords_norm}) < -1"
    assert hr_x_coords_norm.max() <= 1, f"Assertion failed: max value of hr_x_coords_norm({hr_x_coords_norm}) > 1"
    assert hr_x_coords_norm.min() >= -1, f"Assertion failed: min value of hr_x_coords_norm({hr_x_coords_norm}) < -1"
    
    
    assert lr_y_coords_norm.max() <= 1, f"Assertion failed: max value of lr_y_coords_norm({lr_y_coords_norm}) > 1"
    assert lr_y_coords_norm.min() >= -1, f"Assertion failed: min value of lr_y_coords_norm({lr_y_coords_norm}) < -1"
    assert hr_y_coords_norm.max() <= 1, f"Assertion failed: max value of hr_y_coords_norm({hr_y_coords_norm}) > 1"
    assert hr_y_coords_norm.min() >= -1, f"Assertion failed: min value of hr_y_coords_norm({hr_y_coords_norm}) < -1"
    
    img = img.view(1,1,h,w)
    lr_grids = torch.stack([lr_x_coords_norm, lr_y_coords_norm], dim = -1).unsqueeze(0)
    lr_sampled_patch = F.grid_sample(img, lr_grids, align_corners = True)
    hr_grids = torch.stack([hr_x_coords_norm, hr_y_coords_norm], dim = -1).unsqueeze(0)
    hr_sampled_patch = F.grid_sample(img, hr_grids, align_corners = True)

    return lr_sampled_patch.squeeze(0), hr_sampled_patch.squeeze(0)
    
def polar_intepolation_1d(lr_patch, lr_grids, hr_num):
    radius_num, thetas_num = lr_grids.shape[:2]
    lr_thetas = lr_grids[0,:,1].cpu()
    hr_thetas = torch.linspace(lr_thetas.min(), lr_thetas.max(), hr_num).cpu()
    inte = torch.zeros((1,radius_num,hr_num)).cpu()
    for i in range(radius_num):  
        interp_func = interp1d(lr_thetas, lr_patch[0, i, :].cpu(), kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated_values = torch.tensor(interp_func(hr_thetas))
        interpolated_values = torch.clamp(interpolated_values, min=-1.0, max=1.0)
        interpolated_values = torch.nan_to_num(interpolated_values, nan=-1.0)
        inte[0, i, :] = interpolated_values
    return inte

def polar2cartesian(polar_data, polar_coords, h_expand):
    h,w = polar_data.shape[-2:]
    radius, thetas = polar_coords[:,:,:,0] * h_expand, polar_coords[:,:,:,1]
    thetas = thetas - (thetas.max() + thetas.min()) / 2
    theta_max, theta_min = thetas.max(), thetas.min()
    r_start, r_end = radius.min(), radius.max()
    vol_h, vol_w = r_end - r_start * torch.cos((theta_max - theta_min) / 2), 2 * r_end * torch.sin((theta_max - theta_min) / 2)
    grid_h, grid_w = int(vol_h), int(vol_w)
    
    x_acquired = r_end - radius * torch.cos(thetas)
    y_acquired = vol_w / 2 + radius * torch.sin(thetas)
    slice_coord = make_coord(grid_h, grid_w).cuda()
    
    r = ((r_end - slice_coord[:,0])**2 + (vol_w / 2 - slice_coord[:,1])**2).sqrt().cuda()
    slice_theta = torch.arcsin((slice_coord[:, 1] - vol_w / 2) / r)

    mask_r = (r > r_start) & (r < r_end)
    mask_theta = (slice_theta > theta_min) & (slice_theta < theta_max)

    mask = mask_r & mask_theta
    slice_coords_cartisan = slice_coord[mask].long()
    X, Y = np.array(slice_coords_cartisan[:,0].cpu()), np.array(slice_coords_cartisan[:,1].cpu())
    interp = LinearNDInterpolator(list(zip(np.array(x_acquired.view(h*w).cpu()), np.array(y_acquired.view(h*w).cpu()))), polar_data[0].view(h*w).cpu(),fill_value=1.0)
    Z = interp(X, Y)
    
    recon_img = np.zeros((1,grid_h,grid_w))
    recon_img[:, X, Y] = Z
    recon_img = (recon_img * 255).clip(0,255).astype(np.uint8)
    return recon_img
    