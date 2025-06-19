from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import pydicom
import glob
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from utils import *
import random

class MicroUSSagittalImageFolder(Dataset):

    def __init__(self, root_list, axis_distance, scale = 8, repeat = 1, select_k = None):
        self.axis_distance = axis_distance
        self.repeat = repeat
        self.scale = scale
        self.root_list = root_list
        self.img_list = []
        self.meta_list = []
        for n, root_path in enumerate(self.root_list):
            filenames = sorted(glob.glob(os.path.join(root_path, '*.dcm')))
            filenames = trim_files(filenames)
            num_files = len(filenames)
            (h_s, h_e), (w_s, w_e) = get_imaging_range(filenames[0])
            
            h, w = h_e - h_s, w_e - w_s
            pixelspacing = pydicom.dcmread(filenames[0]).PixelSpacing[0]
            h_expand = int(axis_distance / pixelspacing) + h

            imgs = torch.zeros((num_files, h, w))
            thetas = torch.zeros((num_files, 1))
            print('Slice for this case', root_path,':', len(filenames))
            for idx, filename in tqdm(enumerate(filenames), desc = 'Loading DICOMs of case ' + str(n+1) + '...', leave = False):
                dicom = pydicom.dcmread(filename)
                imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e] / 255) - 1, dtype = torch.float32)
                thetas[idx] = dicom.SliceLocation * np.pi / 180
            
            thetas = thetas - (thetas.max() + thetas.min()) / 2  # theta calibration
            theta_min, theta_max = thetas.min(), thetas.max()
            thetas, indices = torch.sort(thetas, dim=0)
            imgs = imgs[indices.squeeze(1)]
            # imgs = imgs.flip(0)

            self.img_list += [imgs[i] for i in range(num_files)]

            meta_info = {
                'theta_min': theta_min,
                'theta_max': theta_max,
                'orig_size': (h, w),
                'h_expand': h_expand,
                'pixelspacing': pixelspacing,
                'hr_theta_num': num_files * scale,
                'lr_theta_num': num_files,
                'scale': scale,
                'axis_distance': axis_distance
            }
            # print(theta_min, theta_max)
            self.meta_list += [meta_info for _ in range(num_files)]
        if select_k: 
            self.img_list = self.img_list[:select_k]
            self.meta_list = self.meta_list[:select_k]
            # self.imgs = random.choices(self.imgs, k = select_k)
        
        
    def __len__(self):
        return len(self.img_list) * self.repeat

    def __getitem__(self, idx):
        x = self.img_list[idx % len(self.img_list)]  
        meta_info = self.meta_list[idx % len(self.img_list)]  
        return x, meta_info



class MicroUSSagittalPatchWrapper(Dataset):

    def __init__(self, dataset, patch_size = 256, random_sampling = True):
        self.dataset = dataset
        self.patch_size = patch_size
        self.scale = self.dataset.scale
        self.random_sampling = random_sampling
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, meta_info = self.dataset[idx]  # 833, 1372
        theta_max, theta_min = 1.57, -1.57 #meta_info['theta_max'], meta_info['theta_min']
        axis_distance, pixelspacing = meta_info['axis_distance'], meta_info['pixelspacing']
        h, w = meta_info['orig_size']
        h_expand = meta_info['h_expand']
        
        # hr_theta_num, lr_theta_num = meta_info['hr_theta_num'], meta_info['lr_theta_num'] # 8*N, N
        lr_theta_num = random.randint(150,500)
        hr_patch_theta_num, lr_patch_theta_num = self.patch_size, int(self.patch_size / self.scale) # 256, 32
        
        if self.random_sampling:
            lr_thetas = torch.sort(torch.rand(lr_theta_num - 2) * (theta_max - theta_min) + theta_min).values
            lr_thetas = torch.cat((torch.tensor([theta_min]), lr_thetas, torch.tensor([theta_max])))
        else:
            lr_thetas = torch.linspace(theta_min, theta_max, lr_theta_num)
            
        lr_patch_thetas_start = random.randint(0, lr_theta_num - lr_patch_theta_num)
        lr_thetas_patch = lr_thetas[lr_patch_thetas_start : lr_patch_thetas_start + lr_patch_theta_num] # 32
        patch_theta_max, patch_theta_min = lr_thetas_patch.max(), lr_thetas_patch.min()
        hr_thetas_patch = torch.linspace(patch_theta_min, patch_theta_max, hr_patch_theta_num)
        
        radius = (torch.flip(torch.arange(h), dims = (0,)) + int(axis_distance / pixelspacing))# / h_expand
        patch_radius_start = random.randint(0, len(radius) - hr_patch_theta_num)
        radius_patch = radius[patch_radius_start : patch_radius_start + hr_patch_theta_num].float()
        
        lr_patch, hr_patch = extract_patches(img, lr_thetas_patch, hr_thetas_patch, radius_patch)
        hr_grids = torch.stack(torch.meshgrid([radius_patch / h_expand, hr_thetas_patch], indexing="ij"), dim=-1)
        lr_grids = torch.stack(torch.meshgrid([radius_patch / h_expand, lr_thetas_patch], indexing="ij"), dim=-1)
        hr_inte = polar_intepolation_1d(lr_patch, lr_grids, self.patch_size)
        
        hr_grids = hr_grids.permute(2,0,1)
        lr_grids = lr_grids.permute(2,0,1)
        lr_radius, lr_thetas = lr_grids[0,:,:], lr_grids[1,:,:]
        lr_thetas_pad = F.pad(lr_thetas, (1,1), 'replicate')
        lr_thetas_left_gap = (lr_thetas_pad[:,1:-1] - lr_thetas_pad[:,:-2]).unsqueeze(-3)
        lr_thetas_right_gap = (lr_thetas_pad[:,2:] - lr_thetas_pad[:,1:-1]).unsqueeze(-3)
        lr_grids = torch.cat([lr_radius.unsqueeze(-3), lr_thetas_left_gap, lr_thetas_right_gap], dim = -3)
        # hr_inte = transforms.ToTensor()(transforms.ToPILImage()(lr_patch).resize((self.patch_size, self.patch_size)))

        return {'lr_img': lr_patch,
                'lr_grids': lr_grids,
                'hr_img': hr_patch,
                'hr_grids': hr_grids,
                'hr_inte': hr_inte,
                'meta_info': meta_info}


class MicroUSAxialImageFolder(Dataset):

    def __init__(self, root_path, axis_distance, scale = 8):
        self.axis_distance = axis_distance
        self.scale = scale
        self.filenames = sorted(glob.glob(os.path.join(root_path, '*.dcm')))
        self.filenames = trim_files(self.filenames)
        self.num_files = len(self.filenames)

        (h_s, h_e), (w_s, w_e) = get_imaging_range(self.filenames[0])
        
        self.h, self.w = h_e - h_s, w_e - w_s
        self.h = (self.h // 16) * 16
        self.pixelspacing = pydicom.dcmread(self.filenames[0]).PixelSpacing[0]
        self.h_expand = int(axis_distance / self.pixelspacing) + self.h

        self.imgs = torch.zeros((self.num_files, self.h, self.w))
        thetas = torch.zeros((self.num_files, 1))
        for idx, filename in tqdm(enumerate(self.filenames), desc = 'Loading DICOMs...', leave = False):
            dicom = pydicom.dcmread(filename)
            self.imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e][ : self.h] / 255) - 1, dtype = torch.float32)
            thetas[idx] = dicom.SliceLocation * np.pi / 180
        thetas = thetas - (thetas.max() + thetas.min()) / 2  # theta calibration
        self.theta_min, self.theta_max = thetas.min(), thetas.max()
        thetas, indices = torch.sort(thetas, dim=0)
        self.imgs = self.imgs[indices.squeeze(1)]

        self.axial_planes = []
        radius = (torch.flip(torch.arange(self.h), dims = (0,)) + int(self.axis_distance / self.pixelspacing))
        self.lr_grids = torch.stack(torch.meshgrid([radius / self.h_expand, thetas.squeeze(-1)], indexing="ij"), dim=-1)
        
        lr_grids_gap = self.lr_grids.permute(2,0,1)
        lr_radius, lr_thetas = lr_grids_gap[0,:,:], lr_grids_gap[1,:,:]
        lr_thetas_pad = F.pad(lr_thetas, (1,1), 'replicate')
        lr_thetas_left_gap = (lr_thetas_pad[:,1:-1] - lr_thetas_pad[:,:-2]).unsqueeze(-3)
        lr_thetas_right_gap = (lr_thetas_pad[:,2:] - lr_thetas_pad[:,1:-1]).unsqueeze(-3)
        self.lr_grids_gap = torch.cat([lr_radius.unsqueeze(-3), lr_thetas_left_gap, lr_thetas_right_gap], dim = -3)
        
        
        hr_thetas = torch.linspace(self.theta_min, self.theta_max, self.num_files * self.scale)
        self.hr_grids = torch.stack(torch.meshgrid([radius / self.h_expand, hr_thetas], indexing="ij"), dim=-1)
        for i in tqdm(range(self.w), desc = 'Converting to axial data...', leave = False):
            axial_slice = torch.zeros((1, self.h, self.num_files)) # [theta, r - dist, intensity]
            for j in range(self.num_files):
                axial_slice[:, :, j] = self.imgs[j][:, self.w - i - 1]
            self.axial_planes.append(axial_slice)

        self.meta_info = {
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'orig_size': (self.h, self.w),
            'h_expand': self.h_expand,
            'pixelspacing': self.pixelspacing,
            'hr_theta_num': len(self.imgs) * self.scale,
            'lr_theta_num': len(self.imgs),
            'scale': self.scale,
            'axis_distance': self.axis_distance
        }
        
    def __len__(self):
        return len(self.axial_planes)

    def __getitem__(self, idx):
        lr_img = self.axial_planes[idx] 
        hr_inte = polar_intepolation_1d(lr_img, self.lr_grids, hr_num = self.num_files * self.scale)
        return {'lr_img': lr_img,
                'lr_grids': self.lr_grids_gap,
                'hr_inte': hr_inte,
                'hr_grids': self.hr_grids,
                'meta_info': self.meta_info}
    
    
# class MicroUSImageFolder(Dataset):

#     def __init__(self, root_path, axis_distance, repeat = 1):
#         self.axis_distance = axis_distance
#         self.repeat = repeat
#         self.filenames = sorted(glob.glob(os.path.join(root_path, '*.dcm')))
#         num_files = len(self.filenames)

#         (h_s, h_e), (w_s, w_e) = get_imaging_range(self.filenames[0])
        
#         self.h, self.w = h_e - h_s, w_e - w_s
#         self.pixelspacing = pydicom.dcmread(self.filenames[0]).PixelSpacing[0]
#         self.h_expand = int(axis_distance / self.pixelspacing) + self.h

#         imgs = torch.zeros((num_files, self.h, self.w))
#         thetas = torch.zeros((num_files, 1))
#         for idx, filename in tqdm(enumerate(self.filenames), desc = 'Loading DICOMs...', leave = False):
#             dicom = pydicom.dcmread(filename)
#             imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e] / 255) - 1, dtype = torch.float32)
#             thetas[idx] = dicom.SliceLocation * np.pi / 180
#         self.theta_min, self.theta_max = thetas.min(), thetas.max()
#         thetas, indices = torch.sort(thetas, dim=0)
#         imgs = imgs[indices.squeeze(1)]

#         self.axial_planes = []
#         for i in tqdm(range(self.w), desc = 'Converting to axial data...', leave = False):
#             axial_slice = torch.zeros((self.h, num_files, 3)) # [theta, r - dist, intensity]
#             for j in range(num_files):
#                 axial_slice[:, j, 0] = imgs[j][:, self.w - i - 1]
#                 axial_slice[:, j, 1] = (torch.flip(torch.arange(self.h), dims = (0,)) + int(axis_distance / self.pixelspacing)) / self.h_expand
#                 axial_slice[:, j, 2] = torch.sin(thetas[j])
#                 # axial_slice[:, j, 3] = (self.w - i - 1) / self.w
#             self.axial_planes.append(axial_slice)

#         self.meta_info = {
#             'theta_min': self.theta_min,
#             'theta_max': self.theta_max,
#             'orig_size': (self.h, self.w),
#             'h_expand': self.h_expand,
#             'pixelspacing': self.pixelspacing
#         }
        
#     def __len__(self):
#         return len(self.axial_planes * self.repeat)

#     def __getitem__(self, idx):
#         x = self.axial_planes[idx % len(self.axial_planes)]  
#         return x, self.meta_info


# class MicroUSPatchWrapper(Dataset):

#     def __init__(self, dataset, inp_size = None, scale_min = 1, scale_max = None, sample_q = None):
#         self.dataset = dataset
#         self.inp_size = inp_size
#         self.scale_min = scale_min
#         if scale_max is None:
#             scale_max = scale_min
#         self.scale_max = scale_max
#         self.sample_q = sample_q

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img, meta_info = self.dataset[idx]
#         s_h = random.uniform(self.scale_min, self.scale_max)
#         s_w = random.uniform(self.scale_min, self.scale_max)

#         if self.inp_size is None:
#             h_lr = math.floor(img.shape[-2] / s_h + 1e-9)
#             w_lr = math.floor(img.shape[-1] / s_w + 1e-9)
#             img = img[:, :round(h_lr * s_h), :round(w_lr * s_w)] # assume round int
#             img_down = random_downsample(img, h_hr, w_hr, h_lr, w_lr)
#             crop_lr, crop_hr = img_down, img
#         else:
#             h_lr, w_lr = self.inp_size, self.inp_size
#             h_hr, w_hr = round(h_lr * s_h), round(w_lr * s_w)
#             x0 = random.randint(0, img.shape[-3] - h_hr)
#             y0 = random.randint(0, img.shape[-2] - w_hr)
#             crop_hr = img[x0 : x0 + h_hr, y0 : y0 + w_hr, :]
#             crop_lr = random_downsample(crop_hr, h_hr, w_hr, h_lr, w_lr)

#         lr_coord, lr_intensities = to_pixel_samples(crop_lr.contiguous())
#         hr_coord, hr_intensities = to_pixel_samples(crop_hr.contiguous())
        

#         if self.sample_q is not None:
#             sample_lst = np.random.choice(
#                 len(hr_coord), self.sample_q, replace=False)
#             hr_coord = hr_coord[sample_lst]
#             hr_intensities = hr_intensities[sample_lst]

#         return {
#             'inp': crop_lr.permute(2, 0, 1),
#             'lr_coord': lr_coord,
#             'hr_coord': hr_coord,
#             'gt': hr_intensities
#         }
    
# class MicroUSDataset(Dataset):
#     def __init__(self, dataroot):
#         self.filenames = sorted(glob.glob(os.path.join(dataroot, '*.dcm')))
#         self.dicoms = []
#         self.voxels = []
#         for filename in self.filenames:
#             dicom = pydicom.dcmread(filename)
#             self.dicoms.append(dicom)
#         self.dicoms.sort(key=lambda dicom: float(dicom.SliceLocation))
#         theta_min, theta_max = self.dicoms[0].SliceLocation, self.dicoms[-1].SliceLocation
#         for dicom in tqdm(self.dicoms, desc = 'Loading DICOMs...', leave = False):
#             img, theta= dicom.pixel_array, dicom.SliceLocation
#             img = img[64:897 :] / 255
#             self.voxels += img_to_voxels(img, theta, theta_min, theta_max)

#     def __len__(self):
#         return len(self.voxels)
    
#     def __getitem__(self, index):
#         data = self.voxels[index]
#         coords = torch.tensor(data['coords'], dtype = torch.float32)
#         intensity = torch.tensor(data['intensity'], dtype = torch.float32)
#         return {
#             'coords': coords,
#             'intensity': intensity
#         }
        
# class MicroUSDatasetLazily(Dataset):
#     def __init__(self, dataroot, imaging_range, coord_mode, axis_distance = 15, sample_size = None):
#         self.sample_size = sample_size
#         self.axis_distance = axis_distance
#         self.coord_mode = coord_mode
#         self.imaging_start, self.imaging_end = imaging_range
#         self.filenames = sorted(glob.glob(os.path.join(dataroot, '*.dcm')))
#         num_files = len(self.filenames)
#         self.h, self.w = pydicom.dcmread(self.filenames[0]).pixel_array[self.imaging_start:self.imaging_end :].shape
#         self.pixelspacing = pydicom.dcmread(self.filenames[0]).PixelSpacing[0]
#         self.h_expand = int(axis_distance / self.pixelspacing) + self.h
#         imgs = torch.zeros((num_files, self.h, self.w))
#         thetas = torch.zeros((num_files, 1))
#         for idx, filename in tqdm(enumerate(self.filenames), desc = 'Loading DICOMs...', leave = False):
#             dicom = pydicom.dcmread(filename)
#             imgs[idx] = torch.tensor(dicom.pixel_array[self.imaging_start:self.imaging_end :] / 255, dtype = torch.float32)
#             thetas[idx] = dicom.SliceLocation
#         self.theta_min, self.theta_max = thetas.min(), thetas.max()
#         self.axial_planes = []
#         for i in tqdm(range(self.w), desc = 'Converting to axial data...', leave = False):
#             axial_slice = torch.zeros((num_files, self.h, 3)) # [theta, r - dist, intensity]
#             for j in range(num_files):
#                 axial_slice[j, :, 0] = thetas[j]
#                 axial_slice[j, :, 1] = torch.arange(self.h)
#                 axial_slice[j, :, 2] = imgs[j][:, i]
#             self.axial_planes.append(axial_slice)

#     def __len__(self):
#         return len(self.axial_planes)
    
#     def __getitem__(self, index):
#         axial_data = self.axial_planes[index] # 300, 962, 3
#         axial_data = axial_data.view(-1, 3) # 300*962, 3
#         sample_lst = np.random.choice(len(axial_data), self.sample_size, replace = False)
#         theta = axial_data[sample_lst, 0]
#         r_minus_dist = axial_data[sample_lst, 1]
#         intensity = axial_data[sample_lst, 2]
#         z_index = index * torch.ones_like(intensity)
#         if self.coord_mode == 'cartesian':
#             coords = pixels_to_voxels(theta, r_minus_dist, z_index, self.h_expand, self.w, self.theta_min, self.theta_max)
#         elif self.coord_mode == 'polar':
#             coord_theta = theta * torch.pi / 180
#             coord_r = (self.h_expand -  r_minus_dist) / self.h_expand
#             coord_z = ((self.w - index - 1) / self.w) * torch.ones_like(intensity)
#             coords = torch.stack([coord_theta, coord_r, coord_z], dim = -1)
#         return {
#             'coords': coords,
#             'intensity': intensity
#         }
    