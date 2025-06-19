import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from data.MicroUS_dataset import *
from torch.utils.data.dataloader import DataLoader
import random
from PIL import Image
import numpy as np
from utils import *
torch.manual_seed(42)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # datasetb 
    # for phase, dataset_opt in opt['datasets'].items():
    #     if phase == 'train' and args.phase != 'val':
    #         train_set = Data.create_dataset(dataset_opt, phase)
    #         train_loader = Data.create_dataloader(
    #             train_set, dataset_opt, phase)
    #     elif phase == 'val':
    #         val_set = Data.create_dataset(dataset_opt, phase)
    #         val_loader = Data.create_dataloader(
    #             val_set, dataset_opt, phase)
    axis_distance = 2 # 15
    dataroot = '/raid/kaifengpang/MicroUS_ToBeReformatted/096'
    # dataroot = '/raid/kaifengpang/MicroUS_ToBeReformatted/107/study/In-Vivo_US_scan_of_prostate'
    
    val_dataset = MicroUSAxialImageFolder(dataroot, axis_distance, scale = 8)
    special_slice = 600 # None
    val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = False, num_workers = 8)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])




    logger.info('Begin Model Evaluation.')

    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    if special_slice:
        for _,  val_data in enumerate(val_loader):
            idx += 1
            if idx == special_slice:
                print('Here we go!')
                diffusion.feed_data(val_data)
                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals()

                # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                inf_img, sr_img = (visuals['INF'] + 1) / 2, (visuals['SR'] + 1) / 2
                polar_coords = val_data['hr_grids']
                h_expand = val_data['meta_info']['h_expand'].cuda()
                inf_img = polar2cartesian(inf_img, polar_coords, h_expand)
                sr_img = polar2cartesian(sr_img, polar_coords, h_expand)
                Image.fromarray(inf_img[0]).convert('L').save('{}/{}_{}_ref.png'.format(result_path, current_step, idx))
                Image.fromarray(sr_img[0]).convert('L').save('{}/{}_{}_sr.png'.format(result_path, current_step, idx))
    else: 
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()

            # hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            # lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            inf_img, sr_img = (visuals['INF'] + 1) / 2, (visuals['SR'] + 1) / 2
            polar_coords = val_data['hr_grids']
            h_expand = val_data['meta_info']['h_expand'].cuda()
            inf_img = polar2cartesian(inf_img, polar_coords, h_expand)
            sr_img = polar2cartesian(sr_img, polar_coords, h_expand)
            Image.fromarray(inf_img[0]).convert('L').save('{}/{}_{}_ref.png'.format(result_path, current_step, idx))
            Image.fromarray(sr_img[0]).convert('L').save('{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        # sr_img_mode = 'grid'
        # if sr_img_mode == 'single':
        #     # single img series
        #     sr_img = visuals['SR']  # uint8
        #     sample_num = sr_img.shape[0]
        #     for iter in range(0, sample_num):
        #         Metrics.save_img(
        #             Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        # else:
        #     # grid img
        #     sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
        #     Metrics.save_img(
        #         sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
        #     Metrics.save_img(
        #         Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        # Metrics.save_img(
        #     hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
        # Metrics.save_img(
        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

        # generation
        # eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        # eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

        # avg_psnr += eval_psnr
        # avg_ssim += eval_ssim

        # if wandb_logger and opt['log_eval']:
        #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

    # avg_psnr = avg_psnr / idx
    # avg_ssim = avg_ssim / idx

    # # log
    # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    # logger_val = logging.getLogger('val')  # validation logger
    # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
    #     current_epoch, current_step, avg_psnr, avg_ssim))

    # if wandb_logger:
    #     if opt['log_eval']:
    #         wandb_logger.log_eval_table()
    #     wandb_logger.log_metrics({
    #         'PSNR': float(avg_psnr),
    #         'SSIM': float(avg_ssim)
    #     })
