# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation                                                                    
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import os
import argparse

from glob import glob
import numpy as np
from PIL import Image
import logging
import torch
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from marigold import CFDiffPipleline
from src.dataset import CFDDataset
from src.dataset.cfd_dataset import \
    (STAT_pressure, STAT_temperature, STAT_velocity)
EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


def get_args():
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/stable-diffusion-2",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--base_data_dir", 
        type=str, 
        default='data/case_data2/fluent_data_map', 
        help="directory of training data"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='output/cfd_results', 
        help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. "
        "For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=1,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions." 
        "This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )

    args = parser.parse_args()
    return args
    

def save_as_npy(depth_pred, output_dir_npy, pred_name_base):
    npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
    if os.path.exists(npy_save_path):
        logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
    np.save(npy_save_path, depth_pred)
    
    
def save_as_png(depth_pred, output_dir_tif, pred_name_base):
    depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
    png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
    if os.path.exists(png_save_path):
        logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
    Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")
   
    
def save_as_color(depth_colored, output_dir_color, pred_name_base):
    colored_save_path = os.path.join(output_dir_color, f"{pred_name_base}.png")
    logging.info(f"{colored_save_path}")
    if os.path.exists(colored_save_path):
        logging.warning(
            f"Existing file: '{colored_save_path}' will be overwritten"
        )
    depth_colored.save(colored_save_path)
    

def load_scale(key):
    if key.__contains__('pressure'):
        return STAT_pressure
    elif key.__contains__('temperature'):
        return STAT_temperature
    elif key.__contains__('velocity'):
        return STAT_velocity
    else:
        raise NotImplementedError


def evaluate(pred, label, key, mask, denormalize=False):
    if len(mask.shape) == 4 or len(mask.shape) == 3:
        mask = mask.squeeze()
        pred = pred.squeeze()
        label = label.squeeze()
    
    if denormalize:
        stat = load_scale(key)
        pred = pred * (stat['max'] - stat['min']) + stat['min']
        label = label * (stat['max'] - stat['min']) + stat['min']

    img_true = (label * mask).astype(np.float32)
    img_pred = (pred * mask).astype(np.float32)
    
    # Flatten arrays to 1D for metric calculations
    y_true = label[mask].flatten()
    y_pred = pred[mask].flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    ssim_value = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min())
    psnr_value = psnr(img_true, img_pred, data_range=img_true.max() - img_true.min())

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'SSIM': ssim_value,
        'PSNR': psnr_value
    }


if "__main__" == __name__:
    args = get_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(args.output_dir, 'infer.log'), 
        filemode='a')
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    
    if args.ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if processing_res == 0 and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution," 
            "due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Output directories
    output_dir_color = os.path.join(args.output_dir, "depth_colored")
    output_dir_tif = os.path.join(args.output_dir, "depth_bw")
    output_dir_npy = os.path.join(args.output_dir, "depth_npy")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {args.output_dir}, {output_dir_color}, {output_dir_npy}, {output_dir_tif}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------
    if args.half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipeline = CFDiffPipleline.from_pretrained(
        args.checkpoint, variant=variant, torch_dtype=dtype
    )
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipeline = pipeline.to(device)
    logging.info(
        f"scale_invariant: {pipeline.scale_invariant}," 
         "shift_invariant: {pipeline.shift_invariant}"
    )

    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{args.checkpoint}`, "
        f"with denoise_steps = {args.denoise_steps or pipeline.default_denoising_steps}, "
        f"ensemble_size = {args.ensemble_size}, "
        f"processing resolution = {processing_res or pipeline.default_processing_resolution}, "
        f"seed = {seed}; "
        f"color_map = {color_map}."
    )
    # -------------------- Test Dataset --------------------
    test_dataset = CFDDataset(
        dataset_dir=args.base_data_dir,
        train_mode=False,
    )
    num = len(test_dataset.filenames)
    logging.info(f'Number of samples in Test: {num}')
    # -------------------- Inference and saving --------------------
    metrics = ['MAE', 'RMSE', 'R2', 'SSIM', 'PSNR']
    fields = ['pressure', 'temperature', 'velocity']
    sum_results = {field: {metric: 0. for metric in metrics} for field in fields}
    avg_results = {field: {metric: 0. for metric in metrics} for field in fields}
    
    with torch.no_grad():
        for test_data in tqdm(test_dataset, desc="Estimating depth", leave=True):
            # Random number generator
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
            # Predict depth
            pipe_out = pipeline(
                rgb_norm=test_data["inputs"],
                flow_in=test_data["flows"],
                text_in=test_data["prompt"],
                valid_mask=test_data["mask"],
                denoising_steps=args.denoise_steps,
                ensemble_size=args.ensemble_size,
                batch_size=batch_size,
                color_map=color_map,
                show_progress_bar=True,
                resample_method=resample_method,
                generator=generator,
            )

            pred: np.ndarray = pipe_out.depth_np
            depth_colored: Image.Image = pipe_out.depth_colored
            prompt = test_data['prompt'].replace(' field', "")
            pred_name_base = test_data['file_name'] + '_' + prompt

            label = (test_data["outputs"] + 1.) / 2.   
            label = torch.clip(label, 0., 1.).squeeze().cpu().numpy()  # (H, W)
            mask = test_data["mask"].squeeze().cpu().numpy()           # (H, W)

            res = evaluate(
                     pred=pred,
                     label=label,
                     mask=mask,
                     key=prompt)
            
            for metric, value in res.items():
                sum_results[prompt][metric] += value
                
            # # Save as npy
            # save_as_npy(depth_pred=depth_pred, 
            #             output_dir_npy=output_dir_npy, 
            #             pred_name_base=pred_name_base)

            # # Save as 8-bit uint png
            # save_as_png(depth_pred=depth_pred,
            #             output_dir_tif=output_dir_tif,
            #             pred_name_base=pred_name_base)

            # Save as Colorize figure
            # save_as_color(depth_colored=depth_colored,
            #               output_dir_color=output_dir_color,
            #               pred_name_base=pred_name_base)
            
        avg_results = {prompt: {metric: value / num for metric, value in value_dict.items()} 
                       for prompt, value_dict in sum_results.items()}
        
        # Create markdown table header
        table =  "| Domain |  MAE  |  RMSE  |   R2   |   SSIM   |   PSNR  |\n"
        table += "|--------|-------|--------|--------|----------|---------|\n"
        
        # Add rows for each domain
        for domain, metrics in avg_results.items():
            table += f"| {domain} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f}| {metrics['R2']:.4f} |{metrics['SSIM']:.4f} | {metrics['PSNR']:.4f} |\n"
        
        logging.info("\nEvaluation Results:\n" + table)