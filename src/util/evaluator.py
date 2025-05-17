import numpy as np
from PIL import Image

import torch
from tqdm.auto import tqdm
from matplotlib import colormaps
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
EXTENSION_LIST = [".jpg", ".jpeg", ".png"]


STAT_pressure={'min': -37.73662186, 'max': 57.6361618}
STAT_temperature={'min': 299.9764404, 'max':310.3595276}
STAT_velocity={'min': 0.0, 'max':0.3930110071636349}


def load_scale(key):
    if key.__contains__('pressure'):
        return STAT_pressure
    elif key.__contains__('temperature'):
        return STAT_temperature
    elif key.__contains__('velocity'):
        return STAT_velocity
    else:
        raise NotImplementedError
    
    
def apply_colors_to_array(x, mask, cmap='Spectral'):
    """
    Args:
        x: numpy array of shape (H, W), values in [0, 1]
    Returns:
        rgb: numpy array of shape (3, H, W), dtype=float32, RGB values in [0, 1]
    """
    # Get the colormap
    color_map = colormaps.get_cmap(cmap)
    # Apply the colormap (returns RGBA)
    rgba = color_map(x)  # shape: (H, W, 4)
    # Drop alpha and transpose to (3, H, W)
    rgb = rgba[..., :3]  # (3, H, W)
    if mask is not None:
        rgb[mask == 0] = 1
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8


def evaluate_batch(preds, labels, field, masks=None, denormalize=False):
    """
    Evaluate metrics for a batch of predictions and labels.

    Args:
        preds: numpy array of shape [B, H, W] or [B, 1, H, W]
        labels: same shape as preds
        field: str, used to load normalization stats
        masks: optional mask array of shape [B, H, W] or [B, 1, H, W]
        denormalize: if True, will rescale predictions and labels

    Returns:
        Dictionary of averaged metrics across the batch
    """
    batch_size = preds.shape[0]
    metrics_sum = {
        'MAE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'SSIM': 0.0,
        'PSNR': 0.0
    }

    for i in range(batch_size):
        pred = preds[i]
        label = labels[i]
        mask = masks[i] if masks is not None else np.ones_like(label)

        # Remove channel dimension if present
        if pred.ndim == 3:
            pred = pred.squeeze()
            label = label.squeeze()
            mask = mask.squeeze()

        if denormalize:
            stat = load_scale(field)
            pred = pred * (stat['max'] - stat['min']) + stat['min']
            label = label * (stat['max'] - stat['min']) + stat['min']

        img_true = (label * mask * 255.).astype(np.uint8)
        img_pred = (pred * mask * 255.).astype(np.uint8)

        # Only use valid (masked) pixels
        y_true = label[mask > 0].flatten()
        y_pred = pred[mask > 0].flatten()

        # Compute metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)
        ssim_value = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min())
        psnr_value = psnr(img_true, img_pred, data_range=img_true.max() - img_true.min())

        metrics_sum['MAE'] += mae
        metrics_sum['RMSE'] += rmse
        metrics_sum['R2'] += r2
        metrics_sum['SSIM'] += ssim_value
        metrics_sum['PSNR'] += psnr_value

    # Average across batch
    averaged_metrics = {k: v / batch_size for k, v in metrics_sum.items()}
    return averaged_metrics


def evaluate_one(pred, label, field, mask, denormalize=False):
    if len(mask.shape) == 4 or len(mask.shape) == 3:
        mask = mask.squeeze()
        pred = pred.squeeze()
    
    if denormalize:
        stat = load_scale(field)
        pred = pred * (stat['max'] - stat['min']) + stat['min']
        label = label * (stat['max'] - stat['min']) + stat['min']

    img_true = (label * mask * 255.).astype(np.uint8)
    img_pred = (pred * mask * 255.).astype(np.uint8)
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


class Evaluator:
    def __init__(self, num_samples, denormalize=False):
        self.metrics = ['MAE', 'RMSE', 'R2', 'SSIM', 'PSNR']
        self.fields = ['pressure', 'temperature', 'velocity']
        self.sum_results = {field: {metric: 0. for metric in self.metrics} 
                            for field in self.fields}
        self.avg_results = {field: {metric: 0. for metric in self.metrics} 
                            for field in self.fields}
        self.denormalize = denormalize
        self.num_samples = num_samples
        
    def evaluate_single(self, pred, label, field, mask):
        res = evaluate_one(pred, label, field, mask, denormalize=self.denormalize)
        
        for metric, value in res.items():
            self.sum_results[field][metric] += value
        return res
    
    def evaluate_batch(self, preds, labels, fields, masks):
        """
        Evaluate metrics for a batch of predictions and labels.

        Args:
            preds: numpy array of shape [B, H, W] or [B, 1, H, W]
            labels: same shape as preds
            field: str, used to load normalization stats
            masks: optional mask array of shape [B, H, W] or [B, 1, H, W]
            denormalize: if True, will rescale predictions and labels

        Returns:
            Dictionary of averaged metrics across the batch
        """
        batch_size = preds.shape[0]

        for i in range(batch_size):
            pred = preds[i]
            label = labels[i]
            mask = masks[i] if masks is not None else np.ones_like(label)

            # Remove channel dimension if present
            if pred.ndim == 3:
                pred = pred.squeeze()
                label = label.squeeze()
                mask = mask.squeeze()

            if self.denormalize:
                stat = load_scale(fields)
                pred = pred * (stat['max'] - stat['min']) + stat['min']
                label = label * (stat['max'] - stat['min']) + stat['min']

            img_true = (label * mask * 255.).astype(np.uint8)
            img_pred = (pred * mask * 255.).astype(np.uint8)

            # Only use valid (masked) pixels
            y_true = label[mask > 0].flatten()
            y_pred = pred[mask > 0].flatten()

            # Compute metrics
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            r2 = r2_score(y_true, y_pred)
            ssim_value = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min())
            psnr_value = psnr(img_true, img_pred, data_range=img_true.max() - img_true.min())

            self.sum_results[fields[i]]['MAE'] += mae
            self.sum_results[fields[i]]['RMSE'] += rmse
            self.sum_results[fields[i]]['R2'] += r2
            self.sum_results[fields[i]]['SSIM'] += ssim_value
            self.sum_results[fields[i]]['PSNR'] += psnr_value
        
    def visualize_single(self, pred, filename, mask, label=None):
        pred_uint8 = apply_colors_to_array(x=pred, mask=mask)
        image = Image.fromarray(pred_uint8)
        image.save(filename)
        
        if label is not None:
            label_uint8 = apply_colors_to_array(x=label, mask=mask)
            img = Image.fromarray(label_uint8)
            img.save(filename.replace('.png', '_gt.png'))
    
    def compute_average(self):
        self.avg_results = \
        {domain: {metric: value / self.num_samples 
                   for metric, value in value_dict.items()} 
            for domain, value_dict in self.sum_results.items()}
        return self.avg_results
    
    def show_average_results(self):
        # Create markdown table header
        table =  "|  Domain    |   MAE   |   RMSE   |   R2   |   SSIM   |   PSNR   |\n"
        table += "|------------|---------|----------|--------|----------|----------|\n"
        
        # Add rows for each domain
        for domain, metrics in self.avg_results.items():
            table += f"| {domain} | {metrics['MAE']:.4f} | {metrics['RMSE']:.4f}| {metrics['R2']:.4f} |{metrics['SSIM']:.4f} | {metrics['PSNR']:.4f} |\n"
        
        return "\nEvaluation Results:\n" + table
        