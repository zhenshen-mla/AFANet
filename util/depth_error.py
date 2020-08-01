import numpy as np
import torch


def depth_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1)
    n = torch.nonzero(binary_mask).size(0)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    abs_err = torch.sum(abs_err) / n
    rel_err = torch.sum(rel_err) / n

    rel_sqr = torch.sum(((x_pred_true - x_output_true)**2) / x_output_true) / n
    log_10 = torch.sum(torch.abs(torch.log10(x_pred_true) - torch.log10(x_output_true))) / n

    RMSE_linear = torch.sqrt(torch.sum((x_pred_true - x_output_true) ** 2) / n)
    RMSE_loge = torch.sqrt(torch.sum((torch.log(x_pred_true) - torch.log(x_output_true)) ** 2) / n)
    RMSE_log10 = torch.sqrt(torch.sum((torch.log10(x_pred_true) - torch.log10(x_output_true)) ** 2) / n)

    return abs_err, rel_err, rel_sqr, log_10, RMSE_linear, RMSE_loge, RMSE_log10


def depth_threshold(x_pred, x_output):
    input_gt_depth_image = x_output.data.squeeze().cpu().numpy().astype(np.float32)
    pred_depth_image = x_pred.data.squeeze().cpu().numpy().astype(np.float32)

    n = np.sum(input_gt_depth_image > 1e-3)  

    idxs = (input_gt_depth_image <= 1e-3)  
    pred_depth_image[idxs] = 1  
    input_gt_depth_image[idxs] = 1

    pred_d_gt = pred_depth_image / input_gt_depth_image
    pred_d_gt[idxs] = 100
    gt_d_pred = input_gt_depth_image / pred_depth_image
    gt_d_pred[idxs] = 100

    Threshold_1_25 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25) / n  # np.maximum返回相对较大的值
    Threshold_1_25_2 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25) / n
    Threshold_1_25_3 = np.sum(np.maximum(pred_d_gt, gt_d_pred) < 1.25 * 1.25 * 1.25) / n

    return Threshold_1_25, Threshold_1_25_2, Threshold_1_25_3