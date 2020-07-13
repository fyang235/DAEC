# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import cv2

from utils.transforms import transform_preds


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def taylor(hm, coord):
    heatmap_height = hm.shape[0]
    heatmap_width = hm.shape[1]
    px = int(coord[0])
    py = int(coord[1])
    if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
        dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
        dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
        dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
        dxy = 0.25 * (hm[py + 1][px + 1] - hm[py - 1][px + 1] - hm[py + 1][px - 1] + hm[py - 1][px - 1])
        dyy = 0.25 * (hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
        derivative = np.matrix([[dx], [dy]])
        hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = hessian.I
            offset = -hessianinv * derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


def gaussian_blur(hm, kernel):
    border = (kernel - 1) // 2
    batch_size = hm.shape[0]
    num_joints = hm.shape[1]
    height = hm.shape[2]
    width = hm.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(hm[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border))
            dr[border:-border, border:-border] = hm[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            hm[i, j] = dr[border:-border, border:-border].copy()
            hm[i, j] *= origin_max / np.max(hm[i, j])
    return hm


def calibrate_coord_with_DAEC(coord_x, coord_y, heat, config):
    if not config.TEST.DAEC.USE_EMPIRICAL_FORMULA:
        expand = int(config.TEST.DAEC.EXPAND_EDGE)
        delta = int(config.TEST.DAEC.DELTA)
    else:
        expand = int(3 * config.MODEL.SIGMA + 1)
        delta = int(config.MODEL.SIGMA + 2)

    x_min = coord_x - expand
    x_max = coord_x + expand + 1 - delta
    y_min = coord_y - expand
    y_max = coord_y + expand + 1 - delta

    h, w = heat.shape

    x_min = x_min if x_min > 0 else 0
    x_max = x_max if x_max < w else w
    y_min = y_min if y_min > 0 else 0
    y_max = y_max if y_max < h else h

    xx = np.array([range(x_min, x_max)])
    xx = np.stack([xx] * (y_max - y_min), axis=1)
    yy = np.array([range(y_min, y_max)])
    yy = np.stack([yy] * (x_max - x_min), axis=2)

    heat = heat[y_min:y_max, x_min:x_max]

    score = np.sum(heat)
    xx = np.sum(xx * heat)
    yy = np.sum(yy * heat)

    if score != 0:
        coord_x = xx / score
        coord_y = yy / score
        return coord_x, coord_y
    else:
        return coord_x, coord_y


def get_final_preds(config, hm, center, scale, mode="DAEC"):
    """
    this function calculates maximum coordinates of heatmap
    """
    mode = config.TEST.DECODE_MODE
    assert mode in ["STANDARD", "SHIFTING", "DARK", "DAEC"]

    coords, maxvals = get_max_preds(hm)
    heatmap_height = hm.shape[2]
    heatmap_width = hm.shape[3]

    # post-processing
    if mode in ["SHIFTING", "DARK", "DAEC"]:
        if mode == "SHIFTING":
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    y, x = coords[n, p]
                    hm[n, p, int(x), int(y)] = 1e-10

            coords_2nd, _ = get_max_preds(hm)

            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    y, x = coords[n, p]
                    y2, x2 = coords_2nd[n, p]
                    dist = np.sqrt((y - y2) * (y - y2) + (x - x2) * (x - x2))
                    y = y + 0.25 * (y2 - y) / dist
                    x = x + 0.25 * (x2 - x) / dist
                    coords[n, p] = y, x

        if mode == "DARK":
            hm = gaussian_blur(hm, config.TEST.BLUR_KERNEL)
            hm = np.maximum(hm, 1e-10)
            hm = np.log(hm)
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    coords[n, p] = taylor(hm[n][p], coords[n][p])

        if mode == "DAEC":
            hm = np.maximum(hm, 1e-10)
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    x, y = coords[n, p]
                    heat = hm[n, p]
                    x, y = calibrate_coord_with_DAEC(int(x), int(y), heat, config)
                    coords[n, p] = x, y

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], [heatmap_width, heatmap_height])

    return preds, maxvals
