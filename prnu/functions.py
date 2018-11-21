# -*- coding: UTF-8 -*-
"""
@author: Luca Bondi (luca.bondi@polimi.it)
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
"""

import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pywt
from PIL import Image
from numpy.fft import fft, ifft, fft2, ifft2
from scipy.ndimage import filters
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


def wiener_dft(im_noise, sigma):
    """
    Adaptive Wiener filter applied to the 2D FFT of the image
    :param im_noise:
    :param sigma:
    :return:
    """
    noise_var = sigma ** 2
    h, w = im_noise.shape

    im_noise_fft = fft2(im_noise)

    im_noise_fft_mag = np.abs(im_noise_fft / (h * w) ** .5)

    im_noise_fft_mag_noise = wiener_adaptive(im_noise_fft_mag, noise_var)

    zeros_y, zeros_x = np.nonzero(im_noise_fft_mag == 0)

    im_noise_fft_mag[zeros_y, zeros_x] = 1
    im_noise_fft_mag_noise[zeros_y, zeros_x] = 0

    im_noise_fft_filt = im_noise_fft * im_noise_fft_mag_noise / im_noise_fft_mag

    im_noise_filt = np.real(ifft2(im_noise_fft_filt))

    return im_noise_filt.astype(np.float32)


def zero_mean(im):
    """
    ZeroMean called with the 'both' argument as from Fridrich toolbox. Same numeric results.
    :param im:
    :return:
    """
    # Adapt the shape ---
    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    # Subtract the 2D mean from each color channel ---
    ch_mean = im.mean(axis=0).mean(axis=0)
    ch_mean.shape = (1, 1, ch)
    Izm = im - ch_mean

    # Compute the 1D mean along each row and each column, then subtract ---
    row_mean = Izm.mean(axis=1)
    col_mean = Izm.mean(axis=0)

    row_mean.shape = (h, 1, ch)
    col_mean.shape = (1, w, ch)

    Izm_r = Izm - row_mean
    Izm_rc = Izm_r - col_mean

    # Restore the shape ---
    if im.shape[2] == 1:
        Izm_rc.shape = im.shape[:2]

    return Izm_rc


def zero_mean_total(im):
    """
    ZeroMeanTotal as from Fridrich toolbox. Same numeric results.
    :param im:
    :return:
    """
    im[0::2, 0::2] = zero_mean(im[0::2, 0::2])
    im[1::2, 0::2] = zero_mean(im[1::2, 0::2])
    im[0::2, 1::2] = zero_mean(im[0::2, 1::2])
    im[1::2, 1::2] = zero_mean(im[1::2, 1::2])
    return im


def rgb2gray(im):
    """
    RGB to gray as from Fridrich toolbox. Same numeric results.
    :param im:
    :return:
    """
    rgb2gray_vector = np.asarray([0.29893602, 0.58704307, 0.11402090]).astype(np.float32)
    rgb2gray_vector.shape = (3, 1)

    if im.ndim == 2:
        Ig = np.copy(im)
    elif im.shape[2] == 1:
        Ig = np.copy(im[:, :, 0])
    elif im.shape[2] == 3:
        w, h = im.shape[:2]
        im = np.reshape(im, (w * h, 3))
        Ig = np.dot(im, rgb2gray_vector)
        Ig.shape = (w, h)
    else:
        raise ValueError('Input image must have 1 or 3 channels')

    return Ig.astype(np.float32)


def threshold(wlet_coeff_energy_avg, noise_var):
    """
    Noise variance theshold as from Fridrich toolbox. Same numeric results.
    :param wlet_coeff_energy_avg:
    :param noise_var:
    :return:
    """
    # return np.maximum(np.zeros(wlet_coeff_energy_avg.shape), wlet_coeff_energy_avg - noise_var)
    res = wlet_coeff_energy_avg - noise_var
    return (res + np.abs(res)) / 2


def wiener_adaptive(x, noise_var, **kwargs):
    """
    WaveNoise as from Fridrich toolbox. Same numeric results.
    Wiener adaptive flter aimed at extracting the noise component
    For each input pixel the average variance over a neighborhoods of different window sizes is first computed.
    The smaller average variance is taken into account when filtering according to Wiener.
    :param x: 2D matrix
    :param noise_var: Power spectral density of the noise we wish to extract (S)
    :param kwargs:
    :return:
    """
    window_size_list = kwargs.pop('window_size_list', [3, 5, 7, 9])

    energy = x ** 2

    avg_win_energy = np.zeros(x.shape + (len(window_size_list),))
    for window_idx, window_size in enumerate(window_size_list):
        avg_win_energy[:, :, window_idx] = filters.uniform_filter(energy,
                                                                  window_size,
                                                                  mode='constant')

    coef_var = threshold(avg_win_energy, noise_var)
    coef_var_min = np.min(coef_var, axis=2)

    """
    coef_var_min(N) is a local estimate of image Power Spectral Density.
    Image content is what we want to remove.

    When coef_var_min(y,x) == 0 the image PSD is null,
    thus we retain all the content of x, which is noise

    When coef_var_min(y,x) >> noise_var the image PSD is higher than the noise PSD,
    thus we remove all the content of x, which is image content
    """

    x = x * noise_var / (coef_var_min + noise_var)

    return x


def noise_extract(im, levels: int = 4, sigma: float = 5) -> np.ndarray:
    """
    NoiseExtract as from Fridrich toolbox. Wavelet are a bit different, thus sigma is set by default to 5 instead of 3

    On a sample dataset:
    sigma	auc cc			auc pce
    3		0.96099997		0.78800005
    4		0.96249998		0.84150004
    *5		0.96349996		0.85825002
    6		0.95674998		0.85550004
    7		0.95225			0.82749999
    9 	    0.94174999		0.77950001
    11		0.93725002		0.75849998


    :param im: uint8
    :param levels: number of Wavelet decomposition levels
    :param sigma: std of gaussian noise to extract
    :return:
    """

    assert (im.dtype == np.uint8)

    im = im.astype(np.float32)

    noise_var = sigma ** 2

    if im.ndim == 2:
        im.shape += (1,)

    W = np.zeros(im.shape, np.float32)

    for ch in range(im.shape[2]):

        wlet = None
        while wlet is None and levels > 0:
            try:
                wlet = pywt.wavedec2(im[:, :, ch], 'db8', level=levels)
            except ValueError:
                levels -= 1
                wlet = None
        if wlet is None:
            raise ValueError('Impossible to compute Wavelet filtering for input size: {}'.format(im.shape))

        wlet_details = wlet[1:]

        wlet_details_filter = [None] * len(wlet_details)
        # Cycle over Wavelet levels 1:levels-1
        for wlet_level_idx, wlet_level in enumerate(wlet_details):
            # Cycle over H,V,D components
            level_coeff_filt = [None] * 3
            for wlet_coeff_idx, wlet_coeff in enumerate(wlet_level):
                level_coeff_filt[wlet_coeff_idx] = wiener_adaptive(wlet_coeff, noise_var)
            wlet_details_filter[wlet_level_idx] = tuple(level_coeff_filt)

        # Set filtered detail coefficients for Levels > 0 ---
        wlet[1:] = wlet_details_filter

        # Set to 0 all Level 0 approximation coefficients ---
        wlet[0][...] = 0

        # Invert wavelet transform ---
        wrec = pywt.waverec2(wlet, 'db8')
        try:
            W[:, :, ch] = wrec
        except ValueError:
            W = np.zeros(wrec.shape[:2] + (im.shape[2],), np.float32)
            W[:, :, ch] = wrec

    if W.shape[2] == 1:
        W.shape = W.shape[:2]

    W = W[:im.shape[0], :im.shape[1]]

    return W


def noise_extract_call(args):
    """
    Arguments dispatched for noise_extract
    :param args:
    :return:
    """
    return noise_extract(*args)


def extract_single(im: np.ndarray, levels: int = 4, sigma: float = 5, wdft_sigma: float = 0) -> np.ndarray:
    """
    Extract PRNU from a single image
    :param im: type np.uint8
    :param levels:
    :param sigma:
    :param wdft_sigma:
    :return:
    """
    W = noise_extract(im, levels, sigma)
    W = rgb2gray(W)
    K = zero_mean_total(W)
    del W  # deallocate memory
    K_std = K.std(ddof=1) if wdft_sigma == 0 else wdft_sigma
    K = wiener_dft(K, K_std).astype(np.float32)

    return K


def inten_scale(im: np.ndarray) -> np.ndarray:
    """
    IntenScale as from Fridrich toolbox
    :param im: type np.uint8
    :return:
    """

    assert (im.dtype == np.uint8)

    T = 252
    v = 6
    out = np.exp(-1 * (im - T) ** 2 / v)
    out[im < T] = im[im < T] / T

    return out


def saturation(im: np.ndarray) -> np.ndarray:
    """
    Saturation as from Fridrich toolbox
    :param im: type np.uint8
    :return:
    """
    assert (isinstance(im, np.ndarray))
    assert (im.dtype == np.uint8)

    if im.ndim == 2:
        im.shape += (1,)

    h, w, ch = im.shape

    if im.max() < 250:
        return np.ones((h, w, ch))

    im_h = im - np.roll(im, (0, 1), (0, 1))
    im_v = im - np.roll(im, (1, 0), (0, 1))
    satur_map = \
        np.bitwise_not(
            np.bitwise_and(
                np.bitwise_and(
                    np.bitwise_and(
                        im_h != 0, im_v != 0
                    ), np.roll(im_h, (0, -1), (0, 1)) != 0
                ), np.roll(im_v, (-1, 0), (0, 1)) != 0
            )
        )

    max_ch = im.max(axis=0).max(axis=0)

    for ch_idx, max_c in enumerate(max_ch):
        if max_c > 250:
            satur_map[:, :, ch_idx] = \
                np.bitwise_not(
                    np.bitwise_and(
                        im[:, :, ch_idx] == max_c, satur_map[:, :, ch_idx]
                    )
                )

    return satur_map


def noise_extract_compact(args):
    """
    Extract residual, multiplied by the image. Useful to save memory in multiprocessing operations
    :param args:
    :return:
    """
    w = noise_extract(*args)
    im = args[0]
    return (w * im / 255.).astype(np.float32)


def inten_sat_compact(args):
    im = args[0]
    return ((inten_scale(im) * saturation(im)) ** 2).astype(np.float32)


def extract_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                             batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
    """
    Extract PRNU from a list of images
    :param imgs: ndarray of size (N,H,W,Ch) and type np.uint8
    :param levels:
    :param sigma:
    :return:
    """
    # TODO merge with rotation aware version, complexity shouldn't increase too much
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape
    n = len(imgs)

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    if processes is None or processes > 1:
        args_list = []
        for im in imgs:
            args_list += [(im, levels, sigma)]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN += ni  # TODO this is slow and single process. can we improve?
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum += wi
            del wi_list

        pool.close()

    else:  # Single process
        for im in tqdm(imgs, disable=tqdm_str is None, desc=tqdm_str, dynamic_ncols=True):
            RPsum += noise_extract_compact((im, levels, sigma))
            NN += (inten_scale(im) * saturation(im)) ** 2

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)

    return K


def crosscorr_2d(k1, k2):
    """
    PRNU 2D cross-correlation
    """
    assert (k1.ndim == 2)
    assert (k2.ndim == 2)

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 -= k1.flatten().mean()
    k2 -= k2.flatten().mean()

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1])], mode='constant', constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1])], mode='constant', constant_values=0)

    k1_fft = fft2(k1, )
    k2_fft = fft2(np.rot90(k2, 2), )

    return np.real(ifft2(k1_fft * k2_fft)).astype(np.float32)


def complex_mul(t1, t2):
    """
    Complex multiplication for torch tensors (real part: channel 0, imag part: channel 1)
    :param t1: (HxWx2) torch Tensor
    :param t2: (HxWx2) torch Tensor
    :return: (HxWx2) torch Tensor
    """
    import torch

    real1 = t1[:, :, :, 0]
    imag1 = t1[:, :, :, 1]
    real2 = t2[:, :, :, 0]
    imag2 = t2[:, :, :, 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def crosscorr_2d_cuda(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """
    PRNU 2D cross-correlation computed on GPU
    :param k1: (HxWxN) numpy array, where N 2D signals are stacked over axis 2
    :param k2: (HxWxN) numpy array, where N 2D signals are stacked over axis 2
    :return: (HxWxN) numpy array, where N 2D cross-correlations are stacked over axis 2
    """
    import torch

    max_height = max(k1.shape[0], k2.shape[0])
    max_width = max(k1.shape[1], k2.shape[1])

    k1 = np.pad(k1, [(0, max_height - k1.shape[0]), (0, max_width - k1.shape[1]), (0, 0)], mode='constant',
                constant_values=0)
    k2 = np.pad(k2, [(0, max_height - k2.shape[0]), (0, max_width - k2.shape[1]), (0, 0)], mode='constant',
                constant_values=0)
    k2 = np.ascontiguousarray(np.rot90(k2, 2, axes=(0, 1)))

    k1 = np.transpose(k1, (2, 0, 1))
    k2 = np.transpose(k2, (2, 0, 1))
    k1_t = torch.Tensor(np.stack([k1, np.zeros_like(k1)], axis=-1)).cuda()
    k2_t = torch.Tensor(np.stack([k2, np.zeros_like(k2)], axis=-1)).cuda()

    k1k2_t_fft = complex_mul(torch.fft(k1_t, 2), torch.fft(k2_t, 2))

    cc = torch.ifft(k1k2_t_fft, 2)[:, :, :, 0].cpu().numpy().astype(np.float32)

    return np.transpose(cc, (1, 2, 0))


def pce(cc: np.ndarray, neigh_radius: int = 2) -> dict:
    """
    PCE position and value
    :param cc: as from crosscorr2d
    :param neigh_radius:
    :return:
    """
    assert (cc.ndim == 2)
    assert (isinstance(neigh_radius, int))

    out = dict()

    max_idx = np.argmax(cc.flatten())
    max_y, max_x = np.unravel_index(max_idx, cc.shape)

    peak_height = cc[max_y, max_x]

    cc_nopeaks = cc.copy()
    cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0

    pce_energy = np.mean(cc_nopeaks.flatten() ** 2)

    out['peak'] = (max_y, max_x)
    out['pce'] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    out['cc'] = peak_height

    return out


def crosscorr_1d(k1, k2):
    """
    PRNU 1D cross-correlation
    """
    assert (k1.shape == k2.shape)
    return np.real(ifft(fft(k1) * fft(k2[..., ::-1]))).astype(np.float32)


def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
    """
    Aligned PRNU cross-correlation
    :param k1: (n1,nk) or (n1,nk1,nk2,...)
    :param k2: (n2,nk) or (n2,nk1,nk2,...)
    :return: (n1,n2)
    """

    # Type cast
    k1 = np.array(k1).astype(np.float32)
    k2 = np.array(k2).astype(np.float32)

    ndim1 = k1.ndim
    ndim2 = k2.ndim
    assert (ndim1 == ndim2)

    k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
    k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

    assert (k1.shape[1] == k2.shape[1])

    k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
    k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

    k2t = np.ascontiguousarray(k2.transpose())

    cc = np.matmul(k1, k2t).astype(np.float32)
    ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

    return {'cc': cc, 'ncc': ncc}


def cut_ctr(x: np.ndarray, shape: tuple) -> np.ndarray:
    x = np.array(x).astype(np.float32)

    y = x
    for dim in np.arange(start=-1, stop=-len(shape) - 1, step=-1):
        idxs = np.arange(x.shape[dim] // 2 - shape[dim] // 2, x.shape[dim] // 2 + (shape[dim] - shape[dim] // 2))
        y = y.take(indices=idxs, axis=dim)

    return y


def single_load_compute_save(args: dict):
    # Loading ---
    im = np.asarray(Image.open(args['in_path']))

    if im.ndim == 0:
        print('Unable to read file: {}'.format(args['in_path']))
        return

    # Extracting ---
    prnu = extract_single(im)

    # Saving ---
    out_dir = os.path.split(args['out_path'])[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(args['out_path'], prnu)


def single_load_compute(in_path: str) -> np.ndarray or None:
    # Loading ---
    im = np.asarray(Image.open(in_path))

    if im.ndim == 0:
        print('Unable to read file: {}'.format(in_path))
        return None

    # Extracting ---
    prnu = extract_single(im).astype(np.float32)

    return prnu


def stats(cc: np.ndarray, gt: np.ndarray, ) -> dict:
    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    assert (cc.shape == gt.shape)
    assert (gt.dtype == np.bool)

    fpr, tpr, th = roc_curve(gt.flatten(), cc.flatten())
    auc_score = auc(fpr, tpr)

    # EER
    eer_idx = np.argmin((fpr - (1 - tpr)) ** 2, axis=0)
    eer = float(fpr[eer_idx])

    outdict = {
        'tpr': tpr,
        'fpr': fpr,
        'th': th,
        'auc': auc_score,
        'eer': eer,
    }

    return outdict


def gt(l1: list or np.ndarray, l2: list or np.ndarray) -> np.ndarray:
    """
    Determine the Ground Truth matrix given the labels
    :param l1: reference labels
    :param l2: residuals labels
    :return:
    """
    l1 = np.array(l1)
    l2 = np.array(l2)

    assert (l1.ndim == 1)
    assert (l2.ndim == 1)

    gt_arr = np.zeros((len(l1), len(l2)), np.bool)

    for l1idx, l1sample in enumerate(l1):
        gt_arr[l1idx, l2 == l1sample] = True

    return gt_arr


def load(path_list: list, prnu_size: tuple):
    k_list = []
    labels_list = []
    for path in path_list:
        label = os.path.splitext(os.path.split(path)[-1])[0]
        k = np.load(path)
        try:
            k = cut_ctr(k, prnu_size)
        except ValueError:
            k = [cut_ctr(i, prnu_size) for i in k if i is not None]
            k = np.stack(k)
        if k.ndim == 2:
            k.shape = (1,) + k.shape
        labels_list += [label] * k.shape[0]
        k_list += [k]
    arr = np.concatenate(k_list)

    return arr, labels_list


def easy_aligned_rgb_cc(k1: np.ndarray, k2: np.ndarray, subsampling: int = 1) -> float:
    assert (k1.ndim == 3)
    assert (k2.ndim == 3)

    k1 = rgb2gray(k1)
    k2 = rgb2gray(k2)

    k1 = k1[::subsampling, ::subsampling]
    k2 = k2[::subsampling, ::subsampling]

    k1 = k1.reshape(1, -1)
    k2 = k2.reshape(1, -1)

    assert (k1.shape[1] == k2.shape[1])

    return aligned_cc(k1, k2)['cc'][0]


def extract_single_for_multiple(probe, ref_w):
    probe_w = noise_extract(probe)
    if probe_w.shape == ref_w.shape:
        cc_0 = easy_aligned_rgb_cc(probe_w, ref_w, 3)
        cc_2 = easy_aligned_rgb_cc(np.rot90(probe_w, 2), ref_w, 3)
        if cc_2 > cc_0:
            probe = np.rot90(probe, 2)
            probe_w = np.rot90(probe_w, 2)
    else:
        cc_1 = easy_aligned_rgb_cc(np.rot90(probe_w, 1), ref_w, 3)
        cc_3 = easy_aligned_rgb_cc(np.rot90(probe_w, 3), ref_w, 3)
        if cc_3 > cc_1:
            probe = np.rot90(probe, 3)
            probe_w = np.rot90(probe_w, 3)
        else:
            probe = np.rot90(probe, 1)
            probe_w = np.rot90(probe_w, 1)
    wi = probe_w * probe / 255.
    nni = (inten_scale(probe) * saturation(probe)) ** 2

    return nni, wi


def extract_multiple_rotation(probes: list, processes: int, batch_size: int, tqdm_str: str = None) -> np.ndarray:
    ref_probe_idx = 0
    while (probes[ref_probe_idx].shape[0] > probes[ref_probe_idx].shape[1]) and ref_probe_idx < len(probes):
        ref_probe_idx += 1
    if ref_probe_idx == len(probes):
        ref_probe_idx = len(probes) - 1
    ref_probe = probes[ref_probe_idx]

    ref_w = noise_extract(ref_probe)
    h, w, ch = ref_probe.shape
    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    with Pool(processes=processes) as pool:
        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(probes)),
                               desc=('Extracting fingerprint' if tqdm_str is None else tqdm_str), dynamic_ncols=True):
            nni_wi = pool.map(partial(extract_single_for_multiple, ref_w=ref_w),
                              probes[batch_idx0:batch_idx0 + batch_size])
            for ni, wi in nni_wi:
                NN += ni
                RPsum += wi
            del nni_wi

    fingerprint = RPsum / (NN + 1)
    fingerprint = rgb2gray(fingerprint)
    fingerprint = zero_mean_total(fingerprint)
    fingerprint = wiener_dft(fingerprint, fingerprint.std(ddof=1)).astype(np.float32)

    return fingerprint


def pce_rot_flip(fingerprint_fft: np.ndarray, residual: np.ndarray, **kwargs) -> dict:
    assert (fingerprint_fft.dtype == np.complex64)
    assert (fingerprint_fft.ndim == 2)
    assert (fingerprint_fft.shape[0] == fingerprint_fft.shape[1])
    assert (residual.dtype == np.float32)
    assert (residual.ndim == 2)

    nfft = fingerprint_fft.shape[0]

    # Compute the rotation and flip invariant embedding
    wz_pad = np.zeros((nfft, nfft), np.float32)
    wz_pad[:residual.shape[0], :residual.shape[1]] = residual[:nfft, :nfft]
    wz_tot = np.zeros((nfft, nfft), np.float32)
    for flip in [False, True]:
        wz_case = wz_pad[::-1] if flip else wz_pad
        for rot_idx in range(4):
            wz_case = np.rot90(wz_case, rot_idx)
            wz_tot += wz_case

    wz_tot_fft = fft2(wz_tot - wz_tot.mean())
    cc2d = np.real(ifft2(fingerprint_fft * wz_tot_fft))
    pce_dict = pce(cc2d, **kwargs)

    return pce_dict
