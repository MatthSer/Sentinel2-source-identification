import os
import iio

import numpy as np
import rasterio
import scipy.stats

from scipy import signal
from scipy import stats
# import vpv

import matplotlib.pyplot as plt


def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def k_mean(img, k=2):
    possible_k = [2, 4, 6, 8]
    if not k in possible_k:
        print(f'Choose k in {possible_k}')
        return exit(0)
    if k == 2:
        k_filter = np.array([[-1, 2, -1]]) * 1 / 2
    elif k == 4:
        k_filter = np.array([[-1, -1, 4, -1, -1]]) * 1 / 4
    elif k == 6:
        k_filter = np.array([[-1, -1, -1, 6, -1, -1, -1]]) * 1 / 6
    elif k == 8:
        k_filter = np.array([[-1, -1, -1, -1, 8, -1, -1, -1, -1]]) * 1 / 8

    filtered_img = signal.convolve2d(img, k_filter, 'same')
    return filtered_img


def compare_signals(sample, signal, position):
    length = len(sample)
    test_signal = signal[length + position:length + position + length]
    corr = np.corrcoef(sample, test_signal)[0, 1]
    return corr


def test_sample(sample, signal):
    sample_size = len(sample)
    signal_size = len(signal)
    if sample_size > signal_size:
        print(f'Sample size {sample_size} exceeds signal size {signal_size}')
        return exit(0)

    # Pad signal to compare
    padded_signal = np.pad(signal, (sample_size, sample_size), 'constant')
    list_corr = []
    for pos in range(signal_size - 1):
        # print(pos)
        corr = compare_signals(sample, padded_signal, pos)
        list_corr.append(corr)
    return list_corr


def list_image_paths(input_path, sensor_id, type, band):
    list_of_paths = []

    dirs = os.listdir(input_path)
    for dir in dirs:
        if not '_GR_' in dir:
            continue
        if not sensor_id in dir:
            continue
        path = os.path.join(input_path, dir, type)
        for file in os.listdir(path):
            if band in file:
                list_of_paths.append(os.path.join(path, file))
    return list_of_paths


def select_crop(img, size_x, size_y):
    border_x = img.shape[0] - size_x - 10
    border_y = img.shape[1] - size_y - 10
    x_start = np.random.randint(10, border_x)
    y_start = np.random.randint(10, border_y)
    crop = img[x_start:x_start + size_x, y_start:y_start + size_y]
    return crop, y_start


def compute_profil_prnu(paths):
    profils = []
    for path in paths:
        profil = get_profil(path)
        profils.append(profil)
    prnu = np.mean(np.array(profils), axis=0)
    return prnu


def compute_prnu(paths):
    residuals = []
    for path in paths:
        residual = get_filtered_img(path)
        residuals.append(residual)
    prnu = np.mean(np.array(residuals), axis=0)
    return prnu


def get_profil_crop(img, size_x, size_y):
    crop, y_start = select_crop(img, size_x, size_y)
    profil = np.mean(crop, 0)
    return profil, y_start


def get_profil(path):
    with rasterio.open(path) as src:
        img = src.read()[-1, :, :]
        filtered_img = k_mean(img, k=2)
        profil = np.mean(filtered_img, axis=(np.uint8(0)))[10:-10]
    return profil


def get_filtered_img(path):
    with rasterio.open(path) as src:
        img = src.read()[-1, :, :]
        filtered_img = k_mean(img, k=2)
    return filtered_img


def prnu_similarity_profil(prnu, residual, crop_size):
    sample_size = len(prnu)
    signal_size = len(residual)
    if not sample_size == signal_size:
        print(f'Sample size {sample_size} must be the same as prnu size {signal_size}')
        return exit(0)

    # Comparing signals
    list_corr = []
    list_corr_i_j = []
    for i in range(0, len(prnu) - 1 - crop_size, crop_size):
        prnu_sample = prnu[i:i + crop_size]
        for j in range(0, len(residual) - 1 - crop_size, crop_size):
            # print(i,j)
            residual_sample = residual[j:j + crop_size]
            corr = np.corrcoef(prnu_sample, residual_sample)[0, 1]
            if not i == j:
                list_corr.append(corr)
            elif i == j:
                list_corr_i_j.append(corr)

    # Similarity test
    ks_pvalue = stats.ks_2samp(list_corr, list_corr_i_j, alternative='greater')[1]

    return ks_pvalue, list_corr, list_corr_i_j


def get_coordinates(shape, crop_size):
    coordinates = []
    for i in range(10, shape[0] - crop_size - 10, crop_size):
        for j in range(10, shape[1] - crop_size - 10, crop_size):
            coordinate = [i, j]
            coordinates.append(coordinate)
    return coordinates


def prnu_similarity(prnu, residual, crop_size):
    sample_size = len(prnu)
    signal_size = len(residual)
    if not sample_size == signal_size:
        print(f'Sample size {sample_size} must be the same as prnu size {signal_size}')
        return exit(0)

    # Get coordinates
    coordinates = get_coordinates(prnu.shape, crop_size)

    # Comparing signals
    list_corr = []
    list_corr_i_j = []
    for prnu_coordinate in coordinates:
        prnu_crop = prnu[prnu_coordinate[0]:prnu_coordinate[0] + crop_size,
                    prnu_coordinate[1]:prnu_coordinate[1] + crop_size]
        prnu_sample = np.mean(prnu_crop, axis=0)
        for residual_coordinate in coordinates:
            residual_crop = residual[residual_coordinate[0]:residual_coordinate[0] + crop_size,
                            residual_coordinate[1]:residual_coordinate[1] + crop_size]
            residual_sample = np.mean(residual_crop, axis=0)

            # Compare profil
            if prnu_coordinate[1] == residual_coordinate[1]:
                list_corr_i_j.append(np.corrcoef(prnu_sample, residual_sample)[0, 1])
            else:
                list_corr.append(np.corrcoef(prnu_sample, residual_sample)[0, 1])

    # Similarity test
    ks_pvalue = stats.ks_2samp(list_corr, list_corr_i_j)[1]
    ks_u_pvalue = 0

    return ks_u_pvalue, ks_pvalue


def get_corr_ditribution(prnu, residual, crop_size):
    sample_size = len(prnu)
    signal_size = len(residual)
    if not sample_size == signal_size:
        print(f'Sample size {sample_size} must be the same as prnu size {signal_size}')
        return exit(0)

    # Get coordinates
    coordinates = get_coordinates(prnu.shape, crop_size)

    # Comparing signals
    list_corr = []
    for coordinate_prnu in coordinates:
        prnu_crop = prnu[coordinate_prnu[0]:coordinate_prnu[0] + crop_size,
                    coordinate_prnu[1]:coordinate_prnu[1] + crop_size]
        for coordinate_residual in coordinates:
            residual_crop = residual[coordinate_residual[0]:coordinate_residual[0] + crop_size,
                            coordinate_residual[1]:coordinate_residual[1] + crop_size]
            prnu_sample = np.mean(prnu_crop, axis=0)
            residual_sample = np.mean(residual_crop, axis=0)
            list_corr.append(np.corrcoef(prnu_sample, residual_sample)[0, 1])

    # return normalize(list_corr)
    return list_corr


def get_pvalue(distribution, value):
    # Scipy
    stats, pvalue = scipy.stats.ttest_1samp(distribution, popmean=value, alternative='greater')
    return np.log10(pvalue)


def get_corr_similarity(ref, test, crop_size, coordinates):
    ref_crop = ref[coordinates[0]:coordinates[0] + crop_size,
               coordinates[1]:coordinates[1] + crop_size]
    test_crop = test[coordinates[0]:coordinates[0] + crop_size,
                coordinates[1]:coordinates[1] + crop_size]
    ref_sample = np.mean(ref_crop, axis=0)
    test_sample = np.mean(test_crop, axis=0)

    return np.corrcoef(ref_sample, test_sample)[0, 1]


def get_sensor_id(path):
    return path.split('/')[-1].split('_')[8]


def get_band_id(path):
    return path.split('/')[-1].split('.')[0].split('_')[-1]
