import os

import numpy as np
import rasterio
import scipy.stats

from scipy import signal
from scipy import stats

from numba import njit, prange
# import vpv




def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def k_mean(img, k=2):
    """
    Applies a convolution filter based on the given k value.

    Args:
        img (numpy.ndarray): Input image.
        k (int): The filter kernel size. Must be in [2, 4, 6, 8].

    Returns:
        numpy.ndarray: The filtered image.
    """
    possible_k = {2, 4, 6, 8}

    if k not in possible_k:
        raise ValueError(f"Choose k from {sorted(possible_k)}")

    # Precomputed kernels as dictionary
    k_filters = {
        2: np.array([[-1, 2, -1]]) / 2,
        4: np.array([[-1, -1, 4, -1, -1]]) / 4,
        6: np.array([[-1, -1, -1, 6, -1, -1, -1]]) / 6,
        8: np.array([[-1, -1, -1, -1, 8, -1, -1, -1, -1]]) / 8,
    }

    # Apply the selected convolution filter
    filtered_img = signal.convolve2d(img.astype(np.float64), k_filters[k], mode='same')

    return filtered_img


def list_img_path(input_dir, sensor_ids, band_ids):
    list_path = []
    for dir in os.listdir(input_dir):
        if not '_GR_' in dir:
            continue
        sensor_id = get_sensor_id(dir)
        path = os.path.join(input_dir, dir, 'IMG_DATA')
        if not sensor_id in sensor_ids:
            continue
        for file in os.listdir(path):
            band_id = get_band_id(file)
            if band_id in band_ids:
                list_path.append(os.path.join(path, file))
    return list_path


def list_image_paths(input_path, sensor_id, img_type, band):
    """
    Retrieves a list of image file paths matching the given criteria.

    Args:
        input_path (str): Root directory containing images.
        sensor_id (str): Sensor ID to filter directories.
        img_type (str): Image type subfolder.
        band (str): Band identifier to filter files.

    Returns:
        list: List of matching image file paths.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    list_of_paths = []

    # Efficiently scan directories
    for dir_entry in os.scandir(input_path):
        if dir_entry.is_dir() and '_GR_' in dir_entry.name and sensor_id in dir_entry.name:
            type_path = os.path.join(dir_entry.path, img_type)

            if not os.path.exists(type_path):
                continue  # Skip if the type subfolder doesn't exist

            # List matching files directly
            list_of_paths.extend([
                os.path.join(type_path, file.name)
                for file in os.scandir(type_path)
                if file.is_file() and band in file.name
            ])

    return list_of_paths


def select_crop(img, size_x, size_y):
    """
    Selects a random crop of the given size from an image.

    Args:
        img (numpy.ndarray): Input image.
        size_x (int): Width of the crop.
        size_y (int): Height of the crop.

    Returns:
        tuple: (cropped image, adjusted x_start position)
    """
    height, width = img.shape[:2]  # Supports both grayscale and RGB images

    # Ensure crop size is within valid range
    if size_x + 20 > width or size_y + 20 > height:
        raise ValueError("Crop size too large for the given image dimensions.")

    # Define the crop boundaries, ensuring valid starting points
    border_x = max(10, width - size_x - 10)
    border_y = max(10, height - size_y - 10)

    x_start = np.random.randint(10, border_x + 1)
    y_start = np.random.randint(10, border_y + 1)

    # Extract the crop
    crop = img[y_start:y_start + size_y, x_start:x_start + size_x]

    return crop, max(0, x_start - 10)


def compute_profil_prnu(paths):
    """
    Computes the PRNU profile as the mean of individual profiles.

    Args:
        paths (list): List of file paths.

    Returns:
        numpy.ndarray: Averaged PRNU profile.
    """
    if not paths:
        raise ValueError("The paths list is empty. Cannot compute PRNU.")

    # Efficient list comprehension for profile extraction
    profils = [get_profil(path) for path in paths]

    # Compute mean PRNU profile
    prnu = np.mean(profils, axis=0)

    return prnu


def compute_prnu(paths):
    """
    Computes the PRNU (Photo Response Non-Uniformity) as the mean of residuals.

    Args:
        paths (list): List of image file paths.

    Returns:
        numpy.ndarray: Averaged PRNU residual.
    """
    if not paths:
        raise ValueError("The paths list is empty. Cannot compute PRNU.")

    # Efficient list comprehension for residual extraction
    residuals = [get_filtered_img(path) for path in paths]

    # Compute mean PRNU residual
    prnu = np.mean(residuals, axis=0)

    return prnu


def get_profil_crop(img, size_x, size_y):
    """
    Extracts a cropped region from the image and computes the mean profile.

    Args:
        img (numpy.ndarray): Input image.
        size_x (int): Width of the crop.
        size_y (int): Height of the crop.

    Returns:
        tuple: (Mean profile of the cropped region, y_start coordinate)
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid or empty image provided.")

    crop, x_start = select_crop(img, size_x, size_y)

    # Compute the mean profile along axis 0 (column-wise)
    profil = np.mean(crop, axis=0)

    return profil, x_start


def get_profil(path):
    """
    Reads an image file, applies filtering, and extracts a profile.

    Args:
        path (str): Path to the raster image.

    Returns:
        numpy.ndarray: Extracted profile from the filtered image.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Invalid path provided.")

    try:
        with rasterio.open(path) as src:
            img = src.read()[-1, :, :]  # Read last band

            if img is None or img.size == 0:
                raise ValueError(f"Empty or corrupted image: {path}")

            # Apply k-mean filtering
            filtered_img = k_mean(img, k=2)

            # Ensure slicing is valid
            if filtered_img.shape[0] <= 20:
                raise ValueError(f"Image height too small for slicing: {filtered_img.shape}")

            # Compute mean profile along axis 0
            profil = np.mean(filtered_img, axis=0)[10:-10]

        return profil

    except rasterio.errors.RasterioIOError:
        raise ValueError(f"Error reading raster file: {path}")


def get_filtered_img(path):
    """
    Reads an image file and applies a k-mean filter.

    Args:
        path (str): Path to the raster image.

    Returns:
        numpy.ndarray: Filtered image.
    """
    if not path or not isinstance(path, str):
        raise ValueError("Invalid file path provided.")

    try:
        with rasterio.open(path) as src:
            img = src.read()[-1, :, :]  # Read the last band

            if img is None or img.size == 0:
                raise ValueError(f"Empty or corrupted image: {path}")

            # Apply k-mean filtering
            filtered_img = k_mean(img, k=2)

        return filtered_img

    except rasterio.errors.RasterioIOError:
        raise ValueError(f"Error reading raster file: {path}")


@njit(fastmath=True)
def compute_correlations(prnu, residual, indices, crop_size):
    num_elements = len(indices)
    list_corr = np.zeros(num_elements * (num_elements - 1), dtype=np.float64)  # Allocate space
    list_corr_i_j = np.zeros(num_elements, dtype=np.float64)  # Allocate space

    index_corr = 0
    index_corr_i_j = 0

    for i in prange(num_elements):
        prnu_sample = prnu[indices[i]:indices[i] + crop_size]
        mean_prnu = np.mean(prnu_sample)
        std_prnu = np.std(prnu_sample)

        for j in prange(num_elements):
            residual_sample = residual[indices[j]:indices[j] + crop_size]
            mean_residual = np.mean(residual_sample)
            std_residual = np.std(residual_sample)

            # Compute Pearson correlation manually (faster than np.corrcoef in Numba)
            corr = np.sum((prnu_sample - mean_prnu) * (residual_sample - mean_residual)) / (
                    crop_size * std_prnu * std_residual)

            if i == j:
                list_corr_i_j[index_corr_i_j] = corr
                index_corr_i_j += 1
            else:
                list_corr[index_corr] = corr
                index_corr += 1

    return list_corr[:index_corr], list_corr_i_j[:index_corr_i_j]


def prnu_similarity_profil(prnu, residual, crop_size):
    sample_size = len(prnu)
    signal_size = len(residual)

    if sample_size != signal_size:
        raise ValueError(f'Sample size {sample_size} must be the same as PRNU size {signal_size}')

    # Precompute valid crop indices
    indices = np.arange(0, sample_size - crop_size, crop_size)

    # Compute correlations using optimized Numba function
    list_corr, list_corr_i_j = compute_correlations(prnu, residual, indices, crop_size)

    # Kolmogorov-Smirnov similarity test
    ks_pvalue = stats.ks_2samp(list_corr, list_corr_i_j, alternative='greater')[1]

    return ks_pvalue, list_corr, list_corr_i_j


def get_witness_corr_distribution(prnu, residual, crop_size):
    sample_size = len(prnu)
    signal_size = len(residual)

    if sample_size != signal_size:
        raise ValueError(f'Sample size {sample_size} must be the same as PRNU size {signal_size}')

    # Precompute valid crop indices
    indices = np.arange(0, sample_size - crop_size, crop_size)

    # Compute correlations using optimized Numba function
    list_corr, list_corr_i_j = compute_correlations(prnu, residual, indices, crop_size)

    # TODO: get back to original
    return np.concatenate((list_corr, list_corr_i_j))
    # return list_corr_i_j


def get_sensor_id(path):
    return path.split('/')[-1].split('_')[8]


def get_band_id(path):
    return path.split('/')[-1].split('.')[0].split('_')[-1]


def get_pvalue(distribution, value):
    stats, pvalue = scipy.stats.ttest_1samp(distribution, popmean=value, alternative='two-sided')
    return pvalue


def patch_source_eval(prnu_profil, residual, size_x, size_y, interval_size):
    """
    Evaluates a patch source by comparing a PRNU profile with a cropped residual.

    Args:
        prnu_profil (numpy.ndarray): Reference PRNU profile.
        residual (numpy.ndarray): Residual image data.
        size_x (int): Width of the cropped profile.
        size_y (int): Height of the cropped profile.
        interval_size (int): Interval size for similarity profiling.

    Returns:
        tuple: (Minimum p-value, position of min p-value, start position of cropped profile)
    """
    if len(prnu_profil) < size_x:
        raise ValueError("PRNU profile length is too short for the given crop size.")

    np.random.seed(1)  # Ensure reproducibility

    # Extract test profile from residual
    crop_profil_test, start = get_profil_crop(residual, size_x, size_y)

    # Pre-allocate NumPy array for p-values
    num_samples = len(prnu_profil) - size_x
    pvalues = np.empty(num_samples, dtype=np.float64)

    # Compute similarity p-values
    for i in range(num_samples):
        sample_ref = prnu_profil[i:i + size_x]
        pvalues[i], _, _ = prnu_similarity_profil(sample_ref, crop_profil_test, interval_size)

    # Compute minimum p-value and its position
    min_pvalue = np.min(pvalues)
    min_pos = np.argmin(pvalues)

    return min_pvalue, min_pos, start


def patch_source_eval_v1(prnu_profil, witness_distribution, residual, size_x, size_y):
    """
    Evaluates a patch source using witness correlation distribution.

    Args:
        prnu_profil (numpy.ndarray): Reference PRNU profile.
        profil_witness (numpy.ndarray): Witness profile for correlation distribution.
        residual (numpy.ndarray): Residual image data.
        size_x (int): Width of the cropped profile.
        size_y (int): Height of the cropped profile.

    Returns:
        tuple: (P-value, position of max correlation, start position of cropped profile)
    """
    if len(prnu_profil) < size_x:
        raise ValueError("PRNU profile length is too short for the given crop size.")

    np.random.seed(1)  # Ensure reproducibility

    # # Get witness correlation distribution
    # witness_distribution = get_witness_corr_distribution(prnu_profil, profil_witness, size_x)

    # Extract residual profile
    residual_sample, start = get_profil_crop(residual, size_x, size_y)

    # Precompute mean and standard deviation for residual sample
    mean_residual = np.mean(residual_sample)
    std_residual = np.std(residual_sample)

    # Number of comparisons
    num_samples = len(prnu_profil) - size_x

    # Pre-allocate NumPy array for correlation coefficients
    corrs = np.empty(num_samples, dtype=np.float64)

    # Compute correlation for each sample
    for i in range(num_samples):
        sample_ref = prnu_profil[i:i + size_x]

        # Compute mean and standard deviation for the PRNU sample
        mean_prnu = np.mean(sample_ref)
        std_prnu = np.std(sample_ref)

        # Compute Pearson correlation manually (faster than np.corrcoef in loops)
        corrs[i] = np.sum((sample_ref - mean_prnu) * (residual_sample - mean_residual)) / (
                size_x * std_prnu * std_residual)

    # Compute max correlation and its position
    max_corr = np.max(corrs)
    max_pos = np.argmax(corrs)

    # Get p-value
    pvalue = get_pvalue(witness_distribution, max_corr)

    return pvalue, max_pos, start
