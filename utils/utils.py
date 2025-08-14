import os

import numpy as np
import rasterio

import matplotlib.pyplot as plt

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


@njit(fastmath=True)
def get_rank(center_val, window):
    count = 0
    for i in prange(window.shape[0]):
        for j in prange(window.shape[1]):
            if window[i, j] < center_val:
                count += 1
    return count


@njit
def rank_1d(arr):
    n = len(arr)
    ranks = np.empty(n, dtype=np.int64)
    for i in range(n):
        rank = 0
        for j in range(n):
            if arr[j] < arr[i]:
                rank += 1
        ranks[i] = rank
    return ranks


@njit(parallel=True, fastmath=True)
def rank_filter(img, patch_size=3):
    pad = patch_size // 2
    H, W = img.shape
    ranked_img = np.zeros_like(img)

    # Manual reflect padding
    for i in prange(H):
        for j in prange(W):
            # Create the patch manually
            window = np.empty((patch_size, patch_size), dtype=img.dtype)
            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    y = i + dy
                    x = j + dx

                    # Reflect padding logic
                    if y < 0:
                        y = -y
                    elif y >= H:
                        y = 2 * H - y - 2

                    if x < 0:
                        x = -x
                    elif x >= W:
                        x = 2 * W - x - 2

                    window[dy + pad, dx + pad] = img[y, x]

            ranked_img[i, j] = get_rank(img[i, j], window)

    return ranked_img


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


def get_profil(path, filter):
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

            # Apply filtering
            if filter == '2-means':
                filtered_img = k_mean(img, k=2)
            elif filter == 'rank':
                filtered_img = rank_filter(img)

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


def ks_rank_test(list_corr, list_corr_i_j):
    aligned = np.ones(len(list_corr_i_j))
    misaligned = np.zeros(len(list_corr))

    all_values = np.concatenate((list_corr, list_corr_i_j))
    index_aligned = np.concatenate((misaligned, aligned))
    rank_corr = rank_1d(all_values)

    rank_aligned = rank_corr[np.where(index_aligned)] / len(all_values)
    # rank_misaligned = rank_corr[np.where(index_aligned == 0)]

    # Perform the KS test against the uniform distribution on [0, 1]
    _, p_value = stats.kstest(rank_aligned, 'uniform')

    return p_value


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
    rank_pvalue = ks_rank_test(list_corr, list_corr_i_j)

    return ks_pvalue, rank_pvalue, list_corr, list_corr_i_j


def align_and_crop(sig1, sig2, normalize=True, use_fft=True, return_corr=False):
    """
    Align two 1D signals of potentially different lengths by maximizing
    their cross-correlation, then return the overlapping (cropped) region
    after alignment.

    Parameters
    ----------
    sig1, sig2 : array-like
        1D input signals.
    normalize : bool, default True
        If True, apply z-score normalization (mean=0, std=1) to make correlation
        insensitive to scale or DC offset.
    use_fft : bool, default True
        If True and SciPy is available, use FFT-based cross-correlation (faster).
    return_corr : bool, default False
        If True, also return the correlation curve and lags array.

    Returns
    -------
    result : dict
        {
          'lag': int,                # optimal shift (sig2 relative to sig1)
          's1_crop': np.ndarray,     # cropped portion of sig1 after alignment
          's2_crop': np.ndarray,     # cropped portion of sig2 after alignment
          'idx1': (start1, end1),    # indices used from sig1 (slice [start1:end1])
          'idx2': (start2, end2),    # indices used from sig2 (slice [start2:end2])
          'corr_peak': float,        # correlation value at optimal lag
          # Optional if return_corr=True:
          'corr': np.ndarray,
          'lags':  np.ndarray,
        }
    """
    s1 = np.asarray(sig1, dtype=float)
    s2 = np.asarray(sig2, dtype=float)

    # Optional z-score normalization
    if normalize:
        def zscore(x):
            std = np.std(x)
            return (x - np.mean(x)) / std if std > 0 else x - np.mean(x)

        s1n, s2n = zscore(s1), zscore(s2)
    else:
        s1n, s2n = s1, s2

    # Cross-correlation (full mode)
    try:
        from scipy.signal import correlate
        method = 'fft' if use_fft else 'direct'
        corr = correlate(s1n, s2n, mode='full', method=method)
    except Exception:
        # Fallback without SciPy (O(n^2))
        corr = np.correlate(s1n, s2n, mode='full')

    lags = np.arange(-len(s2n) + 1, len(s1n))
    best_idx = int(np.argmax(corr))
    best_lag = int(lags[best_idx])
    corr_peak = float(corr[best_idx])

    # Compute overlap indices after alignment
    start1 = max(best_lag, 0)
    start2 = max(-best_lag, 0)
    L = max(0, min(len(s1) - start1, len(s2) - start2))

    s1_crop = s1[start1:start1 + L]
    s2_crop = s2[start2:start2 + L]

    result = {
        'lag': best_lag,
        's1_crop': s1_crop,
        's2_crop': s2_crop,
        'idx1': (start1, start1 + L),
        'idx2': (start2, start2 + L),
        'corr_peak': corr_peak,
    }
    if return_corr:
        result['corr'] = corr
        result['lags'] = lags
    return result


def save_histo_gauss(list_missaligned, list_aligned, pvalue):
    # Set up number of bins
    num_bin = 50
    bin_lims = np.linspace(-1, 1, num_bin + 1)
    bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    bin_widths = bin_lims[1:] - bin_lims[:-1]

    # Computing histograms
    hist1, _ = np.histogram(list_missaligned, bins=bin_lims)
    hist2, _ = np.histogram(list_aligned, bins=bin_lims)

    # Normalizing
    hist1b = hist1 / np.max(hist1)
    hist2b = hist2 / np.max(hist2)

    # Plot
    plt.bar(bin_centers, hist1b, width=bin_widths, align='center', alpha=0.7, edgecolor='black',
            label=r"$\rho(P^r_i, P^p_j):i \neq j$")
    plt.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.7, edgecolor='black',
            label=r"$\rho(P^r_i, P^p_i):i=1,...K$")
    plt.title(f'p_value = {pvalue:.2e}')
    plt.legend(loc='upper right')
    plt.savefig('./outputs/histo.png')


def save_histo_rank(list_missaligned, list_aligned, pvalue):
    aligned = np.ones(len(list_aligned))
    misaligned = np.zeros(len(list_missaligned))

    all_values = np.concatenate((list_missaligned, list_aligned))
    index_aligned = np.concatenate((misaligned, aligned))
    rank_corr = rank_1d(all_values)

    rank_aligned = rank_corr[np.where(index_aligned)]
    # rank_misaligned = rank_corr[np.where(index_aligned == 0)]

    # Define histogram parameters
    num_bins = 50
    bin_edges = np.linspace(0, max(rank_aligned), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Compute histograms
    hist1, _ = np.histogram(rank_aligned, bins=bin_edges)

    plt.bar(bin_centers, hist1, width=bin_widths, align='center', alpha=0.7, edgecolor='black',
            label=r"$\rho(P^r_i, P^p_j): i = j$")
    plt.legend(loc='best')
    plt.savefig('./outputs/histo_rank.png')


def get_sensor_id(path):
    return path.split('/')[-1].split('_')[8]


def get_band_id(path):
    return path.split('/')[-1].split('.')[0].split('_')[-1]
