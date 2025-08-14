import os
import iio
import numpy as np

from utils.utils import *
import matplotlib.pyplot as plt

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))


def main(reference, test, crop_size, filter):

    # Compute profiles
    s1 = get_profil(reference, filter)
    s2 = get_profil(test, filter)

    # Crop
    s1 = s1
    s2 = s2[1000:2000]

    # Compare signals
    if len(s1) == len(s2):
        p_value, list_corr, list_corr_ij = prnu_similarity_profil(s1, s2, crop_size)
    else:
        results = align_and_crop(s1, s2, normalize=True, use_fft=True, return_corr=False)
        s1_crop = results['s1_crop']
        s2_crop = results['s2_crop']
        offset = results['lag']
        p_value, list_corr, list_corr_ij = prnu_similarity_profil(s1_crop, s2_crop, crop_size)

    print(f'p_value: {p_value}')
    print(f'best offset: {offset}')

    if p_value < 1e-6:
        print(f'Images come from the same source')
    else:
        print(f'Images come from a different source')

    # Create outputs
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # Set up number of bins
    num_bin = 50
    bin_lims = np.linspace(-1, 1, num_bin + 1)
    bin_centers = 0.5 * (bin_lims[:-1] + bin_lims[1:])
    bin_widths = bin_lims[1:] - bin_lims[:-1]

    # Computing histograms
    hist1, _ = np.histogram(list_corr, bins=bin_lims)
    hist2, _ = np.histogram(list_corr_ij, bins=bin_lims)

    # Normalizing
    hist1b = hist1 / np.max(hist1)
    hist2b = hist2 / np.max(hist2)

    # Plot
    plt.bar(bin_centers, hist1b, width=bin_widths, align='center', alpha=0.7, edgecolor='black', label=r"$\rho(P^r_i, P^p_j):i \neq j$")
    plt.bar(bin_centers, hist2b, width=bin_widths, align='center', alpha=0.7, edgecolor='black', label=r"$\rho(P^r_i, P^p_i):i=1,...K$")
    plt.title(f'p_value = {p_value:.2e}')
    plt.legend(loc='upper right')
    plt.savefig('./outputs/histo.png')

    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("-t", "--test", type=str, required=True)
    parser.add_argument("-s", "--crop_size", type=int, default=20, required=False)
    parser.add_argument("-f", "--filter", type=str, default='rank', required=False)

    args = parser.parse_args()
    main(args.reference, args.test, args.crop_size, args.filter)
