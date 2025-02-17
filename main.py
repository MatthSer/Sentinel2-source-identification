import os
import iio
import numpy as np

from utils.utils import *
import matplotlib.pyplot as plt

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))


def main(refence, test, crop_size):
    # Compute profils
    prnu_ref = get_profil(refence)
    prnu_test = get_profil(test)

    # Compare signals
    p_value, list_corr, list_corr_ij = prnu_similarity_profil(prnu_ref, prnu_test, crop_size)
    print(f'p_value: {p_value}')

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

    args = parser.parse_args()
    main(args.reference, args.test, args.crop_size)
