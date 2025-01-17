import os
import iio
import numpy as np

from utils.utils import *

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

    # Display histograms
    # Merged
    plt.hist(list_corr, bins='auto', alpha=0.7, edgecolor='black', label=r"$\rho(P^r_i, P^p_j):i \neq j$")
    plt.hist(list_corr_ij, bins='auto', alpha=0.7, edgecolor='black', label=r"$\rho(P^r_i, P^p_j):i=j$")
    plt.title(f'p_value = {p_value:.2e}')
    plt.legend(loc='upper right')
    plt.savefig('./outputs/histo.png')
    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("-t", "--test", type=str, required=True)
    parser.add_argument("-s", "--crop_size", type=int, default=200, required=False)

    args = parser.parse_args()
    main(args.reference, args.test, args.crop_size)
