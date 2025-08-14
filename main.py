import os
import iio
import numpy as np

from utils.utils import *


# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))


def main(reference, test, crop_size, filter, crop=False):

    # Compute profiles
    s1, _, _ = get_profil(reference, filter)
    s2, x_start, y_start = get_profil(test, filter, crop)

    # Compare signals
    if len(s1) == len(s2):
        p_value_gauss, p_value_rank, list_corr, list_corr_ij = prnu_similarity_profil(s1, s2, crop_size)
    else:
        results = align_and_crop(s1, s2, normalize=True, use_fft=True, return_corr=False)
        s1_crop = results['s1_crop']
        s2_crop = results['s2_crop']
        offset = results['lag']
        p_value_gauss, p_value_rank, list_corr, list_corr_ij = prnu_similarity_profil(s1_crop, s2_crop, crop_size)

    print(f'p_value Gauss test: {p_value_gauss}')
    print(f'p_value Rank test: {p_value_rank}')
    print(f'best offset: {offset}')

    if p_value_gauss < 1e-6:
        print(f'Images come from the same source')
    else:
        print(f'Images come from a different source')

    # Create outputs
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    save_histo_gauss(list_corr, list_corr_ij, p_value_gauss)
    save_histo_rank(list_corr, list_corr_ij, p_value_rank)
    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--reference", type=str, required=True)
    parser.add_argument("-t", "--test", type=str, required=True)
    parser.add_argument("-s", "--crop_size", type=int, default=20, required=False)
    parser.add_argument("-f", "--filter", type=str, default='rank', required=False)
    parser.add_argument("-c", "--crop", type=bool, default=False, required=False)

    args = parser.parse_args()
    main(args.reference, args.test, args.crop_size, args.filter)
