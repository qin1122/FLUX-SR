# coding=utf-8

import argparse
import glob
import os
from PIL import Image


def main(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = args.scale

    for scale in scale_list:
        os.makedirs(os.path.join(args.output+str(scale)), exist_ok=True)

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            rlt = img.resize(
                (int(width / scale), int(height / scale)), resample=Image.LANCZOS)
            print(rlt.size)
            rlt.save(os.path.join(args.output+str(scale), f'{basename}.png'))


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='./datasets/DIV2K_train_HR', help='Input folder')
    parser.add_argument(
        '--output', type=str, default='./datasets/DIV2K_train_LR_x', help='Output folder')
    parser.add_argument('--scale', type=int,
                        default=4, nargs='+', help='Scale factor')
    args = parser.parse_args()

    main(args)
