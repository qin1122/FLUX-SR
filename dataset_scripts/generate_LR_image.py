import argparse
import glob
import os
from PIL import Image


def main(args):
    # For DF2K, we consider the following three scales,
    # and the smallest image whose shortest edge is 400
    scale_list = [1/8]
    shortest_edge = 200

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output_fix_size, exist_ok=True)

    path_list = sorted(glob.glob(os.path.join(args.input, '*')))
    for path in path_list:
        print(path)
        basename = os.path.splitext(os.path.basename(path))[0]

        img = Image.open(path)
        width, height = img.size
        for idx, scale in enumerate(scale_list):
            rlt = img.resize(
                (int(width * scale), int(height * scale)), resample=Image.LANCZOS)
            print(rlt.size)
            rlt.save(os.path.join(args.output, f'{basename}.png'))

        # save the smallest image which the shortest edge is 400
        if width < height:
            ratio = height / width
            width = shortest_edge
            height = int(width * ratio)
        else:
            ratio = width / height
            height = shortest_edge
            width = int(height * ratio)
        rlt = img.resize((int(width), int(height)), resample=Image.LANCZOS)
        print(rlt.size)
        rlt.save(os.path.join(args.output_fix_size, f'{basename}.png'))


if __name__ == '__main__':
    """Generate multi-scale versions for GT images with LANCZOS resampling.
    It is now used for DF2K dataset (DIV2K + Flickr 2K)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_HR', help='Input folder')
    parser.add_argument(
        '--output', type=str, default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_LR_x8', help='Output folder')
    parser.add_argument('--output_fix_size', type=str,
                        default='/root/Homeworks/NLP/FLUX_SR/datasets/DIV2K_train_LR_x8_fixed', help='Output folder for fixed size')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    main(args)
