import os
import argparse
from datasets import Dataset, Image
from PIL import Image as PILImage
from tqdm import tqdm


def build_hf_dataset(control_dir, target_dir, caption_dir, output_dir):
    # 获取文件列表并排序（确保匹配）
    control_files = sorted(os.listdir(control_dir))
    target_files = sorted(os.listdir(target_dir))
    caption_files = sorted(os.listdir(caption_dir))

    # 检查数量是否匹配
    assert len(control_files) == len(caption_files), "图片和文本数量不匹配！"

    # 读取数据
    data = []
    for con_file, img_file, txt_file in tqdm(zip(control_files, target_files, caption_files), total=len(target_files)):
        con_path = os.path.join(control_dir, con_file)
        img_path = os.path.join(target_dir, img_file)
        txt_path = os.path.join(caption_dir, txt_file)

        # 读取文本
        with open(txt_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()

        # 读取图片（PIL 格式）
        image = PILImage.open(img_path).convert("RGB")
        control = PILImage.open(con_path).convert("RGB")

        data.append({"image": image, "text": caption, "control": control})

    # 转换为 Hugging Face 数据集
    dataset = Dataset.from_list(data).cast_column("image", Image())

    # 保存数据集
    dataset.save_to_disk(output_dir)

    # 打印信息
    print("Dataset saved to:", output_dir)
    print(dataset)
    print(dataset[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build HF dataset from images and captions")
    parser.add_argument('--control_dir', type=str, required=True,
                        help='Path to control images (e.g., LR)')
    parser.add_argument('--target_dir', type=str, required=True,
                        help='Path to target images (e.g., HR)')
    parser.add_argument('--caption_dir', type=str,
                        required=True, help='Path to text captions')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to save Hugging Face dataset')

    args = parser.parse_args()

    build_hf_dataset(args.control_dir, args.target_dir,
                     args.caption_dir, args.output_dir)
