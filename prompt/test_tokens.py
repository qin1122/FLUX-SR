import os
import openai
import base64
from transformers import CLIPTokenizer

# 配置 OpenAI API Key
api_key = "sk-b6HJ632M3ZxNeG7BsbHuPOcnXFxnYrSTBop8SkWfEgSARSSl"
base_url = "https://aihubmax.com/v1"  # 确保 base_url 正确
client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=600)

# 加载 CLIP 的 tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def count_tokens(text):
    """
    计算文本的 tokens 数量。
    """
    tokenized = tokenizer(text, truncation=False)  # 不截断，计算完整 token
    return len(tokenized["input_ids"])


def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_new_prompt(image_path, old_prompt):
    """调用 GPT API 处理图片，生成文本描述"""
    base64_image = encode_image(image_path)
    prompt = "\""+old_prompt+"\"" + \
        "This prompt is too long. Please refer to the image, shorten it to within 77 tokens while preserving key details and location information.Require generating a piece of plain text without line breaks."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            top_p=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        with open(error_path, "a", encoding="utf-8") as f:
            f.write(f"Error processing image {image_path}: {e}\n")
        return None


# 文件路径
root_folder = "/root/image_reso/dataset/DIV2K_train_prompt_short"  # 替换为你的文本文件所在目录
image_folder = "/root/image_reso/dataset/DIV2K_train_HR"
output_log_path = "/root/image_reso/dataset/DIV2K_train_need_to_short.txt"
error_path = "/root/image_reso/dataset/error.txt"

# 处理所有文本文件
for subdir, _, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".txt"):  # 只处理 .txt 文件
            file_path = os.path.join(subdir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # 计算 token 数量
            token_count = count_tokens(content)
            if token_count > 77:
                # 生成对应的图片路径 (假设图片后缀是 .jpg)
                image_path = os.path.join(
                    image_folder, file.replace(".txt", ".png"))
                # image_path = file_path.replace(".txt", ".jpg")  # 根据实际情况调整后缀

                # 检查图片是否存在
                if os.path.exists(image_path):
                    new_prompt = generate_new_prompt(image_path, content)

                    # 重新写入新的 prompt
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_prompt)

                    # 记录超出限制的文件
                    with open(output_log_path, "a", encoding="utf-8") as log_f:
                        log_f.write(f"{file} 需要修改, 旧 Token: {token_count}\n")

                    print(f"{file} 处理完成，已生成新描述。")
                else:
                    print(f"警告: {image_path} 不存在，跳过。")
