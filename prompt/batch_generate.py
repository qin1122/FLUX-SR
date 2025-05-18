import openai
import base64
import os
import time

# OpenAI API 配置
api_key = "sk-sztWT24Dm7AU4oC8e39QFTCh65cJKpRT7HTkWoBl035RAo9j"
base_url = "https://aihubmax.com/v1"  # 确保 base_url 正确
client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=600)

# 图片文件夹路径
image_folder = "/Users/qinhaonan/Downloads/image_reso/DIV2K_train_missing_"
output_folder = "/Users/qinhaonan/Downloads/image_reso/DIV2K_train_missing"
error_path = "/Users/qinhaonan/Downloads/image_reso/DIV2K_train_errors.txt"
folders = [1]
for i in folders:
    os.makedirs(output_folder+str(i)+"_result", exist_ok=True)


def encode_image(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_image(image_path):
    """调用 GPT API 处理图片，生成文本描述"""
    base64_image = encode_image(image_path)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe the components of this image and their positions within the image. I want the answer to be used as a prompt.Require generating a piece of plain text without line breaks."},
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


# 遍历文件夹中的所有图片
for i in folders:
    for filename in os.listdir(os.path.join(image_folder+str(i))):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(image_folder+str(i), filename)
            print(f"Processing {filename}...")
            prompt_text = process_image(image_path)
            if prompt_text:
                output_file = os.path.join(
                    output_folder+str(i)+"_result", f"{os.path.splitext(filename)[0]}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                print(f"Saved: {output_file}")

            # # 添加休眠时间，防止请求过载
            # time.sleep(2)  # 休眠2秒，可根据需要调整

print("Batch processing complete!")
