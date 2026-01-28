import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple
import os

def read_caption_file(txt_path: str) -> str:
    """
    尝试用 utf-8 读取 .txt 文件，失败则尝试 gbk。
    返回去除首尾空白的字符串。
    """
    for encoding in ["utf-8", "gbk", "gb2312"]:
        try:
            with open(txt_path, "r", encoding=encoding) as f:
                content = f.read().strip()
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # 非编码错误（如文件不存在）直接抛出
            raise e
    # 如果所有编码都失败
    raise ValueError(f"Unable to decode text file {txt_path} with utf-8, gbk, or gb2312.")

# ----------------------------
# 初始化 CLIP 模型和处理器
# ----------------------------
# Load model directly
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ----------------------------
# 定义情感类别与 Prompt 模板
# ----------------------------
emotions = ["negative", "neutral", "positive"]
# 可尝试不同 prompt 提升效果（prompt engineering）
prompt_template = "a photo showing {} emotion"

# 为每个情感生成文本描述
texts = [prompt_template.format(emotion) for emotion in emotions]
print("Using prompts:", texts)


# ----------------------------
# 推理函数：输入图像路径 + 输入文本路径 → 输出情感预测
# ----------------------------
def predict_sentiment(
        image_path: str,
        caption_txt_path: str
) -> int:
    """
    使用 CLIP 进行 zero-shot 多模态情感分类，返回最可能的情感类别。

    Args:
        image_path (str): 图像文件路径
        caption_txt_path (str): 包含文本描述的 .txt 文件路径

    Returns:
        str: 预测的情感类别，取值为 "negative"、"neutral" 或 "positive"
    """
    # 读取 caption 文本
    if not os.path.isfile(caption_txt_path):
        raise FileNotFoundError(f"Caption file not found: {caption_txt_path}")

    caption = read_caption_file(caption_txt_path)
    if not caption:
        raise ValueError(f"Caption file is empty: {caption_txt_path}")

    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 构造多模态输入：将 caption 融入 prompt
    combined_texts = [f"{prompt}. {caption}" for prompt in texts]

    # CLIP 处理
    inputs = processor(
        text=combined_texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 获取 logits 和概率
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1, 3]
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]  # shape: (3,)

    # 找到概率最大的类别索引
    predicted_idx = int(probs.argmax())
    return predicted_idx

# ----------------------------
# 示例使用
# ----------------------------
if __name__ == "__main__":
    import csv

    # 配置路径（请根据你的实际目录结构调整）
    label_file = "train.txt"
    image_dir = "./data"  # 图像存放目录
    caption_dir = "./data"  # caption 文本存放目录

    total = 0
    correct = 0
    results = []  # 可选：保存详细结果用于调试

    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过 header: guid,tag

        for row in reader:
            if not row:
                continue
            guid = row[0].strip()
            true_label = row[1].strip()
            if true_label == 'negative':
                true_label = 0
            elif true_label == 'neutral':
                true_label = 1
            elif true_label == 'positive':
                true_label = 2

            # 构造文件路径
            img_path = os.path.join(image_dir, f"{guid}.jpg")
            txt_path = os.path.join(caption_dir, f"{guid}.txt")

            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            if not os.path.exists(txt_path):
                print(f"Warning: Caption not found: {txt_path}")
                continue

            try:
                pred_label = predict_sentiment(img_path, txt_path)
                total += 1
                if pred_label == true_label:
                    correct += 1
                results.append((guid, true_label, pred_label))
            except Exception as e:
                print(f"Error processing guid={guid}: {e}")
                continue

            if total % 500 == 0:
                print(f"predicted {total}")


    # 计算并输出准确率
    if total == 0:
        print("No valid samples processed.")

    else:
        accuracy = correct / total
        print(f"\nTotal samples: {total}")
        print(f"\nCorrect predictions: {correct}")
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")

        # 可选：打印错误案例
        # errors = [(g, t, p) for g, t, p in results if t != p]
        # print("\n❌ Errors:", errors)