import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForMaskedLM, CLIPProcessor


class ImageTextLabelDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None, label_map=None):
        """
        Args:
            data_dir (str): 图像和文本描述所在目录（包含 xxx.jpg 和 xxx.txt）
            label_file (str): train.txt 路径，格式: guid,tag
            transform (callable, optional): 图像预处理
            label_map (dict, optional): 如 {'negative': 0, 'neutral': 1, 'positive': 2}
        """
        self.data_dir = data_dir
        self.transform = transform

        # 读取标签文件
        df = pd.read_csv(label_file)
        self.guids = df['guid'].astype(str).tolist()
        self.tags = df['tag'].tolist()

        # 构建标签映射
        if label_map is None:
            unique_tags = sorted(set(self.tags))
            self.label_map = {tag: idx for idx, tag in enumerate(unique_tags)}
        else:
            self.label_map = label_map

        # 验证文件是否存在且可读
        self.valid_indices = []
        for i, guid in enumerate(self.guids):
            img_path = os.path.join(data_dir, f"{guid}.jpg")
            txt_path = os.path.join(data_dir, f"{guid}.txt")

            # 检查图像是否存在
            if not os.path.exists(img_path):
                print(f"Warning: Missing image file for guid={guid}")
                continue

            # 检查文本是否存在
            if not os.path.exists(txt_path):
                print(f"Warning: Missing text file for guid={guid}")
                continue

            # 尝试读取文本（先 UTF-8，再 GBK）
            readable = False
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    f.read()
                readable = True
            except UnicodeDecodeError:
                try:
                    with open(txt_path, 'r', encoding='gbk') as f:
                        f.read()
                    readable = True
                except Exception as e:
                    print(f"Warning: Cannot decode text file for guid={guid} ({e})")
            except Exception as e:
                print(f"Warning: Unexpected error reading text file for guid={guid} ({e})")

            if readable:
                self.valid_indices.append(i)
            else:
                print(f"Skipped guid={guid} due to unreadable caption.")


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        guid = self.guids[real_idx]
        tag = self.tags[real_idx]

        # 加载图像
        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 加载文本描述（保持原始字符串，分词在 collate_fn 中进行）
        # 加载文本描述
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        caption = ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        except UnicodeDecodeError:
            with open(txt_path, 'r', encoding='gbk') as f:
                caption = f.read().strip()


        if pd.isna(tag):
            label_int = 10
        else:
            label_int = self.label_map[tag]


        return {
            'image': image,
            'caption': caption,          # 原始字符串
            'label': label_int,
            'guid': guid
        }

def get_data_loader():
    # 图像变换
    # === 训练集使用的图像变换（含数据增强）===
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # === 验证集使用的图像变换（无增强，仅 resize + to tensor）===
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 标签映射
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def collate_fn(batch):
        """自定义 collate 函数，用于 DataLoader"""
        images = [item['image'] for item in batch]
        captions = [item['caption'] for item in batch]
        labels = [item['label'] for item in batch]
        guids = [item['guid'] for item in batch]

        # 堆叠图像
        images = torch.stack(images, dim=0)  # [B, 3, 224, 224]

        # 批量 tokenize captions（自动 padding 和 attention mask）
        encoded = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            'image': images,
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': labels,
            'guid': guids
        }

    # 创建完整数据集
    full_dataset = ImageTextLabelDataset(
        data_dir='data',
        label_file='train.txt',
        transform=train_transform,
        label_map=label_mapping
    )

    # 划分训练集和验证集（8:2）
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=generator)

    #为两个子数据集分别设置不同的transform
    # 因为 random_split 不复制 transform，我们需要手动覆盖
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,  # 验证时通常不打乱
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def get_clip_loader():
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    # 使用 CLIPProcessor 自动处理图像和文本
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 创建完整数据集（注意：不再传 transform）
    full_dataset = ImageTextLabelDataset(
        data_dir='data',
        label_file='train.txt',
        label_map=label_mapping
    )

    # 划分训练/验证集
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 自定义 collate_fn：使用 CLIPProcessor 批量处理
    def collate_fn(batch):
        guids = [item['guid'] for item in batch]
        images = [item['image'] for item in batch]           # List[PIL.Image]
        captions = [item['caption'] for item in batch]       # List[str]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        # CLIPProcessor 自动做：resize(224), normalize, tokenize, padding, etc.
        inputs = processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        return {
            'pixel_values': inputs['pixel_values'],      # [B, 3, 224, 224]
            'input_ids': inputs['input_ids'],            # [B, 77]
            'attention_mask': inputs['attention_mask'],  # [B, 77]
            'label': labels,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader



def get_clip_test_loader():
    # 注意：测试集无标签，所以 label_mapping 可能不会被使用，但仍保留以兼容 Dataset
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 加载测试集（无标签）
    test_dataset = ImageTextLabelDataset(
        data_dir='data',
        label_file='test_without_label.txt',
        label_map=label_mapping  # 如果 Dataset 能处理无标签情况，可保留；否则需修改 Dataset
    )

    def collate_fn(batch):
        guids = [item['guid'] for item in batch]
        images = [item['image'] for item in batch]      # List[PIL.Image]
        captions = [item['caption'] for item in batch]  # List[str]

        # 测试集无真实标签，所以不构造 labels 张量
        inputs = processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )

        return {
            'guid': guids,
            'pixel_values': inputs['pixel_values'],      # [B, 3, 224, 224]
            'input_ids': inputs['input_ids'],            # [B, 77]
            'attention_mask': inputs['attention_mask'],  # [B, 77]
            # 注意：没有 'label' 字段
        }

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return test_loader