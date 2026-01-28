import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from train_validate import train_clip, validate, clip_validate

# 导入你提供的数据加载函数
from load_data import get_data_loader,get_clip_loader

class FineTunedCLIPModel(nn.Module):
    def __init__(self, num_classes=3, freeze_clip=True):
        super(FineTunedCLIPModel, self).__init__()
        # 加载预训练 CLIP 模型（默认 ViT-B/32）
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # 可选：冻结 CLIP 编码器参数（仅训练分类头）
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 获取图像和文本嵌入维度（CLIP 输出为 512 维）
        embed_dim = self.clip_model.config.projection_dim  # 通常是 512

        # 分类头：融合图像和文本特征后进行分类
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取 CLIP 的图像和文本嵌入（已归一化并投影到联合空间）
        outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        image_embeds = outputs.image_embeds  # [B, 512]
        text_embeds = outputs.text_embeds    # [B, 512]

        # 拼接图像和文本特征
        combined = torch.cat([image_embeds, text_embeds], dim=1)  # [B, 1024]

        # 分类
        logits = self.classifier(combined)  # [B, num_classes]
        return logits

if __name__ == '__main__':
    device = torch.device("cuda")
    train_loader,val_loader = get_clip_loader()  # 你的数据加载器

    model = FineTunedCLIPModel(num_classes=3,freeze_clip=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_clip(model=model, optimizer=optimizer, dataloader=train_loader, num_epochs=20, device=device, log_interval=50,
               save_path="model_params/fine_tuned_clip.pth")


    model = FineTunedCLIPModel()
    model.to(device)
    checkpoint = torch.load("model_params/fine_tuned_clip.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)


    acc,loss = clip_validate(model, val_loader, device)
    print(f"validate accuracy:{acc}")
    print(f"validate loss:{loss}")