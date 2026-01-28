import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, MobileNet_V2_Weights, \
    mobilenet_v2
from load_data import get_data_loader
from train_validate import train_model,validate
from thop import profile

class SimpleConcatMultimodalModel(nn.Module):
    def __init__(self, num_classes=3, freeze_vision=False, freeze_text=False):
        super().__init__()

        # --- 图像编码器：MobileNetV2 ---
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        self.vision_model = mobilenet_v2(weights=weights)
        # MobileNetV2 的特征输出在 classifier[1] 之前，维度是 1280
        self.vision_model.classifier = nn.Identity()  # 输出 [B, 1280]
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # --- 文本编码器 ---
        self.text_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        if freeze_text:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # --- 融合与分类头 ---
        combined_dim = 1280 + 768  # MobileNetV2 + BERT [CLS]
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # 图像特征
        img_feat = self.vision_model(image)  # [B, 2048]

        # 文本特征：取 [CLS] token 的输出
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = text_outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # 简单拼接
        fused = torch.cat([img_feat, txt_feat], dim=1)  # [B, 1280+768]

        # 分类
        logits = self.classifier(fused)  # [B, 3]
        return logits

if __name__ == '__main__':
    # 1. 准备组件
    device = torch.device("cuda")
    train_loader,val_loader = get_data_loader()  # 你的数据加载器

    model = SimpleConcatMultimodalModel(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


    # 2. 调用通用训练函数
    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        num_epochs=10,
        device=device,
        log_interval=50,
        save_path="model_params/simple_concat.pth"
    )



    model = SimpleConcatMultimodalModel()
    model.to(device)
    checkpoint = torch.load("model_params/simple_concat.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    acc,loss = validate(model,val_loader,device)
    print(f"validate accuracy:{acc}")
    print(f"validate loss:{loss}")