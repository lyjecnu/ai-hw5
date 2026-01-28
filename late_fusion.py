import torch
import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2, MobileNet_V2_Weights
from transformers import BertModel, BertTokenizer
from load_data import get_data_loader
from train_validate import train_model,validate


class LateFusionSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, fusion_method='weighted_sum', dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method


        # --- 图像分支 (MobileNetV2) ---
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.img_backbone = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # 确保输出为 [B, 1280, 1, 1]
            nn.Flatten(start_dim=1)  # 展平为 [B, 1280]
        )
        self.img_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)  # MobileNetV2 的特征维度是 1280
        )

        # --- 文本分支 ---
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.txt_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )

        # --- 晚期融合头（可选）---
        if fusion_method == 'concat_then_project':
            self.fusion_layer = nn.Linear(num_classes * 2, num_classes)
        elif fusion_method == 'learned_weight':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self,image, input_ids, attention_mask):
        # 图像路径
        img_feat = self.img_backbone(image)  # [B, 512, 1, 1]
        img_feat = img_feat.flatten(1)  # [B, 512]
        img_logits = self.img_classifier(img_feat)  # [B, 3]

        # 文本路径
        txt_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = txt_out.last_hidden_state[:, 0]  # [B, 768]
        txt_logits = self.txt_classifier(cls_repr)  # [B, 3]

        # 晚期融合（在 logits 空间）
        if self.fusion_method == 'average':
            fused_logits = (img_logits + txt_logits) / 2
        elif self.fusion_method == 'weighted_sum':
            # 固定权重（也可设为超参数）
            fused_logits = 0.4 * img_logits + 0.6 * txt_logits
        elif self.fusion_method == 'learned_weight':
            alpha = torch.sigmoid(self.alpha)  # 限制在 [0,1]
            fused_logits = alpha * img_logits + (1 - alpha) * txt_logits
        elif self.fusion_method == 'concat_then_project':
            concat_logits = torch.cat([img_logits, txt_logits], dim=1)  # [B, 6]
            fused_logits = self.fusion_layer(concat_logits)
        else:
            raise ValueError("Unsupported fusion method")

        return fused_logits


if __name__ == '__main__':
    device = torch.device("cuda")
    train_loader,val_loader = get_data_loader()  # 你的数据加载器

    model = LateFusionSentimentClassifier(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        num_epochs=10,
        device=device,
        log_interval=50,
        save_path="model_params/late_fusion.pth"
    )


    model = LateFusionSentimentClassifier()
    model.to(device)
    checkpoint = torch.load("model_params/late_fusion.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    acc,loss = validate(model, val_loader, device)
    print(f"validate accuracy:{acc}")
    print(f"validate loss:{loss}")