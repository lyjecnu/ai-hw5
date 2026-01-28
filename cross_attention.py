import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

from load_data import get_data_loader
from train_validate import train_model,validate
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        # text_feat: [B, L_txt, D]
        # img_feat:  [B, L_img, D]  (e.g., L_img = 49 for 7x7 grid)
        attn_output, _ = self.cross_attn(
            query=text_feat,
            key=img_feat,
            value=img_feat
        )
        # 可选：残差连接 + LayerNorm
        fused = self.layer_norm(text_feat + attn_output)
        return fused  # [B, L_txt, D]


class MultimodalSentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, embed_dim=768, dropout=0.3):
        super().__init__()
        # Text encoder
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        self.text_proj = nn.Linear(768, embed_dim)

        # Image encoder (MobileNetV2 outputs 1280 channels)
        self.img_encoder = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.img_proj = nn.Linear(1280, embed_dim)

        # Fusion
        self.cross_attn = CrossAttentionFusion(embed_dim, num_heads=4)

        # Classifier (uses [CLS] token at position 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        # Text: [B, L] -> [B, L, 768] -> [B, L, embed_dim]
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_out.last_hidden_state)  # [B, L_txt, D]

        # Image
        img_feat = self.img_encoder(image)  # [B, 1280, 7, 7]
        B, C, H, W = img_feat.shape
        img_feat = img_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        img_feat = self.img_proj(img_feat)  # [B, 49, embed_dim]

        # Cross-attention: text queries attend to image regions
        fused_text = self.cross_attn(text_feat, img_feat)  # [B, L_txt, D]

        # Use [CLS] token (assumes input_ids starts with [CLS])
        cls_repr = fused_text[:, 0, :]  # [B, embed_dim]

        logits = self.classifier(self.dropout(cls_repr))
        return logits

if __name__ == '__main__':
    device = torch.device("cuda")
    train_loader,val_loader = get_data_loader()  # 你的数据加载器

    model = MultimodalSentimentClassifier(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        num_epochs=10,
        device=device,
        log_interval=50,
        save_path="model_params/cross_attention.pth"
    )


    model = MultimodalSentimentClassifier()
    model.to(device)
    checkpoint = torch.load("model_params/cross_attention.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    acc,loss = validate(model, val_loader, device)
    print(f"validate accuracy:{acc}")
    print(f"validate loss:{loss}")