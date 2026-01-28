import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

from load_data import get_data_loader
from train_validate import train_model,validate
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights



class TextOnlySentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, embed_dim=768, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image=None, input_ids=None, attention_mask=None):
        # 忽略 image 输入
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = text_out.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(self.dropout(cls_repr))
        return logits

class ImageOnlySentimentClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.img_encoder = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        # Global average pooling instead of flattening all spatial tokens
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 1280, 1, 1]
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, image=None, input_ids=None, attention_mask=None):
        # 忽略 text inputs
        img_feat = self.img_encoder(image)  # [B, 1280, 7, 7]
        pooled = self.global_pool(img_feat).flatten(1)  # [B, 1280]
        logits = self.classifier(self.dropout(pooled))
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loader()
    # --- Ablation 1: Text Only ---
    print("Training Text-Only Model...")
    text_model = TextOnlySentimentClassifier(num_classes=3).to(device)
    optimizer_text = torch.optim.AdamW(text_model.parameters(), lr=2e-5, weight_decay=1e-4)
    train_model(
        model=text_model,
        optimizer=optimizer_text,
        dataloader=train_loader,
        num_epochs=10,
        device=device,
        log_interval=50,
        save_path="model_params/text_only.pth"
    )

    acc_text, loss_text = validate(text_model, val_loader, device)
    print(f"[Text-Only] Val Acc: {acc_text:.4f}, Val Loss: {loss_text:.4f}")

    # --- Ablation 2: Image Only ---
    print("Training Image-Only Model...")
    img_model = ImageOnlySentimentClassifier(num_classes=3).to(device)
    optimizer_img = torch.optim.AdamW(img_model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_model(
        model=img_model,
        optimizer=optimizer_img,
        dataloader=train_loader,
        num_epochs=10,
        device=device,
        log_interval=50,
        save_path="model_params/image_only.pth"
    )

    acc_img, loss_img = validate(img_model, val_loader, device)
    print(f"[Image-Only] Val Acc: {acc_img:.4f}, Val Loss: {loss_img:.4f}")

